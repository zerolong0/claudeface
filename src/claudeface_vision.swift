/// ClaudeFace Vision - Native macOS face landmark detection + ASCII portrait.
///
/// Captures a single frame from the built-in camera via AVFoundation,
/// runs VNDetectFaceLandmarksRequest (Vision framework), and outputs
/// normalized landmark coordinates as JSON to stdout.
///
/// Build:  swiftc -O -o bin/claudeface-vision src/claudeface_vision.swift \
///             -framework AVFoundation -framework Vision -framework CoreMedia \
///             -framework CoreGraphics -framework CoreImage
///
/// Usage:  bin/claudeface-vision          → single detect, JSON to stdout
///         bin/claudeface-vision --ascii   → ASCII portrait to stdout
///         bin/claudeface-vision --ascii W H  → custom size (default 80x35)
///         bin/claudeface-vision --check   → permission/camera check only

import AVFoundation
import CoreGraphics
import CoreImage
import CoreMedia
import Foundation
import Vision

// MARK: - ASCII Art Character Ramp

// Unicode half-block characters for 2x vertical resolution pixel art
let BLOCK_FULL  = "\u{2588}" // █ both pixels bright
let BLOCK_UPPER = "\u{2580}" // ▀ top bright, bottom dark
let BLOCK_LOWER = "\u{2584}" // ▄ top dark, bottom bright
// space = both pixels dark

// MARK: - Frame Capturer (shared by detect + ascii modes)

final class FrameCapturer: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session = AVCaptureSession()
    private let semaphore = DispatchSemaphore(value: 0)
    private var frameCount = 0
    private let warmupFrames = 10
    private var pixelBufferResult: CVPixelBuffer?
    private var useHighRes: Bool

    init(highRes: Bool = true) {
        self.useHighRes = highRes
        super.init()
    }

    /// Capture a single frame and return the raw pixel buffer.
    func captureFrame() -> CVPixelBuffer? {
        let authStatus = AVCaptureDevice.authorizationStatus(for: .video)
        switch authStatus {
        case .denied, .restricted:
            fputs("[error] Camera access denied.\n", stderr)
            return nil
        case .notDetermined:
            let sem = DispatchSemaphore(value: 0)
            var granted = false
            AVCaptureDevice.requestAccess(for: .video) { g in
                granted = g
                sem.signal()
            }
            sem.wait()
            if !granted { return nil }
        default:
            break
        }

        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device)
        else {
            fputs("[error] No camera device found.\n", stderr)
            return nil
        }

        session.sessionPreset = useHighRes ? .high : .medium
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let queue = DispatchQueue(label: "claudeface.capture")
        output.setSampleBufferDelegate(self, queue: queue)
        session.addOutput(output)

        session.startRunning()
        let timeout = semaphore.wait(timeout: .now() + 8)
        session.stopRunning()

        if timeout == .timedOut {
            fputs("[error] Camera capture timed out.\n", stderr)
            return nil
        }

        return pixelBufferResult
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        frameCount += 1
        guard frameCount > warmupFrames else { return }

        if let videoOutput = output as? AVCaptureVideoDataOutput {
            videoOutput.setSampleBufferDelegate(nil, queue: nil)
        }

        pixelBufferResult = CMSampleBufferGetImageBuffer(sampleBuffer)
        semaphore.signal()
    }
}

// MARK: - Face Landmark Detection

func detectLandmarks(from pixelBuffer: CVPixelBuffer) -> [String: Any] {
    let request = VNDetectFaceLandmarksRequest()
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

    do {
        try handler.perform([request])
    } catch {
        return ["status": "error", "message": error.localizedDescription]
    }

    guard let observations = request.results, !observations.isEmpty else {
        return ["status": "no_face"]
    }

    let face = observations[0]
    var data: [String: Any] = ["status": "ok"]

    let box = face.boundingBox
    data["bbox"] = [
        round(Double(box.origin.x) * 10000) / 10000,
        round(Double(box.origin.y) * 10000) / 10000,
        round(Double(box.width) * 10000) / 10000,
        round(Double(box.height) * 10000) / 10000,
    ]

    if let lm = face.landmarks {
        var landmarks: [String: [[Double]]] = [:]

        func extract(_ name: String, _ region: VNFaceLandmarkRegion2D?) {
            guard let r = region else { return }
            landmarks[name] = r.normalizedPoints.map { pt in
                [round(Double(pt.x) * 10000) / 10000,
                 round(Double(pt.y) * 10000) / 10000]
            }
        }

        extract("leftEye", lm.leftEye)
        extract("rightEye", lm.rightEye)
        extract("leftEyebrow", lm.leftEyebrow)
        extract("rightEyebrow", lm.rightEyebrow)
        extract("outerLips", lm.outerLips)
        extract("innerLips", lm.innerLips)
        extract("nose", lm.nose)

        data["landmarks"] = landmarks
    }

    return data
}

// MARK: - ASCII Portrait Rendering

func renderASCII(pixelBuffer: CVPixelBuffer, width: Int = 80, height: Int = 40) {
    // Each terminal row encodes 2 pixel rows via half-block characters.
    let pixelH = height * 2

    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    let imgWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imgHeight = CVPixelBufferGetHeight(pixelBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        fputs("[error] Cannot access pixel buffer.\n", stderr)
        return
    }

    let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)

    // Block size: how many source pixels map to one target cell
    let blockW = Double(imgWidth) / Double(width)
    let blockH = Double(imgHeight) / Double(pixelH)

    // --- Step 1: Block-average sampling (reduces noise vs point sampling) ---
    var grid = [[Double]](repeating: [Double](repeating: 0, count: width), count: pixelH)

    for row in 0..<pixelH {
        let srcYStart = Int(Double(row) * blockH)
        let srcYEnd = min(Int(Double(row + 1) * blockH), imgHeight)

        for col in 0..<width {
            // Mirror horizontally (selfie mirror)
            let mirrorCol = width - 1 - col
            let srcXStart = Int(Double(mirrorCol) * blockW)
            let srcXEnd = min(Int(Double(mirrorCol + 1) * blockW), imgWidth)

            var sum: Double = 0
            var count: Double = 0
            for sy in srcYStart..<srcYEnd {
                for sx in srcXStart..<srcXEnd {
                    let offset = sy * bytesPerRow + sx * 4
                    let b = Double(buffer[offset])
                    let g = Double(buffer[offset + 1])
                    let r = Double(buffer[offset + 2])
                    sum += 0.299 * r + 0.587 * g + 0.114 * b
                    count += 1
                }
            }
            grid[row][col] = count > 0 ? sum / count : 0
        }
    }

    // --- Step 2: Face-region contrast (use center 50% for percentile calc) ---
    // Webcam portraits have the face in the center; using center region
    // prevents bright backgrounds from washing out facial detail.
    let roiY0 = pixelH / 6
    let roiY1 = pixelH * 5 / 6
    let roiX0 = width / 4
    let roiX1 = width * 3 / 4

    var faceValues: [Double] = []
    for row in roiY0..<roiY1 {
        for col in roiX0..<roiX1 {
            faceValues.append(grid[row][col])
        }
    }
    faceValues.sort()
    let lo = faceValues[max(0, Int(Double(faceValues.count) * 0.05))]
    let hi = faceValues[min(faceValues.count - 1, Int(Double(faceValues.count) * 0.95))]
    let range = max(hi - lo, 1.0)

    for row in 0..<pixelH {
        for col in 0..<width {
            // Normalize to 0-1
            var v = max(0, min(1, (grid[row][col] - lo) / range))
            // Single S-curve for moderate contrast boost
            v = v * v * (3.0 - 2.0 * v)
            grid[row][col] = v * 255.0
        }
    }

    // --- Step 3: Local adaptive threshold for detail preservation ---
    // Compare each pixel to its local neighborhood mean.
    // Pixels darker than local mean → black (feature), otherwise → white (skin/bg).
    let radius = max(width / 16, 3) // neighborhood radius
    let bias: Double = 12.0 // sensitivity: higher = less black detail, cleaner look

    // Build integral image for fast box-mean computation
    var integral = [[Double]](repeating: [Double](repeating: 0, count: width + 1), count: pixelH + 1)
    for row in 0..<pixelH {
        var rowSum: Double = 0
        for col in 0..<width {
            rowSum += grid[row][col]
            integral[row + 1][col + 1] = integral[row][col + 1] + rowSum
        }
    }

    var binary = [[Double]](repeating: [Double](repeating: 0, count: width), count: pixelH)
    for row in 0..<pixelH {
        for col in 0..<width {
            let y0 = max(0, row - radius)
            let y1 = min(pixelH, row + radius + 1)
            let x0 = max(0, col - radius)
            let x1 = min(width, col + radius + 1)
            let area = Double((y1 - y0) * (x1 - x0))
            let sum = integral[y1][x1] - integral[y0][x1] - integral[y1][x0] + integral[y0][x0]
            let localMean = sum / area
            binary[row][col] = grid[row][col] > (localMean - bias) ? 255 : 0
        }
    }
    // --- Step 4: Half-block character output ---
    var lines: [String] = []

    for termRow in 0..<height {
        let topRow = termRow * 2
        let botRow = topRow + 1
        var line = ""

        for col in 0..<width {
            let top = binary[topRow][col] > 127
            let bot = botRow < pixelH ? binary[botRow][col] > 127 : false

            if top && bot {
                line += BLOCK_FULL
            } else if top {
                line += BLOCK_UPPER
            } else if bot {
                line += BLOCK_LOWER
            } else {
                line += " "
            }
        }
        lines.append(line)
    }

    print(lines.joined(separator: "\n"))
}

// MARK: - Permission Check

func checkOnly() -> [String: Any] {
    let auth = AVCaptureDevice.authorizationStatus(for: .video)
    let hasCamera = AVCaptureDevice.default(for: .video) != nil

    return [
        "camera_available": hasCamera,
        "permission": "\(auth.rawValue)",
        "permission_label": {
            switch auth {
            case .authorized: return "authorized"
            case .denied: return "denied"
            case .restricted: return "restricted"
            case .notDetermined: return "not_determined"
            @unknown default: return "unknown"
            }
        }(),
    ]
}

// MARK: - JSON Output

func printJSON(_ dict: [String: Any]) {
    if let jsonData = try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys]),
       let jsonString = String(data: jsonData, encoding: .utf8)
    {
        print(jsonString)
    } else {
        print("{\"status\":\"error\",\"message\":\"json_encoding_failed\"}")
    }
}

// MARK: - Main

let args = CommandLine.arguments

if args.count > 1 && args[1] == "--check" {
    printJSON(checkOnly())
} else if args.count > 1 && args[1] == "--ascii" {
    // Parse optional width/height: --ascii [W] [H]
    let w = args.count > 2 ? Int(args[2]) ?? 80 : 80
    let h = args.count > 3 ? Int(args[3]) ?? 35 : 35

    let capturer = FrameCapturer(highRes: false) // medium res is enough for ASCII
    if let pb = capturer.captureFrame() {
        renderASCII(pixelBuffer: pb, width: w, height: h)
        _ = pb // ARC managed
    } else {
        fputs("[error] Failed to capture frame.\n", stderr)
    }
} else {
    // Default: face landmark detection (JSON output)
    let capturer = FrameCapturer(highRes: true)
    if let pb = capturer.captureFrame() {
        let result = detectLandmarks(from: pb)
        printJSON(result)
        _ = pb // ARC managed
    } else {
        printJSON(["status": "no_camera", "message": "Failed to capture frame."])
    }
}
