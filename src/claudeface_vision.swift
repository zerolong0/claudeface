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

let ASCII_RAMP = " .,:;i1tfLCG08@"

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

func renderASCII(pixelBuffer: CVPixelBuffer, width: Int = 80, height: Int = 35) {
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
    let rampCount = ASCII_RAMP.count

    // Terminal characters are ~2x taller than wide, so we sample accordingly
    let scaleX = Double(imgWidth) / Double(width)
    let scaleY = Double(imgHeight) / Double(height)

    var lines: [String] = []

    for row in 0..<height {
        var lineChars: [Character] = []
        // Sample from image (top-to-bottom, but camera image is not flipped)
        let srcY = Int(Double(row) * scaleY)
        let clampedY = min(srcY, imgHeight - 1)

        for col in 0..<width {
            // Mirror horizontally so it looks like a mirror
            let srcX = imgWidth - 1 - Int(Double(col) * scaleX)
            let clampedX = max(0, min(srcX, imgWidth - 1))

            let offset = clampedY * bytesPerRow + clampedX * 4
            // BGRA format
            let b = Double(buffer[offset])
            let g = Double(buffer[offset + 1])
            let r = Double(buffer[offset + 2])

            // Luminance
            let luma = 0.299 * r + 0.587 * g + 0.114 * b
            let idx = min(Int(luma / 256.0 * Double(rampCount)), rampCount - 1)
            let char = ASCII_RAMP[ASCII_RAMP.index(ASCII_RAMP.startIndex, offsetBy: idx)]
            lineChars.append(char)
        }
        lines.append(String(lineChars))
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
