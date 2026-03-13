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
/// Usage:  bin/claudeface-vision              → single detect, JSON to stdout
///         bin/claudeface-vision --image       → auto-detect best terminal protocol
///         bin/claudeface-vision --image W H   → native image (iTerm2/Kitty/Sixel/ANSI)
///         bin/claudeface-vision --image W H proto → force protocol (iterm2/kitty/sixel/ansi)
///         bin/claudeface-vision --pixel W H   → ANSI half-block pixel art
///         bin/claudeface-vision --ascii W H   → B&W ASCII half-block portrait
///         bin/claudeface-vision --ascii-mini W H → Braille mini portrait
///         bin/claudeface-vision --check       → permission/camera check only

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

    // 3D head orientation
    if let roll = face.roll { data["roll"] = Double(truncating: roll) }
    if let yaw = face.yaw { data["yaw"] = Double(truncating: yaw) }
    if let pitch = face.pitch { data["pitch"] = Double(truncating: pitch) }

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
        extract("faceContour", lm.faceContour)
        extract("noseCrest", lm.noseCrest)
        extract("medianLine", lm.medianLine)
        extract("leftPupil", lm.leftPupil)
        extract("rightPupil", lm.rightPupil)

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

// MARK: - RGB + Color Pixel Art Rendering

struct RGB {
    var r: UInt8
    var g: UInt8
    var b: UInt8
}

/// Quantize a channel value to the given number of levels for pixel-art style.
func quantize(_ v: Double, levels: Int) -> UInt8 {
    let step = 255.0 / Double(levels - 1)
    return UInt8(min(255, max(0, Int(round(v / step) * step))))
}

/// Render a colored pixel-art portrait using ANSI 24-bit true color + half-block characters.
/// Uses adaptive threshold edge detection (from ASCII mode) to create clear outlines,
/// then fills non-edge areas with quantized color from the camera.
func renderPixelArt(
    pixelBuffer: CVPixelBuffer,
    faceBBox: [Double]?,  // [x, y, w, h] normalized (Vision coords, origin=bottom-left), nil=full frame
    width: Int = 60,
    height: Int = 30,
    quantizeLevels: Int = 12
) {
    let pixelH = height * 2  // half-block doubles vertical resolution

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

    // --- Step 1: Determine crop region (face bbox with padding, or full frame) ---
    var cropX: Int = 0
    var cropY: Int = 0
    var cropW: Int = imgWidth
    var cropH: Int = imgHeight

    if let bbox = faceBBox, bbox.count == 4 {
        let padding = 0.4
        let bx = bbox[0]
        let by = bbox[1]
        let bw = bbox[2]
        let bh = bbox[3]

        let centerX = bx + bw / 2.0
        let centerY = by + bh / 2.0
        let padW = bw * (1.0 + padding * 2)
        let padH = bh * (1.0 + padding * 2)

        let pixCX = centerX * Double(imgWidth)
        let pixCY = (1.0 - centerY) * Double(imgHeight)
        let pixW = padW * Double(imgWidth)
        let pixH = padH * Double(imgHeight)

        cropX = max(0, Int(pixCX - pixW / 2.0))
        cropY = max(0, Int(pixCY - pixH / 2.0))
        cropW = min(imgWidth - cropX, Int(pixW))
        cropH = min(imgHeight - cropY, Int(pixH))

        if cropW < 100 || cropH < 100 {
            cropX = 0; cropY = 0; cropW = imgWidth; cropH = imgHeight
        }
    }

    // --- Step 2: Block-average downsample into RGB + luminance grids ---
    let blockW = Double(cropW) / Double(width)
    let blockH = Double(cropH) / Double(pixelH)

    var gridR = [[Double]](repeating: [Double](repeating: 0, count: width), count: pixelH)
    var gridG = [[Double]](repeating: [Double](repeating: 0, count: width), count: pixelH)
    var gridB = [[Double]](repeating: [Double](repeating: 0, count: width), count: pixelH)
    var gridLum = [[Double]](repeating: [Double](repeating: 0, count: width), count: pixelH)

    for row in 0..<pixelH {
        let srcYStart = cropY + Int(Double(row) * blockH)
        let srcYEnd = min(cropY + Int(Double(row + 1) * blockH), cropY + cropH)

        for col in 0..<width {
            let mirrorCol = width - 1 - col
            let srcXStart = cropX + Int(Double(mirrorCol) * blockW)
            let srcXEnd = min(cropX + Int(Double(mirrorCol + 1) * blockW), cropX + cropW)

            var sumR: Double = 0, sumG: Double = 0, sumB: Double = 0
            var count: Double = 0
            for sy in srcYStart..<srcYEnd {
                for sx in srcXStart..<srcXEnd {
                    let offset = sy * bytesPerRow + sx * 4
                    sumB += Double(buffer[offset])
                    sumG += Double(buffer[offset + 1])
                    sumR += Double(buffer[offset + 2])
                    count += 1
                }
            }
            if count > 0 {
                let r = sumR / count
                let g = sumG / count
                let b = sumB / count
                gridR[row][col] = r
                gridG[row][col] = g
                gridB[row][col] = b
                gridLum[row][col] = 0.299 * r + 0.587 * g + 0.114 * b
            }
        }
    }

    // --- Step 3: Luminance contrast stretch (face-region percentiles) + S-curve ---
    let roiY0 = pixelH / 6
    let roiY1 = pixelH * 5 / 6
    let roiX0 = width / 4
    let roiX1 = width * 3 / 4

    var faceValues: [Double] = []
    for row in roiY0..<roiY1 {
        for col in roiX0..<roiX1 {
            faceValues.append(gridLum[row][col])
        }
    }
    faceValues.sort()
    let lo = faceValues[max(0, Int(Double(faceValues.count) * 0.05))]
    let hi = faceValues[min(faceValues.count - 1, Int(Double(faceValues.count) * 0.95))]
    let range = max(hi - lo, 1.0)

    for row in 0..<pixelH {
        for col in 0..<width {
            var v = max(0, min(1, (gridLum[row][col] - lo) / range))
            v = v * v * (3.0 - 2.0 * v)  // S-curve
            gridLum[row][col] = v * 255.0
        }
    }

    // --- Step 4: Adaptive threshold for edge detection (same as ASCII mode) ---
    let radius = max(width / 16, 3)
    let bias: Double = 12.0

    // Integral image for fast box-mean
    var integral = [[Double]](repeating: [Double](repeating: 0, count: width + 1), count: pixelH + 1)
    for row in 0..<pixelH {
        var rowSum: Double = 0
        for col in 0..<width {
            rowSum += gridLum[row][col]
            integral[row + 1][col + 1] = integral[row][col + 1] + rowSum
        }
    }

    var isEdge = [[Bool]](repeating: [Bool](repeating: false, count: width), count: pixelH)
    for row in 0..<pixelH {
        for col in 0..<width {
            let y0 = max(0, row - radius)
            let y1 = min(pixelH, row + radius + 1)
            let x0 = max(0, col - radius)
            let x1 = min(width, col + radius + 1)
            let area = Double((y1 - y0) * (x1 - x0))
            let sum = integral[y1][x1] - integral[y0][x1] - integral[y1][x0] + integral[y0][x0]
            let localMean = sum / area
            isEdge[row][col] = gridLum[row][col] <= (localMean - bias)
        }
    }

    // --- Step 5: Per-channel contrast stretch for color fill ---
    func channelStretch(_ grid: inout [[Double]]) {
        var vals: [Double] = []
        for row in roiY0..<roiY1 {
            for col in roiX0..<roiX1 {
                vals.append(grid[row][col])
            }
        }
        vals.sort()
        let clo = vals[max(0, Int(Double(vals.count) * 0.05))]
        let chi = vals[min(vals.count - 1, Int(Double(vals.count) * 0.95))]
        let cr = max(chi - clo, 1.0)
        for row in 0..<grid.count {
            for col in 0..<grid[row].count {
                grid[row][col] = max(0, min(255, (grid[row][col] - clo) / cr * 255.0))
            }
        }
    }

    channelStretch(&gridR)
    channelStretch(&gridG)
    channelStretch(&gridB)

    // Saturation boost (1.3x)
    for row in 0..<pixelH {
        for col in 0..<width {
            let r = gridR[row][col]
            let g = gridG[row][col]
            let b = gridB[row][col]
            let lum = 0.299 * r + 0.587 * g + 0.114 * b
            gridR[row][col] = max(0, min(255, lum + (r - lum) * 1.3))
            gridG[row][col] = max(0, min(255, lum + (g - lum) * 1.3))
            gridB[row][col] = max(0, min(255, lum + (b - lum) * 1.3))
        }
    }

    // --- Step 6: Render with optimized ANSI output (coalesce same colors, reduce escape codes) ---
    let edgeR: UInt8 = 20, edgeG: UInt8 = 15, edgeB: UInt8 = 10  // near-black warm edge

    var lines: [String] = []

    for termRow in 0..<height {
        let topRow = termRow * 2
        let botRow = topRow + 1
        var line = ""
        var prevFg: (UInt8, UInt8, UInt8) = (255, 255, 254)  // force first emit
        var prevBg: (UInt8, UInt8, UInt8) = (255, 255, 254)

        for col in 0..<width {
            let topIsEdge = isEdge[topRow][col]
            let botIsEdge = botRow < pixelH ? isEdge[botRow][col] : false

            let tr: UInt8, tg: UInt8, tb: UInt8
            if topIsEdge {
                tr = edgeR; tg = edgeG; tb = edgeB
            } else {
                tr = UInt8(min(255, max(0, Int(gridR[topRow][col]))))
                tg = UInt8(min(255, max(0, Int(gridG[topRow][col]))))
                tb = UInt8(min(255, max(0, Int(gridB[topRow][col]))))
            }

            let br: UInt8, bg: UInt8, bb: UInt8
            if botIsEdge {
                br = edgeR; bg = edgeG; bb = edgeB
            } else if botRow < pixelH {
                br = UInt8(min(255, max(0, Int(gridR[botRow][col]))))
                bg = UInt8(min(255, max(0, Int(gridG[botRow][col]))))
                bb = UInt8(min(255, max(0, Int(gridB[botRow][col]))))
            } else {
                br = 0; bg = 0; bb = 0
            }

            // Optimize: only emit escape codes when colors change
            let fgChanged = tr != prevFg.0 || tg != prevFg.1 || tb != prevFg.2
            let bgChanged = br != prevBg.0 || bg != prevBg.1 || bb != prevBg.2

            if tr == br && tg == bg && tb == bb {
                // Both halves same color — use background + space (saves fg escape)
                if bgChanged {
                    line += "\u{1B}[48;2;\(br);\(bg);\(bb)m"
                }
                line += " "
                prevBg = (br, bg, bb)
                prevFg = (255, 255, 254)  // invalidate fg cache
            } else {
                if fgChanged && bgChanged {
                    line += "\u{1B}[38;2;\(tr);\(tg);\(tb);48;2;\(br);\(bg);\(bb)m"
                } else if fgChanged {
                    line += "\u{1B}[38;2;\(tr);\(tg);\(tb)m"
                } else if bgChanged {
                    line += "\u{1B}[48;2;\(br);\(bg);\(bb)m"
                }
                line += "\u{2580}"
                prevFg = (tr, tg, tb)
                prevBg = (br, bg, bb)
            }
        }
        line += "\u{1B}[0m"
        lines.append(line)
    }

    print(lines.joined(separator: "\n"))
}

// MARK: - Braille Mini Portrait (for statusline)

/// Render a compact B&W portrait using Braille characters (U+2800..U+28FF).
/// Each Braille character encodes a 2x4 dot matrix, giving much finer detail
/// than half-block characters at the same character count.
///
/// `width` = characters wide, `height` = characters tall.
/// Actual pixel resolution: (width*2) x (height*4).
func renderBrailleMini(
    pixelBuffer: CVPixelBuffer,
    faceBBox: [Double]?,
    width: Int = 15,
    height: Int = 7
) {
    // Braille dot bit positions within a 2x4 cell:
    //   col0  col1
    //   0x01  0x08   row 0
    //   0x02  0x10   row 1
    //   0x04  0x20   row 2
    //   0x40  0x80   row 3
    let dotBits: [[UInt32]] = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]

    let pixW = width * 2   // pixel columns
    let pixH = height * 4  // pixel rows

    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    let imgWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imgHeight = CVPixelBufferGetHeight(pixelBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
    let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)

    // Face crop region — generous padding to include hair + shoulders
    var cropX = 0, cropY = 0, cropW = imgWidth, cropH = imgHeight
    if let bbox = faceBBox, bbox.count == 4 {
        let padding = 0.5
        let centerX = bbox[0] + bbox[2] / 2.0
        let centerY = bbox[1] + bbox[3] / 2.0
        let padW = bbox[2] * (1.0 + padding * 2)
        let padH = bbox[3] * (1.0 + padding * 2) * 1.3  // extend down for shoulders
        let pixCX = centerX * Double(imgWidth)
        let pixCY = (1.0 - centerY) * Double(imgHeight)
        let pW = padW * Double(imgWidth)
        let pH = padH * Double(imgHeight)
        cropX = max(0, Int(pixCX - pW / 2.0))
        cropY = max(0, Int(pixCY - pH / 2.0))
        cropW = min(imgWidth - cropX, Int(pW))
        cropH = min(imgHeight - cropY, Int(pH))
        if cropW < 50 || cropH < 50 { cropX = 0; cropY = 0; cropW = imgWidth; cropH = imgHeight }
    }

    let blockW = Double(cropW) / Double(pixW)
    let blockH = Double(cropH) / Double(pixH)

    // Block-average luminance at full pixel resolution
    var grid = [[Double]](repeating: [Double](repeating: 0, count: pixW), count: pixH)
    for row in 0..<pixH {
        let srcYStart = cropY + Int(Double(row) * blockH)
        let srcYEnd = min(cropY + Int(Double(row + 1) * blockH), cropY + cropH)
        for col in 0..<pixW {
            let mirrorCol = pixW - 1 - col
            let srcXStart = cropX + Int(Double(mirrorCol) * blockW)
            let srcXEnd = min(cropX + Int(Double(mirrorCol + 1) * blockW), cropX + cropW)
            var sum: Double = 0, count: Double = 0
            for sy in srcYStart..<srcYEnd {
                for sx in srcXStart..<srcXEnd {
                    let off = sy * bytesPerRow + sx * 4
                    sum += 0.299 * Double(buffer[off + 2]) + 0.587 * Double(buffer[off + 1]) + 0.114 * Double(buffer[off])
                    count += 1
                }
            }
            grid[row][col] = count > 0 ? sum / count : 0
        }
    }

    // Contrast stretch (face-region percentiles)
    let roiY0 = pixH / 6, roiY1 = pixH * 5 / 6
    let roiX0 = pixW / 4, roiX1 = pixW * 3 / 4
    var faceVals: [Double] = []
    for row in roiY0..<roiY1 { for col in roiX0..<roiX1 { faceVals.append(grid[row][col]) } }
    faceVals.sort()
    let lo = faceVals[max(0, Int(Double(faceVals.count) * 0.05))]
    let hi = faceVals[min(faceVals.count - 1, Int(Double(faceVals.count) * 0.95))]
    let range = max(hi - lo, 1.0)
    for row in 0..<pixH {
        for col in 0..<pixW {
            var v = max(0, min(1, (grid[row][col] - lo) / range))
            v = v * v * (3.0 - 2.0 * v)  // S-curve
            grid[row][col] = v * 255.0
        }
    }

    // Adaptive threshold for clean edge detection
    let radius = max(pixW / 8, 3)
    let bias: Double = 10.0
    var integral = [[Double]](repeating: [Double](repeating: 0, count: pixW + 1), count: pixH + 1)
    for row in 0..<pixH {
        var rowSum: Double = 0
        for col in 0..<pixW {
            rowSum += grid[row][col]
            integral[row + 1][col + 1] = integral[row][col + 1] + rowSum
        }
    }

    var binary = [[Bool]](repeating: [Bool](repeating: false, count: pixW), count: pixH)
    for row in 0..<pixH {
        for col in 0..<pixW {
            let y0 = max(0, row - radius)
            let y1 = min(pixH, row + radius + 1)
            let x0 = max(0, col - radius)
            let x1 = min(pixW, col + radius + 1)
            let area = Double((y1 - y0) * (x1 - x0))
            let sum = integral[y1][x1] - integral[y0][x1] - integral[y1][x0] + integral[y0][x0]
            binary[row][col] = grid[row][col] > (sum / area - bias)
        }
    }

    // Encode into Braille characters
    var lines: [String] = []
    for termRow in 0..<height {
        var line = ""
        for termCol in 0..<width {
            var codepoint: UInt32 = 0x2800  // blank braille
            for dy in 0..<4 {
                for dx in 0..<2 {
                    let py = termRow * 4 + dy
                    let px = termCol * 2 + dx
                    if py < pixH && px < pixW && !binary[py][px] {
                        // Dark pixel → dot ON (ink-on-paper: dark = feature)
                        codepoint |= dotBits[dy][dx]
                    }
                }
            }
            line += String(UnicodeScalar(codepoint)!)
        }
        lines.append(line)
    }
    print(lines.joined(separator: "\n"))
}

// MARK: - Person Segmentation Mask (silhouette only, no photo)

/// Extract person segmentation mask as a downsampled binary grid.
/// Returns only the silhouette outline — no facial detail, no RGB data.
/// The mask shows head shape (including hair), shoulders, and body outline.
func extractSegmentationMask(
    from pixelBuffer: CVPixelBuffer,
    faceBBox: [Double],
    targetWidth: Int,
    targetHeight: Int
) -> [[Bool]]? {
    let segRequest = VNGeneratePersonSegmentationRequest()
    segRequest.qualityLevel = .balanced
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

    do {
        try handler.perform([segRequest])
    } catch {
        return nil
    }

    guard let result = segRequest.results?.first,
          let maskBuffer = result.pixelBuffer as CVPixelBuffer? else {
        return nil
    }

    CVPixelBufferLockBaseAddress(maskBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(maskBuffer, .readOnly) }

    let maskW = CVPixelBufferGetWidth(maskBuffer)
    let maskH = CVPixelBufferGetHeight(maskBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(maskBuffer)
    guard let base = CVPixelBufferGetBaseAddress(maskBuffer) else { return nil }
    let ptr = base.assumingMemoryBound(to: UInt8.self)

    // Crop to face region with padding (same logic as portrait)
    let imgWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imgHeight = CVPixelBufferGetHeight(pixelBuffer)
    let padding = 0.6  // more padding to include hair + shoulders
    let centerX = faceBBox[0] + faceBBox[2] / 2.0
    let centerY = faceBBox[1] + faceBBox[3] / 2.0
    let padW = faceBBox[2] * (1.0 + padding * 2)
    let padH = faceBBox[3] * (1.0 + padding * 2) * 1.5  // extend down for shoulders

    // Map face bbox to mask coordinates
    let mCX = centerX * Double(maskW)
    let mCY = (1.0 - centerY) * Double(maskH)
    let mW = padW * Double(maskW)
    let mH = padH * Double(maskH)

    let cropX = max(0, Int(mCX - mW / 2.0))
    let cropY = max(0, Int(mCY - mH / 2.0))
    let cropW = min(maskW - cropX, Int(mW))
    let cropH = min(maskH - cropY, Int(mH))

    guard cropW > 10 && cropH > 10 else { return nil }

    // Downsample mask to target grid
    let blockW = Double(cropW) / Double(targetWidth)
    let blockH = Double(cropH) / Double(targetHeight)

    var grid = [[Bool]](repeating: [Bool](repeating: false, count: targetWidth), count: targetHeight)

    for row in 0..<targetHeight {
        let srcYStart = cropY + Int(Double(row) * blockH)
        let srcYEnd = min(cropY + Int(Double(row + 1) * blockH), cropY + cropH)
        for col in 0..<targetWidth {
            let mirrorCol = targetWidth - 1 - col  // selfie mirror
            let srcXStart = cropX + Int(Double(mirrorCol) * blockW)
            let srcXEnd = min(cropX + Int(Double(mirrorCol + 1) * blockW), cropX + cropW)

            var sum = 0, count = 0
            for sy in srcYStart..<srcYEnd {
                for sx in srcXStart..<srcXEnd {
                    sum += Int(ptr[sy * bytesPerRow + sx])
                    count += 1
                }
            }
            grid[row][col] = count > 0 && (sum / count) > 128
        }
    }
    return grid
}

// MARK: - Landmark-based Line Art Portrait (Privacy-preserving)

/// Render a face portrait using ONLY Vision landmark coordinates.
/// No camera pixels are used — just the detected feature point positions.
/// This creates a stylized line-drawing that's unique to each user's face shape.
func renderLandmarkPortrait(
    landmarks: [String: [[Double]]],
    bbox: [Double],
    width: Int = 30,
    height: Int = 15,
    segmentationMask: [[Bool]]? = nil  // person silhouette from VNGeneratePersonSegmentation
) {
    // Braille dot bit positions within a 2x4 cell
    let dotBits: [[UInt32]] = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]

    let pixW = width * 2
    let pixH = height * 4

    // Canvas: pixel grid for drawing
    var canvas = [[Bool]](repeating: [Bool](repeating: false, count: pixW), count: pixH)

    // Landmarks are normalized to face bounding box (0-1).
    // Map to pixel grid with mirroring (selfie).
    func mapPoint(_ pt: [Double]) -> (Int, Int) {
        let x = pixW - 1 - Int(pt[0] * Double(pixW - 1))  // mirror X
        let y = pixH - 1 - Int(pt[1] * Double(pixH - 1))  // flip Y (Vision origin = bottom-left)
        return (max(0, min(pixW - 1, x)), max(0, min(pixH - 1, y)))
    }

    // Line thickness scales with canvas size
    let thickness = 1  // thin lines for clean look

    // Draw a single pixel
    func setPixel(_ x: Int, _ y: Int) {
        if x >= 0 && x < pixW && y >= 0 && y < pixH {
            canvas[y][x] = true
        }
    }

    // Draw a thick pixel (filled circle of given radius)
    func setThickPixel(_ x: Int, _ y: Int, _ t: Int) {
        if t <= 1 {
            setPixel(x, y)
            return
        }
        for dy in -t...t {
            for dx in -t...t {
                if dx * dx + dy * dy <= t * t {
                    setPixel(x + dx, y + dy)
                }
            }
        }
    }

    // Bresenham line drawing with thickness
    func drawLine(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int, thick: Int = 0) {
        let t = thick > 0 ? thick : thickness
        var x0 = x0, y0 = y0
        let dx = abs(x1 - x0)
        let dy = abs(y1 - y0)
        let sx = x0 < x1 ? 1 : -1
        let sy = y0 < y1 ? 1 : -1
        var err = dx - dy

        while true {
            setThickPixel(x0, y0, t)
            if x0 == x1 && y0 == y1 { break }
            let e2 = 2 * err
            if e2 > -dy { err -= dy; x0 += sx }
            if e2 < dx { err += dx; y0 += sy }
        }
    }

    // Catmull-Rom spline interpolation for smooth curves
    func catmullRom(_ p0: [Double], _ p1: [Double], _ p2: [Double], _ p3: [Double], _ t: Double) -> [Double] {
        let t2 = t * t, t3 = t2 * t
        let x = 0.5 * ((2 * p1[0]) +
            (-p0[0] + p2[0]) * t +
            (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
            (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
        let y = 0.5 * ((2 * p1[1]) +
            (-p0[1] + p2[1]) * t +
            (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
            (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
        return [x, y]
    }

    // Subdivide points using Catmull-Rom for smooth curves
    func smoothPoints(_ points: [[Double]], close: Bool, subdivisions: Int = 8) -> [[Double]] {
        guard points.count >= 2 else { return points }
        var result: [[Double]] = []
        let n = points.count

        for i in 0..<(close ? n : n - 1) {
            let p0 = points[close ? (i - 1 + n) % n : max(0, i - 1)]
            let p1 = points[i]
            let p2 = points[(i + 1) % n]
            let p3 = points[close ? (i + 2) % n : min(n - 1, i + 2)]

            for s in 0..<subdivisions {
                let t = Double(s) / Double(subdivisions)
                result.append(catmullRom(p0, p1, p2, p3, t))
            }
        }
        if !close { result.append(points.last!) }
        return result
    }

    // Draw connected smooth line strip
    func drawStrip(_ points: [[Double]], close: Bool = true, thick: Int = 0) {
        let smooth = smoothPoints(points, close: close)
        guard smooth.count >= 2 else { return }
        for i in 0..<(smooth.count - 1) {
            let (x0, y0) = mapPoint(smooth[i])
            let (x1, y1) = mapPoint(smooth[i + 1])
            drawLine(x0, y0, x1, y1, thick: thick)
        }
        if close && smooth.count > 2 {
            let (x0, y0) = mapPoint(smooth.last!)
            let (x1, y1) = mapPoint(smooth.first!)
            drawLine(x0, y0, x1, y1, thick: thick)
        }
    }

    // Draw a filled dot
    func drawDot(_ pt: [Double], radius: Int) {
        let (cx, cy) = mapPoint(pt)
        for dy in -radius...radius {
            for dx in -radius...radius {
                if dx * dx + dy * dy <= radius * radius {
                    setPixel(cx + dx, cy + dy)
                }
            }
        }
    }

    // Draw a circle outline (for iris)
    func drawCircle(_ pt: [Double], radius: Int, thick: Int = 1) {
        let (cx, cy) = mapPoint(pt)
        let steps = max(36, radius * 8)
        for i in 0..<steps {
            let angle = Double(i) / Double(steps) * 2.0 * Double.pi
            let x = cx + Int(round(Double(radius) * cos(angle)))
            let y = cy + Int(round(Double(radius) * sin(angle)))
            setThickPixel(x, y, thick)
        }
    }

    // Flood fill a closed region (scanline fill between boundary points)
    func fillPolygon(_ points: [[Double]]) {
        let mapped = points.map { mapPoint($0) }
        guard !mapped.isEmpty else { return }
        let minY = mapped.map { $0.1 }.min()!
        let maxY = mapped.map { $0.1 }.max()!

        for y in minY...maxY {
            var intersections: [Int] = []
            let n = mapped.count
            for i in 0..<n {
                let j = (i + 1) % n
                let (x0, y0) = mapped[i]
                let (x1, y1) = mapped[j]
                if (y0 <= y && y1 > y) || (y1 <= y && y0 > y) {
                    let t = Double(y - y0) / Double(y1 - y0)
                    intersections.append(Int(Double(x0) + t * Double(x1 - x0)))
                }
            }
            intersections.sort()
            var idx = 0
            while idx + 1 < intersections.count {
                for x in intersections[idx]...intersections[idx + 1] {
                    setPixel(x, y)
                }
                idx += 2
            }
        }
    }

    // --- Draw segmentation mask outline (head/hair/shoulder silhouette) ---
    if let mask = segmentationMask, mask.count == pixH, mask[0].count == pixW {
        // Extract edge pixels from the binary mask
        for y in 0..<pixH {
            for x in 0..<pixW {
                if mask[y][x] {
                    // Check if this is a boundary pixel (neighbor is false)
                    let isEdge = (x == 0 || !mask[y][x-1]) ||
                                 (x == pixW-1 || !mask[y][x+1]) ||
                                 (y == 0 || !mask[y-1][x]) ||
                                 (y == pixH-1 || !mask[y+1][x])
                    if isEdge {
                        setThickPixel(x, y, thickness)
                    }
                }
            }
        }
    }

    // --- Draw each facial feature ---

    // Face contour (jawline) — thicker line (skip if we have segmentation mask)
    if let contour = landmarks["faceContour"] {
        drawStrip(contour, close: false)

        // Add forehead arc: connect top of contour with a curved top
        if contour.count >= 2 {
            let leftTop = contour.last!
            let rightTop = contour.first!
            // Create arc points above the face
            let midX = (leftTop[0] + rightTop[0]) / 2.0
            let topY = max(leftTop[1], rightTop[1]) + 0.25  // above eyes
            let steps = 12
            var arcPoints: [[Double]] = [leftTop]
            for i in 1..<steps {
                let t = Double(i) / Double(steps)
                // Quadratic bezier: left -> top -> right
                let p0 = leftTop
                let p1 = [midX, topY]  // control point (top of head)
                let p2 = rightTop
                let x = (1 - t) * (1 - t) * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
                let y = (1 - t) * (1 - t) * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
                arcPoints.append([x, y])
            }
            arcPoints.append(rightTop)
            drawStrip(arcPoints, close: false)
        }
    }

    // Eyebrows — filled (smooth + fill)
    if let lb = landmarks["leftEyebrow"] {
        let smooth = smoothPoints(lb, close: false)
        // Create a thin filled shape by offsetting
        var topPts = smooth
        var botPts = smooth.map { [$0[0], $0[1] - 0.015] }  // offset down slightly
        let fillPts = topPts + botPts.reversed()
        fillPolygon(fillPts)
        drawStrip(lb, close: false)
    }
    if let rb = landmarks["rightEyebrow"] {
        let smooth = smoothPoints(rb, close: false)
        var topPts = smooth
        var botPts = smooth.map { [$0[0], $0[1] - 0.015] }
        let fillPts = topPts + botPts.reversed()
        fillPolygon(fillPts)
        drawStrip(rb, close: false)
    }

    // Eyes (closed loops, smooth)
    if let le = landmarks["leftEye"] { drawStrip(le, close: true, thick: thickness) }
    if let re = landmarks["rightEye"] { drawStrip(re, close: true, thick: thickness) }

    // Iris circle + filled pupil
    let irisR = max(2, min(pixW, pixH) / 24)
    let pupilR = max(1, irisR / 2)
    if let lp = landmarks["leftPupil"], !lp.isEmpty {
        drawCircle(lp[0], radius: irisR, thick: thickness)
        drawDot(lp[0], radius: pupilR)
    }
    if let rp = landmarks["rightPupil"], !rp.isEmpty {
        drawCircle(rp[0], radius: irisR, thick: thickness)
        drawDot(rp[0], radius: pupilR)
    }

    // Nose (smooth curves)
    if let nose = landmarks["nose"] { drawStrip(nose, close: false, thick: thickness) }
    if let crest = landmarks["noseCrest"] { drawStrip(crest, close: false, thick: thickness) }

    // Lips — outer filled, inner line
    if let ol = landmarks["outerLips"] {
        let smooth = smoothPoints(ol, close: true)
        fillPolygon(smooth)
        drawStrip(ol, close: true)
    }
    if let il = landmarks["innerLips"] { drawStrip(il, close: true, thick: thickness) }

    // --- Encode to Braille ---
    var lines: [String] = []
    for termRow in 0..<height {
        var line = ""
        for termCol in 0..<width {
            var codepoint: UInt32 = 0x2800
            for dy in 0..<4 {
                for dx in 0..<2 {
                    let py = termRow * 4 + dy
                    let px = termCol * 2 + dx
                    if py < pixH && px < pixW && canvas[py][px] {
                        codepoint |= dotBits[dy][dx]
                    }
                }
            }
            line += String(UnicodeScalar(codepoint)!)
        }
        lines.append(line)
    }
    print(lines.joined(separator: "\n"))
}

// MARK: - Terminal Image Protocol Support (Progressive Degradation)

/// Supported terminal image protocols, in order of preference.
enum TerminalImageProtocol: String {
    case iterm2  // iTerm2 inline images (best macOS support)
    case kitty   // Kitty graphics protocol
    case sixel   // DEC Sixel
    case ansi    // ANSI half-block fallback (current --pixel)
}

/// Detect the best available terminal image protocol from environment.
func detectTerminalProtocol() -> TerminalImageProtocol {
    let env = ProcessInfo.processInfo.environment

    // iTerm2: LC_TERMINAL=iTerm2 or TERM_PROGRAM=iTerm.app or WezTerm
    if let lcTerm = env["LC_TERMINAL"], lcTerm.lowercased().contains("iterm") {
        return .iterm2
    }
    if let termProg = env["TERM_PROGRAM"] {
        let tp = termProg.lowercased()
        if tp.contains("iterm") { return .iterm2 }
        if tp.contains("wezterm") { return .iterm2 }  // WezTerm supports all, prefer iTerm2 (simplest)
    }

    // Kitty: TERM=xterm-kitty or TERM_PROGRAM=Ghostty
    if let term = env["TERM"], term.contains("kitty") {
        return .kitty
    }
    if env["KITTY_WINDOW_ID"] != nil {
        return .kitty
    }
    if let termProg = env["TERM_PROGRAM"]?.lowercased() {
        if termProg.contains("ghostty") { return .kitty }
    }

    // Sixel: check TERM for terminals known to support it (mintty, mlterm, foot)
    // Most modern terminals that support Sixel also support iTerm2/Kitty,
    // so this is mainly a fallback for Linux terminals.
    // On macOS, iTerm2 supports Sixel too but we prefer native protocol.

    return .ansi  // universal fallback
}

/// Crop a CVPixelBuffer to the face region and return PNG data.
func cropFaceToPNG(
    pixelBuffer: CVPixelBuffer,
    faceBBox: [Double]?,
    maxWidth: Int = 400,
    maxHeight: Int = 400
) -> Data? {
    let imgWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imgHeight = CVPixelBufferGetHeight(pixelBuffer)

    // Determine crop rect
    var cropRect = CGRect(x: 0, y: 0, width: imgWidth, height: imgHeight)

    if let bbox = faceBBox, bbox.count == 4 {
        let padding = 0.4
        let centerX = bbox[0] + bbox[2] / 2.0
        let centerY = bbox[1] + bbox[3] / 2.0
        let padW = bbox[2] * (1.0 + padding * 2)
        let padH = bbox[3] * (1.0 + padding * 2)

        let pixCX = centerX * Double(imgWidth)
        let pixCY = (1.0 - centerY) * Double(imgHeight)  // flip Y
        let pixW = padW * Double(imgWidth)
        let pixH = padH * Double(imgHeight)

        let x = max(0, pixCX - pixW / 2.0)
        let y = max(0, pixCY - pixH / 2.0)
        let w = min(Double(imgWidth) - x, pixW)
        let h = min(Double(imgHeight) - y, pixH)

        if w >= 50 && h >= 50 {
            cropRect = CGRect(x: x, y: y, width: w, height: h)
        }
    }

    // Create CIImage from pixel buffer, crop, mirror, resize, export PNG
    var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    ciImage = ciImage.cropped(to: cropRect)

    // Mirror horizontally (selfie)
    ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: -1, y: 1)
        .translatedBy(x: -ciImage.extent.width, y: 0))

    // Scale down to target size
    let scaleX = Double(maxWidth) / ciImage.extent.width
    let scaleY = Double(maxHeight) / ciImage.extent.height
    let scale = min(scaleX, scaleY, 1.0)  // don't upscale
    if scale < 1.0 {
        ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }

    let context = CIContext()
    guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
        return nil
    }

    // Encode to PNG
    let mutableData = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(mutableData, "public.png" as CFString, 1, nil) else {
        return nil
    }
    CGImageDestinationAddImage(dest, cgImage, nil)
    guard CGImageDestinationFinalize(dest) else {
        return nil
    }

    return mutableData as Data
}

/// Get raw RGBA pixel data from a face-cropped region.
func cropFaceToRGBA(
    pixelBuffer: CVPixelBuffer,
    faceBBox: [Double]?,
    targetWidth: Int = 200,
    targetHeight: Int = 200
) -> (data: [UInt8], width: Int, height: Int)? {
    let imgWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imgHeight = CVPixelBufferGetHeight(pixelBuffer)

    var cropRect = CGRect(x: 0, y: 0, width: imgWidth, height: imgHeight)

    if let bbox = faceBBox, bbox.count == 4 {
        let padding = 0.4
        let centerX = bbox[0] + bbox[2] / 2.0
        let centerY = bbox[1] + bbox[3] / 2.0
        let padW = bbox[2] * (1.0 + padding * 2)
        let padH = bbox[3] * (1.0 + padding * 2)
        let pixCX = centerX * Double(imgWidth)
        let pixCY = (1.0 - centerY) * Double(imgHeight)
        let pixW = padW * Double(imgWidth)
        let pixH = padH * Double(imgHeight)
        let x = max(0, pixCX - pixW / 2.0)
        let y = max(0, pixCY - pixH / 2.0)
        let w = min(Double(imgWidth) - x, pixW)
        let h = min(Double(imgHeight) - y, pixH)
        if w >= 50 && h >= 50 {
            cropRect = CGRect(x: x, y: y, width: w, height: h)
        }
    }

    var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    ciImage = ciImage.cropped(to: cropRect)
    ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: -1, y: 1)
        .translatedBy(x: -ciImage.extent.width, y: 0))

    let scaleX = Double(targetWidth) / ciImage.extent.width
    let scaleY = Double(targetHeight) / ciImage.extent.height
    let scale = min(scaleX, scaleY, 1.0)
    if scale < 1.0 {
        ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }

    let ctx = CIContext()
    let extent = ciImage.extent
    let w = Int(extent.width)
    let h = Int(extent.height)

    guard let cgImage = ctx.createCGImage(ciImage, from: extent) else { return nil }

    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let bitmapCtx = CGContext(
        data: &rgba, width: w, height: h,
        bitsPerComponent: 8, bytesPerRow: w * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    bitmapCtx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
    return (rgba, w, h)
}

// MARK: - iTerm2 Inline Image Protocol

/// Render using iTerm2 inline image protocol (OSC 1337).
/// Format: ESC ] 1337 ; File=inline=1;size=N;width=Xpx;height=Ypx;preserveAspectRatio=1 : BASE64 BEL
func renderITerm2(pixelBuffer: CVPixelBuffer, faceBBox: [Double]?, widthPx: Int, heightPx: Int) -> Bool {
    guard let pngData = cropFaceToPNG(pixelBuffer: pixelBuffer, faceBBox: faceBBox,
                                       maxWidth: widthPx, maxHeight: heightPx) else {
        return false
    }

    let b64 = pngData.base64EncodedString()
    let size = pngData.count

    // Use pixel dimensions for sizing (px suffix)
    print("\u{1B}]1337;File=inline=1;size=\(size);width=\(widthPx)px;height=\(heightPx)px;preserveAspectRatio=1:\(b64)\u{07}", terminator: "")
    print("")  // newline after image
    return true
}

// MARK: - Kitty Graphics Protocol

/// Render using Kitty graphics protocol (APC).
/// Sends PNG data as chunked base64.
func renderKitty(pixelBuffer: CVPixelBuffer, faceBBox: [Double]?, widthPx: Int, heightPx: Int) -> Bool {
    guard let pngData = cropFaceToPNG(pixelBuffer: pixelBuffer, faceBBox: faceBBox,
                                       maxWidth: widthPx, maxHeight: heightPx) else {
        return false
    }

    let b64 = pngData.base64EncodedString()
    let chunkSize = 4096

    // Send in chunks: first chunk with metadata, subsequent chunks continuation
    var offset = 0
    var isFirst = true

    while offset < b64.count {
        let end = min(offset + chunkSize, b64.count)
        let startIdx = b64.index(b64.startIndex, offsetBy: offset)
        let endIdx = b64.index(b64.startIndex, offsetBy: end)
        let chunk = String(b64[startIdx..<endIdx])
        let more = end < b64.count ? 1 : 0

        if isFirst {
            // a=T (transmit+display), f=100 (PNG), t=d (direct data), m=more
            print("\u{1B}_Ga=T,f=100,t=d,m=\(more);\(chunk)\u{1B}\\", terminator: "")
            isFirst = false
        } else {
            print("\u{1B}_Gm=\(more);\(chunk)\u{1B}\\", terminator: "")
        }

        offset = end
    }

    print("")  // newline after image
    return true
}

// MARK: - Sixel Protocol

/// Render using DEC Sixel protocol.
/// Each Sixel row = 6 vertical pixels. Uses 256-color palette with median-cut quantization.
func renderSixel(pixelBuffer: CVPixelBuffer, faceBBox: [Double]?, targetWidth: Int, targetHeight: Int) -> Bool {
    guard let rgbaResult = cropFaceToRGBA(
        pixelBuffer: pixelBuffer, faceBBox: faceBBox,
        targetWidth: targetWidth, targetHeight: targetHeight
    ) else {
        return false
    }

    let (rgba, w, h) = rgbaResult

    // Step 1: Build color palette (uniform quantization to 216 colors = 6^3)
    let levels = 6
    let step = 255 / (levels - 1)

    func quantizeChannel(_ v: UInt8) -> UInt8 {
        return UInt8(min(255, (Int(v) + step / 2) / step * step))
    }

    // Build palette: 216 colors (6x6x6 cube)
    var palette: [(r: UInt8, g: UInt8, b: UInt8)] = []
    for ri in 0..<levels {
        for gi in 0..<levels {
            for bi in 0..<levels {
                palette.append((UInt8(ri * step), UInt8(gi * step), UInt8(bi * step)))
            }
        }
    }

    // Map each pixel to nearest palette index
    var indexed = [Int](repeating: 0, count: w * h)
    for i in 0..<(w * h) {
        let r = rgba[i * 4]
        let g = rgba[i * 4 + 1]
        let b = rgba[i * 4 + 2]

        let ri = min(levels - 1, (Int(r) + step / 2) / step)
        let gi = min(levels - 1, (Int(g) + step / 2) / step)
        let bi = min(levels - 1, (Int(b) + step / 2) / step)

        indexed[i] = ri * levels * levels + gi * levels + bi
    }

    // Step 2: Encode Sixel
    var sixel = ""

    // DCS: P0;1;q  (P0=normal aspect, 1=background, q=begin sixel data)
    sixel += "\u{1B}P0;1;q"

    // Raster attributes: "W;H (pixel dimensions)
    sixel += "\"1;1;\(w);\(h)"

    // Color definitions (#index;2;R%;G%;B%)
    // Only define colors that are actually used
    var usedColors = Set<Int>()
    for idx in indexed { usedColors.insert(idx) }

    for idx in usedColors.sorted() {
        let c = palette[idx]
        let rPct = Int(round(Double(c.r) / 255.0 * 100.0))
        let gPct = Int(round(Double(c.g) / 255.0 * 100.0))
        let bPct = Int(round(Double(c.b) / 255.0 * 100.0))
        sixel += "#\(idx);2;\(rPct);\(gPct);\(bPct)"
    }

    // Step 3: Output sixel rows (each row = 6 vertical pixels)
    let sixelRows = (h + 5) / 6

    for sixelRow in 0..<sixelRows {
        let y0 = sixelRow * 6

        // For each color used in this row band, output a line
        var colorsInBand = Set<Int>()
        for dy in 0..<6 {
            let y = y0 + dy
            if y >= h { break }
            for x in 0..<w {
                colorsInBand.insert(indexed[y * w + x])
            }
        }

        var isFirstColor = true
        for colorIdx in colorsInBand.sorted() {
            sixel += "#\(colorIdx)"

            // Build sixel data for this color
            var rleData: [UInt8] = []
            for x in 0..<w {
                var bits: UInt8 = 0
                for dy in 0..<6 {
                    let y = y0 + dy
                    if y < h && indexed[y * w + x] == colorIdx {
                        bits |= (1 << dy)
                    }
                }
                rleData.append(bits + 63)  // Sixel char = bits + 63
            }

            // RLE compression
            var i = 0
            while i < rleData.count {
                let ch = rleData[i]
                var run = 1
                while i + run < rleData.count && rleData[i + run] == ch && run < 255 {
                    run += 1
                }
                if run >= 3 {
                    sixel += "!\(run)\(Character(UnicodeScalar(ch)))"
                } else {
                    for _ in 0..<run {
                        sixel += String(Character(UnicodeScalar(ch)))
                    }
                }
                i += run
            }

            // $ = carriage return (same row, next color)
            if !isFirstColor || colorsInBand.count > 1 {
                // After each color except the last, use $ to return to start of line
            }
            isFirstColor = false

            // If not the last color in this band, use $ (CR)
            if colorIdx != colorsInBand.sorted().last {
                sixel += "$"
            }
        }

        // - = new line (next sixel row)
        if sixelRow < sixelRows - 1 {
            sixel += "-"
        }
    }

    // ST: end sixel
    sixel += "\u{1B}\\"

    print(sixel, terminator: "")
    print("")
    return true
}

// MARK: - Auto Image Renderer

/// Render face portrait using the best available terminal image protocol.
/// Falls back through: iTerm2 → Kitty → Sixel → ANSI half-block pixel art.
func renderImage(
    pixelBuffer: CVPixelBuffer,
    faceBBox: [Double]?,
    protocol proto: TerminalImageProtocol? = nil,
    width: Int = 300,
    height: Int = 300
) {
    let selectedProto = proto ?? detectTerminalProtocol()

    switch selectedProto {
    case .iterm2:
        if renderITerm2(pixelBuffer: pixelBuffer, faceBBox: faceBBox, widthPx: width, heightPx: height) {
            return
        }
        // Fallback to kitty
        if renderKitty(pixelBuffer: pixelBuffer, faceBBox: faceBBox, widthPx: width, heightPx: height) {
            return
        }
        renderPixelArt(pixelBuffer: pixelBuffer, faceBBox: faceBBox, width: 40, height: 10)

    case .kitty:
        if renderKitty(pixelBuffer: pixelBuffer, faceBBox: faceBBox, widthPx: width, heightPx: height) {
            return
        }
        renderPixelArt(pixelBuffer: pixelBuffer, faceBBox: faceBBox, width: 40, height: 10)

    case .sixel:
        if renderSixel(pixelBuffer: pixelBuffer, faceBBox: faceBBox, targetWidth: width, targetHeight: height) {
            return
        }
        renderPixelArt(pixelBuffer: pixelBuffer, faceBBox: faceBBox, width: 40, height: 10)

    case .ansi:
        renderPixelArt(pixelBuffer: pixelBuffer, faceBBox: faceBBox, width: 40, height: 10)
    }
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
} else if args.count > 1 && args[1] == "--landmark" {
    // Privacy-preserving landmark line-art portrait: --landmark [W] [H]
    // Uses only Vision framework landmark coordinates, no camera pixels.
    let w = args.count > 2 ? Int(args[2]) ?? 30 : 30
    let h = args.count > 3 ? Int(args[3]) ?? 15 : 15

    let capturer = FrameCapturer(highRes: false)
    if let pb = capturer.captureFrame() {
        let result = detectLandmarks(from: pb)
        if let landmarks = result["landmarks"] as? [String: [[Double]]],
           let bbox = result["bbox"] as? [Double] {
            // Extract segmentation mask for silhouette (head shape, hair, shoulders)
            let pixW = w * 2
            let pixH = h * 4
            let mask = extractSegmentationMask(from: pb, faceBBox: bbox,
                                                targetWidth: pixW, targetHeight: pixH)
            renderLandmarkPortrait(landmarks: landmarks, bbox: bbox, width: w, height: h,
                                   segmentationMask: mask)
        } else {
            fputs("[error] No face landmarks detected.\n", stderr)
        }
        _ = pb
    } else {
        fputs("[error] Failed to capture frame.\n", stderr)
    }
} else if args.count > 1 && args[1] == "--image" {
    // Auto-detect best terminal image protocol: --image [W] [H] [protocol]
    // protocol: auto (default), iterm2, kitty, sixel, ansi
    let w = args.count > 2 ? Int(args[2]) ?? 300 : 300
    let h = args.count > 3 ? Int(args[3]) ?? 300 : 300
    let protoName = args.count > 4 ? args[4] : "auto"

    let proto: TerminalImageProtocol? = {
        switch protoName {
        case "iterm2": return .iterm2
        case "kitty": return .kitty
        case "sixel": return .sixel
        case "ansi": return .ansi
        default: return nil  // auto-detect
        }
    }()

    let capturer = FrameCapturer(highRes: true)  // high res for native image protocols
    if let pb = capturer.captureFrame() {
        let result = detectLandmarks(from: pb)
        let bbox = result["bbox"] as? [Double]
        renderImage(pixelBuffer: pb, faceBBox: bbox, protocol: proto, width: w, height: h)
        _ = pb
    } else {
        fputs("[error] Failed to capture frame.\n", stderr)
    }
} else if args.count > 1 && args[1] == "--pixel" {
    // Colored pixel art portrait: --pixel [W] [H]
    let w = args.count > 2 ? Int(args[2]) ?? 60 : 60
    let h = args.count > 3 ? Int(args[3]) ?? 30 : 30

    let capturer = FrameCapturer(highRes: false)
    if let pb = capturer.captureFrame() {
        // Detect face bbox for cropping
        let result = detectLandmarks(from: pb)
        let bbox = result["bbox"] as? [Double]
        renderPixelArt(pixelBuffer: pb, faceBBox: bbox, width: w, height: h)
        _ = pb
    } else {
        fputs("[error] Failed to capture frame.\n", stderr)
    }
} else if args.count > 1 && args[1] == "--ascii-mini" {
    // Braille mini portrait for statusline: --ascii-mini [W] [H]
    let w = args.count > 2 ? Int(args[2]) ?? 15 : 15
    let h = args.count > 3 ? Int(args[3]) ?? 7 : 7

    let capturer = FrameCapturer(highRes: false)
    if let pb = capturer.captureFrame() {
        let result = detectLandmarks(from: pb)
        let bbox = result["bbox"] as? [Double]
        renderBrailleMini(pixelBuffer: pb, faceBBox: bbox, width: w, height: h)
        _ = pb
    } else {
        fputs("[error] Failed to capture frame.\n", stderr)
    }
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
