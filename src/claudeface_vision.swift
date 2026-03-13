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

    // Face crop region
    var cropX = 0, cropY = 0, cropW = imgWidth, cropH = imgHeight
    if let bbox = faceBBox, bbox.count == 4 {
        let padding = 0.3
        let centerX = bbox[0] + bbox[2] / 2.0
        let centerY = bbox[1] + bbox[3] / 2.0
        let padW = bbox[2] * (1.0 + padding * 2)
        let padH = bbox[3] * (1.0 + padding * 2)
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

    // Adaptive threshold
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
