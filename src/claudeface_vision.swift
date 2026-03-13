/// ClaudeFace Vision - Native macOS face landmark detection.
///
/// Captures a single frame from the built-in camera via AVFoundation,
/// runs VNDetectFaceLandmarksRequest (Vision framework), and outputs
/// normalized landmark coordinates as JSON to stdout.
///
/// Build:  swiftc -O -o bin/claudeface-vision src/claudeface_vision.swift \
///             -framework AVFoundation -framework Vision -framework CoreMedia
///
/// Usage:  bin/claudeface-vision          → single detect, JSON to stdout
///         bin/claudeface-vision --check   → permission/camera check only

import AVFoundation
import CoreMedia
import Foundation
import Vision

// MARK: - Single-Frame Capture + Face Landmark Detection

final class FaceLandmarkDetector: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session = AVCaptureSession()
    private let semaphore = DispatchSemaphore(value: 0)
    private var frameCount = 0
    private let warmupFrames = 10 // skip first N frames (camera auto-exposure warmup)
    private var capturedResult: [String: Any]?

    /// Run one capture cycle: open camera → skip warmup → detect landmarks → return JSON-ready dict.
    func detect() -> [String: Any] {
        // Camera authorization
        let authStatus = AVCaptureDevice.authorizationStatus(for: .video)
        switch authStatus {
        case .denied, .restricted:
            return ["status": "no_permission",
                    "message": "Camera access denied. Grant permission in System Settings > Privacy & Security > Camera."]
        case .notDetermined:
            let sem = DispatchSemaphore(value: 0)
            var granted = false
            AVCaptureDevice.requestAccess(for: .video) { g in
                granted = g
                sem.signal()
            }
            sem.wait()
            if !granted {
                return ["status": "no_permission",
                        "message": "Camera access was not granted."]
            }
        default:
            break
        }

        // Open camera
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device)
        else {
            return ["status": "no_camera",
                    "message": "No camera device found."]
        }

        session.sessionPreset = .high // use highest available resolution for reliable detection
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let queue = DispatchQueue(label: "claudeface.capture")
        output.setSampleBufferDelegate(self, queue: queue)
        session.addOutput(output)

        session.startRunning()

        // Wait up to 8 seconds for a result
        let timeout = semaphore.wait(timeout: .now() + 8)
        session.stopRunning()

        if timeout == .timedOut {
            return ["status": "timeout",
                    "message": "Camera capture timed out after 8 seconds."]
        }

        return capturedResult ?? ["status": "error", "message": "Unknown error"]
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        frameCount += 1
        guard frameCount > warmupFrames else { return }

        // Stop receiving further frames
        if let videoOutput = output as? AVCaptureVideoDataOutput {
            videoOutput.setSampleBufferDelegate(nil, queue: nil)
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            capturedResult = ["status": "error", "message": "Failed to get pixel buffer"]
            semaphore.signal()
            return
        }

        // Run Vision face landmark detection
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        do {
            try handler.perform([request])
        } catch {
            capturedResult = ["status": "error", "message": error.localizedDescription]
            semaphore.signal()
            return
        }

        guard let observations = request.results, !observations.isEmpty else {
            capturedResult = ["status": "no_face"]
            semaphore.signal()
            return
        }

        // Use the first (largest/most prominent) face
        let face = observations[0]
        var data: [String: Any] = ["status": "ok"]

        // Bounding box (normalized to full image)
        let box = face.boundingBox
        data["bbox"] = [
            round(Double(box.origin.x) * 10000) / 10000,
            round(Double(box.origin.y) * 10000) / 10000,
            round(Double(box.width) * 10000) / 10000,
            round(Double(box.height) * 10000) / 10000,
        ]

        // Extract key landmark regions
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

        capturedResult = data
        semaphore.signal()
    }
}

// MARK: - Permission Check Mode

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
} else {
    let detector = FaceLandmarkDetector()
    let result = detector.detect()
    printJSON(result)
}
