"""Camera Capture Module using OpenCV.

Provides single-shot and background-threaded frame capture from a webcam.
Uses opencv-python-headless (cv2) with camera index 0.
"""

import argparse
import atexit
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np


class CameraCapture:
    """Captures frames from a webcam via OpenCV."""

    def __init__(self, camera_index: int = 0):
        self._camera_index = camera_index
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        # Register cleanup handlers
        atexit.register(self.stop)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_frame(self) -> Optional[np.ndarray]:
        """Open camera, capture a single frame, release camera, return frame.

        Returns:
            numpy array (BGR image) on success, None on failure.
        """
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            self._print_permission_error()
            return None

        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[CameraCapture] Failed to read frame from camera.")
                return None
            return frame
        finally:
            cap.release()

    def start_background_capture(self, interval_sec: float = 10) -> None:
        """Start a background thread that captures frames at *interval_sec*.

        The latest frame is stored internally and retrievable via
        ``get_latest_frame()``.
        """
        if self._capture_thread is not None and self._capture_thread.is_alive():
            print("[CameraCapture] Background capture is already running.")
            return

        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._background_loop,
            args=(interval_sec,),
            daemon=True,
        )
        self._capture_thread.start()
        print(
            f"[CameraCapture] Background capture started "
            f"(interval={interval_sec}s)."
        )

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame captured by the background thread."""
        with self._lock:
            return self._latest_frame

    def stop(self) -> None:
        """Stop the background capture thread and clean up resources."""
        if self._stop_event.is_set():
            return
        self._stop_event.set()

        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)
            self._capture_thread = None

        print("[CameraCapture] Stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _background_loop(self, interval_sec: float) -> None:
        """Continuously capture frames until stop is requested."""
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            self._print_permission_error()
            return

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                self._stop_event.wait(timeout=interval_sec)
        finally:
            cap.release()

    def _signal_handler(self, signum: int, _frame) -> None:
        """Handle SIGINT / SIGTERM for graceful shutdown."""
        print(f"\n[CameraCapture] Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    @staticmethod
    def _print_permission_error() -> None:
        """Print a user-friendly error about camera access on macOS."""
        print(
            "[CameraCapture] ERROR: Unable to open camera (index 0).\n"
            "\n"
            "  On macOS, the application may not have camera permission.\n"
            "  Please check:\n"
            "    System Settings > Privacy & Security > Camera\n"
            "  and ensure that Terminal (or your IDE) is allowed to access\n"
            "  the camera.\n"
        )


# ------------------------------------------------------------------
# CLI test mode
# ------------------------------------------------------------------

def _cli_test() -> None:
    """Capture one frame and save it to test_frame.jpg."""
    cam = CameraCapture()
    frame = cam.capture_frame()
    if frame is None:
        print("Test FAILED: could not capture a frame.")
        sys.exit(1)

    output_path = "test_frame.jpg"
    cv2.imwrite(output_path, frame)
    h, w = frame.shape[:2]
    print(f"Test PASSED: saved {output_path} ({w}x{h})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Capture Module")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Capture one frame and save to test_frame.jpg",
    )
    args = parser.parse_args()

    if args.test:
        _cli_test()
    else:
        parser.print_help()
