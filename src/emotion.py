"""
Emotion Detection Module using the FER (Facial Expression Recognition) library.

Detects facial emotions from BGR image frames and provides history/trend analysis.
"""

import argparse
import json
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
try:
    from fer import FER
except ImportError:
    from fer.fer import FER

# The 7 FER emotions
FER_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class EmotionDetector:
    """Detects facial emotions from image frames using FER with MTCNN backend."""

    def __init__(self) -> None:
        self._detector = FER(mtcnn=True)
        self._history: deque = deque(maxlen=20)

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect emotions for all faces in a BGR image frame.

        Args:
            frame: numpy array representing a BGR image.

        Returns:
            List of dicts, each containing:
                - emotion (str): dominant emotion label
                - confidence (float): confidence of dominant emotion
                - all_emotions (dict): scores for all 7 emotions
                - bbox (list[int]): [x, y, w, h] bounding box
            Returns an empty list when no face is detected.
        """
        if frame is None or frame.size == 0:
            return []

        try:
            raw_results = self._detector.detect_emotions(frame)
        except Exception:
            return []

        if not raw_results:
            return []

        results = []
        for face in raw_results:
            emotions: dict = face.get("emotions", {})
            box: tuple = face.get("box", (0, 0, 0, 0))

            if not emotions:
                continue

            dominant_label = max(emotions, key=emotions.get)
            dominant_conf = emotions[dominant_label]

            results.append({
                "emotion": dominant_label,
                "confidence": round(dominant_conf, 4),
                "all_emotions": {k: round(v, 4) for k, v in emotions.items()},
                "bbox": list(box),
            })

        return results

    def get_dominant_emotion(self, frame: np.ndarray) -> Optional[tuple[str, float]]:
        """Return the dominant emotion for the first detected face.

        Args:
            frame: numpy array representing a BGR image.

        Returns:
            Tuple of (emotion_label, confidence), or None if no face detected.
        """
        detections = self.detect(frame)
        if not detections:
            return None

        first = detections[0]
        return (first["emotion"], first["confidence"])

    def get_emotion_summary(self, frame: np.ndarray) -> str:
        """Return a human-readable summary of the first detected face's emotions.

        Example output: "happy (85%), hint of surprise (10%)"

        Args:
            frame: numpy array representing a BGR image.

        Returns:
            Descriptive string, or "no face detected" when no face is found.
        """
        detections = self.detect(frame)
        if not detections:
            return "no face detected"

        emotions = detections[0]["all_emotions"]
        # Sort emotions by confidence descending
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

        parts: list[str] = []
        for label, conf in sorted_emotions:
            pct = int(round(conf * 100))
            if pct <= 0:
                continue
            if not parts:
                # Primary emotion
                parts.append(f"{label} ({pct}%)")
            elif pct >= 5:
                # Secondary emotions shown as hints
                parts.append(f"hint of {label} ({pct}%)")

        return ", ".join(parts) if parts else "no face detected"

    # ------------------------------------------------------------------
    # History & trend analysis
    # ------------------------------------------------------------------

    def add_to_history(self, emotion: str, confidence: float) -> None:
        """Record an emotion detection to the rolling history buffer.

        Args:
            emotion: emotion label (e.g. "happy").
            confidence: confidence score (0-1).
        """
        self._history.append({
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "timestamp": time.time(),
        })

    def get_emotion_trend(self) -> dict:
        """Analyse the recent emotion history and return a trend summary.

        Returns:
            Dict with:
                - current (str): most recent emotion, or "unknown"
                - trend (str): "stable", "rising", or "falling"
                - duration_sec (float): seconds the current emotion has persisted
            Returns {"current": "unknown", "trend": "stable", "duration_sec": 0}
            when history is empty.
        """
        if not self._history:
            return {"current": "unknown", "trend": "stable", "duration_sec": 0}

        entries = list(self._history)
        current_emotion = entries[-1]["emotion"]
        current_conf = entries[-1]["confidence"]
        now = entries[-1]["timestamp"]

        # Duration: how long the current emotion has been dominant (consecutive)
        duration_start = now
        for entry in reversed(entries[:-1]):
            if entry["emotion"] == current_emotion:
                duration_start = entry["timestamp"]
            else:
                break
        duration_sec = round(now - duration_start, 2)

        # Trend: compare average confidence of the current emotion in the
        # first half vs second half of history.
        same_emotion_entries = [e for e in entries if e["emotion"] == current_emotion]
        if len(same_emotion_entries) < 2:
            trend = "stable"
        else:
            mid = len(same_emotion_entries) // 2
            first_half_avg = sum(e["confidence"] for e in same_emotion_entries[:mid]) / mid
            second_half_avg = sum(
                e["confidence"] for e in same_emotion_entries[mid:]
            ) / (len(same_emotion_entries) - mid)

            diff = second_half_avg - first_half_avg
            if diff > 0.05:
                trend = "rising"
            elif diff < -0.05:
                trend = "falling"
            else:
                trend = "stable"

        return {
            "current": current_emotion,
            "trend": trend,
            "duration_sec": duration_sec,
        }


# ----------------------------------------------------------------------
# CLI test mode
# ----------------------------------------------------------------------

def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="Emotion Detection CLI test mode")
    parser.add_argument("--test", type=str, required=True, metavar="IMAGE_PATH",
                        help="Path to an image file to analyse")
    args = parser.parse_args()

    image = cv2.imread(args.test)
    if image is None:
        print(json.dumps({"error": f"Could not load image: {args.test}"}, indent=2))
        return

    detector = EmotionDetector()
    detections = detector.detect(image)

    if not detections:
        print(json.dumps({"result": "no face detected"}, indent=2))
        return

    # Also populate history for the first face
    first = detections[0]
    detector.add_to_history(first["emotion"], first["confidence"])

    output = {
        "detections": detections,
        "summary": detector.get_emotion_summary(image),
        "trend": detector.get_emotion_trend(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    _cli_main()
