"""Landmark-Based Emotion Detection Module.

Derives facial emotions from Vision framework face landmark coordinates.
No ML model needed — uses geometric rules on 76 normalized landmark points.

Supported emotions: happy, surprise, sad, angry, neutral.
"""

import json
import sys
import time
from collections import deque
from typing import Optional


EMOTIONS = ["happy", "surprise", "sad", "angry", "neutral"]


class LandmarkEmotionDetector:
    """Detects emotions from face landmark coordinates using geometric rules."""

    def __init__(self) -> None:
        self._history: deque = deque(maxlen=20)

    # ------------------------------------------------------------------
    # Core detection from landmarks
    # ------------------------------------------------------------------

    def detect_from_landmarks(self, landmarks: dict) -> Optional[dict]:
        """Analyze face landmarks and return emotion detection result.

        Args:
            landmarks: Dict of landmark regions from Vision framework.
                Keys: leftEye, rightEye, leftEyebrow, rightEyebrow,
                      outerLips, innerLips, nose.
                Values: list of [x, y] normalized coordinates (0-1),
                        relative to face bounding box,
                        (0,0) = bottom-left, (1,1) = top-right.

        Returns:
            Dict with emotion, confidence, all_emotions, metrics.
            None if landmarks are insufficient.
        """
        if not landmarks:
            return None

        metrics = self._compute_metrics(landmarks)
        if not metrics:
            return None

        scores = self._classify(metrics)
        dominant = max(scores, key=scores.get)
        confidence = scores[dominant]

        return {
            "emotion": dominant,
            "confidence": round(confidence, 4),
            "all_emotions": {k: round(v, 4) for k, v in scores.items()},
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
        }

    # ------------------------------------------------------------------
    # Metric computation from landmark geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(lm: dict) -> dict:
        """Compute facial expression metrics from landmark coordinates."""
        metrics = {}

        # --- Smile detection from outer lips ---
        if "outerLips" in lm and len(lm["outerLips"]) >= 6:
            ol = lm["outerLips"]
            n = len(ol)
            # Points go clockwise from left corner:
            # 0 = left corner, n//2 = right corner
            # n//4 = top center, 3*n//4 = bottom center
            left_corner = ol[0]
            right_corner = ol[n // 2]
            top_center = ol[n // 4]
            bottom_center = ol[3 * n // 4]

            # Smile score: corners above the bottom lip center
            # In a smile, corners rise above the bottom lip; in a frown they drop
            corner_avg_y = (left_corner[1] + right_corner[1]) / 2
            metrics["smile_score"] = corner_avg_y - bottom_center[1]

            # Mouth width-to-height ratio
            mouth_w = abs(right_corner[0] - left_corner[0])
            mouth_h = abs(top_center[1] - bottom_center[1])
            metrics["mouth_width_ratio"] = mouth_w / max(mouth_h, 0.001)

        # --- Mouth openness from inner lips ---
        if "innerLips" in lm and len(lm["innerLips"]) >= 4:
            il = lm["innerLips"]
            n = len(il)
            top_idx = n // 4
            bottom_idx = 3 * n // 4
            inner_gap = abs(il[top_idx][1] - il[bottom_idx][1])
            metrics["mouth_openness"] = inner_gap

        # --- Eye openness ---
        for eye_name in ["leftEye", "rightEye"]:
            if eye_name in lm and len(lm[eye_name]) >= 4:
                eye = lm[eye_name]
                ys = [p[1] for p in eye]
                xs = [p[0] for p in eye]
                height = max(ys) - min(ys)
                width = max(xs) - min(xs)
                metrics[f"{eye_name}_openness"] = height / max(width, 0.001)

        if "leftEye_openness" in metrics and "rightEye_openness" in metrics:
            metrics["eye_openness"] = (
                metrics["leftEye_openness"] + metrics["rightEye_openness"]
            ) / 2

        # --- Brow height relative to eyes ---
        for side in ["left", "right"]:
            brow_key = f"{side}Eyebrow"
            eye_key = f"{side}Eye"
            if brow_key in lm and eye_key in lm:
                brow_avg_y = sum(p[1] for p in lm[brow_key]) / len(lm[brow_key])
                eye_avg_y = sum(p[1] for p in lm[eye_key]) / len(lm[eye_key])
                metrics[f"{side}_brow_height"] = brow_avg_y - eye_avg_y

        if "left_brow_height" in metrics and "right_brow_height" in metrics:
            metrics["brow_height"] = (
                metrics["left_brow_height"] + metrics["right_brow_height"]
            ) / 2

        return metrics

    # ------------------------------------------------------------------
    # Emotion classification from metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(metrics: dict) -> dict:
        """Map facial metrics to emotion scores."""
        smile = metrics.get("smile_score", 0)
        eye_open = metrics.get("eye_openness", 0.3)
        mouth_open = metrics.get("mouth_openness", 0)
        brow_h = metrics.get("brow_height", 0.15)
        mouth_w_ratio = metrics.get("mouth_width_ratio", 3)

        scores = {
            "happy": 0.0,
            "surprise": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "neutral": 0.2,  # baseline
        }

        # Happy: smile (corners elevated above bottom lip) + wide mouth
        if smile > 0.04:
            scores["happy"] += min((smile - 0.03) * 8, 0.8)
        if mouth_w_ratio > 4.5:
            scores["happy"] += 0.15

        # Surprise: eyes wide open + mouth open
        if eye_open > 0.45:
            scores["surprise"] += (eye_open - 0.4) * 3
        if mouth_open > 0.10:
            scores["surprise"] += min(mouth_open * 3, 0.5)

        # Sad: corners at or below bottom lip center
        if smile < 0.02:
            scores["sad"] += min((0.03 - smile) * 8, 0.7)

        # Angry: brow low + eyes narrowed
        if brow_h < 0.10:
            scores["angry"] += (0.15 - brow_h) * 5
        if eye_open < 0.22:
            scores["angry"] += (0.28 - eye_open) * 3

        # Normalize to sum=1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    # ------------------------------------------------------------------
    # History & trend analysis
    # ------------------------------------------------------------------

    def add_to_history(self, emotion: str, confidence: float) -> None:
        """Record an emotion detection to the rolling history buffer."""
        self._history.append({
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "timestamp": time.time(),
        })

    def get_emotion_trend(self) -> dict:
        """Analyse the recent emotion history and return a trend summary.

        Returns:
            Dict with: current, trend ("stable"/"rising"/"falling"), duration_sec.
        """
        if not self._history:
            return {"current": "unknown", "trend": "stable", "duration_sec": 0}

        entries = list(self._history)
        current_emotion = entries[-1]["emotion"]
        now = entries[-1]["timestamp"]

        # Duration: how long the current emotion has been consecutive
        duration_start = now
        for entry in reversed(entries[:-1]):
            if entry["emotion"] == current_emotion:
                duration_start = entry["timestamp"]
            else:
                break
        duration_sec = round(now - duration_start, 2)

        # Trend: compare confidence first-half vs second-half
        same = [e for e in entries if e["emotion"] == current_emotion]
        if len(same) < 2:
            trend = "stable"
        else:
            mid = len(same) // 2
            first_avg = sum(e["confidence"] for e in same[:mid]) / mid
            second_avg = sum(e["confidence"] for e in same[mid:]) / (len(same) - mid)
            diff = second_avg - first_avg
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

    def get_emotion_summary(self, all_emotions: dict) -> str:
        """Return a human-readable summary from emotion scores.

        Example: "happy (72%), hint of neutral (18%)"
        """
        if not all_emotions:
            return "no face detected"

        sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
        parts: list[str] = []
        for label, conf in sorted_emotions:
            pct = int(round(conf * 100))
            if pct <= 0:
                continue
            if not parts:
                parts.append(f"{label} ({pct}%)")
            elif pct >= 5:
                parts.append(f"hint of {label} ({pct}%)")

        return ", ".join(parts) if parts else "no face detected"


# ----------------------------------------------------------------------
# CLI test mode
# ----------------------------------------------------------------------

def _cli_main() -> None:
    """Test with sample landmark data or a JSON file."""
    if len(sys.argv) < 3 or sys.argv[1] != "--test":
        print("Usage: python src/emotion.py --test <landmarks.json>")
        print("  landmarks.json should contain the output of claudeface-vision")
        sys.exit(1)

    with open(sys.argv[2]) as f:
        data = json.load(f)

    if data.get("status") != "ok":
        print(json.dumps({"error": f"Vision status: {data.get('status')}"}, indent=2))
        return

    detector = LandmarkEmotionDetector()
    result = detector.detect_from_landmarks(data.get("landmarks", {}))

    if result is None:
        print(json.dumps({"result": "insufficient landmarks"}, indent=2))
        return

    detector.add_to_history(result["emotion"], result["confidence"])
    output = {
        "detection": result,
        "summary": detector.get_emotion_summary(result["all_emotions"]),
        "trend": detector.get_emotion_trend(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    _cli_main()
