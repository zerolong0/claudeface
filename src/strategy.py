"""Interaction Strategy Engine for ClaudeFace.

Maps detected emotions to interaction strategies that guide Claude's
communication style, providing tone, context, and behavioral suggestions.
"""

import json
import sys
from typing import Optional


class InteractionStrategy:
    """Determines interaction strategy based on detected emotion state."""

    STRATEGY_MAP = {
        "happy": {
            "tone": "positive",
            "context": "用户心情不错，保持积极高效的互动。",
            "suggestions": ["celebrate progress", "maintain energy"],
            "priority": "low",
        },
        "sad": {
            "tone": "supportive",
            "context": "用户似乎有些沮丧，请温和鼓励，提供清晰简洁的帮助。",
            "suggestions": [
                "offer encouragement",
                "simplify explanations",
                "acknowledge effort",
            ],
            "priority": "medium",
        },
        "angry": {
            "tone": "calm",
            "context": "用户可能遇到了挫折，请保持冷静专业，直奔问题核心。",
            "suggestions": ["be direct", "focus on solutions", "avoid filler"],
            "priority": "high",
        },
        "surprise": {
            "tone": "curious",
            "context": "用户可能遇到了意外情况，帮助他们理解发生了什么。",
            "suggestions": ["explain clearly", "provide context"],
            "priority": "low",
        },
        "neutral": {
            "tone": "balanced",
            "context": "用户状态平稳，正常互动即可。",
            "suggestions": ["maintain pace"],
            "priority": "low",
        },
    }

    FATIGUE_STRATEGY = {
        "tone": "caring",
        "context": "用户可能有些疲惫，适时建议休息一下。",
        "suggestions": ["suggest a short break", "keep responses concise"],
        "priority": "high",
    }

    def get_strategy(
        self,
        emotion: str,
        confidence: float = 1.0,
        trend: Optional[str] = None,
        duration_sec: float = 0,
    ) -> dict:
        """Return an interaction strategy dict for the given emotion state.

        Args:
            emotion: Detected emotion label (e.g. "happy", "sad", "neutral").
            confidence: Detection confidence score, 0.0 to 1.0.
            trend: Optional trend indicator (e.g. "improving", "declining").
            duration_sec: How long this emotion has been sustained, in seconds.

        Returns:
            Dict with keys: tone, context, suggestions, priority.
        """
        # Fatigue detection: prolonged low-confidence neutral
        if (
            emotion == "neutral"
            and confidence < 0.5
            and duration_sec > 300
        ):
            return dict(self.FATIGUE_STRATEGY)

        base = self.STRATEGY_MAP.get(emotion, self.STRATEGY_MAP["neutral"])
        return dict(base)

    def get_context_prompt(
        self,
        emotion: str,
        confidence: float = 1.0,
        trend: Optional[str] = None,
        duration_sec: float = 0,
    ) -> str:
        """Return just the context string for hook injection."""
        strategy = self.get_strategy(emotion, confidence, trend, duration_sec)
        return strategy["context"]

    def format_hook_output(self, emotion_data: dict) -> str:
        """Format emotion data into a JSON string for Claude Code hook additionalContext.

        Args:
            emotion_data: Dict with at least "emotion" key, plus optional
                          "confidence", "trend", "duration_sec".

        Returns:
            JSON string suitable for hook additionalContext.
        """
        emotion = emotion_data.get("emotion", "neutral")
        confidence = emotion_data.get("confidence", 1.0)
        trend = emotion_data.get("trend", None)
        duration_sec = emotion_data.get("duration_sec", 0)

        strategy = self.get_strategy(emotion, confidence, trend, duration_sec)

        hook_output = {
            "emotion": emotion,
            "confidence": confidence,
            "strategy": strategy,
        }
        if trend is not None:
            hook_output["trend"] = trend

        return json.dumps(hook_output, ensure_ascii=False)


def main():
    """CLI test mode: python src/strategy.py --test <emotion>"""
    if len(sys.argv) >= 3 and sys.argv[1] == "--test":
        emotion = sys.argv[2].lower()
        engine = InteractionStrategy()
        strategy = engine.get_strategy(emotion)
        print(json.dumps(strategy, indent=2, ensure_ascii=False))
    else:
        print("Usage: python src/strategy.py --test <emotion>")
        print("Emotions: happy, sad, angry, surprise, neutral")
        sys.exit(1)


if __name__ == "__main__":
    main()
