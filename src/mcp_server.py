"""ClaudeFace MCP Server.

Exposes tools for Claude to query user emotional state:
  - get_user_mood: current emotion + confidence + trend
  - get_user_portrait: capture fresh frame, return base64 image
  - get_interaction_suggestion: context-aware interaction advice
"""

import json
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from strategy import InteractionStrategy

STATE_DIR = Path.home() / ".claudeface"
STATE_FILE = STATE_DIR / "state.json"
PID_FILE = STATE_DIR / "daemon.pid"

mcp = FastMCP("claudeface", instructions="ClaudeFace emotion sensing tools")
strategy_engine = InteractionStrategy()


def _read_state() -> dict | None:
    """Read the current state from the daemon's state.json."""
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _is_daemon_running() -> bool:
    """Check whether the daemon process is alive."""
    import os
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ValueError, OSError):
        return False


@mcp.tool()
def get_user_mood() -> str:
    """Get the user's current emotional state detected via webcam.

    Returns JSON with: emotion, confidence, trend, duration_sec, summary.
    Use this to understand how the user is feeling before responding.
    """
    state = _read_state()
    if state is None:
        return json.dumps({
            "status": "unavailable",
            "message": "ClaudeFace daemon is not running or no state available.",
        }, ensure_ascii=False)

    age = time.time() - state.get("timestamp", 0)
    if age > 60:
        return json.dumps({
            "status": "stale",
            "message": f"Emotion data is {age:.0f}s old. Daemon may have stopped.",
            "last_emotion": state.get("emotion"),
        }, ensure_ascii=False)

    result = {
        "status": state.get("status", "unknown"),
        "emotion": state.get("emotion"),
        "confidence": state.get("confidence", 0),
        "trend": state.get("trend", "unknown"),
        "duration_sec": state.get("duration_sec", 0),
        "summary": state.get("summary", ""),
        "all_emotions": state.get("all_emotions", {}),
    }
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_user_portrait() -> str:
    """Capture a fresh photo from the user's webcam and return it as base64.

    Returns JSON with: image_base64 (JPEG), emotion (if detected).
    Use this when you want to see what the user looks like right now.
    """
    import base64
    try:
        from camera import CameraCapture
        from emotion import EmotionDetector

        cam = CameraCapture()
        frame = cam.capture_frame()
        if frame is None:
            return json.dumps({"error": "Camera not available"})

        import cv2
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        detector = EmotionDetector()
        dominant = detector.get_dominant_emotion(frame)

        result = {
            "image_base64": b64,
            "format": "jpeg",
        }
        if dominant:
            result["emotion"] = dominant[0]
            result["confidence"] = dominant[1]

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_interaction_suggestion() -> str:
    """Get a suggestion for how to interact with the user based on their current mood.

    Returns JSON with: tone, context (natural language guidance), suggestions, priority.
    Call this before responding to understand the recommended communication style.
    """
    state = _read_state()
    if state is None or state.get("emotion") is None:
        return json.dumps({
            "tone": "balanced",
            "context": "无法检测用户情绪，正常互动即可。",
            "suggestions": ["maintain normal pace"],
            "priority": "low",
        }, ensure_ascii=False)

    strat = strategy_engine.get_strategy(
        emotion=state["emotion"],
        confidence=state.get("confidence", 1.0),
        trend=state.get("trend"),
        duration_sec=state.get("duration_sec", 0),
    )
    return json.dumps(strat, ensure_ascii=False)


@mcp.tool()
def get_daemon_status() -> str:
    """Check the health of the ClaudeFace daemon process.

    Returns JSON with: running (bool), pid, state_age_sec.
    """
    running = _is_daemon_running()
    result = {"running": running}

    try:
        result["pid"] = int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        result["pid"] = None

    state = _read_state()
    if state:
        result["state_age_sec"] = round(time.time() - state.get("timestamp", 0), 1)
        result["state_status"] = state.get("status")
    else:
        result["state_age_sec"] = None

    return json.dumps(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
