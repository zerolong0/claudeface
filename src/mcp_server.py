"""ClaudeFace MCP Server.

Exposes tools for Claude to query user emotional state:
  - get_user_mood: current emotion + confidence + trend
  - get_interaction_suggestion: context-aware interaction advice
  - get_daemon_status: health check
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
VISION_BINARY = _PROJECT_ROOT / "bin" / "claudeface-vision"

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
def get_interaction_suggestion() -> str:
    """Get a suggestion for how to interact with the user based on their current mood.

    Returns JSON with: tone, context (natural language guidance), suggestions, priority.
    Call this before responding to understand the recommended communication style.
    """
    state = _read_state()
    if state is None or state.get("emotion") is None:
        return json.dumps({
            "tone": "balanced",
            "context": "Unable to detect user emotion. Interact normally.",
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
def get_user_portrait() -> str:
    """Get a colored pixel art portrait of the user captured via webcam.

    Returns ANSI 24-bit true color pixel art using Unicode half-block characters.
    The portrait uses face detection to crop and center on the user's face.
    """
    import subprocess

    if not VISION_BINARY.exists():
        return json.dumps({
            "status": "error",
            "message": "Vision binary not found. Run setup.sh to compile.",
        })

    try:
        result = subprocess.run(
            [str(VISION_BINARY), "--pixel", "40", "10"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return json.dumps({
                "status": "error",
                "message": result.stderr.strip() or "Failed to capture portrait.",
            })
        return result.stdout
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "error", "message": "Portrait capture timed out."})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def get_daemon_status() -> str:
    """Check the health of the ClaudeFace daemon process.

    Returns JSON with: running (bool), pid, state_age_sec, binary_exists.
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

    result["binary_exists"] = VISION_BINARY.exists()

    return json.dumps(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
