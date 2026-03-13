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
import config as cfg

STATE_DIR = Path.home() / ".claudeface"
STATE_FILE = STATE_DIR / "state.json"
MINI_PORTRAIT_FILE = STATE_DIR / "mini_portrait.txt"
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
    """Get user portrait from daemon cache.

    Returns the portrait based on user's configured mode:
    - "safe": Landmark line-art (Vision coordinates only, no camera pixels, privacy-safe)
    - "clear": Color pixel art (uses camera data, more detailed)

    The mode can be changed with set_portrait_mode tool.
    Portraits are cached by daemon every 30 seconds. No additional photo is taken.
    """
    mode = cfg.get_portrait_mode()

    safe_file = STATE_DIR / "portrait_landmark.txt"
    clear_file = STATE_DIR / "portrait_color.txt"

    # Return based on configured mode
    if mode == "safe":
        try:
            portrait = safe_file.read_text().strip()
            if portrait:
                return portrait
        except FileNotFoundError:
            pass
    elif mode == "clear":
        try:
            portrait = clear_file.read_text().strip()
            if portrait:
                return portrait
        except FileNotFoundError:
            pass
    elif mode == "both":
        result = {}
        try:
            result["safe"] = safe_file.read_text().strip()
        except FileNotFoundError:
            pass
        try:
            result["clear"] = clear_file.read_text().strip()
        except FileNotFoundError:
            pass
        if result:
            return json.dumps(result, ensure_ascii=False)

    return json.dumps({
        "status": "unavailable",
        "message": f"No cached portrait (mode={mode}). Daemon may not be running.",
    })


@mcp.tool()
def set_portrait_mode(mode: str) -> str:
    """Switch portrait mode.

    Modes:
    - "safe": Privacy-safe line-art portrait. Uses only Vision framework coordinate
      points to draw face outline. NO camera pixels stored or transmitted. Recommended
      for privacy-conscious users.
    - "clear": Color pixel art portrait. Uses camera image data to render a low-resolution
      colored portrait. More detailed and recognizable. All data stays local only.
    - "both": Cache both modes, return both when queried.

    Privacy notice:
    - Safe mode: Only facial landmark coordinates (x,y points) are used. No image data.
    - Clear mode: Camera pixels are downsampled to ~40x10 character blocks (~400 colored cells).
      Data is cached locally at ~/.claudeface/ and never uploaded anywhere.
    """
    return cfg.set_portrait_mode(mode)


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
