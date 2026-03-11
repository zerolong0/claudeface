#!/usr/bin/env bash
# ClaudeFace SessionStart Hook
# Starts the daemon if not running, captures a portrait, renders it in terminal.

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PLUGIN_DIR/.venv"
STATE_DIR="$HOME/.claudeface"
STATE_FILE="$STATE_DIR/state.json"
PID_FILE="$STATE_DIR/daemon.pid"
PYTHON="$VENV/bin/python"

# Ensure venv exists
if [ ! -f "$PYTHON" ]; then
  echo "[ClaudeFace] Virtual environment not found. Run setup.sh first." >&2
  exit 0
fi

# Start daemon if not running
daemon_running=false
if [ -f "$PID_FILE" ]; then
  pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    daemon_running=true
  fi
fi

if [ "$daemon_running" = false ]; then
  "$PYTHON" "$PLUGIN_DIR/src/daemon.py" start &>/dev/null &
  # Wait briefly for first capture
  sleep 3
fi

# Capture and render portrait
"$PYTHON" -c "
import sys, json
sys.path.insert(0, '$PLUGIN_DIR/src')
from camera import CameraCapture
from emotion import EmotionDetector
from renderer import TerminalRenderer

cam = CameraCapture()
frame = cam.capture_frame()
if frame is None:
    print('[ClaudeFace] Camera not available.', file=sys.stderr)
    sys.exit(0)

detector = EmotionDetector()
dominant = detector.get_dominant_emotion(frame)
emotion = dominant[0] if dominant else None

TerminalRenderer.render_portrait(frame, emotion_label=emotion)
if emotion and dominant:
    conf = dominant[1]
    print(f'  ClaudeFace: Detected {emotion} ({conf:.0%})', file=sys.stderr)
" 2>&1 || true
