#!/usr/bin/env bash
# ClaudeFace SessionStart Hook
# Renders ASCII portrait + emotion, then starts the daemon.

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PLUGIN_DIR/.venv"
STATE_DIR="$HOME/.claudeface"
STATE_FILE="$STATE_DIR/state.json"
PID_FILE="$STATE_DIR/daemon.pid"
PYTHON="$VENV/bin/python"
VISION_BINARY="$PLUGIN_DIR/bin/claudeface-vision"

# Ensure prerequisites
if [ ! -f "$PYTHON" ]; then
  echo "[ClaudeFace] Virtual environment not found. Run setup.sh first." >&2
  exit 0
fi
if [ ! -f "$VISION_BINARY" ]; then
  echo "[ClaudeFace] Vision binary not found. Run setup.sh to compile." >&2
  exit 0
fi

# --- ASCII Portrait + Emotion Detection ---
# This is the "wow" moment: Claude Code recognizes you!
DETECT_JSON=$("$VISION_BINARY" 2>/dev/null || echo '{"status":"error"}')
DETECT_STATUS=$(echo "$DETECT_JSON" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin).get('status','error'))" 2>/dev/null || echo "error")

if [ "$DETECT_STATUS" = "ok" ]; then
  # Render portrait using best available terminal protocol (bypasses Claude context capture)
  # Auto-detects: iTerm2 → Kitty → Sixel → ANSI half-block pixel art
  "$VISION_BINARY" --image 300 300 > /dev/tty 2>/dev/null || true

  # Show emotion as compact one-liner (this goes to Claude context, intentionally small)
  "$PYTHON" -c "
import sys, json
sys.path.insert(0, '$PLUGIN_DIR/src')
from emotion import LandmarkEmotionDetector

data = json.loads('''$DETECT_JSON''')
d = LandmarkEmotionDetector()
r = d.detect_from_landmarks(data.get('landmarks', {}))
if r:
    emojis = {'happy': ':)', 'sad': ':(', 'angry': '>:(', 'surprise': ':O', 'neutral': ':|'}
    e = r['emotion']
    emoji = emojis.get(e, '')
    summary = d.get_emotion_summary(r['all_emotions'])
    print(f'ClaudeFace {emoji} {summary}')
" 2>/dev/null || true
elif [ "$DETECT_STATUS" = "no_face" ]; then
  echo "  [ClaudeFace] Camera active, no face detected" >&2
fi

# --- Start Daemon ---
daemon_running=false
if [ -f "$PID_FILE" ]; then
  pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    daemon_running=true
  fi
fi

if [ "$daemon_running" = false ]; then
  "$PYTHON" "$PLUGIN_DIR/src/daemon.py" start &>/dev/null &
fi
