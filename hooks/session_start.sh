#!/usr/bin/env bash
# ClaudeFace SessionStart Hook
# Starts the daemon if not running, reports initial emotion state.

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PLUGIN_DIR/.venv"
STATE_DIR="$HOME/.claudeface"
STATE_FILE="$STATE_DIR/state.json"
PID_FILE="$STATE_DIR/daemon.pid"
PYTHON="$VENV/bin/python"
VISION_BINARY="$PLUGIN_DIR/bin/claudeface-vision"

# Ensure venv exists
if [ ! -f "$PYTHON" ]; then
  echo "[ClaudeFace] Virtual environment not found. Run setup.sh first." >&2
  exit 0
fi

# Ensure vision binary exists
if [ ! -f "$VISION_BINARY" ]; then
  echo "[ClaudeFace] Vision binary not found. Run setup.sh to compile." >&2
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
  sleep 4
fi

# Report current state
if [ -f "$STATE_FILE" ]; then
  "$PYTHON" -c "
import json, sys, time
try:
    with open('$STATE_FILE') as f:
        state = json.load(f)
    age = time.time() - state.get('timestamp', 0)
    emotion = state.get('emotion', 'unknown')
    confidence = state.get('confidence', 0)
    status = state.get('status', 'unknown')
    if status == 'active' and emotion:
        print(f'[ClaudeFace] Detected: {emotion} ({confidence:.0%})', file=sys.stderr)
    elif status == 'no_face':
        print('[ClaudeFace] No face detected', file=sys.stderr)
    else:
        print(f'[ClaudeFace] Status: {status}', file=sys.stderr)
except Exception:
    pass
" 2>&1 || true
fi
