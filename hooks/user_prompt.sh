#!/usr/bin/env bash
# ClaudeFace UserPromptSubmit Hook
# Reads emotion state and outputs additionalContext JSON for Claude.

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PLUGIN_DIR/.venv"
STATE_DIR="$HOME/.claudeface"
STATE_FILE="$STATE_DIR/state.json"
PYTHON="$VENV/bin/python"

# Ensure venv exists
if [ ! -f "$PYTHON" ]; then
  exit 0
fi

# Read state file
if [ ! -f "$STATE_FILE" ]; then
  exit 0
fi

# Check state freshness and generate context
"$PYTHON" -c "
import json, sys, time
sys.path.insert(0, '$PLUGIN_DIR/src')
from strategy import InteractionStrategy

state_path = '$STATE_FILE'
try:
    with open(state_path) as f:
        state = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(0)

# Skip if data is too stale (>120s)
age = time.time() - state.get('timestamp', 0)
if age > 120:
    sys.exit(0)

emotion = state.get('emotion')
if not emotion:
    sys.exit(0)

engine = InteractionStrategy()
strategy = engine.get_strategy(
    emotion=emotion,
    confidence=state.get('confidence', 1.0),
    trend=state.get('trend'),
    duration_sec=state.get('duration_sec', 0),
)

# Build context string
confidence = state.get('confidence', 0)
trend = state.get('trend', 'stable')
context_parts = [
    f'用户当前情绪: {emotion} ({confidence:.0%})',
    f'趋势: {trend}',
    strategy['context'],
]
context = '。'.join(context_parts)

output = {
    'hookSpecificOutput': {
        'additionalContext': context,
    }
}
print(json.dumps(output, ensure_ascii=False))
" 2>/dev/null || true
