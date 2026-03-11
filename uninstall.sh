#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLAUDE_DIR="$HOME/.claude"
MCP_FILE="$CLAUDE_DIR/.mcp.json"
SETTINGS_FILE="$CLAUDE_DIR/settings.json"
STATE_DIR="$HOME/.claudeface"

echo ""
echo "  ClaudeFace Uninstaller"
echo "  ─────────────────────"
echo ""

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" &>/dev/null; then
    PYTHON="$candidate"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  PYTHON="python3"
fi

# ──────────────────────────────────────────
# Step 1: Stop daemon
# ──────────────────────────────────────────
echo "[1/4] Stopping daemon..."
PID_FILE="$STATE_DIR/daemon.pid"
if [ -f "$PID_FILE" ]; then
  pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    echo "  ✓ Daemon stopped (pid=$pid)"
  else
    echo "  ✓ Daemon not running"
  fi
  rm -f "$PID_FILE"
else
  echo "  ✓ Daemon not running"
fi

# ──────────────────────────────────────────
# Step 2: Remove MCP server
# ──────────────────────────────────────────
echo "[2/4] Removing MCP server..."
if [ -f "$MCP_FILE" ]; then
  "$PYTHON" -c "
import json
with open('$MCP_FILE') as f: d = json.load(f)
if 'mcpServers' in d and 'claudeface' in d['mcpServers']:
    del d['mcpServers']['claudeface']
    with open('$MCP_FILE', 'w') as f: json.dump(d, f, indent=2)
    print('  ✓ Removed from $MCP_FILE')
else:
    print('  ✓ Not found in $MCP_FILE')
" 2>/dev/null || echo "  ⚠ Could not update $MCP_FILE"
else
  echo "  ✓ No MCP config found"
fi

# ──────────────────────────────────────────
# Step 3: Remove hooks
# ──────────────────────────────────────────
echo "[3/4] Removing hooks..."
if [ -f "$SETTINGS_FILE" ]; then
  "$PYTHON" <<'PYEOF'
import json

settings_file = "$SETTINGS_FILE"
# Re-read with proper path
PYEOF

  "$PYTHON" -c "
import json

with open('$SETTINGS_FILE') as f:
    settings = json.load(f)

hooks = settings.get('hooks', {})
changed = False

for event in ['SessionStart', 'UserPromptSubmit']:
    if event in hooks:
        original = hooks[event]
        filtered = [
            entry for entry in original
            if not any('claudeface' in h.get('command', '') for h in entry.get('hooks', []))
        ]
        if len(filtered) != len(original):
            hooks[event] = filtered
            changed = True
        # Remove empty arrays
        if not hooks[event]:
            del hooks[event]

if not hooks and 'hooks' in settings:
    del settings['hooks']

if changed:
    with open('$SETTINGS_FILE', 'w') as f:
        json.dump(settings, f, indent=2)
    print('  ✓ Hooks removed from $SETTINGS_FILE')
else:
    print('  ✓ No ClaudeFace hooks found')
" 2>/dev/null || echo "  ⚠ Could not update $SETTINGS_FILE"
else
  echo "  ✓ No settings file found"
fi

# ──────────────────────────────────────────
# Step 4: Clean up state directory
# ──────────────────────────────────────────
echo "[4/4] Cleaning up..."
if [ -d "$STATE_DIR" ]; then
  rm -rf "$STATE_DIR"
  echo "  ✓ Removed $STATE_DIR"
else
  echo "  ✓ No state directory found"
fi

echo ""
echo "  ✓ ClaudeFace uninstalled"
echo ""
echo "  Note: The plugin source code at $SCRIPT_DIR is preserved."
echo "  To reinstall: bash $SCRIPT_DIR/setup.sh"
echo "  To fully remove: rm -rf $SCRIPT_DIR"
echo ""
