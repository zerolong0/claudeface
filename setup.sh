#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CLAUDE_DIR="$HOME/.claude"
MCP_FILE="$CLAUDE_DIR/.mcp.json"
SETTINGS_FILE="$CLAUDE_DIR/settings.json"
STATE_DIR="$HOME/.claudeface"

echo ""
echo "  ╔═══════════════════════════════════╗"
echo "  ║     ClaudeFace Installer v0.1     ║"
echo "  ║  Emotion-Aware Claude Code Plugin ║"
echo "  ╚═══════════════════════════════════╝"
echo ""

# ──────────────────────────────────────────
# Step 1: Check Python >= 3.10
# ──────────────────────────────────────────
echo "[1/5] Checking Python version..."

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" &>/dev/null; then
    ver=$("$candidate" -c 'import sys; print(sys.version_info.minor)')
    major=$("$candidate" -c 'import sys; print(sys.version_info.major)')
    if [[ "$major" -eq 3 && "$ver" -ge 10 ]]; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "  ❌ Python >= 3.10 is required."
  echo "     Install via: brew install python@3.12"
  exit 1
fi
echo "  ✓ $($PYTHON --version)"

# ──────────────────────────────────────────
# Step 2: Create venv & install dependencies
# ──────────────────────────────────────────
echo "[2/5] Installing dependencies..."

if [ ! -d ".venv" ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/pip install --upgrade pip -q 2>/dev/null
.venv/bin/pip install -r requirements.txt -q 2>/dev/null
echo "  ✓ Python dependencies installed"

# ──────────────────────────────────────────
# Step 3: Create state directory
# ──────────────────────────────────────────
echo "[3/5] Creating state directory..."
mkdir -p "$STATE_DIR"
echo "  ✓ $STATE_DIR"

# ──────────────────────────────────────────
# Step 4: Register MCP Server
# ──────────────────────────────────────────
echo "[4/5] Registering MCP server..."

mkdir -p "$CLAUDE_DIR"

# Build the new MCP server entry
CLAUDEFACE_MCP=$(cat <<EOF
{
  "command": "$SCRIPT_DIR/.venv/bin/python",
  "args": ["$SCRIPT_DIR/src/mcp_server.py"],
  "env": {
    "PYTHONPATH": "$SCRIPT_DIR/src"
  }
}
EOF
)

if [ -f "$MCP_FILE" ]; then
  # Check if claudeface already registered
  if "$PYTHON" -c "import json; d=json.load(open('$MCP_FILE')); exit(0 if 'claudeface' in d.get('mcpServers',{}) else 1)" 2>/dev/null; then
    # Update existing entry
    "$PYTHON" -c "
import json
with open('$MCP_FILE') as f: d = json.load(f)
d['mcpServers']['claudeface'] = json.loads('''$CLAUDEFACE_MCP''')
with open('$MCP_FILE', 'w') as f: json.dump(d, f, indent=2)
"
    echo "  ✓ MCP server updated in $MCP_FILE"
  else
    # Add new entry
    "$PYTHON" -c "
import json
with open('$MCP_FILE') as f: d = json.load(f)
if 'mcpServers' not in d: d['mcpServers'] = {}
d['mcpServers']['claudeface'] = json.loads('''$CLAUDEFACE_MCP''')
with open('$MCP_FILE', 'w') as f: json.dump(d, f, indent=2)
"
    echo "  ✓ MCP server added to $MCP_FILE"
  fi
else
  # Create new .mcp.json
  "$PYTHON" -c "
import json
d = {'mcpServers': {'claudeface': json.loads('''$CLAUDEFACE_MCP''')}}
with open('$MCP_FILE', 'w') as f: json.dump(d, f, indent=2)
"
  echo "  ✓ Created $MCP_FILE"
fi

# ──────────────────────────────────────────
# Step 5: Register Hooks
# ──────────────────────────────────────────
echo "[5/5] Registering hooks..."

SESSION_HOOK_CMD="bash $SCRIPT_DIR/hooks/session_start.sh"
PROMPT_HOOK_CMD="bash $SCRIPT_DIR/hooks/user_prompt.sh"

# Make hooks executable
chmod +x hooks/*.sh

if [ -f "$SETTINGS_FILE" ]; then
  "$PYTHON" <<PYEOF
import json

with open("$SETTINGS_FILE") as f:
    settings = json.load(f)

hooks = settings.setdefault("hooks", {})

# SessionStart hook
session_hooks = hooks.get("SessionStart", [])
already_has_session = any(
    any(h.get("command", "").find("claudeface") >= 0 for h in entry.get("hooks", []))
    for entry in session_hooks
)
if not already_has_session:
    session_hooks.append({
        "matcher": "",
        "hooks": [{
            "type": "command",
            "command": "$SESSION_HOOK_CMD",
            "timeout": 15
        }]
    })
    hooks["SessionStart"] = session_hooks

# UserPromptSubmit hook
prompt_hooks = hooks.get("UserPromptSubmit", [])
already_has_prompt = any(
    any(h.get("command", "").find("claudeface") >= 0 for h in entry.get("hooks", []))
    for entry in prompt_hooks
)
if not already_has_prompt:
    prompt_hooks.append({
        "matcher": "",
        "hooks": [{
            "type": "command",
            "command": "$PROMPT_HOOK_CMD",
            "timeout": 5
        }]
    })
    hooks["UserPromptSubmit"] = prompt_hooks

with open("$SETTINGS_FILE", "w") as f:
    json.dump(settings, f, indent=2)
PYEOF
  echo "  ✓ Hooks registered in $SETTINGS_FILE"
else
  "$PYTHON" -c "
import json
settings = {
    'hooks': {
        'SessionStart': [{'matcher': '', 'hooks': [{'type': 'command', 'command': '$SESSION_HOOK_CMD', 'timeout': 15}]}],
        'UserPromptSubmit': [{'matcher': '', 'hooks': [{'type': 'command', 'command': '$PROMPT_HOOK_CMD', 'timeout': 5}]}]
    }
}
with open('$SETTINGS_FILE', 'w') as f: json.dump(settings, f, indent=2)
"
  echo "  ✓ Created $SETTINGS_FILE with hooks"
fi

# ──────────────────────────────────────────
# Done
# ──────────────────────────────────────────
echo ""
echo "  ╔═══════════════════════════════════╗"
echo "  ║      ✓ Installation Complete      ║"
echo "  ╚═══════════════════════════════════╝"
echo ""
echo "  Next steps:"
echo "    1. Restart Claude Code (exit and run 'claude' again)"
echo "    2. Grant camera permission to your terminal app:"
echo "       System Settings → Privacy & Security → Camera"
echo "    3. Start chatting — ClaudeFace will auto-detect your mood!"
echo ""
echo "  MCP tools available to Claude:"
echo "    • get_user_mood        — check your emotional state"
echo "    • get_user_portrait    — capture your photo"
echo "    • get_interaction_suggestion — get communication advice"
echo ""
echo "  To uninstall: bash $SCRIPT_DIR/uninstall.sh"
echo ""
