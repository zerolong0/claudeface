# ClaudeFace 😊

> Emotion-aware plugin for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — lets Claude sense your mood through your MacBook camera and adapt its communication style accordingly.

ClaudeFace captures your facial expressions via webcam, detects emotions using deep learning, and injects emotional context into every Claude Code conversation. When you're happy, Claude stays upbeat. When you're frustrated, Claude gets straight to the point. When you're tired, Claude suggests a break.

## How It Works

```
MacBook Camera → OpenCV (every 10s) → FER Emotion Detection → state.json
                                                                   │
  ┌────────────────────────────────────────────────────────────────┘
  │
  ├── SessionStart Hook → renders your portrait in terminal
  ├── UserPromptSubmit Hook → injects mood context into every message
  └── MCP Tools → Claude can actively query your emotional state
```

## Features

- **7 Emotion Detection**: happy, sad, angry, fear, surprise, disgust, neutral
- **Trend Analysis**: tracks mood changes over time (rising/falling/stable)
- **Fatigue Detection**: prolonged low-confidence neutral → suggests a break
- **Terminal Portrait**: iTerm2 inline image or colored ASCII art fallback
- **Adaptive Strategies**: maps emotions to communication styles automatically
- **MCP Tools**: Claude can check your mood, take your photo, get interaction advice

## Requirements

- macOS with built-in camera (MacBook)
- Python >= 3.10
- Claude Code CLI

## Installation

```bash
git clone https://github.com/zerolong0/claudeface.git
cd claudeface
bash setup.sh
```

The installer will:
1. Create a Python virtual environment and install dependencies
2. Register the MCP server in `~/.claude/.mcp.json`
3. Register hooks in `~/.claude/settings.json`

Then **restart Claude Code** (exit and run `claude` again).

### Camera Permission

On first run, macOS will ask for camera permission. Grant it to your terminal app (iTerm2 / Terminal.app), then **restart the terminal**.

Manage at: **System Settings → Privacy & Security → Camera**

## Usage

Once installed, ClaudeFace works **automatically**:

- **On session start**: captures your portrait and renders it in the terminal
- **On every message**: reads your emotion and injects context for Claude
- Claude adapts its tone — encouraging when you're down, energetic when you're up

### Ask Claude directly

You can also ask Claude to use the MCP tools:

- "What's my mood right now?" → `get_user_mood`
- "Take my photo" → `get_user_portrait`
- "How should you interact with me?" → `get_interaction_suggestion`
- "Is the daemon running?" → `get_daemon_status`

### Emotion → Strategy Mapping

| Emotion | Tone | Claude's Approach |
|---------|------|-------------------|
| 😊 happy | positive | Celebrates progress, maintains energy |
| 😢 sad | supportive | Offers encouragement, simplifies explanations |
| 😠 angry | calm | Direct, solution-focused, no filler |
| 😨 fear | reassuring | Step-by-step guidance, checks understanding |
| 😲 surprise | curious | Clear explanations, provides context |
| 🤢 disgust | professional | Neutral, fact-focused |
| 😐 neutral | balanced | Normal pace |
| 😴 fatigue* | caring | Suggests breaks, keeps responses concise |

*Fatigue detected when neutral emotion persists with low confidence for >5 minutes.

### Manual Daemon Control

```bash
cd /path/to/claudeface

# Check status
.venv/bin/python src/daemon.py status

# Stop daemon
.venv/bin/python src/daemon.py stop

# Start with custom interval
.venv/bin/python src/daemon.py start --interval 5
```

## Uninstall

```bash
cd /path/to/claudeface
bash uninstall.sh
```

Cleanly removes all hooks, MCP config, and state files. Your other Claude Code settings are preserved.

## Architecture

```
claudeface/
├── setup.sh / uninstall.sh    # Install/uninstall scripts
├── hooks/
│   ├── session_start.sh       # SessionStart: launch daemon + render portrait
│   └── user_prompt.sh         # UserPromptSubmit: inject emotion context
├── src/
│   ├── camera.py              # OpenCV webcam capture
│   ├── emotion.py             # FER emotion detection (7 emotions)
│   ├── renderer.py            # Terminal rendering (iTerm2/ASCII)
│   ├── strategy.py            # Emotion → interaction strategy mapping
│   ├── daemon.py              # Background capture-detect loop
│   └── mcp_server.py          # MCP server (4 tools)
└── requirements.txt           # Python dependencies
```

## Tech Stack

- **Camera**: OpenCV (opencv-python-headless)
- **Emotion Detection**: [FER](https://github.com/justinshenk/fer) (FER2013 dataset, MTCNN face detection)
- **Terminal Rendering**: iTerm2 inline image protocol + ANSI 256-color ASCII art
- **MCP Server**: Python [MCP SDK](https://github.com/modelcontextprotocol/python-sdk) (FastMCP)
- **IPC**: File-based (`~/.claudeface/state.json`)

## License

MIT
