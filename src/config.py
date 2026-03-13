"""ClaudeFace configuration management."""

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".claudeface"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS = {
    "portrait_mode": "safe",  # "safe" | "clear" | "both"
}


def load() -> dict:
    """Load config, creating with defaults if missing."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cfg = json.loads(CONFIG_FILE.read_text())
        # Merge with defaults for any missing keys
        for k, v in DEFAULTS.items():
            cfg.setdefault(k, v)
        return cfg
    except (FileNotFoundError, json.JSONDecodeError):
        save(DEFAULTS)
        return dict(DEFAULTS)


def save(cfg: dict) -> None:
    """Write config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    tmp.replace(CONFIG_FILE)


def get_portrait_mode() -> str:
    return load().get("portrait_mode", "clear")


def set_portrait_mode(mode: str) -> str:
    """Set portrait mode. Returns confirmation message."""
    if mode not in ("safe", "clear", "both"):
        return f"Invalid mode '{mode}'. Choose: safe, clear, both"
    cfg = load()
    old = cfg.get("portrait_mode", "clear")
    cfg["portrait_mode"] = mode
    save(cfg)
    return f"Portrait mode changed: {old} → {mode}"
