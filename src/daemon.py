"""Background daemon that periodically invokes the native Swift vision binary,
detects emotions from landmarks, and writes state to ~/.claudeface/state.json.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Resolve project root so sibling imports work when invoked from anywhere
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from emotion import LandmarkEmotionDetector

STATE_DIR = Path.home() / ".claudeface"
STATE_FILE = STATE_DIR / "state.json"
MINI_PORTRAIT_FILE = STATE_DIR / "mini_portrait.txt"
PORTRAIT_LANDMARK_FILE = STATE_DIR / "portrait_landmark.txt"
PORTRAIT_COLOR_FILE = STATE_DIR / "portrait_color.txt"
PID_FILE = STATE_DIR / "daemon.pid"
VISION_BINARY = _PROJECT_ROOT / "bin" / "claudeface-vision"
DEFAULT_INTERVAL = 10  # seconds
DEFAULT_IDLE_TIMEOUT = 7200  # 2 hours
MINI_PORTRAIT_INTERVAL = 30  # refresh mini portrait every 30s


def _write_state(data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(STATE_FILE)


def _write_pid() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def _remove_pid() -> None:
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def _read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _call_vision_binary() -> dict | None:
    """Call the Swift vision binary and parse its JSON output."""
    if not VISION_BINARY.exists():
        return None

    try:
        result = subprocess.run(
            [str(VISION_BINARY)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


def _update_mini_portrait() -> None:
    """Capture two portrait modes and cache to files:
    1. Landmark (privacy-safe, only coordinate points)
    2. Color pixel art (clear, uses camera pixels)
    Also updates the legacy mini_portrait.txt for statusline.
    """
    if not VISION_BINARY.exists():
        return

    # Landmark portrait (safe mode - no camera pixels)
    try:
        result = subprocess.run(
            [str(VISION_BINARY), "--landmark", "30", "15"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            tmp = PORTRAIT_LANDMARK_FILE.with_suffix(".tmp")
            tmp.write_text(result.stdout.strip())
            tmp.replace(PORTRAIT_LANDMARK_FILE)
            # Also use as default mini portrait for statusline
            tmp2 = MINI_PORTRAIT_FILE.with_suffix(".tmp")
            tmp2.write_text(result.stdout.strip())
            tmp2.replace(MINI_PORTRAIT_FILE)
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Color pixel art portrait (clear mode - uses camera pixels)
    try:
        result = subprocess.run(
            [str(VISION_BINARY), "--pixel", "40", "10"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            tmp = PORTRAIT_COLOR_FILE.with_suffix(".tmp")
            tmp.write_text(result.stdout.strip())
            tmp.replace(PORTRAIT_COLOR_FILE)
    except (subprocess.TimeoutExpired, OSError):
        pass


# ------------------------------------------------------------------
# Main daemon loop
# ------------------------------------------------------------------

def run_daemon(interval: float = DEFAULT_INTERVAL,
               idle_timeout: float = DEFAULT_IDLE_TIMEOUT) -> None:
    """Run the capture-detect loop until stopped or idle timeout."""

    detector = LandmarkEmotionDetector()
    start_time = time.time()
    last_mini_update = 0.0
    running = True

    def _shutdown(signum, _frame):
        nonlocal running
        print(f"\n[daemon] Received signal {signum}, shutting down...")
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    _write_pid()
    print(f"[daemon] Started (pid={os.getpid()}, interval={interval}s, "
          f"idle_timeout={idle_timeout}s)")
    print(f"[daemon] Vision binary: {VISION_BINARY}")

    try:
        while running:
            elapsed = time.time() - start_time
            if elapsed > idle_timeout:
                print("[daemon] Idle timeout reached, shutting down.")
                break

            vision_data = _call_vision_binary()

            if vision_data is None:
                state = {
                    "status": "no_binary",
                    "emotion": None,
                    "confidence": 0,
                    "all_emotions": {},
                    "trend": "unknown",
                    "duration_sec": 0,
                    "timestamp": time.time(),
                    "error": f"Vision binary not found: {VISION_BINARY}",
                }
                _write_state(state)
                time.sleep(interval)
                continue

            vision_status = vision_data.get("status", "error")

            if vision_status == "ok":
                landmarks = vision_data.get("landmarks", {})
                detection = detector.detect_from_landmarks(landmarks)

                if detection:
                    detector.add_to_history(
                        detection["emotion"], detection["confidence"]
                    )
                    trend_info = detector.get_emotion_trend()
                    summary = detector.get_emotion_summary(
                        detection["all_emotions"]
                    )

                    state = {
                        "status": "active",
                        "emotion": detection["emotion"],
                        "confidence": detection["confidence"],
                        "all_emotions": detection["all_emotions"],
                        "summary": summary,
                        "trend": trend_info["trend"],
                        "duration_sec": trend_info["duration_sec"],
                        "timestamp": time.time(),
                    }
                else:
                    state = {
                        "status": "no_face",
                        "emotion": None,
                        "confidence": 0,
                        "all_emotions": {},
                        "trend": "unknown",
                        "duration_sec": 0,
                        "timestamp": time.time(),
                    }
            elif vision_status == "no_face":
                state = {
                    "status": "no_face",
                    "emotion": None,
                    "confidence": 0,
                    "all_emotions": {},
                    "trend": "unknown",
                    "duration_sec": 0,
                    "timestamp": time.time(),
                }
            else:
                state = {
                    "status": vision_status,
                    "emotion": None,
                    "confidence": 0,
                    "all_emotions": {},
                    "trend": "unknown",
                    "duration_sec": 0,
                    "timestamp": time.time(),
                    "error": vision_data.get("message", "Unknown error"),
                }

            _write_state(state)

            # Refresh mini portrait periodically
            now = time.time()
            if now - last_mini_update >= MINI_PORTRAIT_INTERVAL:
                _update_mini_portrait()
                last_mini_update = now

            time.sleep(interval)

    finally:
        _remove_pid()
        print("[daemon] Stopped.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def cmd_start(args):
    existing = _read_pid()
    if existing and _is_running(existing):
        print(f"[daemon] Already running (pid={existing}).")
        return

    # Check that the vision binary exists
    if not VISION_BINARY.exists():
        print(f"[daemon] ERROR: Vision binary not found at {VISION_BINARY}")
        print(f"[daemon] Run setup.sh to compile it, or:")
        print(f"  swiftc -O -o {VISION_BINARY} {_PROJECT_ROOT}/src/claudeface_vision.swift "
              f"-framework AVFoundation -framework Vision -framework CoreMedia")
        return

    if args.foreground:
        run_daemon(interval=args.interval, idle_timeout=args.idle_timeout)
    else:
        # Fork to background
        pid = os.fork()
        if pid > 0:
            print(f"[daemon] Started in background (pid={pid}).")
            return
        # Child process
        os.setsid()
        run_daemon(interval=args.interval, idle_timeout=args.idle_timeout)


def cmd_stop(_args):
    pid = _read_pid()
    if pid is None:
        print("[daemon] Not running (no PID file).")
        return
    if not _is_running(pid):
        print(f"[daemon] Stale PID file (pid={pid}), removing.")
        _remove_pid()
        return
    os.kill(pid, signal.SIGTERM)
    print(f"[daemon] Sent SIGTERM to pid={pid}.")


def cmd_status(_args):
    pid = _read_pid()
    if pid and _is_running(pid):
        print(f"[daemon] Running (pid={pid}).")
    else:
        print("[daemon] Not running.")

    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        ts = state.get("timestamp", 0)
        age = time.time() - ts
        print(f"[state]  Last update: {age:.0f}s ago")
        print(f"[state]  Status: {state.get('status')}")
        print(f"[state]  Emotion: {state.get('emotion')} "
              f"({state.get('confidence', 0):.0%})")
        print(f"[state]  Trend: {state.get('trend')}")
    else:
        print("[state]  No state file found.")

    if not VISION_BINARY.exists():
        print(f"[binary] WARNING: {VISION_BINARY} not found. Run setup.sh.")
    else:
        print(f"[binary] OK: {VISION_BINARY}")


def main():
    parser = argparse.ArgumentParser(description="ClaudeFace Daemon")
    sub = parser.add_subparsers(dest="command")

    p_start = sub.add_parser("start", help="Start the daemon")
    p_start.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                         help=f"Capture interval in seconds (default: {DEFAULT_INTERVAL})")
    p_start.add_argument("--idle-timeout", type=float, default=DEFAULT_IDLE_TIMEOUT,
                         help=f"Idle timeout in seconds (default: {DEFAULT_IDLE_TIMEOUT})")
    p_start.add_argument("--foreground", action="store_true",
                         help="Run in foreground (do not fork)")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop the daemon")
    p_stop.set_defaults(func=cmd_stop)

    p_status = sub.add_parser("status", help="Check daemon status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
