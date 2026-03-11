"""Background daemon that periodically captures camera frames,
detects emotions, and writes state to ~/.claudeface/state.json.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

# Resolve project root so sibling imports work when invoked from anywhere
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from camera import CameraCapture
from emotion import EmotionDetector

STATE_DIR = Path.home() / ".claudeface"
STATE_FILE = STATE_DIR / "state.json"
PID_FILE = STATE_DIR / "daemon.pid"
DEFAULT_INTERVAL = 10  # seconds
DEFAULT_IDLE_TIMEOUT = 7200  # 2 hours


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


# ------------------------------------------------------------------
# Main daemon loop
# ------------------------------------------------------------------

def run_daemon(interval: float = DEFAULT_INTERVAL,
               idle_timeout: float = DEFAULT_IDLE_TIMEOUT) -> None:
    """Run the capture-detect loop until stopped or idle timeout."""

    camera = CameraCapture()
    detector = EmotionDetector()
    start_time = time.time()
    running = True

    def _shutdown(signum, _frame):
        nonlocal running
        print(f"\n[daemon] Received signal {signum}, shutting down…")
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    _write_pid()
    print(f"[daemon] Started (pid={os.getpid()}, interval={interval}s, "
          f"idle_timeout={idle_timeout}s)")

    try:
        while running:
            elapsed = time.time() - start_time
            if elapsed > idle_timeout:
                print("[daemon] Idle timeout reached, shutting down.")
                break

            frame = camera.capture_frame()
            if frame is None:
                state = {
                    "status": "no_camera",
                    "emotion": None,
                    "confidence": 0,
                    "all_emotions": {},
                    "trend": "unknown",
                    "duration_sec": 0,
                    "timestamp": time.time(),
                    "error": "Camera not available",
                }
                _write_state(state)
                time.sleep(interval)
                continue

            detections = detector.detect(frame)

            if detections:
                face = detections[0]
                detector.add_to_history(face["emotion"], face["confidence"])
                trend_info = detector.get_emotion_trend()
                summary = detector.get_emotion_summary(frame)

                state = {
                    "status": "active",
                    "emotion": face["emotion"],
                    "confidence": face["confidence"],
                    "all_emotions": face["all_emotions"],
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

            _write_state(state)
            time.sleep(interval)

    finally:
        camera.stop()
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
