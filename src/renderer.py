"""Terminal Portrait Renderer.

Renders images in the terminal using iTerm2 inline image protocol,
Kitty graphics protocol, or colored/plain ASCII art fallback.
"""

import argparse
import base64
import os
import sys
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

EMOTION_EMOJIS = {
    "happy": "\U0001f60a",
    "sad": "\U0001f622",
    "angry": "\U0001f620",
    "fear": "\U0001f628",
    "surprise": "\U0001f632",
    "disgust": "\U0001f922",
    "neutral": "\U0001f610",
}

ASCII_CHARS = " .:-=+*#%@"


class TerminalRenderer:
    """Render images in a terminal using the best available protocol."""

    # ------------------------------------------------------------------
    # Protocol detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_terminal_protocol() -> str:
        """Return the best rendering protocol for the current terminal.

        Returns:
            ``"iterm2"`` for iTerm2 / WezTerm,
            ``"kitty"`` for Kitty,
            ``"ascii"`` for everything else.
        """
        term_program = os.environ.get("TERM_PROGRAM", "").lower()
        if term_program in ("iterm2", "iterm.app", "wezterm"):
            return "iterm2"
        if term_program == "kitty" or "kitty" in term_program:
            return "kitty"
        return "ascii"

    @staticmethod
    def _supports_256color() -> bool:
        """Check whether the terminal advertises 256-color support."""
        term = os.environ.get("TERM", "")
        colorterm = os.environ.get("COLORTERM", "").lower()
        return "256color" in term or colorterm in ("truecolor", "24bit")

    # ------------------------------------------------------------------
    # iTerm2 inline image protocol
    # ------------------------------------------------------------------

    @staticmethod
    def render_iterm2(image_bytes: bytes) -> None:
        """Print an image using the iTerm2 inline-image protocol.

        Args:
            image_bytes: Raw image file bytes (PNG, JPEG, etc.).
        """
        b64_data = base64.b64encode(image_bytes).decode("ascii")
        sys.stdout.write(
            f"\033]1337;File=inline=1;width=20;height=10:{b64_data}\a"
        )
        sys.stdout.write("\n")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # ASCII art rendering
    # ------------------------------------------------------------------

    @classmethod
    def render_ascii(cls, frame: np.ndarray, width: int = 60) -> None:
        """Convert a BGR numpy image to ASCII art and print it.

        If the terminal supports 256 colors the output will include ANSI
        color escape sequences so the ASCII art is colorised.

        Args:
            frame: A numpy array in BGR color order (OpenCV convention).
            width: The desired character width of the output.
        """
        h, w = frame.shape[:2]
        # Characters are roughly twice as tall as they are wide, so we
        # halve the vertical resolution to preserve the aspect ratio.
        aspect = h / w
        height = int(width * aspect * 0.45)

        # Resize the image to the target character grid.
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Convert to RGB for colour output and compute grey for char mapping.
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        use_color = cls._supports_256color()
        lines: list[str] = []

        for y in range(height):
            row_chars: list[str] = []
            for x in range(width):
                brightness = grey[y, x]
                char_idx = int(brightness / 255 * (len(ASCII_CHARS) - 1))
                char = ASCII_CHARS[char_idx]

                if use_color:
                    r, g, b = int(rgb[y, x, 0]), int(rgb[y, x, 1]), int(rgb[y, x, 2])
                    # Map to the 6x6x6 ANSI-256 colour cube (indices 16-231).
                    ansi = 16 + (36 * (r * 5 // 255)) + (6 * (g * 5 // 255)) + (b * 5 // 255)
                    row_chars.append(f"\033[38;5;{ansi}m{char}\033[0m")
                else:
                    row_chars.append(char)

            lines.append("".join(row_chars))

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # High-level render
    # ------------------------------------------------------------------

    @classmethod
    def render_portrait(
        cls,
        frame: np.ndarray,
        emotion_label: str | None = None,
    ) -> None:
        """Render a portrait image using the best available method.

        Args:
            frame: BGR numpy image (OpenCV format).
            emotion_label: Optional emotion string (e.g. ``"happy"``).
                If provided, an emoji badge is printed below the image.
        """
        protocol = cls.detect_terminal_protocol()

        if protocol == "iterm2":
            # Encode the frame as PNG bytes for the inline protocol.
            success, buf = cv2.imencode(".png", frame)
            if success:
                cls.render_iterm2(buf.tobytes())
            else:
                # Fall back to ASCII on encoding failure.
                cls.render_ascii(frame)
        elif protocol == "kitty":
            # Kitty uses its own graphics protocol but for simplicity we
            # fall back to ASCII here.  A full Kitty implementation would
            # use the APC escape sequence.
            cls.render_ascii(frame)
        else:
            cls.render_ascii(frame)

        if emotion_label:
            emoji = EMOTION_EMOJIS.get(emotion_label.lower(), "")
            label = f"{emoji} {emotion_label}" if emoji else emotion_label
            sys.stdout.write(f"  {label}\n")
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # File-based convenience
    # ------------------------------------------------------------------

    @classmethod
    def render_from_file(
        cls,
        image_path: str,
        emotion_label: str | None = None,
    ) -> None:
        """Load an image from *image_path* and render it.

        Args:
            image_path: Path to an image file readable by OpenCV.
            emotion_label: Optional emotion label for the badge.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            sys.stderr.write(f"Error: could not load image '{image_path}'\n")
            sys.exit(1)
        cls.render_portrait(frame, emotion_label=emotion_label)


# ----------------------------------------------------------------------
# CLI test mode
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal Portrait Renderer — test mode",
    )
    parser.add_argument(
        "--test",
        metavar="IMAGE_PATH",
        required=True,
        help="Path to an image file to render in the terminal.",
    )
    parser.add_argument(
        "--emotion",
        default=None,
        help="Optional emotion label (happy, sad, angry, fear, surprise, disgust, neutral).",
    )
    args = parser.parse_args()

    TerminalRenderer.render_from_file(args.test, emotion_label=args.emotion)


if __name__ == "__main__":
    main()
