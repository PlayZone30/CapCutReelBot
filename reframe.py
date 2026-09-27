"""
reframe.py — center-crop landscape clips to vertical 1080x1920.

Only runs for clips whose *displayed* pixel grid (after accounting for any
rotation tag) is landscape. Clips already vertical are passed through
untouched — this is the expected default per the implementation plan.

Approach: scale so the shorter dimension covers the target, then center-crop
to exactly 1080x1920. This avoids letterboxing and keeps the subject roughly
centered, which is the standard "reels" reframe.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from probe import ClipInfo, probe_clip

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920


class ReframeError(RuntimeError):
    pass


def needs_reframe(info: ClipInfo) -> bool:
    """True only if the clip's actual displayed pixel grid is landscape."""
    return not info.is_portrait


def _ffmpeg_filter(info: ClipInfo) -> str:
    """
    Build a filter chain that:
    1. Applies any rotation tag explicitly (so ffmpeg doesn't silently
       auto-rotate differently than we expect).
    2. Scales to cover 1080x1920 (upscale/downscale as needed).
    3. Center-crops to exactly 1080x1920.
    """
    filters = []

    if info.rotation == 90:
        filters.append("transpose=1")  # 90 clockwise
    elif info.rotation == 180:
        filters.append("transpose=1,transpose=1")
    elif info.rotation == 270:
        filters.append("transpose=2")  # 90 counter-clockwise

    filters.append(
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase"
    )
    filters.append(f"crop={TARGET_WIDTH}:{TARGET_HEIGHT}")

    return ",".join(filters)


def reframe_clip(src_path: str | Path, dest_path: str | Path, info: ClipInfo | None = None) -> Path:
    """
    Center-crop a landscape clip to 1080x1920 vertical. If the clip is
    already vertical, just returns src_path unchanged (no-op, no copy).
    """
    src_path = Path(src_path)
    dest_path = Path(dest_path)
    info = info or probe_clip(src_path)

    if not needs_reframe(info):
        return src_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    filter_chain = _ffmpeg_filter(info)

    cmd = [
        "ffmpeg", "-y",
        # ffmpeg >= 5 auto-applies the display-matrix rotation on decode by
        # default. Since we apply our own transpose below (based on the
        # rotation tag probe.py read), auto-rotate would double up and
        # cancel it out — same issue as probe.normalize_rotation().
        "-noautorotate",
        "-i", str(src_path),
        "-vf", filter_chain,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy",
        str(dest_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise ReframeError("ffmpeg not found. Install ffmpeg (e.g. `brew install ffmpeg`).") from e
    except subprocess.CalledProcessError as e:
        raise ReframeError(f"ffmpeg reframe failed on {src_path}: {e.stderr.strip()}") from e

    return dest_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python reframe.py <src_video> <dest_video>")
        raise SystemExit(1)

    src, dest = sys.argv[1], sys.argv[2]
    info = probe_clip(src)
    if needs_reframe(info):
        out = reframe_clip(src, dest, info)
        print(f"Reframed -> {out}")
    else:
        print(f"{src} is already vertical ({info.display_width}x{info.display_height}); no reframe needed.")
