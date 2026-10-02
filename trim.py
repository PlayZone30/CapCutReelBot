"""
trim.py — cut a clip down to a specific [start_seconds, end_seconds) sub-range.

Used to turn Gemini's per-shot in/out recommendations (see gemini_score.py's
ClipAnalysis/Shot) into an actual rendered clip before any further
processing (speed_process.py, reframe.py) touches it. Trimming first means
every downstream step — most importantly RIFE frame interpolation in
speed_process.py — only ever runs on the footage that actually survives
into the final cut, not the full raw clip. See the discussion in
implementationplan.md / the PR notes for why trim-before-slow-mo is the
right order (compute, Gemini cost, and creative judgment all favor it).

Uses -ss/-to as OUTPUT options (placed after -i) rather than as an input
option before -i. That's slower — it forces a full re-encode instead of a
keyframe-aligned stream copy — but it's frame-accurate, which matters here
since a single raw clip can be split into several short shots and an
inaccurate (keyframe-snapped) trim could visibly clip into the wrong moment.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import config


class TrimError(RuntimeError):
    pass


def trim_clip(src_path: str | Path, dest_path: str | Path,
              start_seconds: float, end_seconds: float) -> Path:
    """
    Re-encode the [start_seconds, end_seconds) sub-range of src_path into
    dest_path. Raises TrimError if the range is invalid or ffmpeg fails.
    """
    src_path = Path(src_path)
    dest_path = Path(dest_path)

    if end_seconds <= start_seconds:
        raise TrimError(
            f"end_seconds ({end_seconds}) must be greater than start_seconds "
            f"({start_seconds}) for {src_path}"
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-ss", str(start_seconds),
        "-to", str(end_seconds),
        *config.video_codec_args(),
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        str(dest_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise TrimError("ffmpeg not found. Install ffmpeg (e.g. `brew install ffmpeg`).") from e
    except subprocess.CalledProcessError as e:
        raise TrimError(
            f"ffmpeg trim failed on {src_path} [{start_seconds}-{end_seconds}]: {e.stderr.strip()}"
        ) from e

    return dest_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python trim.py <src_video> <dest_video> <start_seconds> <end_seconds>")
        raise SystemExit(1)

    src, dest, start, end = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    out = trim_clip(src, dest, start, end)
    print(f"-> {out}")
