"""
probe.py — ffprobe wrapper.

For each clip, returns fps, duration, resolution, and rotation metadata.
This is the "ground truth" step every other stage in the pipeline depends
on: reframe.py and speed_process.py both branch on what this reports.

Rotation handling: phone footage sometimes carries a landscape pixel grid
(e.g. 1920x1080) with a `rotate` side-data tag (90/180/270) instead of true
portrait pixels. `effective_size()` below accounts for that so downstream
code always works off the *displayed* orientation, not the raw pixel grid.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import config


class ProbeError(RuntimeError):
    pass


@dataclass
class ClipInfo:
    path: str
    duration_s: float
    fps: float
    width: int
    height: int
    rotation: int  # degrees: 0, 90, 180, 270 (normalized, clockwise)
    has_audio: bool

    @property
    def duration_us(self) -> int:
        """Duration in microseconds, matching CapCut's draft_info.json units."""
        return round(self.duration_s * 1_000_000)

    @property
    def display_width(self) -> int:
        """Pixel width as actually displayed, after applying rotation."""
        w, _ = self._display_dims()
        return w

    @property
    def display_height(self) -> int:
        """Pixel height as actually displayed, after applying rotation."""
        _, h = self._display_dims()
        return h

    @property
    def is_portrait(self) -> bool:
        return self.display_height >= self.display_width

    def _display_dims(self) -> tuple[int, int]:
        if self.rotation in (90, 270):
            return self.height, self.width
        return self.width, self.height


def _run_ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise ProbeError(
            "ffprobe not found. Install ffmpeg (e.g. `brew install ffmpeg`)."
        ) from e
    except subprocess.CalledProcessError as e:
        raise ProbeError(f"ffprobe failed on {path}: {e.stderr.strip()}") from e
    return json.loads(result.stdout)


def _extract_rotation(video_stream: dict) -> int:
    """
    Rotation can show up in a few places depending on ffmpeg version:
    - stream.tags.rotate (older convention)
    - stream.side_data_list[].rotation (newer convention, signed degrees)
    Normalize to 0/90/180/270 clockwise.
    """
    tags = video_stream.get("tags", {}) or {}
    if "rotate" in tags:
        try:
            return int(tags["rotate"]) % 360
        except (TypeError, ValueError):
            pass

    for side_data in video_stream.get("side_data_list", []) or []:
        if "rotation" in side_data:
            try:
                # side_data rotation is typically negative for clockwise display rotation
                deg = int(round(float(side_data["rotation"])))
                return (-deg) % 360
            except (TypeError, ValueError):
                continue

    return 0


def probe_clip(path: str | Path) -> ClipInfo:
    """Run ffprobe on a single file and return a normalized ClipInfo."""
    path = Path(path)
    if not path.exists():
        raise ProbeError(f"File not found: {path}")

    data = _run_ffprobe(path)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ProbeError(f"No video stream found in {path}")

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    # fps: prefer avg_frame_rate, fall back to r_frame_rate
    fps_raw = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    duration_s = float(
        video_stream.get("duration") or fmt.get("duration") or 0.0
    )

    return ClipInfo(
        path=str(path),
        duration_s=duration_s,
        fps=round(fps, 4),
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        rotation=_extract_rotation(video_stream),
        has_audio=audio_stream is not None,
    )


def probe_folder(folder: str | Path, extensions: tuple[str, ...] = (".mp4", ".mov", ".m4v")) -> list[ClipInfo]:
    """Probe every video file in a folder (non-recursive)."""
    folder = Path(folder)
    clips = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in extensions:
            clips.append(probe_clip(f))
    return clips


def normalize_rotation(src_path: str | Path, dest_path: str | Path,
                        info: ClipInfo | None = None) -> Path:
    """
    Bake any rotation tag into the actual pixels via ffmpeg's transpose
    filter, so downstream steps (reframe, draft_writer) can always treat
    width/height as display dims with rotation=0.

    This matters because a rotation *tag* survives unmodified copies just
    fine in most players, but draft_writer.py writes material width/height
    from ffprobe's raw values and hardcodes the segment's clip.rotation to
    0.0 (copied from the reference draft's two non-rotated sample clips).
    Without normalizing first, a rotated clip would get the wrong
    width/height recorded in draft_info.json and could render sideways or
    with a mismatched canvas in CapCut.

    No-op passthrough (returns src_path unchanged) if rotation == 0.
    """
    src_path = Path(src_path)
    dest_path = Path(dest_path)
    info = info or probe_clip(src_path)

    if info.rotation == 0:
        return src_path

    if info.rotation == 90:
        vf = "transpose=1"
    elif info.rotation == 180:
        vf = "transpose=1,transpose=1"
    elif info.rotation == 270:
        vf = "transpose=2"
    else:
        raise ProbeError(f"Unsupported rotation value {info.rotation} on {src_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        # ffmpeg >= 5 auto-applies the display-matrix rotation on decode by
        # default. Without -noautorotate here, our explicit transpose filter
        # stacks on top of that auto-correction and the two cancel out,
        # leaving the output rotated wrong. See probe.py rotation handling.
        "-noautorotate",
        "-i", str(src_path),
        "-vf", vf,
        *config.video_codec_args(),
        "-c:a", "copy",
        str(dest_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise ProbeError("ffmpeg not found. Install ffmpeg (e.g. `brew install ffmpeg`).") from e
    except subprocess.CalledProcessError as e:
        raise ProbeError(f"ffmpeg rotation normalize failed on {src_path}: {e.stderr.strip()}") from e

    return dest_path


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        print("Usage: python probe.py <video_file_or_folder>")
        raise SystemExit(1)

    p = Path(target)
    infos = [probe_clip(p)] if p.is_file() else probe_folder(p)
    for info in infos:
        print(
            f"{Path(info.path).name}: {info.display_width}x{info.display_height} "
            f"(raw {info.width}x{info.height}, rotation={info.rotation}) "
            f"@ {info.fps}fps, {info.duration_s:.3f}s, "
            f"{'portrait' if info.is_portrait else 'landscape'}, "
            f"audio={info.has_audio}"
        )
