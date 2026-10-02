"""
speed_process.py — bake speed changes into the rendered file, before it
ever touches draft_info.json.

Per the implementation plan: every segment we later write to the draft
keeps `speed: 1.0` — we never generate a reverse-engineered speed-curve.
Instead, slow motion is "baked in" here:

    effective_fps = source_fps * speed_factor
        < ~24 fps  -> RIFE frame interpolation (smooth slow-mo)
        else       -> plain `setpts` conform (cheap, no interpolation needed)

speed_factor < 1.0 slows the clip down (more frames needed to fill the same
wall-clock time -> effective_fps drops -> RIFE kicks in below the threshold).
speed_factor > 1.0 speeds it up (setpts branch, no interpolation needed).
speed_factor == 1.0 is a no-op passthrough.

RIFE requires a separate binary (rife-ncnn-vulkan) and model weights that
aren't part of this repo's dependencies. If speed_process picks the RIFE
branch and the binary isn't found, it raises clearly rather than silently
falling back — slow motion without interpolation looks noticeably choppy,
so a silent fallback would produce a worse result than failing loudly.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import config
from probe import ClipInfo, probe_clip

RIFE_FPS_THRESHOLD = 24.0
RIFE_BINARY_NAME = "rife-ncnn-vulkan"


class SpeedProcessError(RuntimeError):
    pass


def effective_fps(info: ClipInfo, speed_factor: float) -> float:
    return info.fps * speed_factor


def choose_branch(info: ClipInfo, speed_factor: float) -> str:
    """Returns 'setpts' or 'rife' depending on the resulting effective fps."""
    if speed_factor == 1.0:
        return "passthrough"
    return "rife" if effective_fps(info, speed_factor) < RIFE_FPS_THRESHOLD else "setpts"


def _run_setpts(src: Path, dest: Path, speed_factor: float, has_audio: bool) -> None:
    # video: setpts scales presentation timestamps; speed_factor>1 = faster => divide PTS by factor
    vf = f"setpts=PTS/{speed_factor}"
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", vf]

    if has_audio:
        # atempo only supports 0.5-2.0 per filter instance; chain if outside that range.
        atempo_chain = _atempo_chain(speed_factor)
        cmd += ["-af", atempo_chain]
    else:
        cmd += ["-an"]

    cmd += [*config.video_codec_args(), str(dest)]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise SpeedProcessError("ffmpeg not found. Install ffmpeg (e.g. `brew install ffmpeg`).") from e
    except subprocess.CalledProcessError as e:
        raise SpeedProcessError(f"ffmpeg setpts conform failed on {src}: {e.stderr.strip()}") from e


def _atempo_chain(speed_factor: float) -> str:
    """atempo only accepts 0.5-2.0; chain multiple instances for extreme factors."""
    remaining = speed_factor
    filters = []
    while remaining < 0.5 or remaining > 2.0:
        step = 2.0 if remaining > 2.0 else 0.5
        filters.append(f"atempo={step}")
        remaining /= step
    filters.append(f"atempo={remaining}")
    return ",".join(filters)


def _run_rife(src: Path, dest: Path, speed_factor: float, info: ClipInfo) -> None:
    rife_bin = shutil.which(RIFE_BINARY_NAME)
    if not rife_bin:
        raise SpeedProcessError(
            f"Effective fps for this speed_factor ({effective_fps(info, speed_factor):.1f}) is below "
            f"{RIFE_FPS_THRESHOLD}fps and needs RIFE frame interpolation, but "
            f"'{RIFE_BINARY_NAME}' isn't installed. Install it "
            "(https://github.com/nihui/rife-ncnn-vulkan) or choose a speed_factor "
            "that keeps effective fps >= threshold to use the plain setpts path instead."
        )

    # rife-ncnn-vulkan operates on frame sequences, not video files directly.
    # This is a thin orchestration wrapper: extract frames -> interpolate -> re-encode.
    tmp_dir = dest.parent / f".rife_tmp_{src.stem}"
    frames_in = tmp_dir / "in"
    frames_out = tmp_dir / "out"
    frames_in.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), str(frames_in / "%08d.png")],
            capture_output=True, text=True, check=True,
        )

        target_fps = effective_fps(info, 1.0)  # interpolate back up to source fps
        # rife-ncnn-vulkan multi-frame interpolation factor
        multiplier = max(2, round(info.fps / max(effective_fps(info, speed_factor), 1)))
        subprocess.run(
            [rife_bin, "-i", str(frames_in), "-o", str(frames_out), "-m", "rife-v4.6", "-n",
             str(int(len(list(frames_in.glob("*.png"))) * multiplier))],
            capture_output=True, text=True, check=True,
        )

        cmd = [
            "ffmpeg", "-y",
            "-r", str(target_fps),
            "-i", str(frames_out / "%08d.png"),
        ]
        if info.has_audio:
            cmd += ["-i", str(src), "-map", "0:v", "-map", "1:a", "-af", _atempo_chain(speed_factor)]
        cmd += [*config.video_codec_args(), str(dest)]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise SpeedProcessError(f"RIFE pipeline failed on {src}: {e.stderr.strip()}") from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def apply_speed(src_path: str | Path, dest_path: str | Path, speed_factor: float,
                 info: ClipInfo | None = None) -> Path:
    """
    Bake `speed_factor` into a rendered copy of the clip. Returns the path
    to the resulting file (src_path unchanged if speed_factor == 1.0).
    """
    src_path = Path(src_path)
    dest_path = Path(dest_path)
    info = info or probe_clip(src_path)

    branch = choose_branch(info, speed_factor)
    if branch == "passthrough":
        return src_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if branch == "rife":
        _run_rife(src_path, dest_path, speed_factor, info)
    else:
        _run_setpts(src_path, dest_path, speed_factor, info.has_audio)

    return dest_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python speed_process.py <src_video> <dest_video> <speed_factor>")
        raise SystemExit(1)

    src, dest, factor = sys.argv[1], sys.argv[2], float(sys.argv[3])
    info = probe_clip(src)
    branch = choose_branch(info, factor)
    print(f"source fps={info.fps}, speed_factor={factor}, effective_fps={effective_fps(info, factor):.2f}, branch={branch}")
    out = apply_speed(src, dest, factor, info)
    print(f"-> {out}")
