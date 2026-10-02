"""
merge_clips.py — greedily bin-pack raw clips into ~3-minute merged videos
before they're sent to Gemini for scoring.

Why: Gemini is billed/rate-limited per request regardless of how short the
uploaded video is. Sending 46 raw clips (many only 5-15s long) as 46
separate requests burns 46 requests against a 20/day free-tier quota for
almost no benefit — most of that quota is spent on network/model overhead,
not on footage Gemini actually needs to reason about. Concatenating clips
into ~3-minute batches first means the SAME footage gets scored using a
small fraction of the requests.

This is a generic bin-packing pass, not specific to any one shoot — it
works on however many clips of whatever individual lengths are in /raw.

Bin-packing rule (as specified):
    Walk clips top to bottom (existing sort order). Keep adding clips to
    the current bin. After adding a clip, check the bin's running
    duration:
      - if it's still under the threshold (default 2:50 / 170s), keep
        adding the next clip to this same bin, even if that pushes the
        bin over 3 minutes (that overshoot is expected and fine — see the
        worked example below).
      - if it's at/over the threshold, close this bin and start a new one
        with the next clip.

    Worked example from the spec: bin is at 2:30 (150s, under the 170s
    threshold) -> merge the next clip even though it's a full minute ->
    bin becomes 3:30 (210s) -> that's fine, since the overshoot check
    only happens on the NEXT clip, not this one. The 3:30 bin then closes
    and a new bin starts.

Gemini's own shot-timestamps (start_seconds/end_seconds) end up relative
to the MERGED video, not the original file — trim.py doesn't care, it
just cuts whatever range it's given out of whatever file it's given, so
no downstream change is needed there. main.py keeps a mapping from each
merged group back to which original clip is "showing" at any given
timestamp only implicitly (via the merged file itself); shots are cut
directly out of the merged file, not reassembled back to per-source-clip
ranges.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import config
from probe import ClipInfo, probe_clip

# 2:50 — a clip already at or past this point closes its bin instead of
# accepting one more clip, per the bin-packing rule in the module docstring.
DEFAULT_TARGET_SECONDS = 170.0


class MergeError(RuntimeError):
    pass


def plan_merge_groups(clips: list[Path], infos: dict[Path, ClipInfo],
                       target_seconds: float = DEFAULT_TARGET_SECONDS) -> list[list[Path]]:
    """
    Greedily group `clips` (in their given order) into bins whose running
    duration is allowed to cross `target_seconds` by at most one clip's
    worth of overshoot. See module docstring for the exact rule + example.
    """
    groups: list[list[Path]] = []
    current: list[Path] = []
    current_duration = 0.0

    for clip in clips:
        if current and current_duration >= target_seconds:
            groups.append(current)
            current = []
            current_duration = 0.0
        current.append(clip)
        current_duration += infos[clip].duration_s

    if current:
        groups.append(current)

    return groups


def merge_clips(group: list[Path], dest_path: Path) -> Path:
    """
    Concatenate `group` (2+ clips) into a single file at dest_path.

    Uses ffmpeg's concat FILTER (re-encoding), not the concat demuxer
    (stream copy), on purpose: the concat demuxer requires every input to
    share identical codec parameters, which isn't guaranteed across
    arbitrary source clips (different phones, different rotation-tag
    normalization history, etc). The filter approach re-encodes and
    tolerates differing codecs/resolutions/frame rates across inputs at
    the cost of that re-encode.

    If any clip in the group has no audio track, the merge drops audio
    entirely for the whole group (video-only concat) rather than crash on
    a mismatched stream count — logged by the caller, not here.

    A single-clip "group" is returned unchanged (no merge, no re-encode).
    """
    if len(group) == 1:
        return group[0]
    if not group:
        raise MergeError("merge_clips called with an empty group.")

    infos = [probe_clip(c) for c in group]
    all_have_audio = all(i.has_audio for i in infos)

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    inputs: list[str] = []
    for clip in group:
        inputs += ["-i", str(clip)]

    n = len(group)
    if all_have_audio:
        filter_parts = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
        filter_complex = f"{filter_parts}concat=n={n}:v=1:a=1[outv][outa]"
        map_args = ["-map", "[outv]", "-map", "[outa]"]
        codec_args = [*config.video_codec_args(), "-c:a", "aac"]
    else:
        filter_parts = "".join(f"[{i}:v:0]" for i in range(n))
        filter_complex = f"{filter_parts}concat=n={n}:v=1:a=0[outv]"
        map_args = ["-map", "[outv]"]
        codec_args = config.video_codec_args()

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, *map_args,
           *codec_args, str(dest_path)]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise MergeError("ffmpeg not found. Install ffmpeg (e.g. `brew install ffmpeg`).") from e
    except subprocess.CalledProcessError as e:
        raise MergeError(
            f"ffmpeg concat failed on {[str(c) for c in group]}: {e.stderr.strip()}"
        ) from e

    return dest_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python merge_clips.py <output_dir> <clip1> [clip2 ...]")
        raise SystemExit(1)

    out_dir = Path(sys.argv[1])
    clip_paths = [Path(p) for p in sys.argv[2:]]
    infos = {c: probe_clip(c) for c in clip_paths}

    groups = plan_merge_groups(clip_paths, infos)
    print(f"{len(clip_paths)} clip(s) -> {len(groups)} merge group(s):")
    for idx, group in enumerate(groups):
        total = sum(infos[c].duration_s for c in group)
        print(f"  group {idx}: {len(group)} clip(s), {total:.1f}s total -> "
              f"{[c.name for c in group]}")

    for idx, group in enumerate(groups):
        dest = out_dir / f"merged_{idx:03d}.mp4"
        out = merge_clips(group, dest)
        print(f"  -> {out}")
