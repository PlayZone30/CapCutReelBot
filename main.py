"""
main.py — orchestrates the full pipeline end to end:

    Drive folder
      -> drive_ingest.py   pulls raw clips to /raw
      -> probe.py          fps/duration/resolution/rotation per clip
      -> probe.py          normalize rotation (bake in any rotation tag)
      -> merge_clips.py     bin-pack normalized clips into ~3-minute merged
                            batches before they ever go to Gemini (see
                            merge_clips.py docstring — this is what keeps a
                            46-clip Drive folder from costing 46 Gemini
                            requests against a 20/day free-tier quota)
      -> gemini_score.py   Gemini segments each MERGED batch into shots,
                            scored at native speed/duration (see
                            gemini_score.py docstring for why scoring
                            happens before any trim or speed change)
      -> trim.py            cut each surviving shot down to its own
                            [start_seconds, end_seconds) range
      -> speed_process.py   bake in a speed change on the TRIMMED shot only
                            (so RIFE, if triggered, only ever processes the
                            footage that survives into the final cut)
      -> reframe.py         only runs on shots whose real pixel grid is landscape
      -> draft_writer.py    clone_draft() + add_clip_to_draft() + register_draft()
      -> open the draft in CapCut for a human pass

Order matters here: trim happens BEFORE speed_process and BEFORE Gemini ever
sees slowed-down footage, for three reasons (see implementationplan.md /
project notes for the full writeup):
  1. RIFE interpolation (the expensive step in speed_process.py) only ever
     touches the seconds of footage that actually make the final cut,
     instead of the whole raw clip.
  2. Gemini's own cost scales with video duration, so scoring happens on
     the native-speed, untrimmed-but-not-yet-slowed clip — the cheapest
     version of the footage that still shows the real camera movement.
  3. Camera-work judgments (steady pan vs shaky, clean zoom vs not) are a
     property of the real capture, not of a stretched-time render, so
     Gemini should judge native-speed footage.

Usage:
    python main.py                          # full pipeline, default draft name
    python main.py --draft-name my_reel      # custom draft name
    python main.py --skip-ingest             # reuse whatever's already in /raw
    python main.py --skip-score              # skip Gemini scoring, use full clips as-is
    python main.py --min-score 6.0           # drop shots scoring below this threshold
    python main.py --speed-factor 0.5        # bake this speed into every surviving shot
    python main.py --merge-target-seconds 170  # override the ~3min merge-batch size
    python main.py --skip-merge              # score each clip separately (old behavior)
    python main.py --resume-from-merged      # skip ingest/probe/normalize/merge entirely;
                                              # start straight from processed/merged*.mp4
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
import draft_writer
import drive_ingest
import gemini_score
import merge_clips
import reframe
import speed_process
from probe import ClipInfo, normalize_rotation, probe_clip
from trim import trim_clip


def run_pipeline(draft_name: str, skip_ingest: bool = False, skip_score: bool = False,
                  min_score: float = 0.0, speed_factor: float = 1.0,
                  skip_merge: bool = False,
                  merge_target_seconds: float = merge_clips.DEFAULT_TARGET_SECONDS,
                  resume_from_merged: bool = False) -> Path:
    # 0. Resume from already-merged batches: skip ingest, probe,
    #    rotation-normalize, and merge entirely, and pick up straight from
    #    whatever processed/merged*.mp4 files already exist on disk. Useful
    #    when a previous run got through merging (the slow, CPU-heavy part)
    #    but stopped before or during Gemini scoring — there's no need to
    #    re-download or re-encode anything to try again.
    if resume_from_merged:
        merged_clips = sorted(config.PROCESSED_DIR.glob("merged*.mp4"))
        if not merged_clips:
            print(f"No merged*.mp4 files found in {config.PROCESSED_DIR}; "
                  "can't use --resume-from-merged. Run without it first.", file=sys.stderr)
            raise SystemExit(1)
        print(f"Resuming from {len(merged_clips)} existing merged batch(es) in {config.PROCESSED_DIR}:")
        infos: dict[Path, ClipInfo] = {}
        for clip in merged_clips:
            infos[clip] = probe_clip(clip)
            print(f"  {clip.name}: {infos[clip].duration_s:.1f}s")
        raw_clips = merged_clips
        return _score_trim_and_write(
            raw_clips, infos, draft_name, skip_score, min_score, speed_factor,
        )

    # 1. Ingest
    if skip_ingest:
        raw_clips = sorted(
            p for p in config.RAW_DIR.iterdir()
            if p.suffix.lower() in (".mp4", ".mov", ".m4v")
        )
        if not raw_clips:
            print(f"No clips found in {config.RAW_DIR} and --skip-ingest was passed.", file=sys.stderr)
            raise SystemExit(1)
        print(f"Reusing {len(raw_clips)} existing clip(s) in {config.RAW_DIR}")
    else:
        print(f"Ingesting from Drive folder {config.DRIVE_FOLDER_ID} -> {config.RAW_DIR}")
        raw_clips = drive_ingest.ingest()
        print(f"  downloaded {len(raw_clips)} clip(s)")

    # 2. Probe
    print("\nProbing clips...")
    infos: dict[Path, ClipInfo] = {}
    for clip in raw_clips:
        info = probe_clip(clip)
        infos[clip] = info
        print(f"  {clip.name}: {info.display_width}x{info.display_height} @ {info.fps}fps, "
              f"{info.duration_s:.1f}s, {'portrait' if info.is_portrait else 'landscape'}")

    # 3. Normalize rotation (bake in any rotation tag before anything else
    #    touches pixels, per the plan: "if rotation metadata present,
    #    normalize orientation now so every downstream step works on
    #    correctly-oriented pixels"). This runs on the whole clip, before
    #    trimming, since it's cheap relative to RIFE and every shot
    #    extracted from this clip needs to inherit the fix.
    #
    #    Each clip's rotation fix is fully independent of every other
    #    clip's, so this runs concurrently (FFMPEG_MAX_WORKERS, default 2 —
    #    kept low since each ffmpeg process is itself multi-threaded and
    #    the machine may be running other applications at the same time).
    print(f"\nNormalizing rotation (only clips with a rotation tag, "
          f"up to {config.FFMPEG_MAX_WORKERS} concurrent)...")

    def _normalize_one(clip: Path) -> tuple[Path, Path | None]:
        info = infos[clip]
        if info.rotation == 0:
            return clip, None
        dest = config.PROCESSED_DIR / f"rot_norm_{clip.name}"
        out = normalize_rotation(clip, dest, info)
        return clip, out

    normalized_clips: list[Path] = []
    with ThreadPoolExecutor(max_workers=config.FFMPEG_MAX_WORKERS) as pool:
        for clip, out in pool.map(_normalize_one, raw_clips):
            if out is None:
                normalized_clips.append(clip)
                continue
            print(f"  {clip.name}: rotation={infos[clip].rotation} -> normalized to {out.name}")
            infos[out] = probe_clip(out)
            normalized_clips.append(out)
    raw_clips = normalized_clips

    # 4. Merge clips into ~3-minute batches before anything gets sent to
    #    Gemini. See merge_clips.py for the bin-packing rule. This step
    #    changes what "a clip" means for every step after it: from here
    #    on, `raw_clips` holds merged batch files (or single clips
    #    untouched, if a clip was already >= the target on its own, or
    #    --skip-merge was passed). Shot start/end seconds Gemini returns
    #    are relative to whichever file it was actually shown, so trimming
    #    downstream works correctly either way with no extra bookkeeping.
    if skip_merge:
        print("\nSkipping merge (--skip-merge); scoring each clip separately.")
    else:
        groups = merge_clips.plan_merge_groups(raw_clips, infos, merge_target_seconds)
        multi_clip_groups = sum(1 for g in groups if len(g) > 1)
        print(f"\nMerging {len(raw_clips)} clip(s) into {len(groups)} batch(es) "
              f"(target ~{merge_target_seconds:.0f}s each, {multi_clip_groups} actually merged, "
              f"up to {config.FFMPEG_MAX_WORKERS} concurrent)...")

        # Each group's merge is independent ffmpeg work, so batches with
        # 2+ clips run concurrently (same FFMPEG_MAX_WORKERS cap as
        # rotation-normalize above). Single-clip "groups" need no ffmpeg
        # call at all — handled inline, not submitted to the pool.
        def _merge_one(idx_group: tuple[int, list[Path]]) -> tuple[int, Path]:
            idx, group = idx_group
            if len(group) == 1:
                return idx, group[0]
            dest = config.PROCESSED_DIR / f"merged{idx:03d}.mp4"
            return idx, merge_clips.merge_clips(group, dest)

        merged_by_idx: dict[int, Path] = {}
        with ThreadPoolExecutor(max_workers=config.FFMPEG_MAX_WORKERS) as pool:
            for idx, out in pool.map(_merge_one, enumerate(groups)):
                merged_by_idx[idx] = out

        merged_clips: list[Path] = []
        for idx, group in enumerate(groups):
            out = merged_by_idx[idx]
            total_s = sum(infos[c].duration_s for c in group)
            if len(group) == 1:
                print(f"  batch {idx}: {group[0].name} ({total_s:.1f}s, left as-is)")
            else:
                print(f"  batch {idx}: {len(group)} clips ({[c.name for c in group]}) "
                      f"-> {out.name} ({total_s:.1f}s)")
                infos[out] = probe_clip(out)
            merged_clips.append(out)
        raw_clips = merged_clips

    return _score_trim_and_write(
        raw_clips, infos, draft_name, skip_score, min_score, speed_factor,
    )


def _score_trim_and_write(raw_clips: list[Path], infos: dict[Path, ClipInfo],
                           draft_name: str, skip_score: bool, min_score: float,
                           speed_factor: float) -> Path:
    """
    Steps 5-9 of the pipeline: Gemini scoring, trim, speed, reframe, draft
    write. Split out from run_pipeline() so --resume-from-merged can jump
    straight here with an existing set of merged batches, instead of
    re-running ingest/probe/normalize/merge.
    """
    # 5. Gemini scoring (optional) — each clip (now possibly a merged
    #    ~3-minute batch) is segmented into shots and scored at native
    #    speed/duration (see gemini_score.py). Clips are scored
    #    concurrently; uploads run fully in parallel and only the metered
    #    inference call is throttled to GEMINI_RPM/GEMINI_RPD (see
    #    rate_limiter.py).
    #
    #    kept_shots holds (clip_path, shot) pairs — one clip (merged batch
    #    or standalone) can contribute zero, one, or several shots to the
    #    final cut.
    kept_shots: list[tuple[Path, "gemini_score.Shot"]] = []
    if skip_score:
        print("\nSkipping Gemini scoring (--skip-score); using full clips as single shots.")
        for clip in raw_clips:
            info = infos[clip]
            whole_clip_shot = gemini_score.Shot(
                shot_type=gemini_score.ShotType.OTHER,
                start_seconds=0.0,
                end_seconds=info.duration_s,
                is_in_focus=True,
                stability="locked",
                score=10.0,
                reasoning="Scoring skipped (--skip-score); using the full clip as one shot.",
            )
            kept_shots.append((clip, whole_clip_shot))
    else:
        print(f"\nScoring {len(raw_clips)} clip(s) with Gemini "
              f"(up to {config.GEMINI_MAX_WORKERS} concurrent, "
              f"{config.GEMINI_RPM}/min, {config.GEMINI_RPD}/day)...")

        # on_result fires the moment EACH clip finishes (not after the
        # whole batch) — without this, score_many() would only return once
        # every clip is done, so with a handful of large merged batches a
        # caller could sit with the first clip's results ready but
        # invisible for many minutes while later clips are still
        # uploading/scoring. It fires in COMPLETION order, not the
        # original clip order, so it's used here only for live printing —
        # kept_shots itself is still built below in raw_clips' original
        # order, so the final draft's clip order matches the source
        # footage order regardless of which clip happened to finish first.
        def _on_result(clip: Path, result) -> None:
            if isinstance(result, Exception):
                print(f"  {clip.name}: SCORING FAILED ({result})")
                return
            print(f"  {clip.name}: {len(result.shots)} shot(s)")
            for shot in result.shots:
                print(f"    [{shot.start_seconds:.1f}s-{shot.end_seconds:.1f}s] {shot.shot_type.value} "
                      f"score={shot.score:.1f} in_focus={shot.is_in_focus} stability={shot.stability} "
                      f"— {shot.reasoning}")

        results = gemini_score.score_many(raw_clips, on_result=_on_result)

        for clip in raw_clips:
            result = results[str(clip)]
            if isinstance(result, Exception):
                print(f"  {clip.name}: keeping whole clip as one shot (scoring failed).")
                info = infos[clip]
                kept_shots.append((clip, gemini_score.Shot(
                    shot_type=gemini_score.ShotType.OTHER,
                    start_seconds=0.0,
                    end_seconds=info.duration_s,
                    is_in_focus=True,
                    stability="locked",
                    score=min_score,  # exactly at the threshold: keep, but don't pretend it's good
                    reasoning=f"Gemini scoring failed: {result}",
                )))
                continue

            for shot in result.shots:
                if shot.score < min_score or not shot.is_in_focus:
                    continue
                kept_shots.append((clip, shot))

    if not kept_shots:
        print("\nNo shots survived scoring; nothing to write to the draft.", file=sys.stderr)
        raise SystemExit(1)

    # 6. Trim each surviving shot out of its source clip. This is the step
    #    that actually acts on Gemini's start/end range — everything after
    #    this operates on the trimmed shot, not the full raw clip.
    print(f"\nTrimming {len(kept_shots)} shot(s)...")
    trimmed_clips: list[Path] = []
    for idx, (clip, shot) in enumerate(kept_shots):
        dest = config.PROCESSED_DIR / f"shot{idx:03d}_{clip.stem}.mp4"
        out = trim_clip(clip, dest, shot.start_seconds, shot.end_seconds)
        print(f"  {clip.name} [{shot.start_seconds:.1f}s-{shot.end_seconds:.1f}s] -> {out.name}")
        trimmed_clips.append(out)

    # 7. Speed process — bake speed_factor into each TRIMMED shot only, so
    #    RIFE (if the effective fps drops below its threshold) only ever
    #    processes the handful of seconds that made the cut. speed_factor
    #    defaults to 1.0 (passthrough, no-op) until a real per-shot speed
    #    decision is wired up; pass --speed-factor to apply one uniformly.
    print(f"\nApplying speed_factor={speed_factor} to trimmed shots...")
    speed_processed_clips: list[Path] = []
    for clip in trimmed_clips:
        if speed_factor == 1.0:
            speed_processed_clips.append(clip)
            continue
        dest = config.PROCESSED_DIR / f"speed_{clip.name}"
        out = speed_process.apply_speed(clip, dest, speed_factor)
        print(f"  {clip.name} -> {out.name}")
        speed_processed_clips.append(out)

    # 8. Reframe (only touches landscape shots)
    print("\nReframing (only shots with a landscape display grid)...")
    processed_clips: list[Path] = []
    for clip in speed_processed_clips:
        info = probe_clip(clip)
        if reframe.needs_reframe(info):
            dest = config.PROCESSED_DIR / f"reframed_{clip.name}"
            out = reframe.reframe_clip(clip, dest, info)
            print(f"  {clip.name}: reframed -> {out.name}")
            processed_clips.append(out)
        else:
            processed_clips.append(clip)

    # 9. Write the CapCut draft
    print(f"\nBuilding draft '{draft_name}'...")
    clip_dicts = []
    for clip in processed_clips:
        # Re-probe processed clips since trim/speed/reframe may have
        # changed duration/width/height.
        info = probe_clip(clip)
        clip_dicts.append({
            "path": str(clip),
            "duration_us": info.duration_us,
            "width": info.width,
            "height": info.height,
        })

    root_meta_path = config.CAPCUT_DRAFT_ROOT / config.ROOT_META_FILENAME
    dest_folder = draft_writer.build_draft(draft_name, clip_dicts, root_meta_path)
    print(f"Draft written to: {dest_folder}")
    print("Open CapCut and look for it in your project list.")
    return dest_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="CapCut reel automation pipeline.")
    parser.add_argument("--draft-name", default="capcut_reel_bot_output",
                         help="Name of the CapCut draft to create.")
    parser.add_argument("--skip-ingest", action="store_true",
                         help="Reuse clips already in ./raw instead of downloading from Drive.")
    parser.add_argument("--skip-score", action="store_true",
                         help="Skip Gemini scoring; treat each full clip as one kept shot.")
    parser.add_argument("--min-score", type=float, default=0.0,
                         help="Drop shots scoring below this threshold (0-10). Ignored with --skip-score.")
    parser.add_argument("--speed-factor", type=float, default=1.0,
                         help="Speed factor baked into every surviving shot after trimming "
                              "(1.0 = no change, <1.0 = slow motion, >1.0 = sped up).")
    parser.add_argument("--skip-merge", action="store_true",
                         help="Score each clip separately instead of bin-packing into "
                              "~3-minute batches first (uses more Gemini requests).")
    parser.add_argument("--merge-target-seconds", type=float,
                         default=merge_clips.DEFAULT_TARGET_SECONDS,
                         help=f"Target batch length in seconds when merging clips before "
                              f"scoring (default: {merge_clips.DEFAULT_TARGET_SECONDS:.0f}).")
    parser.add_argument("--resume-from-merged", action="store_true",
                         help="Skip ingest/probe/normalize/merge entirely; start straight "
                              "from existing processed/merged*.mp4 files. Use this to retry "
                              "Gemini scoring (or anything after it) without re-downloading "
                              "or re-encoding a previous run's merged batches.")
    args = parser.parse_args()

    run_pipeline(
        draft_name=args.draft_name,
        skip_ingest=args.skip_ingest,
        skip_score=args.skip_score,
        min_score=args.min_score,
        speed_factor=args.speed_factor,
        skip_merge=args.skip_merge,
        merge_target_seconds=args.merge_target_seconds,
        resume_from_merged=args.resume_from_merged,
    )


if __name__ == "__main__":
    main()
