"""
main.py — orchestrates the full pipeline end to end:

    Drive folder
      -> drive_ingest.py   pulls raw clips to /raw
      -> probe.py          fps/duration/resolution/rotation per clip
      -> gemini_score.py   Gemini scores each clip, returns best in/out range
      -> speed_process.py  (no-op unless --speed is passed; speed is 1.0 by default)
      -> reframe.py        only runs on clips whose real pixel grid is landscape
      -> draft_writer.py   clone_draft() + add_clip_to_draft() + register_draft()
      -> open the draft in CapCut for a human pass

Usage:
    python main.py                          # full pipeline, default draft name
    python main.py --draft-name my_reel      # custom draft name
    python main.py --skip-ingest             # reuse whatever's already in /raw
    python main.py --skip-score              # skip Gemini scoring, use full clips as-is
    python main.py --min-score 6.0           # drop clips scoring below this threshold
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
import draft_writer
import drive_ingest
import gemini_score
import reframe
from probe import ClipInfo, normalize_rotation, probe_clip


def run_pipeline(draft_name: str, skip_ingest: bool = False, skip_score: bool = False,
                  min_score: float = 0.0) -> Path:
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
    #    correctly-oriented pixels").
    print("\nNormalizing rotation (only clips with a rotation tag)...")
    normalized_clips: list[Path] = []
    for clip in raw_clips:
        info = infos[clip]
        if info.rotation != 0:
            dest = config.PROCESSED_DIR / f"rot_norm_{clip.name}"
            out = normalize_rotation(clip, dest, info)
            print(f"  {clip.name}: rotation={info.rotation} -> normalized to {out.name}")
            infos[out] = probe_clip(out)
            normalized_clips.append(out)
        else:
            normalized_clips.append(clip)
    raw_clips = normalized_clips

    # 4. Gemini scoring (optional) — clips are scored concurrently; uploads
    #    run fully in parallel and only the metered inference call is
    #    throttled to GEMINI_RPM/GEMINI_RPD (see rate_limiter.py).
    kept_clips: list[Path] = []
    if skip_score:
        print("\nSkipping Gemini scoring (--skip-score); using full clips as-is.")
        kept_clips = list(raw_clips)
    else:
        print(f"\nScoring {len(raw_clips)} clip(s) with Gemini "
              f"(up to {config.GEMINI_MAX_WORKERS} concurrent, "
              f"{config.GEMINI_RPM}/min, {config.GEMINI_RPD}/day)...")
        results = gemini_score.score_many(raw_clips)
        for clip in raw_clips:
            result = results[str(clip)]
            if isinstance(result, Exception):
                print(f"  {clip.name}: SCORING FAILED ({result}) — keeping clip as-is.")
                kept_clips.append(clip)
                continue
            print(f"  {clip.name}: score={result.score:.1f} in={result.in_seconds:.1f}s "
                  f"out={result.out_seconds:.1f}s — {result.reasoning}")
            if result.score >= min_score:
                kept_clips.append(clip)
            else:
                print(f"    -> dropped (below min_score={min_score})")

    if not kept_clips:
        print("\nNo clips survived scoring; nothing to write to the draft.", file=sys.stderr)
        raise SystemExit(1)

    # 5. Reframe (only touches landscape clips)
    print("\nReframing (only clips with a landscape display grid)...")
    processed_clips: list[Path] = []
    for clip in kept_clips:
        info = infos[clip]
        if reframe.needs_reframe(info):
            dest = config.PROCESSED_DIR / f"reframed_{clip.name}"
            out = reframe.reframe_clip(clip, dest, info)
            print(f"  {clip.name}: reframed -> {out.name}")
            processed_clips.append(out)
        else:
            processed_clips.append(clip)

    # 6. Write the CapCut draft
    print(f"\nBuilding draft '{draft_name}'...")
    clip_dicts = []
    for clip in processed_clips:
        # Re-probe processed clips since reframe may have changed width/height/duration.
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
                         help="Skip Gemini scoring; use all downloaded clips as-is.")
    parser.add_argument("--min-score", type=float, default=0.0,
                         help="Drop clips scoring below this threshold (0-10). Ignored with --skip-score.")
    args = parser.parse_args()

    run_pipeline(
        draft_name=args.draft_name,
        skip_ingest=args.skip_ingest,
        skip_score=args.skip_score,
        min_score=args.min_score,
    )


if __name__ == "__main__":
    main()
