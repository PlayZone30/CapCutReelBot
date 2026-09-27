"""
gemini_score.py — score each clip against a rubric using Gemini's native
video understanding, and get back the best in/out range + reasoning.

Uses the Gemini Interactions API (client.interactions.create) via the
`google-genai` SDK, confirmed against ai.google.dev/gemini-api/docs:
- video-understanding.md   -> Files API upload + video input shape
- interactions/structured-output.md -> response_format + pydantic schema

Files API is used (not inline base64) since these are raw phone-camera
clips that will regularly exceed the 20MB inline-request ceiling.

Concurrency: file uploads (client.files.upload) don't count against the
Gemini quota, so they run fully in parallel via a thread pool. The actual
scoring call (client.interactions.create) is the metered operation, so
every call to it goes through a shared RateLimiter that enforces
GEMINI_RPM / GEMINI_RPD from config.py (defaults: 5/min, 20/day — the
free tier's limits). This lets score_folder() process a whole batch of
clips as fast as the quota allows instead of serially waiting on each one.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

from google import genai
from pydantic import BaseModel, Field

import config
from rate_limiter import RateLimiter, RateLimitExceeded

# A single raw clip regularly contains several distinct camera moments back
# to back (e.g. a soft rack-focus opening followed by a static detail shot).
# Scoring the whole clip as one unit loses that structure, so Gemini is
# asked to segment each clip into a sequential list of shots and score each
# one independently — see DEFAULT_RUBRIC below.
DEFAULT_RUBRIC = """
You are analyzing raw footage for a short-form vertical highlight reel (Instagram
Reels). Watch the ENTIRE clip and segment it into a sequential list of distinct
camera shots — every contiguous stretch where the camera is doing one clear thing.
Cover the full clip with no gaps and no overlaps: the end of one shot's range
should be the start of the next.

For each shot, classify shot_type using EXACTLY one of these values:
- static             locked off, no camera movement
- push_in            camera or zoom moves toward the subject, framing tightens
- pull_out           camera or zoom moves away from the subject, framing widens
- pan                camera rotates left/right
- tilt               camera rotates up/down
- whip_pan           fast blurred pan, usually used as a transition
- rack_focus         focus shifts from soft to sharp (or sharp to soft); framing
                     may or may not also be moving
- tracking           camera physically moves alongside or around the subject
- handheld_walk      walking POV or handheld movement through the space
- detail_insert       close, mostly static shot on texture/detail (fabric, stitching,
                     buttons, a small product detail)
- wide_establishing   wide shot showing the space/context
- other              anything that doesn't cleanly fit the above

A shot that starts blurry and sharpens partway through must be split into two
shots: a short rack_focus segment, then whatever the sharp shot becomes. Don't
lump a soft opening into the same shot as the sharp footage that follows it.

For each shot also report:
- is_in_focus: false if that range is soft/blurry for its whole duration
- stability: "locked", "stable_handheld", or "shaky"
- score (0-10): how usable this specific shot is in a polished highlight reel —
  penalize persistent blur, bad exposure, and shaky footage; do NOT penalize a
  shot just for being a wide or a detail shot, both are valuable if clean
- reasoning: one sentence on why you scored it that way

Return every shot you find, including low-scoring ones — filtering happens
downstream. Do not merge visually distinct camera movements into one shot just
because they're both "good."
"""


class ShotType(str, Enum):
    STATIC = "static"
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"
    PAN = "pan"
    TILT = "tilt"
    WHIP_PAN = "whip_pan"
    RACK_FOCUS = "rack_focus"
    TRACKING = "tracking"
    HANDHELD_WALK = "handheld_walk"
    DETAIL_INSERT = "detail_insert"
    WIDE_ESTABLISHING = "wide_establishing"
    OTHER = "other"


class Shot(BaseModel):
    shot_type: ShotType
    start_seconds: float
    end_seconds: float
    is_in_focus: bool = Field(description="False if this range is soft/blurry for its whole duration.")
    stability: str = Field(description='One of "locked", "stable_handheld", "shaky".')
    score: float = Field(description="0.0 (unusable) to 10.0 (excellent) for a polished highlight reel.")
    reasoning: str = Field(description="Brief explanation of the score and classification.")


class ClipAnalysis(BaseModel):
    shots: list[Shot] = Field(
        description="Every distinct camera shot in the clip, in order, covering the full duration with no gaps."
    )


class GeminiScoreError(RuntimeError):
    pass


def _get_client() -> genai.Client:
    return genai.Client(api_key=config.require_gemini_key())


def _get_rate_limiter() -> RateLimiter:
    return RateLimiter(
        per_minute=config.GEMINI_RPM,
        per_day=config.GEMINI_RPD,
        state_file=config.GEMINI_RATE_STATE_FILE,
    )


def _upload_and_wait(client: genai.Client, path: Path):
    """
    Upload a file and poll until it's ACTIVE. This does NOT count against
    the inference quota, so callers may run this concurrently across
    many clips without a rate limiter.
    """
    myfile = client.files.upload(file=str(path))
    while not myfile.state or myfile.state.name != "ACTIVE":
        if myfile.state and myfile.state.name == "FAILED":
            raise GeminiScoreError(f"Gemini file processing failed for {path}")
        time.sleep(2)
        myfile = client.files.get(name=myfile.name)
    return myfile


def score_clip(clip_path: str | Path, rubric: str = DEFAULT_RUBRIC,
               client: genai.Client | None = None,
               rate_limiter: RateLimiter | None | bool = True) -> ClipAnalysis:
    """
    Upload a single clip to Gemini and get back a structured ClipAnalysis:
    the full clip segmented into sequential shots, each independently
    classified and scored.

    Scoring runs against the clip at its native (raw) speed/duration —
    trimming and any slow-motion processing happen downstream, after a
    shot's range has already been decided here. This keeps Gemini's video
    input as short as possible (Gemini's cost scales with duration) and
    means shot classification reflects real camera movement, not movement
    stretched by a speed change applied later.

    `rate_limiter` defaults to True, which builds a fresh RateLimiter that
    reads/writes the shared state file — this makes every direct call
    (including ad-hoc `python gemini_score.py <file>` runs) get recorded
    against the daily quota, not just calls routed through score_many().
    Pass an existing RateLimiter instance to share state across a batch
    (score_many does this), or pass False to skip throttling entirely
    (not recommended — nothing stops you from blowing through the quota).
    """
    clip_path = Path(clip_path)
    client = client or _get_client()

    myfile = _upload_and_wait(client, clip_path)

    if rate_limiter is True:
        rate_limiter = _get_rate_limiter()
    if rate_limiter:
        rate_limiter.acquire()

    interaction = client.interactions.create(
        model=config.GEMINI_MODEL,
        input=[
            {"type": "video", "uri": myfile.uri, "mime_type": myfile.mime_type},
            {"type": "text", "text": rubric},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": ClipAnalysis.model_json_schema(),
        },
    )

    try:
        return ClipAnalysis.model_validate_json(interaction.output_text)
    except Exception as e:
        raise GeminiScoreError(
            f"Failed to parse Gemini response for {clip_path}: {e}\nRaw: {interaction.output_text}"
        ) from e


def score_many(clip_paths: list[str | Path], rubric: str = DEFAULT_RUBRIC,
               max_workers: int | None = None) -> dict[str, ClipAnalysis | Exception]:
    """
    Score multiple clips concurrently. Uploads happen in parallel across
    up to `max_workers` threads; the actual scoring calls are serialized
    through a shared RateLimiter so the batch never exceeds GEMINI_RPM /
    GEMINI_RPD regardless of how many workers are running.

    Each genai.Client is not guaranteed thread-safe for concurrent calls
    from the SDK's perspective, so each worker thread gets its own client
    (cheap — it's just an HTTP client wrapper, no separate auth round trip).

    Returns a dict keyed by str(path). Failed clips map to the raised
    exception instead of a ClipAnalysis, so one bad clip doesn't abort the
    whole batch.
    """
    clip_paths = [Path(p) for p in clip_paths]
    max_workers = max_workers or config.GEMINI_MAX_WORKERS
    rate_limiter = _get_rate_limiter()
    results: dict[str, ClipAnalysis | Exception] = {}

    def _worker(path: Path) -> ClipAnalysis:
        client = _get_client()  # one client per thread, cheap to construct
        return score_clip(path, rubric, client=client, rate_limiter=rate_limiter)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_path = {pool.submit(_worker, p): p for p in clip_paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results[str(path)] = future.result()
            except (GeminiScoreError, RateLimitExceeded) as e:
                results[str(path)] = e

    return results


def score_folder(folder: str | Path, rubric: str = DEFAULT_RUBRIC,
                  extensions: tuple[str, ...] = (".mp4", ".mov", ".m4v"),
                  max_workers: int | None = None) -> dict[str, ClipAnalysis | Exception]:
    """Score every video file in a folder (non-recursive), in parallel."""
    folder = Path(folder)
    files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in extensions]
    return score_many(files, rubric, max_workers=max_workers)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        print("Usage: python gemini_score.py <video_file_or_folder>")
        raise SystemExit(1)

    def _print_analysis(name: str, analysis: ClipAnalysis) -> None:
        print(f"{name}: {len(analysis.shots)} shot(s)")
        for shot in analysis.shots:
            print(f"  [{shot.start_seconds:.1f}s-{shot.end_seconds:.1f}s] {shot.shot_type.value} "
                  f"score={shot.score:.1f} in_focus={shot.is_in_focus} stability={shot.stability}")
            print(f"    reasoning: {shot.reasoning}")

    p = Path(target)
    if p.is_file():
        _print_analysis(p.name, score_clip(p))
    else:
        for path_str, result in score_folder(p).items():
            if isinstance(result, Exception):
                print(f"{Path(path_str).name}: FAILED — {result}")
                continue
            _print_analysis(Path(path_str).name, result)
