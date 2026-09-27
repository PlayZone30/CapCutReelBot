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
from pathlib import Path

from google import genai
from pydantic import BaseModel, Field

import config
from rate_limiter import RateLimiter, RateLimitExceeded

DEFAULT_RUBRIC = """
Score this clip for use in a short vertical highlight reel. Consider:
- visual clarity and framing (is the subject clearly visible and in focus?)
- energy / interest (does something notable happen?)
- lack of dead air at the very start/end of the chosen range
Pick the single best contiguous sub-range of the clip (it may be the whole
clip) that would work best in a fast-cut reel.
"""


class ClipScore(BaseModel):
    score: float = Field(description="Overall quality score from 0.0 (unusable) to 10.0 (excellent).")
    in_seconds: float = Field(description="Start of the best sub-range, in seconds from clip start.")
    out_seconds: float = Field(description="End of the best sub-range, in seconds from clip start.")
    reasoning: str = Field(description="Brief explanation of the score and chosen range.")


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
               rate_limiter: RateLimiter | None = None) -> ClipScore:
    """
    Upload a single clip to Gemini and get back a structured ClipScore
    (score, best in/out range, reasoning).

    If `rate_limiter` is given, it's only consulted right before the
    metered client.interactions.create call — the upload above runs
    unthrottled.
    """
    clip_path = Path(clip_path)
    client = client or _get_client()

    myfile = _upload_and_wait(client, clip_path)

    if rate_limiter is not None:
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
            "schema": ClipScore.model_json_schema(),
        },
    )

    try:
        return ClipScore.model_validate_json(interaction.output_text)
    except Exception as e:
        raise GeminiScoreError(
            f"Failed to parse Gemini response for {clip_path}: {e}\nRaw: {interaction.output_text}"
        ) from e


def score_many(clip_paths: list[str | Path], rubric: str = DEFAULT_RUBRIC,
               max_workers: int | None = None) -> dict[str, ClipScore | Exception]:
    """
    Score multiple clips concurrently. Uploads happen in parallel across
    up to `max_workers` threads; the actual scoring calls are serialized
    through a shared RateLimiter so the batch never exceeds GEMINI_RPM /
    GEMINI_RPD regardless of how many workers are running.

    Each genai.Client is not guaranteed thread-safe for concurrent calls
    from the SDK's perspective, so each worker thread gets its own client
    (cheap — it's just an HTTP client wrapper, no separate auth round trip).

    Returns a dict keyed by str(path). Failed clips map to the raised
    exception instead of a ClipScore, so one bad clip doesn't abort the
    whole batch.
    """
    clip_paths = [Path(p) for p in clip_paths]
    max_workers = max_workers or config.GEMINI_MAX_WORKERS
    rate_limiter = _get_rate_limiter()
    results: dict[str, ClipScore | Exception] = {}

    def _worker(path: Path) -> ClipScore:
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
                  max_workers: int | None = None) -> dict[str, ClipScore | Exception]:
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

    p = Path(target)
    if p.is_file():
        result = score_clip(p)
        print(f"{p.name}: score={result.score} in={result.in_seconds}s out={result.out_seconds}s")
        print(f"  reasoning: {result.reasoning}")
    else:
        for path_str, result in score_folder(p).items():
            if isinstance(result, Exception):
                print(f"{Path(path_str).name}: FAILED — {result}")
                continue
            print(f"{Path(path_str).name}: score={result.score} in={result.in_seconds}s out={result.out_seconds}s")
            print(f"  reasoning: {result.reasoning}")
