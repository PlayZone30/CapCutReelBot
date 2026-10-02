"""
config.py — environment + OS-specific paths for the CapCut reel bot.

Loads `.env` (via python-dotenv) and exposes constants used by every other
module in the pipeline. Nothing in here talks to the network or the
filesystem beyond reading the .env file and resolving default paths.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (safe to call even if the file is missing).
REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")


def _getenv(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val else default


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
GEMINI_API_KEY = _getenv("GEMINI_API_KEY")
GEMINI_MODEL = _getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Free-tier rate limits (requests to client.interactions.create only —
# file uploads don't count against this quota). Override in .env if your
# key is on a paid tier with higher limits.
# https://ai.google.dev/gemini-api/docs/rate-limits
GEMINI_RPM = int(_getenv("GEMINI_RPM", "5"))
GEMINI_RPD = int(_getenv("GEMINI_RPD", "20"))
GEMINI_RATE_STATE_FILE = REPO_ROOT / ".gemini_rate_state.json"

# How many clips to upload/score concurrently. Uploads run fully parallel;
# the rate limiter below throttles only the inference calls to GEMINI_RPM.
# This is intentionally independent of FFMPEG_MAX_WORKERS — uploading a
# merged batch to Gemini is bandwidth-bound, not CPU-bound, so the
# reasoning that caps ffmpeg concurrency (CPU/RAM headroom for other apps)
# doesn't apply here at all. Don't default this off of FFMPEG_MAX_WORKERS.
GEMINI_MAX_WORKERS = int(_getenv("GEMINI_MAX_WORKERS", "5"))

# ---------------------------------------------------------------------------
# Google Drive
# ---------------------------------------------------------------------------
DRIVE_FOLDER_ID = _getenv("DRIVE_FOLDER_ID")
GOOGLE_SERVICE_ACCOUNT_FILE = _getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

# Plain API key (Drive API enabled, no OAuth/service account needed) for
# reading public/link-shared folders via the real Drive REST API instead
# of scraping the anonymous drive.google.com/uc download page (gdown's
# approach, which is subject to an undocumented "too many viewers"
# per-file throttle separate from — and much stricter than — the official
# Drive API quotas). Falls back to gdown automatically if unset.
# https://console.cloud.google.com/apis/credentials -> Create API key,
# then enable the "Google Drive API" for the project.
GOOGLE_API_KEY = _getenv("GOOGLE_API_KEY")

# The google-genai SDK's env-var auto-detection (used only as a fallback
# when genai.Client() isn't given an explicit api_key — which every call
# site in this project does provide, via require_gemini_key() below) reads
# GOOGLE_API_KEY straight out of os.environ and logs a "Both GOOGLE_API_KEY
# and GEMINI_API_KEY are set" warning whenever both exist there, regardless
# of which one actually ends up used. Since GOOGLE_API_KEY here is for
# Drive downloads only and has nothing to do with Gemini, unset it from the
# process environment once we've captured its value above — every other
# module in this project reads config.GOOGLE_API_KEY (the Python variable),
# not os.environ directly, so this has no effect on Drive ingestion.
os.environ.pop("GOOGLE_API_KEY", None)

# How many files to download concurrently when using the Drive API path.
# Each download costs 200 quota units; per-minute-per-user cap is 325,000,
# so even a generous worker count here is far from that ceiling for any
# realistic folder size.
DRIVE_MAX_WORKERS = int(_getenv("DRIVE_MAX_WORKERS", "6"))

# Which video encoder every ffmpeg-calling module uses (probe.py's
# normalize_rotation, reframe.py, merge_clips.py, trim.py, speed_process.py).
#   "videotoolbox" -> Apple hardware H.264 encoder (macOS only). Measured on
#       the M1 this project was developed on: ~1.6x faster per job and
#       under half the CPU (210% vs 502%) than libx264 below, BUT multiple
#       concurrent hardware sessions do NOT parallelize on this chip
#       (2 concurrent sessions measured at ~2x one session's time, i.e.
#       they serialize) — so FFMPEG_MAX_WORKERS defaults to 1 for this
#       encoder, see below. Real quality trade-off: uses -q:v (0-100, not
#       a direct equivalent of -crf) and is widely reported lower quality
#       per bitrate than libx264 at comparable settings.
#   "libx264"     -> software encoder. Genuinely parallelizes across
#       multiple concurrent ffmpeg processes (each using its own CPU
#       cores), and gives predictable, tunable quality via -crf.
# Measured end-to-end on this project's real 45-clip / ~1000s-footage
# shoot: videotoolbox (1 worker) finished the rotation-normalize stage in
# ~302s vs libx264 (2 workers) at ~383s — videotoolbox won by ~21% despite
# no concurrency, because its per-job speed advantage was larger than what
# 2-way software parallelism bought back. That result is specific to an
# 8-core/8GB M1 and to this project's clip-length mix; re-measure before
# assuming it holds on different hardware or very different footage.
VIDEO_ENCODER = _getenv("VIDEO_ENCODER", "videotoolbox")

# How many concurrent ffmpeg processes to run for the rotation-normalize
# and merge pipeline stages. Defaults depend on VIDEO_ENCODER: 1 for
# videotoolbox (concurrency doesn't help it, see above), 2 for libx264
# (software encoding genuinely parallelizes, but each process is itself
# multi-threaded, and the user may be running other applications at the
# same time, so this stays conservative rather than scaling to core
# count). Override explicitly in .env if you know you have more headroom.
_FFMPEG_MAX_WORKERS_DEFAULT = "1" if VIDEO_ENCODER == "videotoolbox" else "2"
FFMPEG_MAX_WORKERS = int(_getenv("FFMPEG_MAX_WORKERS", _FFMPEG_MAX_WORKERS_DEFAULT))


def video_codec_args(quality: str = "high") -> list[str]:
    """
    Return the -c:v ... ffmpeg args for the configured VIDEO_ENCODER.
    `quality` is a rough knob ("high"/"medium") mapped to encoder-specific
    settings, since -crf (libx264) and -q:v (videotoolbox) aren't on
    directly comparable scales.
    """
    if VIDEO_ENCODER == "videotoolbox":
        q = "65" if quality == "high" else "50"
        return ["-c:v", "h264_videotoolbox", "-q:v", q]
    crf = "18" if quality == "high" else "23"
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", crf]

# ---------------------------------------------------------------------------
# CapCut draft locations (OS-dependent)
# ---------------------------------------------------------------------------
_SYSTEM = platform.system()  # "Darwin", "Windows", "Linux"

if _SYSTEM == "Darwin":
    _DEFAULT_DRAFT_ROOT = str(
        Path.home() / "Movies" / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft"
    )
    # Confirmed against a real captured draft on macOS (CapCut 9.5.0).
    DRAFT_FILENAME = "draft_info.json"
elif _SYSTEM == "Windows":
    _DEFAULT_DRAFT_ROOT = str(
        Path(os.environ.get("USERPROFILE", str(Path.home())))
        / "AppData"
        / "Local"
        / "JianyingPro"
        / "User Data"
        / "Projects"
        / "com.lveditor.draft"
    )
    # Unverified — community tooling reports Windows CapCut/JianyingPro uses
    # this filename instead of draft_info.json. Confirm on a real Windows
    # install before relying on this branch.
    DRAFT_FILENAME = "draft_content.json"
else:
    # Linux has no native CapCut client; default to the macOS-style layout
    # so things don't hard-crash, but this is unverified/unsupported.
    _DEFAULT_DRAFT_ROOT = str(Path.home() / ".capcut_reel_bot" / "drafts")
    DRAFT_FILENAME = "draft_info.json"

CAPCUT_DRAFT_ROOT = Path(_getenv("CAPCUT_DRAFT_ROOT") or _DEFAULT_DRAFT_ROOT)

# ---------------------------------------------------------------------------
# Repo-relative working directories
# ---------------------------------------------------------------------------
RAW_DIR = REPO_ROOT / "raw"
PROCESSED_DIR = REPO_ROOT / "processed"
TEMPLATE_DRAFT_DIR = REPO_ROOT / "templates" / "reference_draft"
ROOT_META_FILENAME = "root_meta_info.json"

for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def require_gemini_key() -> str:
    """Raise a clear error if GEMINI_API_KEY isn't set, otherwise return it."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or `export GEMINI_API_KEY=...` before running."
        )
    return GEMINI_API_KEY


if __name__ == "__main__":
    print(f"System:            {_SYSTEM}")
    print(f"CAPCUT_DRAFT_ROOT: {CAPCUT_DRAFT_ROOT}")
    print(f"DRAFT_FILENAME:    {DRAFT_FILENAME}")
    print(f"ROOT_META_FILENAME:{ROOT_META_FILENAME}")
    print(f"TEMPLATE_DRAFT_DIR:{TEMPLATE_DRAFT_DIR}")
    print(f"GEMINI_API_KEY set: {bool(GEMINI_API_KEY)}")
    print(f"GEMINI_MODEL:      {GEMINI_MODEL}")
    print(f"DRIVE_FOLDER_ID:   {DRIVE_FOLDER_ID}")
