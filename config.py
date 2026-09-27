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
GEMINI_MAX_WORKERS = int(_getenv("GEMINI_MAX_WORKERS", "5"))

# ---------------------------------------------------------------------------
# Google Drive
# ---------------------------------------------------------------------------
DRIVE_FOLDER_ID = _getenv("DRIVE_FOLDER_ID")
GOOGLE_SERVICE_ACCOUNT_FILE = _getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

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
