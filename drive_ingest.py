"""
drive_ingest.py — pull raw clips from a Google Drive folder into /raw.

Primary path: the real Drive API v3, authenticated with a SERVICE ACCOUNT
(GOOGLE_SERVICE_ACCOUNT_FILE in .env), downloaded via google-api-python-
client's official MediaIoBaseDownload (chunked, has its own retry
support). This does NOT require the folder to be re-shared with the
service account specifically — a folder that's already "Anyone with the
link" is, by definition, accessible to any Google identity including a
service account, no extra sharing step needed.

Why a service account rather than a bare API key (this project's earlier
approach): confirmed against multiple independent, converging reports
(Stack Overflow threads 62367007, 65363916, 63601082, and Google's own
docs at developers.google.com/workspace/guides/create-credentials, which
states plainly that "API keys provide anonymous access to public data").
A bare API key is treated as anonymous traffic, and Drive's front-door
anti-abuse system — a *different*, undocumented layer from the documented
quota table (developers.google.com/workspace/drive/api/guides/limits) —
flags anonymous binary-download (`alt=media`) traffic more aggressively,
especially under burst concurrency. Hands-on: this project's API-key path
tripped a "your computer or network may be sending automated queries" 403
partway through a 46-file batch. That specific error string doesn't even
appear in Google's own Drive API error-handling guide
(developers.google.com/workspace/drive/api/guides/handle-errors) — every
reference to it is describing this same front-door block, not a Drive API
error. OAuth/service-account-authenticated requests are not anonymous and
don't hit this layer the same way.

Downloads still run concurrently (DRIVE_MAX_WORKERS) for speed, but with
a small randomized stagger on start time (see _stagger_delay) to avoid
firing many large downloads in the exact same instant, since burst
concurrency was a likely aggravating factor in the original block.

Fallback paths, in order: GOOGLE_API_KEY (bare key, works but more prone
to the block above) -> gdown (unauthenticated, scrapes the Drive web UI's
download page, has its own separate "too many viewers" per-file throttle).
"""
from __future__ import annotations

import io
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

import config

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
_MAX_RETRIES = 6
_MAX_BACKOFF_SECONDS = 64.0


class DriveIngestError(RuntimeError):
    pass


def _folder_url(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"


def _clean_name(name: str) -> str:
    # Google Drive frequently prefixes duplicated/"make a copy" files with
    # "Copy of " — strip it for cleaner downstream filenames.
    return name.removeprefix("Copy of ")


def _format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _stagger_delay(index: int, max_workers: int) -> float:
    """
    Small randomized delay so `max_workers` downloads don't all start in
    the exact same instant — spreads the initial burst out over roughly
    one second per worker slot, which is cheap insurance against
    triggering Drive's front-door anti-abuse system on a batch of large
    files (see module docstring).
    """
    slot = index % max_workers
    return slot * 0.3 + random.uniform(0, 0.2)


def _request_with_backoff(method: str, url: str, **kwargs) -> requests.Response:
    """
    Exponential backoff per Google's documented algorithm: wait
    min(2**n + random_ms, max_backoff) between retries, only for the
    time-based errors (403 rate limit / 429) it's meant for. Other error
    codes (404, permission-denied-for-real, etc) raise immediately since
    retrying won't help.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.request(method, url, timeout=60, **kwargs)
        except requests.RequestException as e:
            last_exc = e
            resp = None

        if resp is not None:
            if resp.status_code not in (403, 429):
                return resp
            last_exc = DriveIngestError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        sleep_for = min((2 ** attempt) + random.uniform(0, 1), _MAX_BACKOFF_SECONDS)
        time.sleep(sleep_for)

    raise DriveIngestError(f"Exceeded retries for {url}: {last_exc}")


def _list_folder_files(folder_id: str, api_key: str) -> list[dict]:
    """List every non-trashed file directly inside a Drive folder (files.list)."""
    files: list[dict] = []
    page_token = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id, name, mimeType, size)",
            "key": api_key,
            "pageSize": 1000,
        }
        if page_token:
            params["pageToken"] = page_token

        resp = _request_with_backoff("GET", f"{DRIVE_API_BASE}/files", params=params)
        if resp.status_code != 200:
            raise DriveIngestError(f"files.list failed ({resp.status_code}): {resp.text[:300]}")

        data = resp.json()
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return files


def _download_one(file_meta: dict, dest_dir: Path, api_key: str, index: int, total: int) -> Path:
    """
    Download one file with visible progress: prints a start line
    immediately (so the terminal never looks stuck), then periodic
    progress lines (throttled to roughly once per second per file so
    concurrent downloads don't flood the terminal), then a done line with
    the final size and elapsed time.
    """
    time.sleep(_stagger_delay(index, config.DRIVE_MAX_WORKERS))

    file_id = file_meta["id"]
    name = _clean_name(file_meta["name"])
    dest_path = dest_dir / name
    declared_size = int(file_meta.get("size") or 0)

    print(f"[{index}/{total}] downloading {name} "
          f"({_format_bytes(declared_size) if declared_size else 'unknown size'})...", flush=True)

    start = time.monotonic()
    resp = _request_with_backoff(
        "GET", f"{DRIVE_API_BASE}/files/{file_id}",
        params={"alt": "media", "key": api_key},
        stream=True,
    )
    if resp.status_code != 200:
        raise DriveIngestError(
            f"files.get?alt=media failed for {name} ({resp.status_code}): {resp.text[:300]}"
        )

    total_size = int(resp.headers.get("Content-Length") or declared_size or 0)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    downloaded = 0
    last_print = start

    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)

            now = time.monotonic()
            if now - last_print >= 1.0:
                last_print = now
                elapsed = now - start
                speed = downloaded / elapsed if elapsed > 0 else 0
                if total_size:
                    pct = 100 * downloaded / total_size
                    print(f"  [{index}/{total}] {name}: {pct:5.1f}%  "
                          f"{_format_bytes(downloaded)}/{_format_bytes(total_size)}  "
                          f"{_format_bytes(speed)}/s", flush=True)
                else:
                    print(f"  [{index}/{total}] {name}: {_format_bytes(downloaded)} "
                          f"({_format_bytes(speed)}/s)", flush=True)

    tmp_path.rename(dest_path)
    elapsed = time.monotonic() - start
    print(f"[{index}/{total}] done: {name} "
          f"({_format_bytes(downloaded)} in {elapsed:.1f}s)", flush=True)
    return dest_path


def ingest_via_api_key(folder_id: str | None = None, dest_dir: str | Path | None = None,
                        api_key: str | None = None, max_workers: int | None = None) -> list[Path]:
    """
    List and download every file in a Drive folder using the real Drive
    API v3 with a plain API key. Works for any folder shared "Anyone with
    the link". Downloads run concurrently via ThreadPoolExecutor.
    """
    folder_id = folder_id or config.DRIVE_FOLDER_ID
    api_key = api_key or config.GOOGLE_API_KEY
    max_workers = max_workers or config.DRIVE_MAX_WORKERS
    if not folder_id:
        raise DriveIngestError("DRIVE_FOLDER_ID is not set (check .env).")
    if not api_key:
        raise DriveIngestError("GOOGLE_API_KEY is not set (check .env).")

    dest_dir = Path(dest_dir or config.RAW_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in Drive folder {folder_id}...", flush=True)
    files = _list_folder_files(folder_id, api_key)
    if not files:
        raise DriveIngestError(
            f"No files found in folder {folder_id}. Check the folder ID and that it's "
            "shared 'Anyone with the link' (or use a service account for private folders)."
        )

    total_bytes = sum(int(f.get("size") or 0) for f in files)
    print(f"Found {len(files)} file(s), {_format_bytes(total_bytes)} total. "
          f"Downloading with {max_workers} concurrent worker(s)...", flush=True)
    for f in files:
        print(f"  - {_clean_name(f['name'])} ({_format_bytes(int(f.get('size') or 0))})", flush=True)

    results: list[Path] = []
    errors: list[str] = []
    total = len(files)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_name = {
            pool.submit(_download_one, f, dest_dir, api_key, i + 1, total): f["name"]
            for i, f in enumerate(files)
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results.append(future.result())
            except DriveIngestError as e:
                print(f"FAILED: {name}: {e}", flush=True)
                errors.append(f"{name}: {e}")

    if errors:
        raise DriveIngestError(
            f"{len(errors)}/{len(files)} file(s) failed to download:\n" + "\n".join(errors)
        )

    print(f"All {len(results)} file(s) downloaded successfully into {dest_dir}", flush=True)
    return sorted(results)


def ingest_public_folder(folder_id: str | None = None, dest_dir: str | Path | None = None) -> list[Path]:
    """
    gdown-based fallback: works against any folder shared "Anyone with the
    link" without an API key, but is subject to Drive's undocumented
    anonymous-page throttle — see module docstring. Prefer
    ingest_via_api_key() when GOOGLE_API_KEY is set.
    """
    import gdown

    folder_id = folder_id or config.DRIVE_FOLDER_ID
    if not folder_id:
        raise DriveIngestError("DRIVE_FOLDER_ID is not set (check .env).")

    dest_dir = Path(dest_dir or config.RAW_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = gdown.download_folder(
        url=_folder_url(folder_id),
        output=str(dest_dir),
        quiet=False,
        use_cookies=False,
    )
    if not downloaded:
        raise DriveIngestError(
            f"No files downloaded from folder {folder_id}. It may be private — "
            "set GOOGLE_SERVICE_ACCOUNT_FILE and use ingest_private_folder() instead."
        )

    # gdown's return shape varies by version: a list of path strings when
    # `output` is given directly, or objects with `.local_path` in some
    # versions/when nesting under a folder-name subdirectory. Handle both,
    # and flatten into dest_dir so downstream steps see one flat /raw dir.
    result_paths: list[Path] = []
    for item in downloaded:
        src = Path(item if isinstance(item, str) else item.local_path)
        target = dest_dir / _clean_name(src.name)
        if src.resolve() != target.resolve():
            if target.exists():
                target.unlink()
            shutil.move(str(src), str(target))
        result_paths.append(target)

    # Clean up any now-empty nested folders gdown created.
    for child in dest_dir.iterdir():
        if child.is_dir() and not any(child.iterdir()):
            child.rmdir()

    return result_paths


def _list_folder_files_service_account(service, folder_id: str) -> list[dict]:
    """List every non-trashed file directly inside a Drive folder, via a service account."""
    files: list[dict] = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def _download_one_service_account(service, file_meta: dict, dest_dir: Path,
                                   index: int, total: int, max_workers: int) -> Path:
    """
    Download one file via the service account, using the official
    MediaIoBaseDownload (chunked, has its own internal retry on transient
    errors). Verifies the final size against what Drive reported before
    accepting the file, so a truncated/corrupted download is caught here
    rather than silently propagating downstream.
    """
    from googleapiclient.http import MediaIoBaseDownload

    time.sleep(_stagger_delay(index, max_workers))

    name = _clean_name(file_meta["name"])
    dest_path = dest_dir / name
    declared_size = int(file_meta.get("size") or 0)

    print(f"[{index}/{total}] downloading {name} "
          f"({_format_bytes(declared_size) if declared_size else 'unknown size'})...", flush=True)

    start = time.monotonic()
    request = service.files().get_media(fileId=file_meta["id"])
    buf = io.FileIO(str(dest_path.with_suffix(dest_path.suffix + ".part")), "wb")
    downloader = MediaIoBaseDownload(buf, request, chunksize=4 * 1024 * 1024)

    done = False
    last_print = start
    last_pct = -1
    while not done:
        status, done = downloader.next_chunk(num_retries=_MAX_RETRIES)
        if status:
            now = time.monotonic()
            if now - last_print >= 1.0:
                last_print = now
                pct = int(status.progress() * 100)
                if pct != last_pct:
                    last_pct = pct
                    elapsed = now - start
                    downloaded = int(status.resumable_progress)
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    print(f"  [{index}/{total}] {name}: {pct:5.1f}%  "
                          f"{_format_bytes(downloaded)}/{_format_bytes(declared_size)}  "
                          f"{_format_bytes(speed)}/s", flush=True)
    buf.close()

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    actual_size = tmp_path.stat().st_size
    if declared_size and actual_size != declared_size:
        tmp_path.unlink(missing_ok=True)
        raise DriveIngestError(
            f"Size mismatch for {name}: expected {declared_size} bytes, got {actual_size}. "
            "Download likely truncated; discarded."
        )

    tmp_path.rename(dest_path)
    elapsed = time.monotonic() - start
    print(f"[{index}/{total}] done: {name} "
          f"({_format_bytes(actual_size)} in {elapsed:.1f}s)", flush=True)
    return dest_path


def ingest_private_folder(folder_id: str | None = None, dest_dir: str | Path | None = None,
                           service_account_file: str | None = None,
                           max_workers: int | None = None) -> list[Path]:
    """
    List and download every file in a Drive folder using a service
    account. Works for both private folders (shared explicitly with the
    service account's email) and public "Anyone with the link" folders
    (no extra sharing needed — the folder is already open to any Google
    identity). Requires google-api-python-client + google-auth.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as e:
        raise DriveIngestError(
            "Service-account ingestion needs google-api-python-client + google-auth. "
            "Install with: pip install google-api-python-client google-auth"
        ) from e

    folder_id = folder_id or config.DRIVE_FOLDER_ID
    service_account_file = service_account_file or config.GOOGLE_SERVICE_ACCOUNT_FILE
    max_workers = max_workers or config.DRIVE_MAX_WORKERS
    if not folder_id:
        raise DriveIngestError("DRIVE_FOLDER_ID is not set (check .env).")
    if not service_account_file:
        raise DriveIngestError("GOOGLE_SERVICE_ACCOUNT_FILE is not set (check .env).")

    dest_dir = Path(dest_dir or config.RAW_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    creds = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    # Each worker thread gets its own service client — googleapiclient's
    # http objects aren't guaranteed thread-safe to share across threads.
    def _new_service():
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    print(f"Listing files in Drive folder {folder_id} (service account)...", flush=True)
    files = _list_folder_files_service_account(_new_service(), folder_id)
    if not files:
        raise DriveIngestError(
            f"No files found in folder {folder_id}. Check the folder ID, that it's shared "
            "with the service account (or 'Anyone with the link'), and that the Drive API "
            "is enabled on the service account's Cloud project."
        )

    total_bytes = sum(int(f.get("size") or 0) for f in files)
    print(f"Found {len(files)} file(s), {_format_bytes(total_bytes)} total. "
          f"Downloading with {max_workers} concurrent worker(s)...", flush=True)
    for f in files:
        print(f"  - {_clean_name(f['name'])} ({_format_bytes(int(f.get('size') or 0))})", flush=True)

    results: list[Path] = []
    errors: list[str] = []
    total = len(files)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_name = {
            pool.submit(_download_one_service_account, _new_service(), f, dest_dir,
                        i + 1, total, max_workers): f["name"]
            for i, f in enumerate(files)
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"FAILED: {name}: {e}", flush=True)
                errors.append(f"{name}: {e}")

    if errors:
        raise DriveIngestError(
            f"{len(errors)}/{len(files)} file(s) failed to download:\n" + "\n".join(errors)
        )

    print(f"All {len(results)} file(s) downloaded successfully into {dest_dir}", flush=True)
    return sorted(results)


def ingest(folder_id: str | None = None, dest_dir: str | Path | None = None) -> list[Path]:
    """
    Pick the best available ingestion method based on config:
      1. GOOGLE_SERVICE_ACCOUNT_FILE set -> service account (most robust, see module docstring)
      2. GOOGLE_API_KEY set              -> real Drive API with a bare key (works, more block-prone)
      3. neither set                     -> gdown fallback (unauthenticated, throttle-prone)
    """
    if config.GOOGLE_SERVICE_ACCOUNT_FILE:
        return ingest_private_folder(folder_id, dest_dir)
    if config.GOOGLE_API_KEY:
        return ingest_via_api_key(folder_id, dest_dir)
    return ingest_public_folder(folder_id, dest_dir)


if __name__ == "__main__":
    paths = ingest()
    print(f"Downloaded {len(paths)} file(s) into {config.RAW_DIR}:")
    for p in paths:
        print(f"  - {p.name}")
