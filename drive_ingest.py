"""
drive_ingest.py — pull raw clips from a Google Drive folder into /raw.

Primary path: `gdown.download_folder`, which works against any folder
shared "Anyone with the link" without needing OAuth or a service account.
Confirmed against the real folder in this project
(https://drive.google.com/drive/folders/<DRIVE_FOLDER_ID>) — it's public
and lists 6 clips.

Fallback path: if GOOGLE_SERVICE_ACCOUNT_FILE is set (folder is NOT
public), use the Drive API v3 with a service account instead. This path
requires `google-api-python-client` + `google-auth`, which aren't in
requirements.txt by default since the primary path doesn't need them —
install them if you actually need this branch.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import gdown

import config


class DriveIngestError(RuntimeError):
    pass


def _folder_url(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"


def ingest_public_folder(folder_id: str | None = None, dest_dir: str | Path | None = None) -> list[Path]:
    """
    Download every file in a publicly-shared Drive folder into dest_dir
    (flattened — gdown may create a subfolder named after the Drive folder;
    we move files up into dest_dir directly).
    """
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
        # Google Drive frequently prefixes duplicated/"make a copy" files
        # with "Copy of " — strip it for cleaner downstream filenames.
        clean_name = src.name.removeprefix("Copy of ")
        target = dest_dir / clean_name
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


def ingest_private_folder(folder_id: str | None = None, dest_dir: str | Path | None = None,
                           service_account_file: str | None = None) -> list[Path]:
    """
    Download every file in a Drive folder the caller has access to via a
    service account. Requires google-api-python-client + google-auth
    (`pip install google-api-python-client google-auth`).
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as e:
        raise DriveIngestError(
            "Private-folder ingestion needs google-api-python-client + google-auth. "
            "Install with: pip install google-api-python-client google-auth"
        ) from e

    folder_id = folder_id or config.DRIVE_FOLDER_ID
    service_account_file = service_account_file or config.GOOGLE_SERVICE_ACCOUNT_FILE
    if not folder_id:
        raise DriveIngestError("DRIVE_FOLDER_ID is not set (check .env).")
    if not service_account_file:
        raise DriveIngestError("GOOGLE_SERVICE_ACCOUNT_FILE is not set (check .env).")

    dest_dir = Path(dest_dir or config.RAW_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    creds = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build("drive", "v3", credentials=creds)

    results: list[Path] = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
        ).execute()

        for f in resp.get("files", []):
            dest_path = dest_dir / f["name"]
            request = service.files().get_media(fileId=f["id"])
            with open(dest_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            results.append(dest_path)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results


def ingest(folder_id: str | None = None, dest_dir: str | Path | None = None) -> list[Path]:
    """Pick public vs. private ingestion based on config."""
    if config.GOOGLE_SERVICE_ACCOUNT_FILE:
        return ingest_private_folder(folder_id, dest_dir)
    return ingest_public_folder(folder_id, dest_dir)


if __name__ == "__main__":
    paths = ingest()
    print(f"Downloaded {len(paths)} file(s) into {config.RAW_DIR}:")
    for p in paths:
        print(f"  - {p.name}")
