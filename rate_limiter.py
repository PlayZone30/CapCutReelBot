"""
rate_limiter.py — a small sliding-window rate limiter for the Gemini API's
free-tier quota (5 requests/minute, 20 requests/day as of this writing —
confirm your actual tier's limits at https://ai.google.dev/gemini-api/docs/rate-limits).

Only the model-inference call (client.interactions.create) counts against
this quota; client.files.upload does not, so uploads can run fully in
parallel while inference calls get throttled to the configured rate.

State is persisted to a small JSON file so the daily cap holds across
multiple separate `python main.py` invocations in the same day, not just
within one process. Thread-safe for concurrent use within one process;
not designed for multiple processes hitting the same state file at once.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class RateLimitExceeded(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int, state_file: str | Path):
        self.per_minute = per_minute
        self.per_day = per_day
        self.state_file = Path(state_file)
        self._lock = threading.Lock()
        self._timestamps: list[float] = self._load()

    def _load(self) -> list[float]:
        if not self.state_file.exists():
            return []
        try:
            data = json.loads(self.state_file.read_text())
            now = time.time()
            return [t for t in data if now - t < 86400]
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self) -> None:
        try:
            self.state_file.write_text(json.dumps(self._timestamps))
        except OSError:
            pass  # best-effort persistence; don't crash the pipeline over it

    def acquire(self) -> None:
        """
        Block until a request slot is available under both the per-minute
        and per-day caps. Raises RateLimitExceeded if the day is fully spent
        (no point waiting up to 24h inside a script run).
        """
        while True:
            with self._lock:
                now = time.time()
                self._timestamps = [t for t in self._timestamps if now - t < 86400]

                if len(self._timestamps) >= self.per_day:
                    oldest = min(self._timestamps)
                    wait_hours = (86400 - (now - oldest)) / 3600
                    raise RateLimitExceeded(
                        f"Daily Gemini quota of {self.per_day} requests reached. "
                        f"Next slot frees up in ~{wait_hours:.1f}h."
                    )

                minute_ts = [t for t in self._timestamps if now - t < 60]
                if len(minute_ts) < self.per_minute:
                    self._timestamps.append(now)
                    self._save()
                    return

                sleep_for = 60 - (now - min(minute_ts)) + 0.05

            time.sleep(max(sleep_for, 0.05))

    def remaining_today(self) -> int:
        with self._lock:
            now = time.time()
            self._timestamps = [t for t in self._timestamps if now - t < 86400]
            return max(0, self.per_day - len(self._timestamps))
