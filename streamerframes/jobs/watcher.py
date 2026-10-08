"""Watch folders (PLAN.md Phase 5 item 5): new videos get an offline job queued automatically.

The first scan of a folder records what is already there without queueing it. A file is queued once its
size and mtime are unchanged across two scans, so files still being copied are left alone.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..cache.store import atomic_write_json, read_json
from ..config import Settings

log = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi", ".ts", ".m2ts"}


class FolderWatcher:
    def __init__(self, settings: Settings, submit, interval: float = 60.0):
        """``submit(path, kind, profile)`` queues a job (JobManager.submit)."""
        self.settings = settings
        self.submit = submit
        self.interval = interval
        self.state_path = Path(settings.cache_root) / "watch_state.json"
        self.state: dict = read_json(self.state_path) or {"dirs": {}, "files": {}}
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    def scan(self) -> list[str]:
        queued = []
        files = self.state["files"]
        for d in self.settings.watch_dirs:
            root = Path(d)
            if not root.is_dir():
                continue
            baseline = d not in self.state["dirs"]
            for path in sorted(root.rglob("*")):
                if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file():
                    continue
                key = str(path)
                st = path.stat()
                sig = [st.st_size, st.st_mtime_ns]
                entry = files.get(key)
                if baseline:
                    files[key] = {"sig": sig, "queued": True, "baseline": True}
                elif entry is None or entry["sig"] != sig:
                    files[key] = {"sig": sig, "queued": False}  # new or still changing: wait one more scan
                elif not entry["queued"]:
                    try:
                        self.submit(key, "offline", self.settings.watch_profile)
                        entry["queued"] = True
                        queued.append(key)
                        log.info("watch folder: queued %s", key)
                    except Exception:  # noqa: BLE001 - e.g. unreadable file; retried next scan
                        log.exception("watch folder: could not queue %s", key)
            self.state["dirs"][d] = True
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.state_path, self.state)
        return queued

    def start(self) -> None:
        if not self.settings.watch_dirs:
            return
        self._halt.clear()
        self._thread = threading.Thread(target=self._loop, name="sf-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._halt.set()

    def _loop(self) -> None:
        while True:
            try:
                self.scan()
            except Exception:  # noqa: BLE001
                log.exception("watch folder scan failed")
            if self._halt.wait(self.interval):
                return
