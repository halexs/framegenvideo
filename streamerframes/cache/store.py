"""On-disk segment cache (PLAN.md section 3.3): the contract between worker, server and export.

<cache_root>/<video_id>/source.json, audio/, <profile_id>/{manifest.json, segments/, index/, progress.json, lock, stop}
A segment is complete only if it is listed in some index/run_*.csv (ffmpeg writes the line after closing the file).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

SEGMENT_RE = re.compile(r"v_(\d+)\.ts$")
STATUSES = ("new", "running", "paused", "complete", "failed")


def atomic_write_json(path: Path, data) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def video_id_for(path: str | Path) -> str:
    p = Path(path).resolve()
    st = p.stat()
    key = f"{str(p).lower()}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def process_create_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def lock_is_live(lock: dict | None) -> bool:
    if not lock or "pid" not in lock:
        return False
    created = process_create_time(int(lock["pid"]))
    return created is not None and abs(created - float(lock.get("create_time", created))) < 1.0


class LockHeld(RuntimeError):
    pass


@dataclass
class ProfileCache:
    root: Path   # <cache_root>/<video_id>/<profile_id>

    def __post_init__(self):
        self.root = Path(self.root)

    # Paths
    @property
    def video_dir(self) -> Path:
        return self.root.parent

    @property
    def video_id(self) -> str:
        return self.root.parent.name

    @property
    def profile_id(self) -> str:
        return self.root.name

    @property
    def segments_dir(self) -> Path:
        return self.root / "segments"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def progress_path(self) -> Path:
        return self.root / "progress.json"

    @property
    def lock_path(self) -> Path:
        return self.root / "lock"

    @property
    def stop_path(self) -> Path:
        return self.root / "stop"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    def segment_path(self, k: int) -> Path:
        return self.segments_dir / f"v_{k:05d}.ts"

    def ensure_dirs(self) -> None:
        for d in (self.segments_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)

    # Manifest / progress
    def manifest(self) -> dict | None:
        return read_json(self.manifest_path)

    def write_manifest(self, data: dict) -> None:
        data["updated_at"] = time.time()
        atomic_write_json(self.manifest_path, data)

    def update_manifest(self, **changes) -> dict:
        data = self.manifest() or {}
        data.update(changes)
        self.write_manifest(data)
        return data

    def progress(self) -> dict | None:
        return read_json(self.progress_path)

    def write_progress(self, data: dict) -> None:
        data["updated_at"] = time.time()
        atomic_write_json(self.progress_path, data)

    # Segment index
    def index_rows(self) -> dict[int, tuple[float, float]]:
        """segment number -> (start, end) seconds as recorded by the segment muxer."""
        rows: dict[int, tuple[float, float]] = {}
        if not self.index_dir.exists():
            return rows
        for csv_path in sorted(self.index_dir.glob("run_*.csv")):
            try:
                with open(csv_path, newline="", encoding="utf-8") as fh:
                    for row in csv.reader(fh):
                        m = SEGMENT_RE.search(row[0]) if row else None
                        if m and len(row) >= 3 and self.segment_path(int(m.group(1))).exists():
                            rows[int(m.group(1))] = (float(row[1]), float(row[2]))
            except (OSError, ValueError, IndexError):
                continue
        return rows

    def completed_segments(self) -> set[int]:
        return set(self.index_rows())

    def next_run_id(self) -> int:
        ids = [int(p.stem.split("_")[1]) for p in self.index_dir.glob("run_*.csv")] if self.index_dir.exists() else []
        return max(ids, default=-1) + 1

    @staticmethod
    def first_missing(completed: set[int], start: int, total: int) -> int | None:
        for k in range(max(start, 0), total):
            if k not in completed:
                return k
        for k in range(0, min(start, total)):  # wrap: fill earlier holes last
            if k not in completed:
                return k
        return None

    @staticmethod
    def gap_end(completed: set[int], k: int, total: int) -> int:
        """End (exclusive) of the hole starting at k: the next completed segment, or total."""
        for e in range(k + 1, total):
            if e in completed:
                return e
        return total

    def contiguous_prefix(self, completed: set[int] | None = None) -> int:
        completed = self.completed_segments() if completed is None else completed
        n = 0
        while n in completed:
            n += 1
        return n

    # Lock and stop
    def acquire_lock(self, run: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        current = read_json(self.lock_path)
        if lock_is_live(current) and int(current["pid"]) != os.getpid():
            raise LockHeld(f"{self.root} is being generated by pid {current['pid']}")
        pid = os.getpid()
        atomic_write_json(self.lock_path, {"pid": pid, "create_time": process_create_time(pid),
                                           "started": time.time(), "run": run})

    def release_lock(self) -> None:
        current = read_json(self.lock_path)
        if current and int(current.get("pid", -1)) == os.getpid():
            self.lock_path.unlink(missing_ok=True)

    def lock(self) -> dict | None:
        current = read_json(self.lock_path)
        return current if lock_is_live(current) else None

    def request_stop(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.stop_path.touch()

    def stop_requested(self) -> bool:
        return self.stop_path.exists()

    def clear_stop(self) -> None:
        self.stop_path.unlink(missing_ok=True)

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


class CacheStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def video_dir(self, video_id: str) -> Path:
        return self.root / video_id

    def profile(self, video_id: str, profile_id: str) -> ProfileCache:
        return ProfileCache(self.root / video_id / profile_id)

    def write_source(self, video_id: str, info_json: dict) -> None:
        d = self.video_dir(video_id)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_json(d / "source.json", info_json)

    def source(self, video_id: str) -> dict | None:
        return read_json(self.video_dir(video_id) / "source.json")

    def audio_dir(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "audio"

    def video_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "source.json").exists())

    def profiles(self, video_id: str) -> list[ProfileCache]:
        d = self.video_dir(video_id)
        if not d.exists():
            return []
        return [ProfileCache(p) for p in sorted(d.iterdir()) if (p / "manifest.json").exists()]

    def remove(self, video_id: str, profile_id: str | None = None) -> None:
        targets = self.profiles(video_id) if profile_id is None else [self.profile(video_id, profile_id)]
        for pc in targets:
            if pc.lock():
                raise LockHeld(f"{pc.root} is in use")
        if profile_id is None:
            shutil.rmtree(self.video_dir(video_id), ignore_errors=True)
        else:
            shutil.rmtree(self.profile(video_id, profile_id).root, ignore_errors=True)

    def gc(self, max_bytes: int, keep: set[tuple[str, str]] = frozenset()) -> list[ProfileCache]:
        """LRU-evict idle, non-exported profile caches until the total fits in ``max_bytes``."""
        entries = []
        total = 0
        for vid in self.video_ids():
            for pc in self.profiles(vid):
                size = pc.size_bytes()
                total += size
                exported = pc.export_dir.exists() and any(pc.export_dir.iterdir())
                if pc.lock() or exported or (vid, pc.profile_id) in keep:
                    continue
                last = (pc.manifest() or {}).get("last_access", (pc.manifest() or {}).get("updated_at", 0))
                entries.append((last, size, pc))
        evicted = []
        for _, size, pc in sorted(entries, key=lambda e: e[0]):
            if total <= max_bytes:
                break
            shutil.rmtree(pc.root, ignore_errors=True)
            total -= size
            evicted.append(pc)
        return evicted
