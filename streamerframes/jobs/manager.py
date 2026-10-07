"""Single-GPU job scheduler (PLAN.md Phase 3 "jobs/manager.py").

- One worker subprocess at a time owns the GPU; jobs persist in <cache_root>/jobs.json.
- Stream jobs outrank offline jobs: a running offline job is stopped gracefully (touch ``stop``), killed with
  its ffmpeg children after ``stop_timeout``, and re-queued so it resumes later.
- ``stream_policy="newest_wins"``: a new stream job pauses older stream jobs.
- On start, running workers from a previous server are adopted (pid + create time); dead ones are re-queued
  (offline, with ``auto_resume``) or paused.
- Watchdog: a live worker whose progress is stale for ``watchdog_seconds`` is killed and marked failed.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import psutil

from ..cache.store import (
    ProfileCache,
    atomic_write_json,
    lock_is_live,
    process_create_time,
    read_json,
)
from ..config import REPO_ROOT, Settings

log = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
FINAL = ("complete", "failed", "cancelled")
PRIORITY = {"stream": 0, "offline": 1}


@dataclass
class JobRecord:
    id: str
    video_id: str
    video_path: str
    profile: str
    profile_id: str
    cache_dir: str
    kind: str = "offline"
    start_segment: int = 0
    state: str = "queued"          # queued | running | paused | complete | failed | cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    pid: int | None = None
    pid_create_time: float | None = None
    stop_reason: str | None = None  # preempt | cancel | superseded | retarget
    stop_requested_at: float | None = None
    error: str | None = None
    runs: int = 0

    @property
    def cache(self) -> ProfileCache:
        return ProfileCache(Path(self.cache_dir))


Resolved = tuple[str, str, str]  # video_id, profile_id, cache_dir


def default_resolver(settings: Settings) -> Callable[[str, str, str], Resolved]:
    def resolve_job(video_path: str, profile: str, kind: str) -> Resolved:
        from ..pipeline.worker import Job, resolve
        _, vid, _, _, cache, _ = resolve(settings, Job(video_path=video_path, profile=profile, kind=kind))
        return vid, cache.profile_id, str(cache.root)
    return resolve_job


def default_worker_cmd(job_file: Path) -> list[str]:
    return [sys.executable, "-m", "streamerframes.pipeline.worker", str(job_file)]


def kill_tree(pid: int, timeout: float = 5.0) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


class JobManager:
    def __init__(self, settings: Settings, *, config_path: str | None = None,
                 resolver: Callable[[str, str, str], Resolved] | None = None,
                 worker_cmd: Callable[[Path], list[str]] = default_worker_cmd,
                 job_defaults: dict | None = None, poll_interval: float = 1.0, stop_timeout: float = 30.0,
                 watchdog_seconds: float = 120.0, on_change: Callable[[JobRecord], None] | None = None):
        self.settings = settings
        self.config_path = config_path
        self.resolver = resolver or default_resolver(settings)
        self.worker_cmd = worker_cmd
        self.job_defaults = job_defaults or {}
        self.poll_interval = poll_interval
        self.stop_timeout = stop_timeout
        self.watchdog_seconds = watchdog_seconds
        self.on_change = on_change
        self.root = Path(settings.cache_root)
        self.jobs_path = self.root / "jobs.json"
        self.jobs: dict[str, JobRecord] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._halt = threading.Event()
        self._load()

    # Persistence
    def _load(self) -> None:
        data = read_json(self.jobs_path) or {}
        for raw in data.get("jobs", []):
            rec = JobRecord(**{k: v for k, v in raw.items() if k in JobRecord.__dataclass_fields__})
            self.jobs[rec.id] = rec

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.jobs_path, {"jobs": [asdict(j) for j in self.jobs.values()]})

    def _changed(self, rec: JobRecord) -> None:
        self._save()
        if self.on_change:
            self.on_change(rec)

    # Lifecycle
    def start(self) -> None:
        self.adopt()
        self._halt.clear()
        self._thread = threading.Thread(target=self._loop, name="sf-jobs", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Stop scheduling. Running workers keep going and are adopted by the next server."""
        self._halt.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._halt.wait(self.poll_interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the scheduler must survive anything
                log.exception("scheduler tick failed")

    def adopt(self) -> None:
        with self._lock:
            for rec in self.jobs.values():
                if rec.state != "running":
                    continue
                if self._alive(rec):
                    log.info("adopted running worker pid %s for job %s", rec.pid, rec.id)
                else:
                    self._on_exit(rec, None)
            tracked = {r.cache_dir for r in self.jobs.values() if r.state == "running"}
            for lock_path in self.root.glob("*/*/lock"):
                cache = ProfileCache(lock_path.parent)
                lock = read_json(lock_path)
                if str(cache.root) in tracked or not lock_is_live(lock):
                    continue
                manifest = cache.manifest() or {}
                rec = JobRecord(id=uuid.uuid4().hex[:12], video_id=cache.video_id,
                                video_path=manifest.get("source", ""), profile=manifest.get("profile_name", "?"),
                                profile_id=cache.profile_id, cache_dir=str(cache.root), state="running",
                                started=lock.get("started"), pid=int(lock["pid"]),
                                pid_create_time=lock.get("create_time"))
                self.jobs[rec.id] = rec
                log.info("adopted untracked worker pid %s on %s", rec.pid, cache.root)
            self._save()

    # API
    def submit(self, video_path: str, kind: str = "offline", profile: str = "realtime",
               start_segment: int = 0) -> JobRecord:
        if kind not in PRIORITY:
            raise ValueError(f"kind must be one of {sorted(PRIORITY)}")
        vid, pid, cache_dir = self.resolver(video_path, profile, kind)
        with self._lock:
            for rec in self.jobs.values():
                if rec.video_id == vid and rec.profile_id == pid and rec.state in ACTIVE:
                    if kind == "stream" and rec.kind == "offline":
                        rec.kind = "stream"  # promote: someone is watching now
                        self._changed(rec)
                    return rec
            rec = JobRecord(id=uuid.uuid4().hex[:12], video_id=vid, video_path=video_path, profile=profile,
                            profile_id=pid, cache_dir=cache_dir, kind=kind, start_segment=start_segment)
            if kind == "stream" and self.settings.stream_policy == "newest_wins":
                for other in self.jobs.values():
                    if other.kind == "stream" and other.state == "queued":
                        other.state = "paused"
                        other.stop_reason = "superseded"
                        self._changed(other)
                    elif other.kind == "stream" and other.state == "running":
                        self._request_stop(other, "superseded")
            self.jobs[rec.id] = rec
            self._changed(rec)
            log.info("queued %s job %s for %s (%s)", kind, rec.id, video_path, profile)
        self.tick()
        return rec

    def resume(self, job_id: str) -> JobRecord:
        with self._lock:
            rec = self.jobs[job_id]
            if rec.state in ("paused", "failed"):
                rec.state, rec.stop_reason, rec.error = "queued", None, None
                self._changed(rec)
        self.tick()
        return rec

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            rec = self.jobs[job_id]
            if rec.state == "queued":
                rec.state, rec.finished = "cancelled", time.time()
                self._changed(rec)
            elif rec.state == "running":
                self._request_stop(rec, "cancel")
            elif rec.state == "paused":
                rec.state = "cancelled"
                self._changed(rec)
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def list(self) -> list[JobRecord]:
        return sorted(self.jobs.values(), key=lambda r: (r.state not in ACTIVE, PRIORITY[r.kind], r.created))

    def running(self) -> JobRecord | None:
        return next((r for r in self.jobs.values() if r.state == "running"), None)

    def active_for(self, video_id: str, profile_id: str | None = None) -> JobRecord | None:
        for rec in self.list():
            if rec.video_id == video_id and rec.state in ACTIVE and profile_id in (None, rec.profile_id):
                return rec
        return None

    def prune(self, keep_final: int = 200) -> None:
        with self._lock:
            final = sorted((r for r in self.jobs.values() if r.state in FINAL), key=lambda r: r.created)
            for rec in final[:-keep_final] if len(final) > keep_final else []:
                del self.jobs[rec.id]
            self._save()

    # Scheduling
    def tick(self) -> None:
        with self._lock:
            now = time.time()
            current = self.running()
            if current:
                if not self._alive(current):
                    self._on_exit(current, self._returncode(current))
                    current = None
                else:
                    self._supervise(current, now)
            if current is None:
                nxt = self._next_queued()
                if nxt:
                    self._launch(nxt)
            elif current.kind == "offline" and current.stop_reason is None:
                nxt = self._next_queued()
                if nxt and nxt.kind == "stream":
                    log.info("preempting offline job %s for stream job %s", current.id, nxt.id)
                    self._request_stop(current, "preempt")

    def _next_queued(self) -> JobRecord | None:
        queued = [r for r in self.jobs.values() if r.state == "queued"]
        return min(queued, key=lambda r: (PRIORITY[r.kind], r.created), default=None)

    def _supervise(self, rec: JobRecord, now: float) -> None:
        if rec.stop_requested_at and now - rec.stop_requested_at > self.stop_timeout:
            log.warning("job %s did not stop within %ss; killing", rec.id, self.stop_timeout)
            kill_tree(rec.pid)
            return
        progress = rec.cache.progress() or {}
        last = max(rec.started or 0, progress.get("updated_at", 0))
        if now - last > self.watchdog_seconds:
            log.error("job %s: no progress for %ss; killing", rec.id, int(now - last))
            kill_tree(rec.pid)
            rec.error = f"watchdog: no progress for {int(now - last)}s"
            rec.stop_reason = "watchdog"

    def _request_stop(self, rec: JobRecord, reason: str) -> None:
        rec.stop_reason = reason
        rec.stop_requested_at = time.time()
        rec.cache.request_stop()
        self._changed(rec)

    def _launch(self, rec: JobRecord) -> None:
        job_dir = self.root / "jobs"
        job_dir.mkdir(parents=True, exist_ok=True)
        job_file = job_dir / f"{rec.id}.json"
        payload = {"job_id": rec.id, "video_path": rec.video_path, "profile": rec.profile, "kind": rec.kind,
                   "start_segment": rec.start_segment, "config": self.config_path, "cache_dir": rec.cache_dir,
                   **self.job_defaults}
        atomic_write_json(job_file, payload)
        cache = rec.cache
        cache.root.mkdir(parents=True, exist_ok=True)
        cache.clear_stop()
        log_file = open(cache.root / "worker.log", "ab")
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True  # survives the server's Ctrl+C; adopted on restart
        proc = subprocess.Popen(self.worker_cmd(job_file), cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL,
                                stdout=log_file, stderr=subprocess.STDOUT, **kwargs)
        log_file.close()
        self._procs[rec.id] = proc
        rec.state, rec.pid, rec.pid_create_time = "running", proc.pid, process_create_time(proc.pid)
        rec.started, rec.finished, rec.stop_reason, rec.stop_requested_at, rec.error = (
            time.time(), None, None, None, None)
        rec.runs += 1
        self._changed(rec)
        log.info("started %s job %s (pid %s)", rec.kind, rec.id, proc.pid)

    def _alive(self, rec: JobRecord) -> bool:
        proc = self._procs.get(rec.id)
        if proc is not None:
            return proc.poll() is None
        if rec.pid is None:
            return False
        created = process_create_time(rec.pid)
        if created is None:
            return False
        if rec.pid_create_time and abs(created - rec.pid_create_time) > 1.0:
            return False
        try:
            return psutil.Process(rec.pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def _returncode(self, rec: JobRecord) -> int | None:
        proc = self._procs.pop(rec.id, None)
        return proc.returncode if proc is not None else None

    def _on_exit(self, rec: JobRecord, rc: int | None) -> None:
        manifest = rec.cache.manifest() or {}
        status = manifest.get("status")
        rec.pid = rec.pid_create_time = None
        rec.cache.clear_stop()
        reason = rec.stop_reason
        if status == "complete":
            rec.state = "complete"
        elif reason == "preempt":
            rec.state = "queued"
        elif reason == "cancel":
            rec.state = "cancelled"
        elif reason in ("superseded", "retarget"):
            rec.state = "paused" if reason == "superseded" else "queued"
        elif reason == "watchdog":
            rec.state = "failed"
        elif rc == 1 or status == "failed":
            rec.state = "failed"
            rec.error = manifest.get("error") or rec.error or f"worker exited with code {rc}; see worker.log"
        elif rc == 3 or status == "paused":
            rec.state = "paused"
        elif rc is None and status in ("running", "new", None):
            # The worker died with the server down (or crashed): resume offline work automatically.
            rec.state = "queued" if (rec.kind == "offline" and self.settings.auto_resume) else "paused"
        else:
            rec.state = "failed"
            rec.error = rec.error or f"worker exited with code {rc} and cache status {status}"
        if rec.state in FINAL:
            rec.finished = time.time()
        if rec.state != "queued":
            rec.stop_reason = None
        rec.stop_requested_at = None
        log.info("job %s -> %s%s", rec.id, rec.state, f" ({rec.error})" if rec.error else "")
        self._changed(rec)


def job_view(rec: JobRecord) -> dict:
    """Job + live progress + derived fields for the API and UI."""
    data = asdict(rec)
    cache = rec.cache
    manifest = cache.manifest() or {}
    progress = cache.progress() or {}
    data["manifest_status"] = manifest.get("status")
    data["progress"] = progress if rec.state == "running" else None
    total = manifest.get("total_segments")
    if total:
        done = cache.completed_segments()
        prefix = cache.contiguous_prefix(done)
        from fractions import Fraction
        seg_secs = float(Fraction(manifest["seg_frames"]) / Fraction(manifest["out_fps"]))
        data.update(segments_done=len(done), segments_total=total, contiguous_segments=prefix,
                    contiguous_seconds=round(prefix * seg_secs, 3),
                    duration_seconds=round(manifest["total_out_frames"] / float(Fraction(manifest["out_fps"])), 3),
                    percent=round(100.0 * len(done) / total, 1))
    return json.loads(json.dumps(data, default=str))
