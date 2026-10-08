"""Run one job to completion: ``python -m streamerframes.pipeline.worker <job.json>``.

Used in-process by ``render`` and as a subprocess by the job manager (one GPU worker at a time).
Exit codes: 0 done (complete, or nothing left to do), 1 failed, 3 paused (stop requested).
"""
from __future__ import annotations

import copy
import json
import logging
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..cache.store import CacheStore, ProfileCache, video_id_for
from ..config import Profile, Settings, load_settings
from ..probe import VideoInfo, probe
from .audio import ensure_audio
from .export import export
from .generator import Generator, RunResult, resolve_scale

log = logging.getLogger(__name__)


@dataclass
class Job:
    video_path: str
    profile: str = "realtime"
    kind: str = "offline"            # offline | stream
    start_segment: int = 0
    overrides: dict | None = None    # profile field overrides, e.g. {"scale": 0.5, "encoder": {"cq": 18}}
    export: bool = False
    export_path: str | None = None
    container: str = "auto"
    device: str = "cuda"
    hwaccel: bool = True
    config: str | None = None
    job_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def build_profile(settings: Settings, job: Job) -> Profile:
    profile = copy.deepcopy(settings.profile(job.profile))
    for key, value in (job.overrides or {}).items():
        if key == "encoder":
            for ek, ev in value.items():
                setattr(profile.encoder, ek, ev)
        elif hasattr(profile, key):
            setattr(profile, key, value)
        else:
            raise ValueError(f"unknown profile override {key!r}")
    if job.overrides:
        profile.name = f"{job.profile}+custom"
    return profile


def resolve(settings: Settings, job: Job, info: VideoInfo | None = None):
    """-> (info, video_id, profile, scale, ProfileCache) without starting anything."""
    info = info or probe(job.video_path, settings.ffprobe)
    vid = video_id_for(job.video_path)
    store = CacheStore(settings.cache_root)
    store.write_source(vid, info.to_json())
    profile = build_profile(settings, job)
    scale = resolve_scale(profile, info.width, info.height, settings=settings, model=profile.model)
    cache = store.profile(vid, profile.profile_id(scale))
    return info, vid, profile, scale, cache


def run_job(job: Job, settings: Settings | None = None, stop_event: threading.Event | None = None,
            on_progress: Callable[[dict], None] | None = None, engine_factory=None) -> RunResult:
    settings = settings or load_settings(job.config)
    stop_event = stop_event or threading.Event()
    info, vid, profile, scale, cache = resolve(settings, job)
    log.info("job %s: %s profile=%s scale=%s cache=%s", job.job_id or "-", job.video_path, profile.name,
             scale, cache.root)
    if job.kind == "stream":
        ensure_audio(info, CacheStore(settings.cache_root).audio_dir(vid))
    gen = Generator(info, profile, cache, scale=scale, model_dir=settings.model_path(profile.model),
                    device=job.device, hwaccel=job.hwaccel, engine_factory=engine_factory,
                    stop_event=stop_event, on_progress=on_progress)
    start = job.start_segment
    while True:
        result = gen.run(start)
        if result.status != "partial" or stop_event.is_set() or cache.stop_requested():
            break
        start = 0  # a hole was filled; fill the next one (first_missing wraps around)
    if result.status == "complete" and job.export:
        export(cache, info, Path(job.export_path) if job.export_path else None, job.container,
               Path(settings.export_dir) if settings.export_dir else None)
    return result


def cache_for(settings: Settings, job: Job) -> ProfileCache:
    return resolve(settings, job)[4]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m streamerframes.pipeline.worker <job.json>", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    job = Job.from_dict(json.loads(Path(argv[0]).read_text(encoding="utf-8")))
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        result = run_job(job, stop_event=stop)
    except Exception:  # noqa: BLE001
        log.exception("worker failed")
        return 1
    log.info("job %s finished: %s", job.job_id or "-", result.status)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
