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
from .generator import Generator, RunResult

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
    export_codec: str = "copy"       # copy (H.264 segments) | hevc
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


def _video_crop(store: CacheStore, vid: str, info: VideoInfo, settings: Settings):
    """Letterbox crop, detected once per video and remembered in source.json."""
    from ..engine.color import Crop
    from .letterbox import detect_letterbox

    source = store.source(vid) or {}
    if "letterbox" not in source:
        crop = detect_letterbox(info, ffmpeg=settings.ffmpeg)
        source["letterbox"] = [crop.x, crop.y, crop.w, crop.h] if crop else None
        store.write_source(vid, source)
    box = source["letterbox"]
    return Crop(*box) if box else None


def _auto_choice(store: CacheStore, vid: str, settings: Settings, profile: Profile, info: VideoInfo, crop,
                 device: str | None) -> tuple[str, float]:
    """Resolve model="auto" and scale="auto" once per (video, profile) and remember the result, so the
    server and every worker agree on the cache directory even if calibration changes later."""
    from ..bench import (
        choose_model_and_scale,
        estimate_ms,
        load_calibration,
        run_bench,
        save_entries,
    )
    from .generator import interps_per_second, out_fps_for_profile

    if profile.model != "auto" and profile.scale != "auto":
        return profile.model, float(profile.scale)
    source = store.source(vid) or {}
    sticky = source.get("auto_choice", {}).get(profile.name)
    if sticky:
        return sticky["model"], float(sticky["scale"])
    models = settings.available_models() if profile.model == "auto" else [profile.model]
    if not models:
        raise RuntimeError(f"no usable model in {settings.model_dir} or {settings.models_dir}")
    scales = (1.0, 0.5) if profile.scale == "auto" else (float(profile.scale),)
    h, w = (crop.h, crop.w) if crop else (info.height, info.width)
    calibration = load_calibration(settings.cache_root)
    if device and str(device).startswith("cuda"):
        for model in models:
            if estimate_ms(calibration, model, w, h, scales[0], profile.cuda_graphs) is None:
                log.info("no calibration for %s at %dx%d; benchmarking (a few seconds)", model, w, h)
                save_entries(settings.cache_root, run_bench(settings.model_path(model), model, w, h, device,
                                                            scales, graphs_options=(profile.cuda_graphs,),
                                                            warmup=3, iters=10))
        calibration = load_calibration(settings.cache_root)
    choice = choose_model_and_scale(calibration, models, w, h,
                                    interps_per_second(info.src_fps, out_fps_for_profile(info.src_fps, profile)),
                                    profile.cuda_graphs, scales)
    if choice is None:  # no calibration (e.g. resolved by the web server): best model, size heuristic
        choice = (models[0], scales[0] if len(scales) == 1 else (1.0 if w * h <= 1280 * 720 else 0.5))
    model, scale = choice
    source.setdefault("auto_choice", {})[profile.name] = {"model": model, "scale": scale}
    store.write_source(vid, source)
    return model, float(scale)


def resolve(settings: Settings, job: Job, info: VideoInfo | None = None, device: str | None = None):
    """-> (info, video_id, profile, scale, ProfileCache, crop) without starting generation.

    ``device`` allows a calibration benchmark for scale=auto (workers pass it; the web server doesn't).
    """
    info = info or probe(job.video_path, settings.ffprobe)
    vid = video_id_for(job.video_path)
    store = CacheStore(settings.cache_root)
    source = store.source(vid) or {}
    source.update(info.to_json())
    store.write_source(vid, source)
    profile = build_profile(settings, job)
    crop = _video_crop(store, vid, info, settings) if profile.letterbox_crop else None
    profile.model, scale = _auto_choice(store, vid, settings, profile, info, crop, device)
    profile.scale = scale
    cache = store.profile(vid, profile.profile_id(scale))
    return info, vid, profile, scale, cache, crop


def run_job(job: Job, settings: Settings | None = None, stop_event: threading.Event | None = None,
            on_progress: Callable[[dict], None] | None = None, engine_factory=None) -> RunResult:
    settings = settings or load_settings(job.config)
    stop_event = stop_event or threading.Event()
    info, vid, profile, scale, cache, crop = resolve(settings, job, device=job.device)
    if job.job_id is None:
        cache.clear_stop()  # a stale stop from a killed CLI run; the job manager clears its own before launch
    log.info("job %s: %s profile=%s scale=%s cache=%s", job.job_id or "-", job.video_path, profile.name,
             scale, cache.root)
    audio_thread = None
    audio_stop = threading.Event()
    if job.kind == "stream":
        # Audio encodes far faster than real time; build it alongside the video instead of delaying playback.
        audio_dir = CacheStore(settings.cache_root).audio_dir(vid)
        audio_thread = threading.Thread(target=_audio_worker, args=(info, audio_dir, audio_stop),
                                        name="sf-audio", daemon=True)
        audio_thread.start()
    result = RunResult("failed", error="not started")
    try:
        gen = Generator(info, profile, cache, scale=scale, model_dir=settings.model_path(profile.model),
                        device=job.device, hwaccel=job.hwaccel, engine_factory=engine_factory,
                        stop_event=stop_event, on_progress=on_progress, crop=crop)
        free_mb = _free_vram_mb(job.device)
        if free_mb is not None and free_mb < settings.min_free_vram_mb:
            msg = (f"only {free_mb} MB of GPU memory free (need {settings.min_free_vram_mb}); "
                   "close other GPU apps (e.g. LM Studio) and retry")
            gen._finish("failed", msg)
            result = RunResult("failed", error=msg)
            return result
        start = job.start_segment
        while True:
            result = gen.run(start)
            if result.status != "partial" or stop_event.is_set() or cache.stop_requested():
                break
            if job.kind == "stream":
                break  # the viewer's stretch is done; the manager queues an offline gap-filler (fill_gaps)
            start = 0  # a hole was filled; fill the next one (first_missing wraps around)
    finally:
        if audio_thread is not None:
            if result.status != "complete":
                audio_stop.set()  # never leave an orphaned ffmpeg behind
            audio_thread.join()
    if result.status == "complete" and job.export:
        export(cache, info, Path(job.export_path) if job.export_path else None, job.container,
               Path(settings.export_dir) if settings.export_dir else None, job.export_codec,
               settings.hevc_encoder)
    return result


def _audio_worker(info: VideoInfo, audio_dir: Path, stop: threading.Event) -> None:
    try:
        ensure_audio(info, audio_dir, stop)
    except Exception:  # noqa: BLE001 - video can still play without audio
        log.exception("audio rendition failed")


def _free_vram_mb(device: str) -> int | None:
    if not str(device).startswith("cuda"):
        return None
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (run `python -m streamerframes check`)")
    free, _ = torch.cuda.mem_get_info()
    return int(free / 2**20)


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
