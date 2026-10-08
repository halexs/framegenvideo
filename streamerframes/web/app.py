"""FastAPI app: movie library, jobs API, HLS playback (PLAN.md Phase 3). Clients only ever send opaque IDs."""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..cache import playlist
from ..cache.store import CacheStore, LockHeld, ProfileCache, video_id_for
from ..config import Settings, load_settings
from ..jobs.manager import JobManager, job_view, segment_ranges
from ..pipeline.audio import audio_state
from ..probe import probe

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi", ".ts", ".m2ts"}
ID_RE = re.compile(r"^[0-9a-f]{8,32}$")
M3U8 = "application/vnd.apple.mpegurl"
NO_CACHE = {"Cache-Control": "no-cache"}
LONG_CACHE = {"Cache-Control": "public, max-age=31536000, immutable"}


def _check_id(value: str) -> str:
    if not ID_RE.match(value):
        raise HTTPException(404, "not found")
    return value


class Library:
    """Maps opaque video IDs to files under the configured movie roots (stat only; cheap to refresh)."""

    def __init__(self, roots: list[str], ttl: float = 30.0):
        self.roots = [Path(r) for r in roots]
        self.ttl = ttl
        self._scanned = 0.0
        self._videos: dict[str, tuple[Path, str]] = {}
        self._lock = threading.Lock()

    def scan(self, force: bool = False) -> dict[str, tuple[Path, str]]:
        with self._lock:
            if force or time.monotonic() - self._scanned > self.ttl:
                videos = {}
                for root in self.roots:
                    if not root.is_dir():
                        log.warning("movies root %s not found", root)
                        continue
                    for path in sorted(root.rglob("*")):
                        if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file():
                            try:
                                vid = video_id_for(path)
                            except OSError:
                                continue
                            title = str(path.relative_to(root)).replace("\\", "/")
                            videos[vid] = (path, title if len(self.roots) == 1 else f"{root.name}/{title}")
                self._videos = videos
                self._scanned = time.monotonic()
            return self._videos

    def path(self, vid: str) -> Path:
        entry = self.scan().get(vid) or self.scan(force=True).get(vid)
        if entry is None:
            raise HTTPException(404, "video not found")
        return entry[0]


def setup_logging(cache_root: Path, level=logging.INFO) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(cache_root / "server.log", maxBytes=5 * 2**20, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # segment fetches would flood the log


LONG_POLL_SECONDS = 25.0
NEAR_FRONTIER = 3  # segments ahead of a run's frontier that are worth waiting for instead of retargeting


def create_app(settings: Settings | None = None, manager: JobManager | None = None,
               config_path: str | None = None, start_manager: bool = True,
               long_poll_seconds: float = LONG_POLL_SECONDS) -> FastAPI:
    settings = settings or load_settings(config_path)
    store = CacheStore(settings.cache_root)
    library = Library(settings.movies_roots)
    manager = manager or JobManager(settings, config_path=config_path)
    exports_running: set[str] = set()

    @asynccontextmanager
    async def lifespan(_app):
        if start_manager:
            manager.start()
        try:
            yield
        finally:
            manager.shutdown()

    app = FastAPI(title="StreamerFrames", lifespan=lifespan)
    app.state.settings, app.state.manager, app.state.library, app.state.store = settings, manager, library, store

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        response = await call_next(request)
        level = logging.INFO if request.url.path.startswith("/api") else logging.DEBUG
        log.log(level, "%s %s -> %s", request.method, request.url.path, response.status_code)
        return response

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # Pages
    def page(name: str) -> FileResponse:
        return FileResponse(STATIC / name, headers=NO_CACHE)

    @app.get("/")
    def index():
        return page("index.html")

    @app.get("/watch/{video_id}")
    def watch(video_id: str):
        _check_id(video_id)
        return page("watch.html")

    @app.get("/compare/{video_id}")
    def compare(video_id: str):
        _check_id(video_id)
        return page("compare.html")

    # Helpers
    def cache_summary(pc: ProfileCache) -> dict:
        m = pc.manifest() or {}
        total = m.get("total_segments") or 0
        done = len(pc.completed_segments()) if total else 0
        exports = [e for e in m.get("exports", []) if Path(e).exists()]
        return {"profile_id": pc.profile_id, "profile_name": m.get("profile_name"), "status": m.get("status"),
                "percent": round(100.0 * done / total, 1) if total else 0.0, "out_fps": m.get("out_fps"),
                "scale": m.get("scale"), "error": m.get("error"), "has_export": bool(exports),
                "export_state": m.get("export_state"), "export_error": m.get("export_error")}

    def video_entry(vid: str, path: Path, title: str) -> dict:
        src = store.source(vid) or {}
        job = manager.active_for(vid)
        return {"video_id": vid, "title": title, "duration": src.get("duration"), "width": src.get("width"),
                "height": src.get("height"), "fps": src.get("src_fps"), "container": path.suffix.lower()[1:],
                "caches": [cache_summary(pc) for pc in store.profiles(vid)],
                "job": job_view(job) if job else None}

    def profile_cache(vid: str, pid: str) -> ProfileCache:
        pc = store.profile(_check_id(vid), _check_id(pid))
        if not pc.manifest_path.exists():
            raise HTTPException(404, "no such cache")
        return pc

    # API
    @app.get("/api/library")
    def api_library(refresh: bool = False):
        videos = library.scan(force=refresh)
        return {"videos": [video_entry(vid, p, t) for vid, (p, t) in sorted(videos.items(), key=lambda e: e[1][1])],
                "profiles": sorted(settings.profiles), "running": job_view(manager.running())
                if manager.running() else None}

    @app.get("/api/videos/{video_id}")
    def api_video(video_id: str):
        path = library.path(_check_id(video_id))
        src = store.source(video_id)
        if not src or "src_fps" not in src:
            info = probe(path, settings.ffprobe)
            src = {**(src or {}), **info.to_json()}
            store.write_source(video_id, src)
        entry = video_entry(video_id, path, library.scan()[video_id][1])
        entry["source"] = {k: v for k, v in src.items() if k != "path"}
        entry["jobs"] = [job_view(j) for j in manager.list() if j.video_id == video_id][:20]
        return entry

    @app.post("/api/jobs")
    def api_submit(body: dict = Body(...)):
        vid = _check_id(str(body.get("video_id", "")))
        kind = body.get("kind", "stream")
        profile = body.get("profile") or ("realtime" if kind == "stream" else "quality")
        if profile not in settings.profiles:
            raise HTTPException(400, f"unknown profile {profile!r}")
        path = library.path(vid)
        vid2, pid, cache_dir = manager.resolver(str(path), profile, kind)
        pc = ProfileCache(Path(cache_dir))
        pc.update_manifest(last_access=time.time()) if pc.manifest_path.exists() else None
        if (pc.manifest() or {}).get("status") == "complete":
            return {"state": "complete", "video_id": vid2, "profile_id": pid, "profile": profile, "kind": kind,
                    "id": None, **{k: v for k, v in cache_summary(pc).items() if k in ("percent",)}}
        try:
            rec = manager.submit(str(path), kind=kind, profile=profile,
                                 start_segment=int(body.get("start_segment", 0)))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return job_view(rec)

    @app.get("/api/status/{video_id}/{profile_id}")
    def api_status(video_id: str, profile_id: str):
        """What the watch page polls: cache state, completed ranges (for seeking) and the active job."""
        pc = profile_cache(video_id, profile_id)
        m = pc.manifest()
        from fractions import Fraction
        seg_secs = float(Fraction(m["seg_frames"]) / Fraction(m["out_fps"]))
        job = manager.active_for(video_id, profile_id)
        return {**cache_summary(pc), "ranges": segment_ranges(pc.completed_segments()), "seg_seconds": seg_secs,
                "segments_total": m["total_segments"],
                "duration_seconds": m["total_out_frames"] / float(Fraction(m["out_fps"])),
                "job": job_view(job) if job else None}

    @app.get("/api/jobs")
    def api_jobs():
        return {"jobs": [job_view(j) for j in manager.list()[:100]]}

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        rec = manager.get(_check_id(job_id))
        if rec is None:
            raise HTTPException(404, "no such job")
        return job_view(rec)

    @app.delete("/api/jobs/{job_id}")
    def api_cancel(job_id: str):
        if manager.get(_check_id(job_id)) is None:
            raise HTTPException(404, "no such job")
        return job_view(manager.cancel(job_id))

    @app.post("/api/jobs/{job_id}/resume")
    def api_resume(job_id: str):
        if manager.get(_check_id(job_id)) is None:
            raise HTTPException(404, "no such job")
        return job_view(manager.resume(job_id))

    @app.post("/api/videos/{video_id}/export")
    def api_export(video_id: str, body: dict = Body(...)):
        pc = profile_cache(video_id, str(body.get("profile_id", "")))
        m = pc.manifest()
        if m.get("status") != "complete":
            raise HTTPException(409, "generation is not complete yet")
        key = str(pc.root)
        if key in exports_running:
            return cache_summary(pc)
        path = library.path(video_id)

        def run():
            from ..pipeline.export import export
            try:
                info = probe(path, settings.ffprobe)
                export(pc, info, export_dir=Path(settings.export_dir) if settings.export_dir else None)
                pc.update_manifest(export_state="done", export_error=None)
            except Exception as exc:  # noqa: BLE001 - shown in the UI
                log.exception("export failed")
                pc.update_manifest(export_state="failed", export_error=str(exc))
            finally:
                exports_running.discard(key)

        exports_running.add(key)
        pc.update_manifest(export_state="running", export_error=None)
        threading.Thread(target=run, name="sf-export", daemon=True).start()
        return cache_summary(pc)

    @app.delete("/api/cache/{video_id}")
    @app.delete("/api/cache/{video_id}/{profile_id}")
    def api_delete_cache(video_id: str, profile_id: str | None = None):
        _check_id(video_id)
        if profile_id is not None:
            _check_id(profile_id)
        if manager.active_for(video_id, profile_id):
            raise HTTPException(409, "a job is using this cache; cancel it first")
        try:
            store.remove(video_id, profile_id)
        except LockHeld as exc:
            raise HTTPException(409, str(exc)) from None
        return {"ok": True}

    # Media
    @app.get("/media/original/{video_id}")
    def media_original(video_id: str):
        path = library.path(_check_id(video_id))
        media = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
                 ".mkv": "video/x-matroska"}.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media)

    @app.get("/hls/{video_id}/{profile_id}/master.m3u8")
    def hls_master(video_id: str, profile_id: str):
        pc = profile_cache(video_id, profile_id)
        has_audio = bool((store.source(video_id) or {}).get("audio")) and \
            audio_state(store.audio_dir(video_id)) != "none"
        return Response(playlist.master_playlist(pc.manifest(), has_audio), media_type=M3U8, headers=NO_CACHE)

    @app.get("/hls/{video_id}/{profile_id}/video.m3u8")
    def hls_video(video_id: str, profile_id: str):
        pc = profile_cache(video_id, profile_id)
        # Phase 4: the whole timeline up front (segment boundaries are deterministic), so players can seek
        # anywhere; segments that aren't generated yet are long-polled below.
        return Response(playlist.video_playlist(pc, full=True), media_type=M3U8, headers=NO_CACHE)

    @app.get("/hls/{video_id}/audio.m3u8")
    def hls_audio(video_id: str):
        audio_dir = store.audio_dir(_check_id(video_id))
        if not audio_dir.exists():
            raise HTTPException(404, "no audio rendition")
        return Response(playlist.audio_playlist(audio_dir), media_type=M3U8, headers=NO_CACHE)

    def ensure_generating(video_id: str, pc: ProfileCache, n: int) -> None:
        """Segment n was requested but isn't ready: wait for the current run, or point generation at it."""
        rec = manager.active_for(video_id, pc.profile_id)
        if rec is not None and rec.state == "running":
            progress = pc.progress() or {}
            current = progress.get("start_segment") == rec.start_segment
            frontier = progress.get("segment_frontier", rec.start_segment) if current else rec.start_segment
            if rec.start_segment <= n <= max(frontier, rec.start_segment) + NEAR_FRONTIER:
                return
        if rec is not None:
            manager.retarget(rec, n)
            return
        m = pc.manifest() or {}
        profile = str(m.get("profile_name") or "realtime").split("+")[0]
        if profile not in settings.profiles:
            return
        manager.stream_from(str(library.path(video_id)), profile, n)

    @app.get("/hls/{video_id}/{profile_id}/seg/{n}.ts")
    async def hls_segment(video_id: str, profile_id: str, n: int):
        pc = profile_cache(video_id, profile_id)
        total = (pc.manifest() or {}).get("total_segments", 0)
        if not 0 <= n < total:
            raise HTTPException(404, "no such segment")

        def ready() -> bool:
            return n in pc.completed_segments()

        if not await run_in_threadpool(ready):
            await run_in_threadpool(ensure_generating, video_id, pc, n)
            deadline = time.monotonic() + long_poll_seconds
            while not await run_in_threadpool(ready):
                if time.monotonic() >= deadline:
                    return Response("generating", status_code=503, headers={"Retry-After": "2", **NO_CACHE})
                await asyncio.sleep(0.25)
        return FileResponse(pc.segment_path(n), media_type="video/mp2t", headers=LONG_CACHE)

    @app.get("/hls/{video_id}/audio/{n}.ts")
    def hls_audio_segment(video_id: str, n: int):
        path = playlist.audio_segment_path(store.audio_dir(_check_id(video_id)), n)
        if path is None or not path.exists():
            raise HTTPException(404, "no such audio segment")
        return FileResponse(path, media_type="video/mp2t", headers=LONG_CACHE)

    @app.get("/exports/{video_id}/{profile_id}")
    def download_export(video_id: str, profile_id: str):
        pc = profile_cache(video_id, profile_id)
        exports = [Path(e) for e in (pc.manifest() or {}).get("exports", []) if Path(e).exists()]
        if not exports:
            raise HTTPException(404, "not exported")
        return FileResponse(exports[-1], filename=exports[-1].name)

    @app.exception_handler(FileNotFoundError)
    async def _not_found(request: Request, exc: FileNotFoundError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    return app
