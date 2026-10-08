"""One generation run: fill the first hole of segments in a profile cache (PLAN.md Phase 1)."""
from __future__ import annotations

import collections
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

from ..cache.store import ProfileCache
from ..config import Profile
from ..probe import VideoInfo
from ..timeline import (
    Timeline,
    out_fps_for_multiplier,
    out_fps_for_target,
    scene_cut_frame,
)
from . import ffmpeg
from .procs import FrameReader, FrameWriter, keep_awake

log = logging.getLogger(__name__)

EXIT_OK, EXIT_FAILED, EXIT_PAUSED = 0, 1, 3


@dataclass
class RunResult:
    status: str                 # complete | partial | paused | failed | noop
    run_id: int | None = None
    first_segment: int | None = None
    frames_written: int = 0
    error: str | None = None

    @property
    def exit_code(self) -> int:
        return {"failed": EXIT_FAILED, "paused": EXIT_PAUSED}.get(self.status, EXIT_OK)


def out_fps_for_profile(src_fps: Fraction, profile: Profile) -> Fraction:
    if profile.mode == "target":
        return out_fps_for_target(src_fps, int(profile.target_fps))
    return out_fps_for_multiplier(src_fps, Fraction(profile.multi))


def interps_per_second(src_fps: Fraction, out_fps: Fraction) -> float:
    """Interpolated (non-source) output frames needed per second of video."""
    return float(out_fps - src_fps)


def _rate_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def frame_tolerance(n: int) -> int:
    """How far the decoded frame count may fall short of the probe estimate before we call it a failure."""
    return max(3, n // 500)


def default_engine_factory(info: VideoInfo, profile: Profile, model_dir: Path, scale: float, device: str,
                           crop=None):
    def build():
        from ..engine.color import ColorSpec, CroppedConverter, YuvConverter
        from ..engine.loader import load_ifnet
        from ..engine.rife import RifeEngine

        conv = CroppedConverter(
            YuvConverter(info.width, info.height,
                         ColorSpec.from_names(info.color_space, info.color_range, info.height), device), crop)
        h, w = conv.size
        net = load_ifnet(model_dir, device)
        engine = RifeEngine(net, h, w, scale, device, scene_ssim=profile.scene_ssim, dup_mad=profile.dup_mad)
        if profile.cuda_graphs:
            engine.enable_cuda_graph()
        return engine, conv
    return build


class _Abort(Exception):
    pass


class Generator:
    def __init__(self, info: VideoInfo, profile: Profile, cache: ProfileCache, *, scale: float,
                 model_dir: Path | None = None, device: str = "cuda", hwaccel: bool = True,
                 engine_factory: Callable | None = None, stop_event: threading.Event | None = None,
                 progress_interval: float = 2.0, on_progress: Callable[[dict], None] | None = None,
                 crop=None, in_ring: int = 6, out_ring: int = 8):
        self.info = info
        self.profile = profile
        self.cache = cache
        self.scale = scale
        self.device = device
        self.hwaccel = hwaccel
        self.crop = crop
        self.in_ring, self.out_ring = in_ring, out_ring
        self.engine_factory = engine_factory or default_engine_factory(info, profile, model_dir, scale, device,
                                                                       crop)
        self.stop_event = stop_event or threading.Event()
        self.progress_interval = progress_interval
        self.on_progress = on_progress
        self._engine = None
        self._conv = None
        self.manifest = self._load_or_create_manifest()

    # Manifest
    def _base_timeline(self) -> Timeline:
        out_fps = out_fps_for_profile(self.info.src_fps, self.profile)
        return Timeline.create(self.info.src_fps, out_fps, self.info.n_src, Fraction(self.profile.seg_seconds))

    def _load_or_create_manifest(self) -> dict:
        tl = self._base_timeline()
        m = self.cache.manifest()
        if m:
            if m.get("seg_frames") != tl.seg_frames or m.get("out_fps") != _rate_str(tl.out_fps):
                raise RuntimeError(f"{self.cache.manifest_path} does not match this profile; delete the cache")
            return m
        self.cache.ensure_dirs()
        m = {
            "version": 1,
            "video_id": self.cache.video_id,
            "profile_id": self.cache.profile_id,
            "profile_name": self.profile.name,
            "profile": self.profile.pixel_settings(),
            "scale": self.scale,
            "crop": [self.crop.x, self.crop.y, self.crop.w, self.crop.h] if self.crop else None,
            "source": self.info.path,
            "width": self.info.width,
            "height": self.info.height,
            "src_fps": _rate_str(tl.src_fps),
            "out_fps": _rate_str(tl.out_fps),
            "n_src": tl.n_src,
            "n_src_actual": None,
            "seg_frames": tl.seg_frames,
            "total_segments": tl.total_segments,
            "total_out_frames": tl.total_out_frames,
            "last_segment_frames": tl.total_out_frames - (tl.total_segments - 1) * tl.seg_frames,
            "status": "new",
            "error": None,
            "error_tail": None,
            "created_at": time.time(),
        }
        self.cache.write_manifest(m)
        return m

    def timeline(self) -> Timeline:
        base = self._base_timeline()
        return base.with_n_src(self.manifest.get("n_src_actual") or self.manifest["n_src"])

    def _set_frame_count(self, n_actual: int) -> Timeline:
        tl = self.timeline().with_n_src(n_actual)
        self.manifest.update(
            n_src_actual=n_actual, total_out_frames=tl.total_out_frames, total_segments=tl.total_segments,
            last_segment_frames=tl.total_out_frames - (tl.total_segments - 1) * tl.seg_frames)
        self.cache.write_manifest(self.manifest)
        return tl

    def _finish(self, status: str, error: str | None = None, tail: list[str] | None = None):
        self.manifest.update(status=status, error=error, error_tail=tail)
        self.cache.write_manifest(self.manifest)

    def verify_complete(self, tl: Timeline) -> str | None:
        """None if every segment exists and durations add up; else the reason it isn't complete."""
        rows = self.cache.index_rows()
        missing = [k for k in range(tl.total_segments) if k not in rows]
        if missing:
            return f"{len(missing)} segments missing (first {missing[0]})"
        if self.manifest.get("n_src_actual") is None:
            return "final frame count unknown (end of stream never decoded)"
        # The muxer logs each segment's end on the output timeline (the first start of a run reads 0, so
        # starts are unreliable). Every segment must end exactly on its frame boundary.
        frame = 1 / float(tl.out_fps)
        for k in range(tl.total_segments):
            expected_end = float(tl.segment_range(k)[1] / tl.out_fps)
            if abs(rows[k][1] - expected_end) > 1.5 * frame:
                return f"segment {k} ends at {rows[k][1]:.3f}s, expected {expected_end:.3f}s"
        return None

    # Engine
    def _engine_pair(self):
        if self._engine is None:
            self._engine, self._conv = self.engine_factory()
        return self._engine, self._conv

    def _stop(self) -> bool:
        return self.stop_event.is_set() or self.cache.stop_requested()

    # Run
    def run(self, start_segment: int = 0) -> RunResult:
        tl = self.timeline()
        completed = self.cache.completed_segments()
        k0 = self.cache.first_missing(completed, start_segment, tl.total_segments)
        if k0 is None:
            reason = self.verify_complete(tl)
            self._finish("complete" if reason is None else "failed", reason)
            return RunResult("complete" if reason is None else "failed", error=reason)
        end = self.cache.gap_end(completed, k0, tl.total_segments)
        run_id = self.cache.next_run_id()
        self.cache.acquire_lock(run_id)
        # Don't clear a pending stop here: a seek can retarget the job while this worker is still starting.
        self.manifest.update(status="running", error=None, error_tail=None)
        self.cache.write_manifest(self.manifest)
        keep_awake(True)
        log.info("run %d: segments [%d, %d) of %d, source frame %d", run_id, k0, end, tl.total_segments,
                 tl.segment_start_source_frame(k0))
        try:
            return self._run(tl, k0, end, run_id)
        except Exception as exc:  # noqa: BLE001 - recorded in the manifest, then re-raised
            log.exception("run %d crashed", run_id)
            self._finish("failed", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            keep_awake(False)
            self.cache.clear_stop()
            self.cache.release_lock()

    def _run(self, tl: Timeline, k0: int, end: int, run_id: int) -> RunResult:
        """Three stages: reader thread -> GPU (this thread) -> writer thread, over recycled host buffers.

        Host buffers are pinned on CUDA so copies are async; the writer waits on a CUDA event per frame,
        so the GPU thread never calls synchronize().
        """
        import torch

        info, n = self.info, tl.seg_frames
        root = self.cache.root
        s0 = tl.segment_start_source_frame(k0)
        frame_size = ffmpeg_frame_size(info)
        cuda = str(self.device).startswith("cuda")
        decoder = FrameReader(ffmpeg.decoder_cmd(info, s0, self.hwaccel), root / "ffmpeg-decode.log", frame_size)
        enc = self.profile.encoder
        encoder = FrameWriter(
            ffmpeg.encoder_cmd(info, tl, k0, run_id, root,
                               encoder="lossless" if enc.codec == "lossless" else "nvenc",
                               preset=enc.preset, cq=enc.cq, maxrate=enc.maxrate),
            root / "ffmpeg-encode.log")

        def ring(count: int) -> queue.Queue:
            q: queue.Queue = queue.Queue()
            for _ in range(count):
                q.put(torch.empty(frame_size, dtype=torch.uint8, pin_memory=cuda))
            return q

        in_free, out_free = ring(self.in_ring), ring(self.out_ring)
        in_q: queue.Queue = queue.Queue()
        out_q: queue.Queue = queue.Queue()
        abort = threading.Event()
        writer_error: list[BaseException] = []

        def read_loop():
            try:
                while not abort.is_set():
                    try:
                        buf = in_free.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if not decoder.read_into(memoryview(buf.numpy())):
                        in_q.put(None)
                        return
                    in_q.put(buf)
            except BaseException as exc:  # noqa: BLE001 - handed to the GPU thread
                in_q.put(exc)

        def write_loop():
            while True:
                item = out_q.get()
                if item is None:
                    return
                buf, event = item
                if not writer_error and not discard.is_set():
                    try:
                        if event is not None:
                            event.synchronize()
                        encoder.write(memoryview(buf.numpy()))
                    except BaseException as exc:  # noqa: BLE001 - encoder died; keep draining
                        writer_error.append(exc)
                        abort.set()
                out_free.put(buf)

        reader = threading.Thread(target=read_loop, name="sf-reader", daemon=True)
        writer = threading.Thread(target=write_loop, name="sf-writer", daemon=True)
        reader.start()
        writer.start()

        host: dict[int, torch.Tensor] = {}   # source index -> host buffer (owned until dropped)
        dev: dict[int, torch.Tensor] = {}    # source index -> yuv on device
        rgb: dict[int, torch.Tensor] = {}    # source index -> padded engine input
        kinds: dict[int, object] = {}
        h2d_events: dict[int, object] = {}
        scratch = None
        next_idx = s0
        n_actual: int | None = None
        j = j_start = k0 * n
        j_end = end * n if end < tl.total_segments else None
        stopped = abandoned = False
        discard = threading.Event()
        cuts = 0
        started = last_report = time.monotonic()
        self._samples = collections.deque()  # (time, out frame) for a rolling speed estimate

        def ensure(idx: int) -> None:
            nonlocal next_idx, n_actual
            while next_idx <= idx and n_actual is None:
                item = in_q.get()
                if isinstance(item, BaseException):
                    raise item
                if item is None:
                    n_actual = next_idx
                    return
                host[next_idx] = item
                next_idx += 1

        def on_device(idx: int) -> torch.Tensor:
            if idx not in dev:
                if cuda:
                    dev[idx] = host[idx].to(self.device, non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record()
                    h2d_events[idx] = ev
                else:
                    dev[idx] = host[idx].clone()
            return dev[idx]

        def to_rgb(idx: int) -> torch.Tensor:
            if idx not in rgb:
                engine, conv = self._engine_pair()
                rgb[idx] = engine.pad(conv.to_rgb(on_device(idx)))
            return rgb[idx]

        def pair_kind(i: int):
            from ..engine.rife import PairKind
            if not (self.profile.scene_detect or self.profile.dup_mad > 0):
                return PairKind.NORMAL
            if i not in kinds:
                engine, _ = self._engine_pair()
                kinds[i] = engine.classify_pair(to_rgb(i), to_rgb(i + 1), check_cut=self.profile.scene_detect)
                if kinds[i] is PairKind.CUT:
                    log.info("scene cut after source frame %d (%.3fs)", i, float(i / tl.src_fps))
            return kinds[i]

        def drop_before(i: int) -> None:
            for old in [x for x in host if x < i]:
                ev = h2d_events.pop(old, None)
                if ev is not None:
                    ev.synchronize()  # long since done; the reader may now overwrite the buffer
                in_free.put(host.pop(old))
                dev.pop(old, None)
                rgb.pop(old, None)
                kinds.pop(old, None)

        def take_out_buffer() -> torch.Tensor:
            while True:
                if writer_error:
                    raise _Abort()
                try:
                    return out_free.get(timeout=0.2)
                except queue.Empty:
                    continue

        try:
            while j_end is None or j < j_end:
                if j > j_start and j % n == 0 and self._stop():
                    stopped = True
                    break
                if j % 8 == 0 and self.cache.stop_now_requested():
                    stopped = abandoned = True  # drop the partial segment; it never reaches the index
                    break
                p = j * tl.ratio
                i = math.floor(p)
                t = p - i
                ensure(i + 1 if t else i)
                if n_actual is not None:
                    if n_actual <= s0:
                        raise RuntimeError(f"decoder produced no frames from source frame {s0}")
                    if j >= tl.total_out_frames_for(n_actual):
                        break
                    if p >= n_actual - 1:
                        i, t = n_actual - 1, Fraction(0)
                source = i
                if t != 0:
                    from ..engine.rife import PairKind
                    kind = pair_kind(i)
                    if kind is PairKind.CUT:
                        source = scene_cut_frame(i, t)
                        cuts += 1
                    elif kind is not PairKind.DUPLICATE:
                        source = None
                out = take_out_buffer()
                if source is not None:
                    out.copy_(host[source])          # pass-through: the exact source bytes
                    out_q.put((out, None))
                else:
                    engine, conv = self._engine_pair()
                    mid = engine.interpolate(to_rgb(i), to_rgb(i + 1), float(t))
                    if scratch is None:
                        scratch = torch.empty(frame_size, dtype=torch.uint8, device=conv.device)
                    conv.from_rgb(mid, on_device(i), scratch)
                    if cuda:
                        out.copy_(scratch, non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record()
                        out_q.put((out, ev))
                    else:
                        out.copy_(scratch)
                        out_q.put((out, None))
                drop_before(i)
                j += 1
                now = time.monotonic()
                if now - last_report >= self.progress_interval:
                    last_report = now
                    self._report(run_id, tl, k0, j_start, j, started, now, cuts)
        except _Abort:
            pass  # encoder died; its exit code and log tail are reported below
        finally:
            if abandoned:
                discard.set()
                encoder.kill()  # killed mid-segment: ffmpeg never lists the partial file
            out_q.put(None)
            writer.join()
            abort.set()
            enc_rc = 0 if abandoned else encoder.close()
            dec_rc = decoder.close()
            reader.join(timeout=10)

        written = j - j_start
        self._report(run_id, tl, k0, j_start, j, started, time.monotonic(), cuts)

        if not abandoned and (enc_rc != 0 or writer_error):
            return self._fail(run_id, k0, written, f"encoder exited with code {enc_rc}", encoder.tail())
        if decoder.eof and dec_rc != 0:
            return self._fail(run_id, k0, written, f"decoder exited with code {dec_rc}", decoder.tail())
        if n_actual is not None:
            expected = self.manifest["n_src"]
            if n_actual < expected - frame_tolerance(expected):
                return self._fail(run_id, k0, written,
                                  f"decoder delivered {n_actual} of ~{expected} source frames", decoder.tail())
            if n_actual != expected:
                log.info("source has %d frames (probe estimated %d)", n_actual, expected)
            tl = self._set_frame_count(n_actual)

        if stopped:
            self._finish("paused")
            return RunResult("paused", run_id, k0, written)
        if self.cache.first_missing(self.cache.completed_segments(), 0, tl.total_segments) is None:
            reason = self.verify_complete(tl)
            if reason:
                return self._fail(run_id, k0, written, reason)
            self._finish("complete")
            return RunResult("complete", run_id, k0, written)
        self._finish("paused")
        return RunResult("partial", run_id, k0, written)

    def _fail(self, run_id, k0, written, error, tail=None) -> RunResult:
        log.error("run %d failed: %s", run_id, error)
        self._finish("failed", error, tail)
        return RunResult("failed", run_id, k0, written, error)

    def _report(self, run_id, tl, k0, j_start, j, started, now, cuts) -> None:
        # Speed over the last ~60 s of output. Averaging from the start of the run would count model
        # loading and cudnn autotuning (often 10-30 s) and show near-zero speed and a huge ETA at first.
        samples = getattr(self, "_samples", None)
        if samples is None:
            samples = self._samples = collections.deque()
        samples.append((now, j))
        while len(samples) > 2 and now - samples[0][0] > 60:
            samples.popleft()
        first = next(((t, f) for t, f in samples if f > j_start), None)
        if first and now - first[0] >= 1.0 and j > first[1]:
            rate = (j - first[1]) / (now - first[0])
        else:
            rate = 0.0
        done = j - j_start
        ratio = float(rate / tl.out_fps)
        remaining = max(tl.total_out_frames - j, 0)
        data = {
            "run": run_id,
            "start_segment": k0,
            "segment_frontier": j // tl.seg_frames,
            "out_frame": j,
            "out_frames_done": done,
            "total_out_frames": tl.total_out_frames,
            "src_fps_measured": round(float(rate * tl.ratio), 3),
            "realtime_ratio": round(ratio, 4),
            "eta_seconds": round(remaining / rate, 1) if rate > 0 else None,
            "scene_cuts": cuts,
            "started_at": time.time() - (now - started),
        }
        gpu = _gpu_mem_mb(self.device)
        if gpu is not None:
            data["gpu_mem_mb"] = gpu
        self.cache.write_progress(data)
        if self.on_progress:
            self.on_progress(data)


def ffmpeg_frame_size(info: VideoInfo) -> int:
    return ffmpeg.frame_bytes(info.width, info.height)


def _gpu_mem_mb(device: str) -> int | None:
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch
        return int(torch.cuda.memory_allocated() / 2**20)
    except Exception:  # noqa: BLE001
        return None
