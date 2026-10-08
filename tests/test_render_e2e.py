"""End-to-end Phase 1 acceptance on CPU: real ffmpeg, lossless encoder, fake IFNet (linear blend)."""
import hashlib
import json
import math
import shutil
import subprocess
import threading
from fractions import Fraction as F

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

from streamerframes.cache.store import CacheStore  # noqa: E402
from streamerframes.config import Settings  # noqa: E402
from streamerframes.pipeline.ffmpeg import frame_bytes  # noqa: E402
from streamerframes.pipeline.worker import Job, resolve, run_job  # noqa: E402

from .fake_model import write_fake_model  # noqa: E402

W, H = 64, 48
FRAME = frame_bytes(W, H)


def make_clip(path, frames=60, rate="24000/1001", audio=True, cut_at=None):
    if cut_at is None:
        cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={rate}"]
    else:  # hard scene cut: testsrc2, then a different pattern
        cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
               f"testsrc2=size={W}x{H}:rate={rate}:d={cut_at * F(rate).denominator / F(rate).numerator}[a];"
               f"mandelbrot=size={W}x{H}:rate={rate}[b];[a][b]concat=n=2:v=1:a=0"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000", "-shortest", "-c:a", "aac"]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-g", "12", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


def raw_frames(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo",
                          "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
    assert len(out) % FRAME == 0
    return [out[i:i + FRAME] for i in range(0, len(out), FRAME)]


def segment_frames(cache, total):
    frames = []
    for k in range(total):
        frames += raw_frames(cache.segment_path(k))
    return frames


@pytest.fixture
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    model = write_fake_model(root / "model")
    settings = Settings(cache_root=str(root / "cache"), model_dir=str(model))
    for p in settings.profiles.values():
        p.encoder.codec = "lossless"
        p.seg_seconds = 0.5
        p.scene_detect = False
    return root, settings


def job_for(path, **kw):
    kw.setdefault("profile", "realtime")
    kw.setdefault("overrides", {"scale": 1.0})
    return Job(video_path=str(path), device="cpu", hwaccel=False, **kw)


def test_2x_render_is_exact(env):
    root, settings = env
    src = make_clip(root / "a.mp4")
    job = job_for(src, export=True)
    result = run_job(job, settings)
    assert result.status == "complete"
    info, vid, profile, scale, cache, crop = resolve(settings, job)
    m = cache.manifest()
    assert m["status"] == "complete" and m["out_fps"] == "48000/1001"
    assert m["n_src_actual"] == 60 and m["total_out_frames"] == 120
    assert m["seg_frames"] == 24 and m["total_segments"] == 5

    src_frames = raw_frames(src)
    out = segment_frames(cache, m["total_segments"])
    assert len(out) == 120
    # Integer source positions pass through bit-exact.
    for j in range(0, 120, 2):
        assert out[j] == src_frames[min(j // 2, 59)], j
    # Midpoints are the blend of the neighbours (fake model), up to YUV->RGB->YUV loss: ~30% of this tiny,
    # saturated test pattern clips at the RGB gamut edge, and 4:2:0 chroma gets resampled.
    for j in range(1, 117, 2):
        a, b = (torch.frombuffer(bytearray(src_frames[j // 2 + d]), dtype=torch.uint8).float() for d in (0, 1))
        mid = torch.frombuffer(bytearray(out[j]), dtype=torch.uint8).float()
        err = (mid - (a + b) / 2).abs()
        assert err[: W * H].mean() < 1.5, j
        assert err[W * H:].mean() < 6, j

    export = m["exports"][0]
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-count_frames", "-of", "json",
                                       export], capture_output=True, text=True, check=True).stdout)
    v = [s for s in probe["streams"] if s["codec_type"] == "video"][0]
    assert int(v["nb_read_frames"]) == 120 and v["r_frame_rate"] == "48000/1001"
    assert any(s["codec_type"] == "audio" for s in probe["streams"])
    assert raw_frames(export) == out


def test_target_60_from_23976(env):
    root, settings = env
    src = make_clip(root / "b.mp4", frames=48, audio=False)
    job = job_for(src, profile="quality")
    assert run_job(job, settings).status == "complete"
    cache = resolve(settings, job)[4]
    m = cache.manifest()
    assert m["out_fps"] == "60000/1001"
    assert m["total_out_frames"] == math.ceil(48 * F(5, 2)) == 120
    assert len(segment_frames(cache, m["total_segments"])) == 120


def test_resume_after_stop_matches_uninterrupted(env):
    root, settings = env
    src = make_clip(root / "c.mp4", frames=72, audio=False)

    reference = job_for(src, overrides={"scale": 1.0, "seg_seconds": 1.0})
    assert run_job(reference, settings).status == "complete"
    ref_cache = resolve(settings, reference)[4]
    ref = segment_frames(ref_cache, ref_cache.manifest()["total_segments"])

    job = job_for(src)  # different seg_seconds -> a different profile cache
    stop = threading.Event()
    seen = []

    def on_progress(p):
        seen.append(p)
        if p["out_frame"] >= 30:
            stop.set()

    cache = resolve(settings, job)[4]
    from streamerframes.pipeline import generator as gen_mod
    orig = gen_mod.Generator.__init__

    def fast_progress(self, *a, **kw):
        kw["progress_interval"] = 0
        orig(self, *a, **kw)

    gen_mod.Generator.__init__ = fast_progress
    try:
        first = run_job(job, settings, stop_event=stop, on_progress=on_progress)
    finally:
        gen_mod.Generator.__init__ = orig
    assert first.status == "paused"
    assert cache.manifest()["status"] == "paused"
    done = cache.completed_segments()
    assert done == {0, 1}  # stopped at the first segment boundary after frame 30 (N=24)

    # A killed run leaves a partial segment file with no index line; it must be redone, not trusted.
    cache.segment_path(2).write_bytes(b"garbage")
    second = run_job(job, settings)
    assert second.status == "complete" and second.first_segment == 2
    m = cache.manifest()
    assert segment_frames(cache, m["total_segments"]) == ref
    assert not cache.lock_path.exists() and not cache.stop_path.exists()


def test_short_decode_is_failed_not_complete(env, monkeypatch):
    root, settings = env
    src = make_clip(root / "d.mp4", frames=40, audio=False)
    import streamerframes.pipeline.worker as worker
    real_probe = worker.probe

    def lying_probe(path, ffprobe="ffprobe"):
        info = real_probe(path, ffprobe)
        info.n_src = 400  # like bug 4: the decoder stops long before the expected end
        return info

    monkeypatch.setattr(worker, "probe", lying_probe)
    job = job_for(src)
    result = run_job(job, settings)
    assert result.status == "failed"
    assert "delivered 40 of ~400" in result.error
    m = resolve(settings, job, lying_probe(src))[4].manifest()
    assert m["status"] == "failed"


def test_scene_cut_never_blends(env):
    root, settings = env
    settings.profiles["realtime"].scene_detect = True
    src = make_clip(root / "e.mp4", frames=24, audio=False, cut_at=12)
    job = job_for(src)
    assert run_job(job, settings).status == "complete"
    cache = resolve(settings, job)[4]
    out = segment_frames(cache, cache.manifest()["total_segments"])
    src_frames = raw_frames(src)
    assert out[23] in (src_frames[11], src_frames[12])  # t=0.5 across the cut -> a real frame


def test_cache_listing_and_corrupt_input(env, capsys):
    root, settings = env
    bad = root / "bad.mp4"
    bad.write_bytes(hashlib.sha256(b"x").digest() * 100)
    with pytest.raises((RuntimeError, ValueError), match="ffprobe failed|no video stream"):
        run_job(job_for(bad), settings)
    assert CacheStore(settings.cache_root).video_ids() == []


def test_stream_job_builds_audio_rendition_once(env):
    root, settings = env
    src = make_clip(root / "f.mp4", frames=48)
    job = job_for(src, kind="stream")
    assert run_job(job, settings).status == "complete"
    vid = resolve(settings, job)[1]
    audio = CacheStore(settings.cache_root).audio_dir(vid)
    assert (audio / "DONE").read_text() == "done"
    assert list(audio.glob("a_*.ts")) and (audio / "audio.csv").exists()
    silent = make_clip(root / "g.mp4", frames=24, audio=False)
    job = job_for(silent, kind="stream")
    assert run_job(job, settings).status == "complete"
    assert (CacheStore(settings.cache_root).audio_dir(resolve(settings, job)[1]) / "DONE").read_text() == "none"


def test_stop_now_never_indexes_the_partial_segment(env):
    root, settings = env
    src = make_clip(root / "h.mp4", frames=72, audio=False)
    reference = job_for(src, overrides={"scale": 1.0, "seg_seconds": 1.0})
    assert run_job(reference, settings).status == "complete"
    ref_cache = resolve(settings, reference)[4]
    ref = segment_frames(ref_cache, ref_cache.manifest()["total_segments"])

    job = job_for(src)
    cache = resolve(settings, job)[4]
    from streamerframes.pipeline import generator as gen_mod
    orig = gen_mod.Generator.__init__

    def fast_progress(self, *a, **kw):
        kw["progress_interval"] = 0
        orig(self, *a, **kw)

    gen_mod.Generator.__init__ = fast_progress
    try:
        # Abandon mid-segment (N=24): frame 40 is inside segment 1.
        result = run_job(job, settings, on_progress=lambda p: p["out_frame"] >= 40 and cache.request_stop(now=True))
    finally:
        gen_mod.Generator.__init__ = orig
    assert result.status == "paused"
    assert cache.completed_segments() == {0}  # segment 1 was cut off and must not count
    assert run_job(job, settings).status == "complete"
    assert segment_frames(cache, cache.manifest()["total_segments"]) == ref
