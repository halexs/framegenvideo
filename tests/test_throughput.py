"""Phase 2: letterbox crop, calibration-driven scale=auto, bench, and pipeline failure handling."""
import json
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

from streamerframes.bench import (  # noqa: E402
    choose_scale,
    entry_key,
    estimate_ms,
    load_calibration,
)
from streamerframes.cache.store import CacheStore  # noqa: E402
from streamerframes.config import REPO_ROOT, Settings  # noqa: E402
from streamerframes.engine.color import (  # noqa: E402
    ColorSpec,
    Crop,
    CroppedConverter,
    YuvConverter,
)
from streamerframes.pipeline.letterbox import (  # noqa: E402
    detect_letterbox,
    parse_cropdetect,
    union_crop,
)
from streamerframes.pipeline.worker import Job, resolve, run_job  # noqa: E402
from streamerframes.probe import probe  # noqa: E402

from .fake_model import write_fake_model  # noqa: E402
from .test_render_e2e import FRAME, H, W, raw_frames, segment_frames  # noqa: E402


def test_parse_and_union_crop():
    assert parse_cropdetect("x\n[Parsed_cropdetect_0] crop=64:32:0:8\n... crop=64:30:0:10") == (64, 30, 0, 10)
    assert parse_cropdetect("crop=-64:-32:10:10") is None
    assert parse_cropdetect("nothing") is None
    # A dark scene detects a smaller area; the union keeps the real picture.
    assert union_crop([(1920, 800, 0, 140), (1200, 400, 300, 300)], 1920, 1080) == Crop(0, 140, 1920, 800)
    assert union_crop([(1920, 1060, 0, 10)], 1920, 1080) is None
    assert union_crop([(1920, 1040, 0, 20)], 1920, 1080) is not None  # < 32 px of bars: not worth it
    assert union_crop([(63, 31, 1, 9)], 64, 48) == Crop(0, 8, 64, 32)  # rounded out to even


def test_cropped_converter_keeps_bars():
    full = YuvConverter(8, 8, ColorSpec.from_names("bt709"), "cpu")
    conv = CroppedConverter(full, Crop(0, 2, 8, 4))
    assert conv.size == (4, 8)
    bg = torch.arange(full.nbytes, dtype=torch.int64).remainder(200).add(16).to(torch.uint8)
    out = torch.empty_like(bg)
    conv.from_rgb(torch.ones(1, 3, 4, 8), bg, out)
    y = out[:64].view(8, 8)
    assert (y[2:6] == 235).all()                         # picture area written
    assert torch.equal(y[:2], bg[:64].view(8, 8)[:2])    # bars untouched
    assert torch.equal(y[6:], bg[:64].view(8, 8)[6:])
    assert torch.equal(conv.to_rgb(out), torch.ones(1, 3, 4, 8))


def test_choose_scale_from_calibration():
    entries = {
        entry_key("gpu", "default", 1920, 800, 1.0, True): {"ms": 50.0},   # 20 interps/s
        entry_key("gpu", "default", 1920, 800, 0.5, True): {"ms": 30.0},   # 33 interps/s
    }
    assert estimate_ms(entries, "default", 1920, 800, 1.0, True) == 50.0
    assert estimate_ms(entries, "default", 960, 800, 1.0, True) == 25.0   # pixel-count scaling
    assert estimate_ms(entries, "default", 1920, 800, 1.0, False) is None
    # 23.976 -> 47.952 needs ~24 interps/s: 1.0 is too slow, 0.5 fits with 10% margin.
    assert choose_scale(entries, "default", 1920, 800, 23.976, True) == 0.5
    assert choose_scale(entries, "default", 1280, 536, 23.976, True) == 1.0
    assert choose_scale(entries, "default", 3840, 1600, 23.976, True) == 0.5  # nothing fits: fastest
    assert choose_scale({}, "default", 1920, 800, 24, True) is None


@pytest.fixture
def env(tmp_path):
    model = write_fake_model(tmp_path / "model")
    settings = Settings(cache_root=str(tmp_path / "cache"), model_dir=str(model))
    for p in settings.profiles.values():
        p.encoder.codec = "lossless"
        p.seg_seconds = 0.5
        p.scene_detect = False
    return tmp_path, settings


def letterboxed_clip(path, frames=24):
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={W}x32:rate=24000/1001",
                    "-vf", f"pad={W}:{H}:0:8:black", "-frames:v", str(frames), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@needs_ffmpeg
def test_letterbox_render_infers_only_the_picture(env):
    root, settings = env
    src = letterboxed_clip(root / "lb.mp4")
    assert detect_letterbox(probe(src), points=4) == Crop(0, 8, W, 32)
    job = Job(video_path=str(src), device="cpu", hwaccel=False, overrides={"scale": 1.0})
    assert run_job(job, settings).status == "complete"
    info, vid, profile, scale, cache, crop = resolve(settings, job)
    assert crop == Crop(0, 8, W, 32) and cache.manifest()["crop"] == [0, 8, W, 32]
    out = segment_frames(cache, cache.manifest()["total_segments"])
    src_frames = raw_frames(src)
    for j in range(1, 47, 2):  # interpolated frames: bars are the exact source bars
        o = torch.frombuffer(bytearray(out[j]), dtype=torch.uint8)
        s = torch.frombuffer(bytearray(src_frames[j // 2]), dtype=torch.uint8)
        assert torch.equal(o[: 8 * W], s[: 8 * W]) and torch.equal(o[40 * W: W * H], s[40 * W: W * H])

    off = Job(video_path=str(src), device="cpu", hwaccel=False, overrides={"scale": 1.0, "letterbox_crop": False})
    assert resolve(settings, off)[5] is None


@needs_ffmpeg
def test_auto_scale_is_sticky_per_video(env, tmp_path):
    root, settings = env
    src = letterboxed_clip(root / "s.mp4", frames=6)
    job = Job(video_path=str(src), device="cpu", hwaccel=False)  # realtime profile: scale=auto
    first = resolve(settings, job)
    assert first[3] == 1.0  # no calibration, small frame -> heuristic
    store = CacheStore(settings.cache_root)
    assert store.source(first[1])["auto_scale"] == {"realtime": 1.0}
    # New calibration says 1.0 is too slow, but the existing choice (and cache) is kept.
    (root / "cache" / "calibration.json").write_text(json.dumps({"entries": {
        entry_key("cpu", "default", W, 32, 1.0, True): {"ms": 1000.0},
        entry_key("cpu", "default", W, 32, 0.5, True): {"ms": 1.0}}}))
    assert resolve(settings, job)[3] == 1.0
    assert resolve(settings, job)[4].root == first[4].root


@needs_ffmpeg
def test_encoder_crash_fails_cleanly(env):
    root, settings = env
    src = letterboxed_clip(root / "e.mp4", frames=12)
    settings.profiles["realtime"].encoder.codec = "h264_nvenc"
    settings.profiles["realtime"].encoder.preset = "no-such-preset"
    job = Job(video_path=str(src), device="cpu", hwaccel=False, overrides={"scale": 1.0})
    result = run_job(job, settings)
    assert result.status == "failed" and "encoder exited" in result.error
    m = resolve(settings, job)[4].manifest()
    assert m["status"] == "failed" and m["error_tail"]


def test_bench_cli_writes_calibration(tmp_path):
    model = write_fake_model(tmp_path / "model")
    cfg = tmp_path / "sf.toml"
    cfg.write_text(f'cache_root = "{(tmp_path / "cache").as_posix()}"\nmodel_dir = "{model.as_posix()}"\n')
    r = subprocess.run([sys.executable, "-m", "streamerframes", "--config", str(cfg), "bench", "--size", "64x48",
                        "--device", "cpu", "--warmup", "1", "--iters", "2"], capture_output=True, text=True,
                       cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr
    entries = load_calibration(tmp_path / "cache")
    assert set(entries) == {entry_key("cpu", "default", 64, 48, s, False) for s in (1.0, 0.5)}
    assert "interpolations/s" in r.stdout


def test_frame_size_constant():
    assert FRAME == W * H * 3 // 2


def test_graph_matches_uses_eager_run_to_run_noise():
    from streamerframes.engine.rife import graph_matches
    base = torch.zeros(1, 3, 4, 4)
    assert graph_matches(base + 5e-4, base, base)[0]              # under the 1e-3 floor
    assert not graph_matches(base + 0.06, base, base)[0]          # the old failure mode, with stable eager
    noisy = base.clone()
    noisy[0, 0, 0, 0] = 0.05                                      # eager itself varies by 0.05...
    assert graph_matches(base + 0.08, base, noisy)[0]             # ...so 0.08 is within 2x that


def test_smooth_pair_is_image_like(env):
    root, settings = env
    from streamerframes.engine.loader import load_ifnet
    from streamerframes.engine.rife import RifeEngine
    eng = RifeEngine(load_ifnet(settings.model_dir, "cpu"), 100, 200, 1.0, "cpu")
    a, b = eng._smooth_pair()
    assert a.shape == (1, 3, 128, 256) and float(a.min()) >= 0 and float(a.max()) <= 1
    assert float((a[..., 1:, :] - a[..., :-1, :]).abs().mean()) < 0.05  # smooth, unlike noise (~0.33)
    assert torch.equal(torch.roll(a, (3, 5), (2, 3)), b)


def test_progress_speed_ignores_startup(env):
    root, settings = env
    from fractions import Fraction as Fr

    from streamerframes.cache.store import ProfileCache
    from streamerframes.pipeline.generator import Generator
    from streamerframes.timeline import Timeline

    class Info:  # just what Generator needs to build its manifest
        src_fps, n_src, width, height, path = Fr(24000, 1001), 10000, 64, 48, "x"

    gen = Generator(Info, settings.profile("realtime"), ProfileCache(root / "c" / "v" / "p"), scale=1.0)
    tl = Timeline.create(Fr(24000, 1001), Fr(48000, 1001), 10000)
    seen = []
    gen.on_progress = seen.append
    gen._samples = __import__("collections").deque()
    gen._report(0, tl, 0, 0, 0, started=0.0, now=20.0, cuts=0)    # 20 s of model load, no frames yet
    gen._report(0, tl, 0, 0, 2, started=0.0, now=22.0, cuts=0)    # first frames appear
    gen._report(0, tl, 0, 0, 98, started=0.0, now=24.0, cuts=0)   # 48 out fps since then
    assert seen[0]["eta_seconds"] is None and seen[0]["realtime_ratio"] == 0
    assert abs(seen[-1]["src_fps_measured"] - 24.0) < 0.1          # not dragged down by the 20 s start-up
    assert abs(seen[-1]["realtime_ratio"] - 1.0) < 0.01
