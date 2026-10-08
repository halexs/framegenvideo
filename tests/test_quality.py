"""Phase 5: model=auto with lite models, duplicate-frame skipping, scene cuts, HEVC export, watch folders."""
import json
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

from streamerframes.bench import choose_model_and_scale, entry_key  # noqa: E402
from streamerframes.cache.store import CacheStore  # noqa: E402
from streamerframes.config import REPO_ROOT, Settings  # noqa: E402
from streamerframes.jobs.watcher import FolderWatcher  # noqa: E402
from streamerframes.pipeline.worker import Job, resolve, run_job  # noqa: E402

from .fake_model import write_fake_model  # noqa: E402
from .test_render_e2e import H, W, raw_frames, segment_frames  # noqa: E402


@pytest.fixture
def env(tmp_path):
    settings = Settings(cache_root=str(tmp_path / "cache"), model_dir=str(write_fake_model(tmp_path / "model")),
                        models_dir=str(tmp_path / "models"))
    for p in settings.profiles.values():
        p.encoder.codec = "lossless"
        p.seg_seconds = 0.5
        p.scene_detect = False
        p.letterbox_crop = False
    return tmp_path, settings


def test_available_models_order(env):
    root, settings = env
    for name in ("4.25.lite", "4.22", "4.22.lite"):
        write_fake_model(root / "models" / name)
    (root / "models" / "broken").mkdir()
    assert settings.available_models() == ["default", "4.22", "4.22.lite", "4.25.lite"]
    settings.model_preference = ["4.25.lite"]
    assert settings.available_models()[0] == "4.25.lite"


def test_choose_model_and_scale():
    e = {entry_key("g", "default", 1920, 800, 1.0, True): {"ms": 60.0},   # 16.7/s
         entry_key("g", "default", 1920, 800, 0.5, True): {"ms": 45.0},   # 22/s
         entry_key("g", "lite", 1920, 800, 1.0, True): {"ms": 30.0},      # 33/s
         entry_key("g", "lite", 1920, 800, 0.5, True): {"ms": 20.0}}
    models = ["default", "lite"]
    assert choose_model_and_scale(e, models, 1920, 800, 15.0, True) == ("default", 1.0)
    assert choose_model_and_scale(e, models, 1920, 800, 19.0, True) == ("default", 0.5)
    assert choose_model_and_scale(e, models, 1920, 800, 23.976, True) == ("lite", 1.0)
    assert choose_model_and_scale(e, models, 1920, 800, 100.0, True) == ("lite", 0.5)  # fastest
    assert choose_model_and_scale(e, ["default"], 1920, 800, 23.976, True, scales=(1.0,)) == ("default", 1.0)


def clip(path, vf=None, frames=24, rate="24000/1001", src="testsrc2"):
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"{src}=size={W}x{H}:rate={rate}"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


@needs_ffmpeg
def test_model_auto_picks_lite_when_full_is_too_slow(env):
    root, settings = env
    write_fake_model(root / "models" / "4.25.lite")
    src = clip(root / "a.mp4")
    (root / "cache").mkdir()
    (root / "cache" / "calibration.json").write_text(json.dumps({"entries": {
        entry_key("g", "default", W, H, 1.0, True): {"ms": 100.0},
        entry_key("g", "default", W, H, 0.5, True): {"ms": 80.0},
        entry_key("g", "4.25.lite", W, H, 1.0, True): {"ms": 10.0}}}))
    job = Job(video_path=str(src), device="cpu", hwaccel=False, overrides={"model": "auto", "scale": "auto"})
    assert run_job(job, settings).status == "complete"
    info, vid, profile, scale, cache, crop = resolve(settings, job)
    assert (profile.model, scale) == ("4.25.lite", 1.0)
    assert cache.manifest()["profile"]["model"] == "4.25.lite"
    assert CacheStore(settings.cache_root).source(vid)["auto_choice"]["realtime+custom"] == {
        "model": "4.25.lite", "scale": 1.0}


@needs_ffmpeg
def test_duplicate_frames_are_copied_not_interpolated(env):
    root, settings = env
    settings.profiles["realtime"].dup_mad = 0.002
    # 12 fps content shown at 23.976: every frame is doubled, like anime on twos.
    src = clip(root / "twos.mp4", vf="fps=24000/1001", rate="12000/1001", frames=24)
    job = Job(video_path=str(src), device="cpu", hwaccel=False, overrides={"scale": 1.0})
    assert run_job(job, settings).status == "complete"
    cache = resolve(settings, job)[4]
    out = segment_frames(cache, cache.manifest()["total_segments"])
    s = raw_frames(src)
    dup_pairs = [i for i in range(len(s) - 1) if s[i] == s[i + 1]]
    assert len(dup_pairs) >= 10
    for i in dup_pairs:
        assert out[2 * i + 1] == s[i]  # bit-exact copy: inference (and YUV round trip) skipped


@needs_ffmpeg
def test_no_blends_across_several_cuts(env):
    root, settings = env
    settings.profiles["realtime"].scene_detect = True
    parts = [("testsrc2", 8), ("mandelbrot", 8), ("smptebars", 8), ("rgbtestsrc", 8)]
    inputs, chains = [], []
    for k, (name, n) in enumerate(parts):
        inputs += ["-f", "lavfi", "-i", f"{name}=size={W}x{H}:rate=24000/1001"]
        chains.append(f"[{k}:v]trim=end_frame={n},setpts=PTS-STARTPTS[v{k}]")
    graph = ";".join(chains) + ";" + "".join(f"[v{k}]" for k in range(len(parts))) + \
        f"concat=n={len(parts)}:v=1:a=0[out]"
    src = root / "cuts.mp4"
    subprocess.run(["ffmpeg", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[out]",
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(src)], check=True)
    job = Job(video_path=str(src), device="cpu", hwaccel=False, overrides={"scale": 1.0})
    assert run_job(job, settings).status == "complete"
    cache = resolve(settings, job)[4]
    out = segment_frames(cache, cache.manifest()["total_segments"])
    s = raw_frames(src)
    for cut in (7, 15, 23):  # last frame of each part
        assert out[2 * cut + 1] in (s[cut], s[cut + 1])


@needs_ffmpeg
def test_hevc_export(env):
    root, settings = env
    settings.hevc_encoder = "libx265"
    src = clip(root / "h.mp4")
    out = root / "out.mp4"
    job = Job(video_path=str(src), device="cpu", hwaccel=False, overrides={"scale": 1.0}, export=True,
              export_path=str(out), export_codec="hevc")
    assert run_job(job, settings).status == "complete"
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json",
                                       str(out)], capture_output=True, text=True, check=True).stdout)
    v = probe["streams"][0]
    assert v["codec_name"] == "hevc" and v["codec_tag_string"] == "hvc1" and int(v["nb_read_frames"]) == 48


def test_watch_folder(env):
    root, settings = env
    watched = root / "incoming"
    watched.mkdir()
    (watched / "old.mp4").write_bytes(b"old")
    settings.watch_dirs = [str(watched)]
    queued = []
    w = FolderWatcher(settings, lambda path, kind, profile: queued.append((path, kind, profile)))
    assert w.scan() == []                                  # baseline: existing files are left alone
    (watched / "new.mkv").write_bytes(b"part")
    (watched / "notes.txt").write_bytes(b"x")
    assert w.scan() == []                                  # first sighting: maybe still copying
    (watched / "new.mkv").write_bytes(b"partial2")
    assert w.scan() == []                                  # still changing
    assert w.scan() == [str(watched / "new.mkv")]          # stable -> queued once
    assert queued == [(str(watched / "new.mkv"), "offline", "quality")]
    assert w.scan() == []
    w2 = FolderWatcher(settings, lambda *a: queued.append(a))  # state survives restarts
    assert w2.scan() == [] and len(queued) == 1


def test_bench_all_models(env):
    root, settings = env
    write_fake_model(root / "models" / "4.25.lite")
    cfg = root / "sf.toml"
    cfg.write_text(f'cache_root = "{(root / "cache").as_posix()}"\nmodel_dir = "{settings.model_dir}"\n'
                   f'models_dir = "{(root / "models").as_posix()}"\n')
    r = subprocess.run([sys.executable, "-m", "streamerframes", "--config", str(cfg), "bench", "--size", "64x48",
                        "--device", "cpu", "--scales", "1.0", "--warmup", "1", "--iters", "1"],
                       capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr
    assert "|default|" in r.stdout and "|4.25.lite|" in r.stdout
