"""Shared setup for web tests: a movies folder, a config file the worker subprocess can load, the app."""
import subprocess
from pathlib import Path

from streamerframes.config import load_settings
from streamerframes.jobs.manager import JobManager
from streamerframes.web.app import create_app

from .fake_model import write_fake_model


def make_movie(path: Path, frames: int = 48, rate: str = "24000/1001", audio: bool = True, vf: str | None = None):
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=64x48:rate={rate}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000", "-shortest", "-c:a", "aac"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


def make_env(tmp_path: Path, seg_seconds: float = 0.5, extra: str = ""):
    movies = tmp_path / "movies"
    movies.mkdir()
    model = write_fake_model(tmp_path / "model")
    cfg = tmp_path / "sf.toml"
    cfg.write_text(
        f'movies_roots = ["{movies.as_posix()}"]\ncache_root = "{(tmp_path / "cache").as_posix()}"\n'
        f'model_dir = "{model.as_posix()}"\n{extra}'
        + "".join(f'[profiles.{p}]\nscale = 1.0\nscene_detect = false\nseg_seconds = {seg_seconds}\n'
                  f'encoder = {{ codec = "lossless" }}\n' for p in ("realtime", "quality")))
    settings = load_settings(cfg, env={})
    manager = JobManager(settings, config_path=str(cfg), job_defaults={"device": "cpu", "hwaccel": False},
                         poll_interval=0.1)
    app = create_app(settings, manager=manager, config_path=str(cfg))
    return movies, settings, manager, app
