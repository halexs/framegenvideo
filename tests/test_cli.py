import json
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

from .fake_model import write_fake_model  # noqa: E402


def run_cli(*args, cwd):
    return subprocess.run([sys.executable, "-m", "streamerframes", *args], capture_output=True, text=True,
                          cwd=cwd)


def test_render_then_cache_ls(tmp_path):
    from streamerframes.config import REPO_ROOT

    model = write_fake_model(tmp_path / "model")
    cfg = tmp_path / "sf.toml"
    cfg.write_text(f'cache_root = "{(tmp_path / "cache").as_posix()}"\nmodel_dir = "{model.as_posix()}"\n'
                   '[profiles.quality]\nscene_detect = false\nseg_seconds = 1.0\n')
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25",
                    "-frames:v", "30", "-pix_fmt", "yuv420p", str(clip)], check=True)
    out = tmp_path / "out.mkv"
    r = run_cli("--config", str(cfg), "render", str(clip), "--device", "cpu", "--no-hwaccel",
                "--encoder", "lossless", "--scale", "1.0", "--skip-check", "--export", str(out), cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr
    assert "complete" in r.stderr
    frames = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                             "stream=nb_read_frames,r_frame_rate", "-of", "json", str(out)],
                            capture_output=True, text=True, check=True).stdout
    stream = json.loads(frames)["streams"][0]
    assert stream["r_frame_rate"] == "60/1" and int(stream["nb_read_frames"]) == 72  # 30 @ 25 -> 60 fps

    r = run_cli("--config", str(cfg), "cache", "ls", cwd=REPO_ROOT)
    assert "complete" in r.stdout and "100.0%" in r.stdout and "exported" in r.stdout

    r = run_cli("--config", str(cfg), "render", str(clip), "--device", "cpu", "--no-hwaccel",
                "--encoder", "lossless", "--scale", "1.0", "--skip-check", "--no-export", cwd=REPO_ROOT)
    assert r.returncode == 0 and "complete" in r.stderr  # already done: no work


def test_plan_and_probe(tmp_path):
    from streamerframes.config import REPO_ROOT

    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24000/1001",
                    "-frames:v", "10", str(clip)], check=True)
    r = run_cli("plan", str(clip), "--target-fps", "60", cwd=REPO_ROOT)
    assert json.loads(r.stdout)["total_out_frames"] == 25
    r = run_cli("probe", str(tmp_path / "nope.mp4"), cwd=REPO_ROOT)
    assert r.returncode == 1 and "error" in r.stderr
