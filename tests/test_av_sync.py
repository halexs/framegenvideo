"""A/V sync through the real HLS output (PLAN.md Phase 3): a white flash on every 24th source frame and a
1 kHz beep starting at the same instant. After generation (with a forced stop/resume in the middle) the HLS
master playlist is decoded over HTTP the way a player would, and flash vs. beep onsets must agree within
one output frame."""
import re
import shutil
import socket
import subprocess
import threading
import time
from fractions import Fraction as F

import pytest

pytest.importorskip("torch")
uvicorn = pytest.importorskip("uvicorn")
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

from streamerframes.pipeline import generator as gen_mod  # noqa: E402
from streamerframes.pipeline.worker import Job, resolve, run_job  # noqa: E402

from .webenv import make_env  # noqa: E402

RATE = F(24000, 1001)
PERIOD = 24          # source frames between flashes
FRAMES = 8 * PERIOD


def flash_clip(path):
    period_s = float(PERIOD / RATE)
    beep = f"0.5*sin(2*PI*1000*t)*lt(mod(t+0.0001,{period_s:.9f}),0.06)"
    subprocess.run([
        "ffmpeg", "-v", "error",
        "-f", "lavfi", "-i", f"color=c=0x202020:size=64x48:rate={RATE.numerator}/{RATE.denominator}",
        "-f", "lavfi", "-i", f"aevalsrc='{beep}':s=48000:d={float(FRAMES / RATE) + 0.5}",
        "-vf", f"drawbox=c=white:t=fill:enable='eq(mod(n,{PERIOD}),0)'",
        "-frames:v", str(FRAMES), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(path)], check=True)
    return path


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def onsets(times, values, threshold):
    out, prev = [], False
    for t, v in zip(times, values):
        on = v > threshold
        if on and not prev:
            out.append(t)
        prev = on
    return out


def test_flash_and_beep_stay_in_sync_across_resume(tmp_path):
    movies, settings, manager, app = make_env(tmp_path, seg_seconds=1.0)
    src = flash_clip(movies / "sync.mp4")
    job = Job(video_path=str(src), kind="stream", profile="realtime", device="cpu", hwaccel=False)

    # First run stops after ~3 s of output; the second resumes from the next segment.
    stop = threading.Event()
    orig = gen_mod.Generator.__init__

    def fast_progress(self, *a, **kw):
        kw["progress_interval"] = 0
        orig(self, *a, **kw)

    gen_mod.Generator.__init__ = fast_progress
    try:
        first = run_job(job, settings, stop_event=stop,
                        on_progress=lambda p: p["out_frame"] > 130 and stop.set())
    finally:
        gen_mod.Generator.__init__ = orig
    assert first.status == "paused"
    second = run_job(job, settings)
    assert second.status == "complete" and second.first_segment > 0
    _, vid, _, _, cache, _ = resolve(settings, job)

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        url = f"http://127.0.0.1:{port}/hls/{vid}/{cache.profile_id}/master.m3u8"
        video = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", url, "-map", "0:v:0", "-vf",
             "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
        audio = subprocess.run(
            ["ffmpeg", "-v", "info", "-i", url, "-map", "0:a:0", "-af",
             "asetnsamples=n=96,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
             "-f", "null", "-"], capture_output=True, text=True, timeout=120)
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert video.returncode == 0, video.stderr
    assert audio.returncode == 0, audio.stderr
    vt = [float(x) for x in re.findall(r"pts_time:([\d.]+)", video.stdout)]
    vy = [float(x) for x in re.findall(r"YAVG=([\d.]+)", video.stdout)]
    at = [float(x) for x in re.findall(r"pts_time:([\d.]+)", audio.stdout)]
    ar = [float(x) if x != "-inf" else -200.0 for x in re.findall(r"RMS_level=(-?[\w.]+)", audio.stdout)]
    assert len(vt) == len(vy) == FRAMES * 2

    flashes = onsets(vt, vy, 200)  # a full-white frame; half-blends next to it stay below
    beeps = onsets(at, ar, -40)
    assert len(flashes) == FRAMES // PERIOD
    assert len(beeps) >= len(flashes)
    frame = 1 / float(RATE * 2)
    window = 96 / 48000  # audio measured in 2 ms windows
    for f in flashes[1:]:  # the first beep starts at t=0, inside the 21 ms the priming fix trims
        nearest = min(beeps, key=lambda b: abs(b - f))
        print(f"flash {f:.4f} beep {nearest:.4f} diff {1000 * (nearest - f):+.1f} ms")
        assert abs(nearest - f) <= frame / 2 + window, (f, nearest)
