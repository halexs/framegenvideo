import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from streamerframes.config import Settings
from streamerframes.jobs.manager import JobManager, job_view

FAKE = str(Path(__file__).with_name("fake_worker.py"))


def make_manager(tmp_path, **kw):
    settings = Settings(cache_root=str(tmp_path / "cache"))
    settings.stream_policy = kw.pop("policy", "newest_wins")

    def resolver(video_path, profile, kind):
        vid = Path(video_path).name.replace(".", "")[:8].ljust(8, "0").encode().hex()[:16]
        pid = ("%012x" % (abs(hash(profile)) % 16**12))
        return vid, pid, str(tmp_path / "cache" / vid / pid)

    kw.setdefault("poll_interval", 0.05)
    return JobManager(settings, resolver=resolver, worker_cmd=lambda f: [sys.executable, FAKE, str(f)], **kw)


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    raise AssertionError("timed out")


def video(tmp_path, name):
    p = tmp_path / "movies" / name
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"")
    return str(p)


def calls(rec):
    path = rec.cache.root / "calls.log"
    return path.read_text().splitlines() if path.exists() else []


@pytest.fixture
def mgr(tmp_path):
    m = make_manager(tmp_path)
    m.start()
    yield m
    m.shutdown()
    for rec in m.jobs.values():  # never leave fake workers behind
        if rec.pid:
            subprocess.run(["kill", "-9", str(rec.pid)], capture_output=True)


def test_runs_to_completion_and_dedupes(mgr, tmp_path):
    rec = mgr.submit(video(tmp_path, "slow.mp4"), "offline", "quality")
    assert mgr.submit(video(tmp_path, "slow.mp4"), "offline", "quality") is rec
    wait_for(lambda: rec.state == "running")
    (rec.cache.root / "finish").touch()
    wait_for(lambda: rec.state == "complete")
    assert rec.finished and rec.pid is None
    saved = json.loads((Path(mgr.settings.cache_root) / "jobs.json").read_text())
    assert saved["jobs"][0]["state"] == "complete"
    assert job_view(rec)["percent"] == 0.0


def test_failure_is_reported(mgr, tmp_path):
    rec = mgr.submit(video(tmp_path, "fail.mp4"))
    wait_for(lambda: rec.state == "failed")
    assert rec.error == "boom"


def test_stream_preempts_offline_which_resumes(mgr, tmp_path):
    offline = mgr.submit(video(tmp_path, "slow.mp4"), "offline", "quality")
    wait_for(lambda: offline.state == "running")
    stream = mgr.submit(video(tmp_path, "ok.mp4"), "stream", "realtime")
    wait_for(lambda: stream.state == "complete")
    # The offline job was stopped gracefully, re-queued, and started again after the stream job.
    wait_for(lambda: offline.state == "running" and offline.runs == 2)
    wait_for(lambda: len(calls(offline)) == 2)  # the restarted worker logs once it is up
    assert calls(offline) == ["offline start=0", "offline start=0"]
    assert mgr.running() is offline
    (offline.cache.root / "finish").touch()
    wait_for(lambda: offline.state == "complete")


def test_newest_stream_wins(mgr, tmp_path):
    first = mgr.submit(video(tmp_path, "slow.mp4"), "stream", "realtime")
    wait_for(lambda: first.state == "running")
    second = mgr.submit(video(tmp_path, "slow.mp4"), "stream", "quality")  # same movie, other profile
    wait_for(lambda: first.state == "paused")
    wait_for(lambda: second.state == "running")
    mgr.cancel(second.id)
    wait_for(lambda: second.state == "cancelled")
    mgr.resume(first.id)
    wait_for(lambda: first.state == "running")
    mgr.cancel(first.id)
    wait_for(lambda: first.state == "cancelled")


def test_queue_policy_keeps_both(tmp_path):
    m = make_manager(tmp_path, policy="queue")
    a = m.submit(video(tmp_path, "slow.mp4"), "stream", "realtime")
    b = m.submit(video(tmp_path, "ok.mp4"), "stream", "realtime")
    assert a.state == "running" and b.state == "queued"
    m.cancel(a.id)
    wait_for(lambda: (m.tick() or True) and b.state == "complete")


def test_offline_promoted_to_stream_on_dedupe(mgr, tmp_path):
    rec = mgr.submit(video(tmp_path, "slow.mp4"), "offline", "realtime")
    assert mgr.submit(video(tmp_path, "slow.mp4"), "stream", "realtime") is rec
    assert rec.kind == "stream"
    mgr.cancel(rec.id)
    wait_for(lambda: rec.state == "cancelled")


def test_restart_adopts_running_worker(tmp_path):
    m1 = make_manager(tmp_path)
    rec = m1.submit(video(tmp_path, "slow.mp4"), "offline", "quality")
    assert rec.state == "running"
    pid = rec.pid
    m1.shutdown()  # server goes away; the worker keeps running

    m2 = make_manager(tmp_path)
    m2.start()
    try:
        adopted = m2.get(rec.id)
        assert adopted.state == "running" and adopted.pid == pid
        m2.tick()
        assert m2.running().id == rec.id  # no duplicate worker
        (adopted.cache.root / "finish").touch()
        wait_for(lambda: adopted.state == "complete")
    finally:
        m2.shutdown()


def test_restart_requeues_dead_offline_worker(tmp_path):
    m1 = make_manager(tmp_path)
    rec = m1.submit(video(tmp_path, "slow.mp4"), "offline", "quality")
    subprocess.run(["kill", "-9", str(rec.pid)])
    m1._procs[rec.id].wait()
    m1.shutdown()

    m2 = make_manager(tmp_path)
    m2.start()
    try:
        again = m2.get(rec.id)
        wait_for(lambda: again.state == "running" and again.runs == 2)  # auto-resumed
        (again.cache.root / "finish").touch()
        wait_for(lambda: again.state == "complete")
    finally:
        m2.shutdown()


def test_unresponsive_worker_is_killed(tmp_path):
    m = make_manager(tmp_path, stop_timeout=0.5, watchdog_seconds=60)
    rec = m.submit(video(tmp_path, "hang.mp4"), "offline", "quality")
    m.cancel(rec.id)
    wait_for(lambda: (m.tick() or True) and rec.state == "cancelled")

    m.watchdog_seconds = 0.3
    rec2 = m.submit(video(tmp_path, "hang.mp4"), "offline", "realtime")
    wait_for(lambda: (m.tick() or True) and rec2.state == "failed")
    assert "watchdog" in rec2.error


def test_retarget_restarts_running_stream_at_segment(mgr, tmp_path):
    rec = mgr.submit(video(tmp_path, "slow.mp4"), "stream", "realtime")
    wait_for(lambda: rec.state == "running")
    assert mgr.retarget(rec, 7)
    assert rec.cache.stop_now_requested() or rec.state != "running"
    wait_for(lambda: rec.runs == 2 and rec.state == "running")
    wait_for(lambda: len(calls(rec)) == 2)
    assert calls(rec) == ["stream start=0", "stream start=7"]
    assert not mgr.retarget(rec, 9)  # debounced: scrubbing doesn't thrash restarts
    assert mgr.retarget(rec, 7)       # already there
    mgr.cancel(rec.id)
    wait_for(lambda: rec.state == "cancelled")


def test_stream_from_reuses_or_starts_job(mgr, tmp_path):
    rec = mgr.stream_from(video(tmp_path, "slow.mp4"), "realtime", 4)
    assert rec.start_segment == 4 and rec.kind == "stream"
    wait_for(lambda: len(calls(rec)) == 1)
    assert calls(rec) == ["stream start=4"]
    mgr.cancel(rec.id)
    wait_for(lambda: rec.state == "cancelled")


def test_finished_stream_stretch_queues_gap_filler(mgr, tmp_path):
    rec = mgr.submit(video(tmp_path, "partial.mp4"), "stream", "realtime", start_segment=5)
    wait_for(lambda: rec.state == "paused")
    filler = next(r for r in mgr.jobs.values() if r is not rec)
    assert filler.kind == "offline" and filler.profile_id == rec.profile_id and filler.start_segment == 0
    wait_for(lambda: filler.state in ("paused", "running", "queued"))
    mgr.settings.fill_gaps = False
    other = mgr.submit(video(tmp_path, "partial.mp4"), "stream", "quality")
    wait_for(lambda: other.state == "paused")
    assert not any(r.profile_id == other.profile_id and r is not other for r in mgr.jobs.values())
