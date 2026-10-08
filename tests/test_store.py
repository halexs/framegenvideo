import os
import subprocess
import sys
import time

from streamerframes.cache.store import (
    CacheStore,
    LockHeld,
    ProfileCache,
    atomic_write_json,
    read_json,
    video_id_for,
)


def _complete(pc: ProfileCache, ks, run=0):
    pc.ensure_dirs()
    lines = []
    for k in ks:
        pc.segment_path(k).write_bytes(b"x")
        lines.append(f"v_{k:05d}.ts,{k}.0,{k + 1}.0\n")
    (pc.index_dir / f"run_{run:04d}.csv").write_text("".join(lines))


def test_atomic_json(tmp_path):
    p = tmp_path / "a.json"
    atomic_write_json(p, {"x": 1})
    assert read_json(p) == {"x": 1}
    assert read_json(tmp_path / "missing.json", "d") == "d"
    p.write_text("{broken")
    assert read_json(p) is None


def test_video_id_changes_with_file(tmp_path):
    f = tmp_path / "m.mp4"
    f.write_bytes(b"a")
    first = video_id_for(f)
    assert first == video_id_for(f) and len(first) == 16
    time.sleep(0.01)
    f.write_bytes(b"ab")
    assert video_id_for(f) != first


def test_completed_segments_need_index_and_file(tmp_path):
    pc = ProfileCache(tmp_path / "v" / "p")
    _complete(pc, [0, 1, 2], run=0)
    _complete(pc, [5], run=1)
    pc.segment_path(3).write_bytes(b"partial, no index line")
    pc.segment_path(1).unlink()
    assert pc.completed_segments() == {0, 2, 5}
    assert pc.next_run_id() == 2
    assert pc.contiguous_prefix() == 1


def test_first_missing_and_gap_end():
    done = {0, 1, 5, 6}
    assert ProfileCache.first_missing(done, 0, 8) == 2
    assert ProfileCache.gap_end(done, 2, 8) == 5
    assert ProfileCache.first_missing(done, 5, 8) == 7
    assert ProfileCache.gap_end(done, 7, 8) == 8
    assert ProfileCache.first_missing({0, 1, 2, 3, 6, 7}, 6, 8) == 4  # wraps to earlier holes
    assert ProfileCache.first_missing(set(range(8)), 3, 8) is None


def test_lock_is_exclusive_and_stale_locks_are_stolen(tmp_path):
    pc = ProfileCache(tmp_path / "v" / "p")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        import psutil
        pc.root.mkdir(parents=True)
        atomic_write_json(pc.lock_path, {"pid": child.pid, "create_time": psutil.Process(child.pid).create_time()})
        assert pc.lock()["pid"] == child.pid
        try:
            pc.acquire_lock(0)
            raise AssertionError("expected LockHeld")
        except LockHeld:
            pass
    finally:
        child.kill()
        child.wait()
    assert pc.lock() is None  # dead pid
    pc.acquire_lock(1)
    assert read_json(pc.lock_path)["pid"] == os.getpid()
    pc.release_lock()
    assert not pc.lock_path.exists()


def test_stop_flag(tmp_path):
    pc = ProfileCache(tmp_path / "v" / "p")
    assert not pc.stop_requested()
    pc.request_stop()
    assert pc.stop_requested()
    pc.clear_stop()
    assert not pc.stop_requested()


def test_store_listing_remove_and_gc(tmp_path):
    store = CacheStore(tmp_path)
    for vid, last in (("aaa", 1), ("bbb", 2), ("ccc", 3)):
        store.write_source(vid, {"path": vid})
        pc = store.profile(vid, "p1")
        _complete(pc, [0])
        pc.segment_path(0).write_bytes(b"x" * 1000)
        pc.write_manifest({"status": "complete", "last_access": last})
    exported = store.profile("bbb", "p1")
    (exported.export_dir).mkdir()
    (exported.export_dir / "out.mp4").write_bytes(b"y")
    assert store.video_ids() == ["aaa", "bbb", "ccc"]

    evicted = store.gc(max_bytes=2500)
    assert [pc.video_id for pc in evicted] == ["aaa"]  # oldest idle one; bbb is exported
    assert store.profiles("aaa") == []

    store.remove("ccc")
    assert store.video_ids() == ["aaa", "bbb"]  # gc keeps source.json; remove() drops the whole video
