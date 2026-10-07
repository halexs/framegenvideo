import shutil
import time

import pytest

pytest.importorskip("torch")
pytest.importorskip("httpx")
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

from fastapi.testclient import TestClient  # noqa: E402

from .webenv import make_env, make_movie  # noqa: E402


@pytest.fixture
def web(tmp_path):
    movies, settings, manager, app = make_env(tmp_path)
    with TestClient(app) as client:
        yield movies, settings, manager, client
    for rec in manager.jobs.values():
        if rec.state == "running":
            manager.cancel(rec.id)


def wait_job(client, job_id, states=("complete", "failed", "cancelled", "paused"), timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in states:
            return job
        time.sleep(0.2)
    raise AssertionError(f"job stuck: {job}")


def test_pages_and_ids(web):
    movies, settings, manager, client = web
    make_movie(movies / "A Movie (2001).mp4", frames=12)
    (movies / "notes.txt").write_text("x")
    assert "StreamerFrames" in client.get("/").text
    lib = client.get("/api/library").json()
    assert [v["title"] for v in lib["videos"]] == ["A Movie (2001).mp4"]
    vid = lib["videos"][0]["video_id"]
    assert client.get(f"/watch/{vid}").status_code == 200
    v = client.get(f"/api/videos/{vid}").json()
    assert v["source"]["src_fps"] == "24000/1001" and "path" not in v["source"]
    # Only opaque IDs are accepted; paths never reach the filesystem.
    for bad in ("../../etc", "..%2F..%2Fetc%2Fpasswd", "ABCDEF0123456789", "deadbeefdeadbeef"):
        assert client.get(f"/api/videos/{bad}").status_code == 404
        assert client.get(f"/media/original/{bad}").status_code == 404
    assert client.post("/api/jobs", json={"video_id": "../x"}).status_code == 404
    assert client.post("/api/jobs", json={"video_id": vid, "profile": "nope"}).status_code == 400


def test_original_supports_range_requests(web):
    movies, settings, manager, client = web
    make_movie(movies / "m.mp4", frames=12)
    vid = client.get("/api/library").json()["videos"][0]["video_id"]
    r = client.get(f"/media/original/{vid}", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100


def test_stream_job_serves_hls_then_export_and_delete(web):
    movies, settings, manager, client = web
    make_movie(movies / "m.mp4", frames=48)
    vid = client.get("/api/library").json()["videos"][0]["video_id"]
    job = client.post("/api/jobs", json={"video_id": vid, "kind": "stream", "profile": "realtime"}).json()
    assert job["state"] in ("queued", "running")
    again = client.post("/api/jobs", json={"video_id": vid, "kind": "stream", "profile": "realtime"}).json()
    assert again["id"] == job["id"]  # deduped
    done = wait_job(client, job["id"])
    assert done["state"] == "complete", done
    assert done["percent"] == 100.0 and done["contiguous_segments"] == done["segments_total"] == 4
    pid = done["profile_id"]

    master = client.get(f"/hls/{vid}/{pid}/master.m3u8")
    assert master.headers["cache-control"] == "no-cache"
    assert 'URI="../audio.m3u8"' in master.text and "FRAME-RATE=47.952" in master.text
    video = client.get(f"/hls/{vid}/{pid}/video.m3u8").text
    assert "#EXT-X-PLAYLIST-TYPE:VOD" in video and "#EXT-X-ENDLIST" in video
    assert video.count("#EXTINF:0.500500,") == 4
    seg = client.get(f"/hls/{vid}/{pid}/seg/0.ts")
    assert seg.status_code == 200 and seg.content[:1] == b"G" and "immutable" in seg.headers["cache-control"]
    assert client.get(f"/hls/{vid}/{pid}/seg/99.ts").status_code == 404
    audio = client.get(f"/hls/{vid}/audio.m3u8").text
    assert "#EXT-X-ENDLIST" in audio and "audio/0.ts" in audio
    assert client.get(f"/hls/{vid}/audio/0.ts").status_code == 200

    # A finished cache answers immediately without starting a worker.
    again = client.post("/api/jobs", json={"video_id": vid, "kind": "stream", "profile": "realtime"}).json()
    assert again["state"] == "complete" and again["id"] is None

    r = client.post(f"/api/videos/{vid}/export", json={"profile_id": pid})
    assert r.status_code == 200
    end = time.time() + 30
    while time.time() < end:
        caches = client.get("/api/library").json()["videos"][0]["caches"]
        if caches[0]["export_state"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert caches[0]["export_state"] == "done", caches[0]["export_error"]
    dl = client.get(f"/exports/{vid}/{pid}")
    assert dl.status_code == 200 and len(dl.content) > 1000

    assert client.delete(f"/api/cache/{vid}").json() == {"ok": True}
    assert client.get(f"/hls/{vid}/{pid}/master.m3u8").status_code == 404


def test_cancel_and_delete_conflict(web):
    movies, settings, manager, client = web
    make_movie(movies / "long.mp4", frames=600, audio=False)
    vid = client.get("/api/library").json()["videos"][0]["video_id"]
    job = client.post("/api/jobs", json={"video_id": vid, "kind": "offline", "profile": "quality"}).json()
    assert client.delete(f"/api/cache/{vid}").status_code == 409
    client.delete(f"/api/jobs/{job['id']}")
    done = wait_job(client, job["id"], states=("cancelled", "complete"))
    assert done["state"] in ("cancelled", "complete")


def test_seek_retargets_generation_and_gaps_get_filled(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_IFNET_DELAY", "0.03")  # ~35 interpolations/s: slow enough to seek ahead of
    movies, settings, manager, app = make_env(tmp_path)
    make_movie(movies / "seek.mp4", frames=360, audio=False)  # 30 segments of 0.5 s at 2x
    with TestClient(app) as client:
        vid = client.get("/api/library").json()["videos"][0]["video_id"]
        job = client.post("/api/jobs", json={"video_id": vid, "kind": "stream", "profile": "realtime"}).json()
        pid = job["profile_id"]
        video = client.get(f"/hls/{vid}/{pid}/video.m3u8").text
        assert video.count("#EXTINF") == 30 and "#EXT-X-ENDLIST" in video  # whole timeline up front
        assert client.get(f"/hls/{vid}/{pid}/seg/30.ts").status_code == 404

        t0 = time.time()
        far = client.get(f"/hls/{vid}/{pid}/seg/25.ts")  # seek near the end: long-poll + retarget
        assert far.status_code == 200 and far.content[:1] == b"G"
        rec = manager.get(job["id"])
        assert rec.start_segment == 25 and rec.runs == 2
        status = client.get(f"/api/status/{vid}/{pid}").json()
        assert [25, 26] in [[a, min(b, 26)] for a, b in status["ranges"]]
        assert time.time() - t0 < 20

        # After the stream stretch reaches the end, an offline job fills the holes and completes the cache.
        end = time.time() + 90
        while time.time() < end:
            st = client.get(f"/api/status/{vid}/{pid}").json()
            if st["status"] == "complete":
                break
            time.sleep(0.3)
        assert st["status"] == "complete" and st["ranges"] == [[0, 30]]
        kinds = sorted(j.kind for j in manager.jobs.values())
        assert kinds == ["offline", "stream"]
