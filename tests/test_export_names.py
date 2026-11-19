"""Export file names: the flat default and the media-server ("library") layout."""
import shutil
import time
from types import SimpleNamespace

import pytest

from streamerframes.config import load_settings
from streamerframes.pipeline.export import default_export_name


def name(path, out_fps, container="mp4", codec="copy", layout="library"):
    return default_export_name(SimpleNamespace(path=path), {"out_fps": out_fps}, container, codec, layout)


def test_library_names():
    assert name("E:/Movies/Batman Begins (2005).mp4", "60000/1001") == \
        "Batman Begins (2005)/Batman Begins (2005) - 60fps.mp4"
    assert name("/m/Film (1999).mkv", "48", container="mkv") == "Film (1999)/Film (1999) - 48fps.mkv"
    assert name("/m/Film (1999).mkv", "48000/1001") == "Film (1999)/Film (1999) - 48fps.mp4"
    assert name("/m/Film (1999).mp4", "60000/1001", codec="hevc") == "Film (1999)/Film (1999) - 60fps HEVC.mp4"


def test_library_names_keep_punctuation_but_not_windows_reserved():
    assert name("/m/Amélie, Part 1: Ready? (2001).mp4", "60000/1001") == \
        "Amélie, Part 1_ Ready_ (2001)/Amélie, Part 1_ Ready_ (2001) - 60fps.mp4"
    assert name("/m/....mp4", "60") == "video/video - 60fps.mp4"


def test_flat_names_unchanged():
    assert name("/m/Batman Begins (2005).mp4", "60000/1001", layout="flat") == "Batman Begins _2005_.59.940fps.mp4"
    assert name("/m/a.mp4", "48", layout="flat") == "a.48fps.mp4"
    assert name("/m/a.mp4", "48000/1001", codec="hevc", layout="flat") == "a.47.952fps.hevc.mp4"
    assert default_export_name(SimpleNamespace(path="/m/a.mp4"), {"out_fps": "48"}, "mp4") == "a.48fps.mp4"


def test_layout_is_validated(tmp_path):
    with pytest.raises(ValueError):
        name("/m/a.mp4", "48", layout="nested")
    cfg = tmp_path / "sf.toml"
    cfg.write_text('export_layout = "jellyfin"\n')
    with pytest.raises(ValueError, match="export_layout"):
        load_settings(cfg, env={})
    cfg.write_text('export_layout = "library"\n')
    assert load_settings(cfg, env={}).export_layout == "library"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_library_layout_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from .webenv import make_env, make_movie

    exports = tmp_path / "Framegen"
    movies, settings, manager, app = make_env(
        tmp_path, extra=f'export_dir = "{exports.as_posix()}"\nexport_layout = "library"\n')
    make_movie(movies / "A Film (2001).mp4", frames=24)
    with TestClient(app) as client:
        vid = client.get("/api/library").json()["videos"][0]["video_id"]
        # What a watch folder queues: an offline quality job that exports when it finishes.
        rec = manager.submit(str(movies / "A Film (2001).mp4"), kind="offline", profile="quality", export=True)
        end = time.time() + 90
        while time.time() < end and manager.get(rec.id).state not in ("complete", "failed"):
            time.sleep(0.2)
        assert manager.get(rec.id).state == "complete", manager.get(rec.id).error
        expected = exports / "A Film (2001)" / "A Film (2001) - 60fps.mp4"
        assert expected.exists()
        assert not list(exports.rglob("*.partial.*"))
        cache = client.get("/api/library").json()["videos"][0]["caches"][0]
        assert cache["has_export"] and cache["export_state"] == "done"
        dl = client.get(f"/exports/{vid}/{rec.profile_id}")
        assert dl.status_code == 200 and dl.content == expected.read_bytes()
