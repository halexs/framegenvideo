import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402

import server  # noqa: E402


@pytest.fixture
def movies(tmp_path, monkeypatch):
    root = tmp_path / "movies"
    root.mkdir()
    (root / "a.mp4").write_bytes(b"x")
    (tmp_path / "secret.mp4").write_bytes(b"x")
    monkeypatch.setattr(server, "MOVIES_ROOT", root)
    return root


def test_get_video_path_ok(movies):
    assert server.get_video_path("a.mp4") == (movies / "a.mp4").resolve()


@pytest.mark.parametrize("name", ["../secret.mp4", "..%2Fsecret.mp4", "missing.mp4"])
def test_get_video_path_rejects_escape_and_missing(movies, name):
    with pytest.raises(HTTPException) as exc:
        server.get_video_path(name)
    assert exc.value.status_code == 404


def test_missing_movies_dir_has_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MOVIES_ROOT", tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="MOVIES_DIR"):
        server.get_movies_root()


def test_sanitize_name():
    assert server.sanitize_name("Dir/My Movie (2020).mp4") == "Dir_My_Movie_2020_.mp4"
