import pytest

from streamerframes.config import REPO_ROOT, load_settings


def test_defaults_without_file(tmp_path):
    s = load_settings(tmp_path / "missing.toml", env={})
    assert s.host == "127.0.0.1"
    assert set(s.profiles) == {"realtime", "quality"}
    assert s.model_path("default") == REPO_ROOT / "train_log"


def test_example_file_loads():
    s = load_settings(REPO_ROOT / "streamerframes.example.toml", env={})
    assert s.profile("quality").encoder.cq == 18
    assert s.profile("realtime").scale == "auto"


def test_file_and_env_overrides(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('port = 9000\n[profiles.realtime]\nscale = 0.5\nencoder = { cq = 25 }\n'
                   '[profiles.anime]\nmode = "target"\ntarget_fps = 60\ndup_mad = 0.002\n')
    s = load_settings(cfg, env={"MOVIES_DIR": "/a", "PORT": "9100", "CACHE_DIR": "/c"})
    assert s.port == 9100 and s.movies_roots == ["/a"] and s.cache_root == "/c"
    rt = s.profile("realtime")
    assert rt.scale == 0.5 and rt.encoder.cq == 25 and rt.encoder.preset == "p4"  # merged onto defaults
    assert s.profile("anime").dup_mad == 0.002


def test_unknown_keys_rejected(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("prot = 1\n")
    with pytest.raises(ValueError, match="prot"):
        load_settings(cfg, env={})
    with pytest.raises(ValueError):
        load_settings(env={}).profile("nope")


def test_profile_id_tracks_pixels_only():
    s = load_settings(env={})
    rt = s.profile("realtime")
    base = rt.profile_id(0.5)
    assert len(base) == 12 and base == rt.profile_id(0.5)
    assert rt.profile_id(1.0) != base
    rt.cuda_graphs = not rt.cuda_graphs
    rt.target_fps = 24  # unused in multi mode
    assert rt.profile_id(0.5) == base
    rt.encoder.cq += 1
    assert rt.profile_id(0.5) != base
