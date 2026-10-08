"""Phase 6 groundwork that runs without a GPU: ONNX export matches PyTorch; TensorRT falls back cleanly."""
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from streamerframes.config import REPO_ROOT, Profile  # noqa: E402
from streamerframes.engine.loader import load_ifnet  # noqa: E402
from streamerframes.engine.onnx_export import (  # noqa: E402
    IFNetForExport,
    export_onnx,
    verify_onnx,
)
from streamerframes.engine.trt_backend import engine_path  # noqa: E402
from streamerframes.pipeline.generator import _tensorrt_engine  # noqa: E402

from .fake_model import write_fake_model  # noqa: E402


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    return load_ifnet(write_fake_model(tmp_path_factory.mktemp("m")), "cpu")


def test_export_wrapper_uses_full_scale_list(net):
    assert IFNetForExport(net, 0.5).scale_list == [32.0, 16.0, 8.0, 4.0, 2.0]


def test_onnx_export_matches_pytorch(net, tmp_path):
    path = export_onnx(net, 64, 128, 1.0, tmp_path / "m.onnx")
    import onnx
    model = onnx.load(str(path))
    assert [i.name for i in model.graph.input] == ["img0", "img1", "timestep"]
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 3, 64, 128]  # static shape per padded resolution
    assert verify_onnx(net, path, 64, 128, 1.0) < 1e-5


def test_tensorrt_backend_falls_back_without_tensorrt(net, tmp_path):
    prof = Profile(backend="tensorrt")
    assert _tensorrt_engine(net, 48, 64, 1.0, "cpu", prof, tmp_path) is None
    p = engine_path(tmp_path, "NVIDIA GeForce GTX 1080 Ti", "4.25", 1920, 832, 1.0)
    assert p.name == "NVIDIA_GeForce_GTX_1080_Ti_4.25_1920x832_s1.0.engine"


def test_profile_backend_does_not_change_cache_id():
    assert Profile(backend="tensorrt").profile_id(1.0) == Profile().profile_id(1.0)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_trt_export_cli(tmp_path):
    model = write_fake_model(tmp_path / "model")
    cfg = tmp_path / "sf.toml"
    cfg.write_text(f'cache_root = "{(tmp_path / "cache").as_posix()}"\nmodel_dir = "{model.as_posix()}"\n')
    r = subprocess.run([sys.executable, "-m", "streamerframes", "--config", str(cfg), "trt", "export",
                        "--size", "64x48", "--device", "cpu"], capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr
    assert "max abs error" in r.stdout and "ok" in r.stdout
    assert (tmp_path / "cache" / "engines" / "default_64x64_s1.0.onnx").exists()
