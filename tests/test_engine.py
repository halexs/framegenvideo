import shutil
import subprocess

import pytest

torch = pytest.importorskip("torch")

from streamerframes.engine.color import (  # noqa: E402
    ColorSpec,
    YuvConverter,
    frame_bytes,
)
from streamerframes.engine.loader import count_blocks, load_ifnet  # noqa: E402
from streamerframes.engine.rife import PairKind, RifeEngine  # noqa: E402

from .fake_model import write_fake_model  # noqa: E402


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    return load_ifnet(write_fake_model(tmp_path_factory.mktemp("model")), "cpu")


def test_loader_strips_prefix_and_counts_blocks(net):
    assert count_blocks(net) == 5
    assert not net.training
    assert all(not p.requires_grad for p in net.parameters())


def test_loader_reports_missing_weights(tmp_path):
    d = write_fake_model(tmp_path / "m")
    torch.save({}, d / "flownet.pkl")
    with pytest.raises(RuntimeError, match="missing weights"):
        load_ifnet(d, "cpu")
    with pytest.raises(FileNotFoundError):
        load_ifnet(tmp_path / "nope", "cpu")


@pytest.mark.parametrize("h,w,scale,ph,pw", [
    (800, 1920, 1.0, 832, 1920), (800, 1920, 0.5, 896, 1920), (1080, 1920, 0.5, 1152, 1920),
    (360, 640, 1.0, 384, 640), (48, 64, 0.25, 256, 256)])
def test_padding_follows_scale(net, h, w, scale, ph, pw):
    eng = RifeEngine(net, h, w, scale, "cpu")
    assert eng.padded_shape == (ph, pw)
    assert eng.scale_list == [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
    assert eng.pad(torch.zeros(1, 3, h, w)).shape == (1, 3, ph, pw)


def test_bad_scale(net):
    with pytest.raises(ValueError):
        RifeEngine(net, 10, 10, 0.3, "cpu")


def test_interpolate_blends_and_crops(net):
    eng = RifeEngine(net, 50, 70, 1.0, "cpu")
    a, b = torch.zeros(1, 3, 50, 70), torch.ones(1, 3, 50, 70)
    out = eng.interpolate(eng.pad(a), eng.pad(b), 0.25)
    assert out.shape == (1, 3, 50, 70)
    assert torch.allclose(out, torch.full_like(out, 0.25))


def test_classify_pair(net):
    eng = RifeEngine(net, 64, 64, 1.0, "cpu", scene_ssim=0.2, dup_mad=0.002)
    g = torch.Generator().manual_seed(0)
    base = torch.rand(1, 3, 64, 64, generator=g)
    assert eng.classify_pair(base, base.clone()) == PairKind.DUPLICATE
    shifted = torch.roll(base, 1, dims=3) * 0.9 + 0.05
    assert eng.classify_pair(base, shifted) == PairKind.NORMAL
    other = torch.rand(1, 3, 64, 64, generator=g)
    assert eng.classify_pair(base, other) == PairKind.CUT
    assert eng.classify_pair(base, other, check_cut=False) == PairKind.NORMAL


def _ramp_yuv(w, h):
    y = torch.arange(w * h).remainder(220).add(16).to(torch.uint8)
    c = (w + 1) // 2 * ((h + 1) // 2)
    u = torch.arange(c).remainder(200).add(28).to(torch.uint8)
    v = torch.arange(c).flip(0).remainder(200).add(28).to(torch.uint8)
    return torch.cat([y, u, v])


@pytest.mark.parametrize("w,h", [(64, 48), (65, 47)])
@pytest.mark.parametrize("space", ["bt709", "bt470bg"])
def test_yuv_round_trip(w, h, space):
    conv = YuvConverter(w, h, ColorSpec.from_names(space), "cpu")
    assert conv.nbytes == frame_bytes(w, h)
    # Flat chroma survives 4:2:0 exactly, so luma must round-trip within one code value.
    buf = _ramp_yuv(w, h)
    buf[w * h:] = 128
    back = conv.from_rgb(conv.to_rgb(buf))
    assert (back.int() - buf.int()).abs().max() <= 1


def test_rgb_to_yuv_known_colors():
    conv = YuvConverter(2, 2, ColorSpec.from_names("bt709"), "cpu")
    white = conv.from_rgb(torch.ones(1, 3, 2, 2))
    assert white.tolist() == [235] * 4 + [128, 128]
    red = torch.zeros(1, 3, 2, 2)
    red[:, 0] = 1
    assert conv.from_rgb(red).tolist() == [63] * 4 + [102, 240]  # BT.709 limited-range red


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
@pytest.mark.parametrize("space", ["bt709", "bt470bg"])
@pytest.mark.parametrize("yuv", [(126, 100, 160), (235, 128, 128), (16, 128, 128), (81, 90, 240),
                                 (145, 54, 34), (41, 240, 110)])
def test_solid_colors_match_ffmpeg_colorspace_filter(space, yuv):
    w = h = 4
    buf = bytes([yuv[0]] * 16 + [yuv[1]] * 4 + [yuv[2]] * 4)
    vf = f"colorspace=all={space}:iall={space}:irange=tv:range=pc,format=yuv444p,scale"
    ref = subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
                          "-i", "-", "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         input=buf, capture_output=True, check=True).stdout
    conv = YuvConverter(w, h, ColorSpec.from_names(space), "cpu")
    ours = conv.to_rgb(torch.frombuffer(bytearray(buf), dtype=torch.uint8))[0, :, 0, 0] * 255
    assert all(abs(o - r) <= 1.0 for o, r in zip(ours.tolist(), ref[:3]))
