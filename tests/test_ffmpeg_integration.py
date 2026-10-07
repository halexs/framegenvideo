"""Run the command builders against real ffmpeg (CPU only). Skipped when ffmpeg is missing."""
import csv
import json
import shutil
import subprocess
from fractions import Fraction as F

import pytest

from streamerframes.pipeline import ffmpeg
from streamerframes.probe import probe
from streamerframes.timeline import Timeline

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                reason="ffmpeg/ffprobe not installed")

W, H, N_SRC = 64, 48, 120


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate=24000/1001",
         "-frames:v", str(N_SRC), "-c:v", "libx264", "-g", "48", "-pix_fmt", "yuv420p", str(path)],
        check=True)
    return probe(path)


def _decode(info, start):
    out = subprocess.run(ffmpeg.decoder_cmd(info, start, hwaccel=False), capture_output=True, check=True)
    size = W * H * 3 // 2
    assert len(out.stdout) % size == 0
    return [out.stdout[i:i + size] for i in range(0, len(out.stdout), size)]


def test_probe_real_file(clip):
    assert clip.src_fps == F(24000, 1001)
    assert clip.n_src == N_SRC
    assert (clip.width, clip.height) == (W, H)


@pytest.mark.parametrize("start", [1, 47, 48, 49, 100])
def test_decoder_seek_is_frame_exact(clip, start):
    full = _decode(clip, 0)
    assert len(full) == N_SRC
    seeked = _decode(clip, start)
    assert seeked[0] == full[start]
    assert seeked == full[start:]


def test_encoder_segments_have_exact_length_and_start_with_keyframes(clip, tmp_path):
    # Short segments keep the test fast: 1 s target at 2x -> 48 frames.
    tl = Timeline.create(clip.src_fps, clip.src_fps * 2, clip.n_src, seg_target_seconds=1)
    (tmp_path / "segments").mkdir()
    (tmp_path / "index").mkdir()
    frames = _decode(clip, 0)
    cmd = ffmpeg.encoder_cmd(clip, tl, 0, 0, tmp_path, encoder="lossless")
    # No RIFE here: emit the nearest-earlier source frame for every output frame.
    payload = b"".join(frames[tl.source_position(j)[0]] for j in range(tl.total_out_frames))
    subprocess.run(cmd, input=payload, check=True, capture_output=True)

    rows = list(csv.reader(open(tmp_path / "index" / "run_0000.csv")))
    assert len(rows) == tl.total_segments
    for k, row in enumerate(rows):
        seg = tmp_path / "segments" / row[0]
        assert seg.name == f"v_{k:05d}.ts"
        probe_out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=key_frame",
             "-of", "json", str(seg)], capture_output=True, text=True, check=True)
        keys = [f["key_frame"] for f in json.loads(probe_out.stdout)["frames"]]
        start, end = tl.segment_range(k)
        assert len(keys) == end - start
        assert keys[0] == 1
