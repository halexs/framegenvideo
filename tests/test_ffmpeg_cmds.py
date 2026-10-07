from fractions import Fraction as F
from pathlib import Path

from streamerframes.pipeline import ffmpeg
from streamerframes.probe import StreamInfo, VideoInfo
from streamerframes.timeline import Timeline


def _info(**kw):
    base = dict(path="in.mp4", width=1920, height=800, src_fps=F(24000, 1001), n_src=1000,
                duration=41.7, start_time=0.0, pix_fmt="yuv420p", bit_depth=8,
                color_space="bt709", color_primaries="bt709", color_transfer="bt709",
                color_range="tv")
    base.update(kw)
    return VideoInfo(**base)


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_decoder_from_start_has_no_seek():
    cmd = ffmpeg.decoder_cmd(_info())
    assert "-ss" not in cmd
    assert _arg(cmd, "-r") == "24000/1001"
    assert cmd[-1] == "pipe:1"


def test_decoder_seek_lands_a_quarter_frame_early():
    cmd = ffmpeg.decoder_cmd(_info(start_time=1.0), start_frame=480, hwaccel=False)
    assert "-hwaccel" not in cmd
    expected = 1.0 + (480 - 0.25) * 1001 / 24000
    assert abs(float(_arg(cmd, "-ss")) - expected) < 1e-6
    assert cmd.index("-ss") < cmd.index("-i")


def test_encoder_segments():
    info = _info()
    tl = Timeline.create(info.src_fps, F(60000, 1001), info.n_src)
    cmd = ffmpeg.encoder_cmd(info, tl, start_segment=3, run_id=7, out_dir=Path("c"))
    assert _arg(cmd, "-framerate") == "60000/1001"
    assert _arg(cmd, "-g") == "240"
    assert _arg(cmd, "-force_key_frames") == "expr:eq(mod(n,240),0)"
    assert _arg(cmd, "-segment_start_number") == "3"
    assert float(_arg(cmd, "-output_ts_offset")) == float(3 * 240 / tl.out_fps)
    assert Path(_arg(cmd, "-segment_list")) == Path("c/index/run_0007.csv")
    assert "-b_ref_mode" not in cmd  # unsupported on Pascal


def test_sd_color_tags():
    args = ffmpeg.color_args(_info(height=480))
    assert _arg(args, "-colorspace") == "bt470bg"


def test_audio_picks_default_stream():
    info = _info(audio=[StreamInfo(1, "ac3", "fra", False), StreamInfo(2, "aac", "eng", True)])
    assert _arg(ffmpeg.audio_cmd(info, Path("a")), "-map") == "0:a:1"


def test_export_container_and_finalize():
    pgs = _info(subtitles=[StreamInfo(3, "hdmv_pgs_subtitle", None, False)])
    assert ffmpeg.export_container(pgs) == "mkv"
    assert ffmpeg.export_container(_info()) == "mp4"
    cmd = ffmpeg.finalize_cmd(_info(), Path("list.txt"), Path("out.mp4"))
    assert "mov_text" in cmd and "+faststart" in cmd
    assert "mov_text" not in ffmpeg.finalize_cmd(_info(), Path("l.txt"), Path("out.mkv"))
