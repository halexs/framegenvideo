from fractions import Fraction as F

import pytest

from streamerframes.probe import parse_ffprobe, parse_rate


def _data(**video):
    v = {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1920, "height": 800,
         "pix_fmt": "yuv420p", "r_frame_rate": "24000/1001", "avg_frame_rate": "24000/1001",
         "nb_frames": "201471", "duration": "8405.396000", "start_time": "0.000000"}
    v.update(video)
    return {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "mjpeg", "width": 1, "height": 1,
             "disposition": {"attached_pic": 1}},
            v,
            {"index": 2, "codec_type": "audio", "codec_name": "aac", "tags": {"language": "eng"},
             "disposition": {"default": 1}},
            {"index": 3, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
        ],
        "format": {"duration": "8405.4"},
    }


def test_parse_rate():
    assert parse_rate("24000/1001") == F(24000, 1001)
    assert parse_rate("0/0") is None
    assert parse_rate(None) is None


def test_parse_typical_film():
    info = parse_ffprobe(_data(), "movie.mkv")
    assert info.width == 1920 and info.height == 800  # skipped the cover-art stream
    assert info.src_fps == F(24000, 1001)
    assert info.n_src == 201471
    assert info.color_space == "bt709" and info.color_range == "tv"
    assert info.audio[0].language == "eng" and info.audio[0].default
    assert info.has_bitmap_subtitles
    assert info.warnings == []
    assert info.to_json()["src_fps"] == "24000/1001"


def test_implausible_nb_frames_falls_back_to_duration():
    info = parse_ffprobe(_data(nb_frames="5", duration="10.010"))
    assert info.n_src == 240


def test_sd_defaults_and_warnings():
    info = parse_ffprobe(_data(height=480, width=720, sample_aspect_ratio="32:27",
                               pix_fmt="yuv420p10le", avg_frame_rate="20/1",
                               side_data_list=[{"rotation": -90}]))
    assert info.color_space == "bt470bg"
    assert info.bit_depth == 10
    assert info.rotation == 270
    assert info.sar == F(32, 27)
    assert len(info.warnings) == 4


def test_no_video_stream():
    with pytest.raises(ValueError):
        parse_ffprobe({"streams": [{"codec_type": "audio"}]})
