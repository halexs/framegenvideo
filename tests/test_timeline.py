from fractions import Fraction as F

import pytest

from streamerframes.timeline import (
    Timeline,
    out_fps_for_multiplier,
    out_fps_for_target,
    scene_cut_frame,
)

NTSC_FILM = F(24000, 1001)

CASES = [
    # (src_fps, out_fps, expected period_out/period_src, expected seg_frames)
    (NTSC_FILM, NTSC_FILM * 2, (2, 1), 192),
    (NTSC_FILM, F(60000, 1001), (5, 2), 240),
    (F(30000, 1001), F(60000, 1001), (2, 1), 240),
    (F(25), F(60), (12, 5), 240),
    (F(30), F(60), (2, 1), 240),
    (F(24), F(60), (5, 2), 240),
]


def test_out_fps_helpers():
    assert out_fps_for_target(NTSC_FILM, 60) == F(60000, 1001)
    assert out_fps_for_target(F(25), 60) == 60
    assert out_fps_for_multiplier(NTSC_FILM, 2) == F(48000, 1001)
    with pytest.raises(ValueError):
        out_fps_for_target(F(60), 60)
    with pytest.raises(ValueError):
        out_fps_for_multiplier(F(30), 1)


@pytest.mark.parametrize("src,out,period,seg", CASES)
def test_period_and_segment_length(src, out, period, seg):
    tl = Timeline.create(src, out, n_src=1000)
    assert (tl.period_out, tl.period_src) == period
    assert tl.seg_frames == seg
    assert tl.seg_frames % tl.period_out == 0


@pytest.mark.parametrize("src,out,period,seg", CASES)
@pytest.mark.parametrize("n_src", [1, 2, 7, 999, 1000, 1003])
def test_segments_cover_every_frame_once_and_start_on_source_frames(src, out, period, seg, n_src):
    tl = Timeline.create(src, out, n_src=n_src)
    covered = []
    for k in range(tl.total_segments):
        start, end = tl.segment_range(k)
        assert 0 < end - start <= tl.seg_frames
        covered.extend(range(start, end))
        s_k = tl.segment_start_source_frame(k)
        assert tl.source_position(start) == (min(s_k, n_src - 1), 0)
        assert tl.segment_start_time(k) == start / tl.out_fps
    assert covered == list(range(tl.total_out_frames))


@pytest.mark.parametrize("src,out,period,seg", CASES)
def test_source_positions_are_monotonic_and_in_range(src, out, period, seg):
    n_src = 500
    tl = Timeline.create(src, out, n_src=n_src)
    # Output is at least as long as the source.
    assert tl.total_out_frames / tl.out_fps >= n_src / tl.src_fps
    assert (tl.total_out_frames - 1) / tl.out_fps < n_src / tl.src_fps
    prev = F(-1)
    for j in range(tl.total_out_frames):
        i, t = tl.source_position(j)
        assert 0 <= t < 1
        assert 0 <= i <= n_src - 1
        if t:
            assert i + 1 <= n_src - 1  # blending needs a next frame
        pos = i + t
        assert pos >= prev
        prev = pos


def test_2x_alternates_source_and_midpoint():
    tl = Timeline.create(NTSC_FILM, NTSC_FILM * 2, n_src=4)
    assert [tl.source_position(j) for j in range(tl.total_out_frames)] == [
        (0, 0), (0, F(1, 2)), (1, 0), (1, F(1, 2)), (2, 0), (2, F(1, 2)), (3, 0), (3, 0)]


def test_23976_to_5994_pattern():
    tl = Timeline.create(NTSC_FILM, F(60000, 1001), n_src=100)
    assert [tl.source_position(j) for j in range(6)] == [
        (0, 0), (0, F(2, 5)), (0, F(4, 5)), (1, F(1, 5)), (1, F(3, 5)), (2, 0)]


def test_batman_begins_numbers():
    # From PLAN.md: 201,471 frames at 24000/1001 fps.
    tl = Timeline.create(NTSC_FILM, F(60000, 1001), n_src=201471)
    assert tl.total_out_frames == 503678
    assert tl.seg_frames == 240
    assert tl.total_segments == 2099
    assert tl.segment_duration(tl.total_segments - 1) == F(503678 - 2098 * 240) / F(60000, 1001)


def test_segment_for_time():
    tl = Timeline.create(NTSC_FILM, F(60000, 1001), n_src=10000)
    assert tl.segment_for_time(0) == 0
    assert tl.segment_for_time(tl.segment_start_time(5)) == 5
    assert tl.segment_for_time(tl.segment_start_time(5) - F(1, 1000)) == 4
    assert tl.segment_for_time(1e9) == tl.total_segments - 1


def test_bad_indices():
    tl = Timeline.create(F(30), F(60), n_src=10)
    with pytest.raises(IndexError):
        tl.source_position(tl.total_out_frames)
    with pytest.raises(IndexError):
        tl.segment_range(tl.total_segments)


def test_scene_cut_frame():
    assert scene_cut_frame(3, F(2, 5)) == 3
    assert scene_cut_frame(3, F(1, 2)) == 4
