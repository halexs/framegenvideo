"""Output-frame schedule and segment <-> frame mapping (PLAN.md section 4).

Pure Python with exact ``Fraction`` arithmetic. Must not import torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction


def out_fps_for_multiplier(src_fps: Fraction, multi: int | Fraction) -> Fraction:
    out = Fraction(src_fps) * Fraction(multi)
    if out <= src_fps:
        raise ValueError(f"multiplier {multi} does not raise the frame rate")
    return out


def out_fps_for_target(src_fps: Fraction, target: int) -> Fraction:
    """60 -> 60000/1001 for NTSC-rate sources (x/1001), else exactly 60."""
    src_fps = Fraction(src_fps)
    out = Fraction(target * 1000, 1001) if src_fps.denominator == 1001 else Fraction(target)
    if out <= src_fps:
        raise ValueError(f"target {target} fps is not above source {float(src_fps):.3f} fps")
    return out


@dataclass(frozen=True)
class Timeline:
    src_fps: Fraction
    out_fps: Fraction
    n_src: int
    seg_frames: int

    @classmethod
    def create(cls, src_fps: Fraction, out_fps: Fraction, n_src: int,
               seg_target_seconds: Fraction | float = 4) -> "Timeline":
        src_fps, out_fps = Fraction(src_fps), Fraction(out_fps)
        if src_fps <= 0 or out_fps <= src_fps:
            raise ValueError("out_fps must be greater than src_fps > 0")
        if n_src < 1:
            raise ValueError("n_src must be >= 1")
        period = (out_fps / src_fps).numerator
        target = Fraction(seg_target_seconds) * out_fps
        seg_frames = max(1, round(target / period)) * period
        return cls(src_fps, out_fps, n_src, seg_frames)

    @property
    def ratio(self) -> Fraction:
        """Source frames advanced per output frame."""
        return self.src_fps / self.out_fps

    @property
    def period_out(self) -> int:
        return (self.out_fps / self.src_fps).numerator

    @property
    def period_src(self) -> int:
        return (self.out_fps / self.src_fps).denominator

    @property
    def total_out_frames(self) -> int:
        return math.ceil(self.n_src * self.out_fps / self.src_fps)

    @property
    def total_segments(self) -> int:
        return -(-self.total_out_frames // self.seg_frames)

    def source_position(self, j: int) -> tuple[int, Fraction]:
        """Map output frame ``j`` to ``(i, t)``: blend source frames i and i+1 at time t in [0, 1).

        Past the last source frame the final frame is repeated (t == 0).
        """
        if not 0 <= j < self.total_out_frames:
            raise IndexError(j)
        return self.position_for(j, self.n_src)

    def position_for(self, j: int, n_src: int) -> tuple[int, Fraction]:
        """``source_position`` against a frame count only known at decode time (probe counts are estimates)."""
        p = j * self.ratio
        if p >= n_src - 1:
            return n_src - 1, Fraction(0)
        i = math.floor(p)
        return i, p - i

    def total_out_frames_for(self, n_src: int) -> int:
        return math.ceil(n_src * self.out_fps / self.src_fps)

    def with_n_src(self, n_src: int) -> "Timeline":
        return Timeline(self.src_fps, self.out_fps, n_src, self.seg_frames)

    def segment_range(self, k: int) -> tuple[int, int]:
        """Output frames ``[start, end)`` of segment ``k``; the last segment may be short."""
        if not 0 <= k < self.total_segments:
            raise IndexError(k)
        start = k * self.seg_frames
        return start, min(start + self.seg_frames, self.total_out_frames)

    def segment_start_source_frame(self, k: int) -> int:
        s = k * self.seg_frames * self.ratio
        assert s.denominator == 1, "segments must start on a source frame"
        return int(s)

    def segment_start_time(self, k: int) -> Fraction:
        return k * self.seg_frames / self.out_fps

    def segment_duration(self, k: int) -> Fraction:
        start, end = self.segment_range(k)
        return (end - start) / self.out_fps

    def segment_for_time(self, seconds: Fraction | float) -> int:
        k = math.floor(Fraction(seconds) * self.out_fps / self.seg_frames)
        return min(max(k, 0), self.total_segments - 1)


def scene_cut_frame(i: int, t: Fraction) -> int:
    """At a hard cut between i and i+1, show the nearer source frame instead of a blend."""
    return i if t < Fraction(1, 2) else i + 1
