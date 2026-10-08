"""Letterbox detection (PLAN.md Phase 2 item 4): sample cropdetect across the file, infer only the picture."""
from __future__ import annotations

import logging
import re
import subprocess

from ..engine.color import Crop
from ..probe import VideoInfo

log = logging.getLogger(__name__)
CROP_RE = re.compile(r"crop=(-?\d+):(-?\d+):(-?\d+):(-?\d+)")


def parse_cropdetect(stderr: str) -> tuple[int, int, int, int] | None:
    found = CROP_RE.findall(stderr)
    if not found:
        return None
    w, h, x, y = map(int, found[-1])
    return (w, h, x, y) if w > 0 and h > 0 and x >= 0 and y >= 0 else None


def default_min_bars(height: int) -> int:
    """32 px for real video; proportionally less for tiny frames."""
    return min(32, max(2, height // 6))


def union_crop(samples: list[tuple[int, int, int, int]], width: int, height: int,
               min_bars: int | None = None) -> Crop | None:
    """Smallest rectangle containing every sample's picture area, if it still removes >= min_bars px."""
    min_bars = default_min_bars(height) if min_bars is None else min_bars
    if not samples:
        return None
    x0 = min(s[2] for s in samples)
    y0 = min(s[3] for s in samples)
    x1 = max(s[2] + s[0] for s in samples)
    y1 = max(s[3] + s[1] for s in samples)
    x0, y0 = x0 - x0 % 2, y0 - y0 % 2
    x1, y1 = min(x1 + x1 % 2, width - width % 2), min(y1 + y1 % 2, height - height % 2)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0 or (width - w) + (height - h) < min_bars:
        return None
    return Crop(x0, y0, w, h)


def detect_letterbox(info: VideoInfo, points: int = 8, ffmpeg: str = "ffmpeg",
                     min_bars: int | None = None) -> Crop | None:
    if info.width % 2 or info.height % 2 or info.duration <= 0:
        return None
    samples = []
    for k in range(points):
        at = info.start_time + info.duration * (k + 0.5) / points
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-ss", f"{at:.3f}", "-i", info.path, "-map", "0:v:0",
               "-vf", "cropdetect=limit=24:round=2:reset=0", "-frames:v", "60", "-f", "null", "-"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            continue
        sample = parse_cropdetect(out.stderr)
        if sample:
            samples.append(sample)
    if len(samples) < max(2, points // 2):
        return None  # too few readable samples to trust
    crop = union_crop(samples, info.width, info.height, min_bars)
    if crop:
        log.info("letterbox: inferring %dx%d at (%d,%d) of %dx%d", crop.w, crop.h, crop.x, crop.y,
                 info.width, info.height)
    return crop
