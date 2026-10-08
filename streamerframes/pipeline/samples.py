"""Real frame pairs from a video for numerical checks (ONNX/TensorRT verification)."""
from __future__ import annotations

import subprocess

import torch

from ..probe import VideoInfo


def sample_frame_pairs(info: VideoInfo, crop=None, points=(0.25, 0.5, 0.75), ffmpeg: str = "ffmpeg"
                       ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Consecutive frame pairs at fractions of the duration, cropped like generation would be.

    Returns 1x3xHxW float RGB in [0, 1] on the CPU (not padded). Unreadable points are skipped.
    """
    w, h = (crop.w, crop.h) if crop else (info.width, info.height)
    pairs = []
    for frac in points:
        at = info.start_time + info.duration * frac
        cmd = [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{at:.3f}", "-i", info.path, "-map", "0:v:0",
               "-frames:v", "2", "-fps_mode", "passthrough"]  # no duplicated first frame after the seek
        if crop:
            cmd += ["-vf", f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y}"]
        cmd += ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        out = subprocess.run(cmd, capture_output=True).stdout
        size = w * h * 3
        if len(out) < 2 * size:
            continue
        frames = [torch.frombuffer(bytearray(out[i * size:(i + 1) * size]), dtype=torch.uint8)
                  .view(h, w, 3).permute(2, 0, 1)[None].float() / 255 for i in (0, 1)]
        pairs.append((frames[0], frames[1]))
    return pairs
