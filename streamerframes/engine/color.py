"""Planar YUV420 (uint8, as ffmpeg -pix_fmt yuv420p) <-> RGB float on the GPU (PLAN.md Phase 1)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Luma coefficients (Kr, Kb) by ffprobe color_space name.
_MATRICES = {
    "bt709": (0.2126, 0.0722),
    "bt470bg": (0.299, 0.114),
    "smpte170m": (0.299, 0.114),
    "bt601": (0.299, 0.114),
    "fcc": (0.30, 0.11),
    "bt2020nc": (0.2627, 0.0593),
    "bt2020c": (0.2627, 0.0593),
}


from ..pipeline.ffmpeg import frame_bytes  # noqa: E402,F401  (re-exported)


@dataclass(frozen=True)
class ColorSpec:
    kr: float
    kb: float
    full_range: bool

    @classmethod
    def from_names(cls, color_space: str, color_range: str = "tv", height: int = 1080) -> "ColorSpec":
        default = "bt709" if height >= 720 else "bt470bg"
        kr, kb = _MATRICES.get(color_space or default, _MATRICES[default])
        return cls(kr, kb, color_range in ("pc", "jpeg", "full"))

    @property
    def kg(self) -> float:
        return 1.0 - self.kr - self.kb


class YuvConverter:
    """Converts between packed planar yuv420p buffers and 1x3xHxW float RGB in [0, 1]."""

    def __init__(self, width: int, height: int, spec: ColorSpec, device: torch.device | str = "cuda"):
        self.w, self.h = width, height
        self.cw, self.ch = (width + 1) // 2, (height + 1) // 2
        self.spec = spec
        self.device = torch.device(device)
        self.nbytes = frame_bytes(width, height)
        self._y_end = width * height
        self._u_end = self._y_end + self.cw * self.ch
        if spec.full_range:
            self._y_off, self._y_scale, self._c_scale = 0.0, 255.0, 255.0
        else:
            self._y_off, self._y_scale, self._c_scale = 16.0, 219.0, 224.0

    def planes(self, buf: torch.Tensor):
        """Y, U, V views (1x1xHxW, 1x1xCHxCW) of a packed yuv420p buffer."""
        flat = buf.reshape(-1)
        y = flat[: self._y_end].view(1, 1, self.h, self.w)
        u = flat[self._y_end: self._u_end].view(1, 1, self.ch, self.cw)
        v = flat[self._u_end: self.nbytes].view(1, 1, self.ch, self.cw)
        return y, u, v

    def to_rgb(self, buf: torch.Tensor) -> torch.Tensor:
        """``buf``: uint8 tensor of ``nbytes`` on ``device``. Returns 1x3xHxW float32."""
        return self.planes_to_rgb(*self.planes(buf))

    def planes_to_rgb(self, y: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        s = self.spec
        h, w = y.shape[-2:]
        y = (y.float() - self._y_off) / self._y_scale
        uv = torch.cat([u, v], 1).float()
        uv = F.interpolate(uv, size=(h, w), mode="bilinear", align_corners=False)
        uv = (uv - 128.0) / self._c_scale
        cb, cr = uv[:, :1], uv[:, 1:]
        r = y + 2 * (1 - s.kr) * cr
        b = y + 2 * (1 - s.kb) * cb
        g = (y - s.kr * r - s.kb * b) / s.kg
        return torch.cat([r, g, b], 1).clamp_(0, 1)

    def rgb_to_planes(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """1x3xHxW float RGB -> (Y 1x1xHxW uint8, UV 1x2xCHxCW uint8)."""
        s = self.spec
        r, g, b = rgb[:, :1], rgb[:, 1:2], rgb[:, 2:3]
        y = s.kr * r + s.kg * g + s.kb * b
        cb = (b - y) / (2 * (1 - s.kb))
        cr = (r - y) / (2 * (1 - s.kr))
        c = F.avg_pool2d(torch.cat([cb, cr], 1), 2, ceil_mode=True)
        y8 = (y * self._y_scale + self._y_off).round_().clamp_(0, 255).to(torch.uint8)
        c8 = (c * self._c_scale + 128.0).round_().clamp_(0, 255).to(torch.uint8)
        return y8, c8

    def from_rgb(self, rgb: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """1x3xHxW float RGB -> packed yuv420p uint8 (written into ``out`` if given)."""
        y8, c8 = self.rgb_to_planes(rgb)
        if out is None:
            out = torch.empty(self.nbytes, dtype=torch.uint8, device=rgb.device)
        y, u, v = self.planes(out)
        y.copy_(y8)
        u.copy_(c8[:, :1])
        v.copy_(c8[:, 1:])
        return out


@dataclass(frozen=True)
class Crop:
    """Active picture area inside a letterboxed frame. Offsets and sizes are even (yuv420 chroma)."""
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self):
        if any(v % 2 for v in (self.x, self.y, self.w, self.h)) or self.w <= 0 or self.h <= 0:
            raise ValueError(f"crop must be positive and even: {self}")


class CroppedConverter:
    """Converts only the active area of each frame; bars are taken from a source frame untouched."""

    def __init__(self, full: YuvConverter, crop: Crop | None):
        self.full = full
        self.crop = crop
        self.exact = crop is None
        self.inner = full if crop is None else YuvConverter(crop.w, crop.h, full.spec, full.device)
        self.device = full.device
        self.nbytes = full.nbytes

    @property
    def size(self) -> tuple[int, int]:
        """(height, width) of what the engine sees."""
        return (self.full.h, self.full.w) if self.exact else (self.crop.h, self.crop.w)

    def _crop_planes(self, buf: torch.Tensor):
        y, u, v = self.full.planes(buf)
        c = self.crop
        cy, cx, ch, cw = c.y // 2, c.x // 2, c.h // 2, c.w // 2
        return (y[..., c.y:c.y + c.h, c.x:c.x + c.w], u[..., cy:cy + ch, cx:cx + cw],
                v[..., cy:cy + ch, cx:cx + cw])

    def to_rgb(self, buf: torch.Tensor) -> torch.Tensor:
        if self.exact:
            return self.full.to_rgb(buf)
        return self.inner.planes_to_rgb(*self._crop_planes(buf))

    def from_rgb(self, rgb: torch.Tensor, background: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Write ``rgb`` (engine-sized) into ``out``; outside the crop, ``out`` is a copy of ``background``."""
        if self.exact:
            return self.full.from_rgb(rgb, out)
        out.copy_(background)
        y8, c8 = self.inner.rgb_to_planes(rgb)
        y, u, v = self._crop_planes(out)
        y.copy_(y8)
        u.copy_(c8[:, :1])
        v.copy_(c8[:, 1:])
        return out
