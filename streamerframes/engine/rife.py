"""RifeEngine: padding, timestep inference, scene-cut and duplicate detection (PLAN.md Phase 1)."""
from __future__ import annotations

import enum

import torch
import torch.nn.functional as F

from .loader import count_blocks


class PairKind(enum.Enum):
    NORMAL = "normal"
    CUT = "cut"          # hard scene cut: show the nearer source frame, never a blend
    DUPLICATE = "dup"    # identical frames: copy, skip inference


def _ssim_matlab():
    # Imported lazily: upstream module needs the repo root on sys.path (the loader adds it).
    from model.pytorch_msssim import ssim_matlab
    return ssim_matlab


class RifeEngine:
    """Wraps an upstream IFNet for one fixed frame size.

    Inputs and outputs are 1x3xHxW float32 RGB in [0, 1] already on ``device``.
    """

    def __init__(self, net: torch.nn.Module, height: int, width: int, scale: float = 1.0,
                 device: torch.device | str = "cuda", scene_ssim: float = 0.2, dup_mad: float = 0.0):
        if scale <= 0 or (64 / scale) != int(64 / scale):
            raise ValueError(f"scale must divide 64 evenly, got {scale}")
        self.net = net
        self.h, self.w = height, width
        self.scale = scale
        self.device = torch.device(device)
        align = int(64 / scale)
        self.ph = -(-height // align) * align
        self.pw = -(-width // align) * align
        blocks = count_blocks(net) or 5
        self.scale_list = [2 ** k / scale for k in reversed(range(blocks))]
        self.scene_ssim = scene_ssim
        self.dup_mad = dup_mad
        self._t = torch.zeros(1, 1, 1, 1, device=self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    @property
    def padded_shape(self) -> tuple[int, int]:
        return self.ph, self.pw

    def pad(self, img: torch.Tensor) -> torch.Tensor:
        if (self.ph, self.pw) == (self.h, self.w):
            return img
        return F.pad(img, (0, self.pw - self.w, 0, self.ph - self.h), mode="replicate")

    @torch.inference_mode()
    def run_padded(self, img0p: torch.Tensor, img1p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Raw network call on padded inputs; ``t`` is a (1,1,1,1) tensor. Returns padded output."""
        _, _, merged = self.net(torch.cat((img0p, img1p), 1), t, self.scale_list)
        return merged[-1]

    @torch.inference_mode()
    def interpolate(self, img0p: torch.Tensor, img1p: torch.Tensor, t: float) -> torch.Tensor:
        """Frame at time ``t`` in (0, 1) between padded inputs. Returns the clamped, unpadded result."""
        self._t.fill_(float(t))
        out = self.run_padded(img0p, img1p, self._t)
        return out[:, :, : self.h, : self.w].clamp(0, 1)

    @torch.inference_mode()
    def classify_pair(self, img0: torch.Tensor, img1: torch.Tensor, check_cut: bool = True) -> PairKind:
        """One small host sync per source pair. Inputs may be padded or not."""
        img0, img1 = img0[:, :3, : self.h, : self.w], img1[:, :3, : self.h, : self.w]
        if self.dup_mad > 0:
            a = F.interpolate(img0, scale_factor=0.125, mode="area")
            b = F.interpolate(img1, scale_factor=0.125, mode="area")
            if float((a - b).abs().mean()) < self.dup_mad:
                return PairKind.DUPLICATE
        if check_cut:
            a = F.interpolate(img0, size=(32, 32), mode="bilinear", align_corners=False)
            b = F.interpolate(img1, size=(32, 32), mode="bilinear", align_corners=False)
            if float(_ssim_matlab()(a, b)) < self.scene_ssim:
                return PairKind.CUT
        return PairKind.NORMAL
