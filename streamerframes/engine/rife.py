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


def graph_matches(graph_out: torch.Tensor, eager_a: torch.Tensor, eager_b: torch.Tensor,
                  floor: float = 1e-3) -> tuple[bool, float, float]:
    """Graph output counts as equal to eager if it is as close as two eager runs are to each other
    (cudnn may pick different kernels run to run), with an absolute floor well below one 8-bit step."""
    err = float((graph_out - eager_a).abs().max())
    baseline = float((eager_b - eager_a).abs().max())
    return err <= max(floor, 2 * baseline), err, baseline


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
        self._graph = None
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def _smooth_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Image-like test inputs: smooth texture and a slightly shifted copy.

        Random noise makes optical flow chaotic, so rounding differences between kernel choices blow up
        into large output differences; real frames behave like this pair, not like noise.
        """
        g = torch.Generator(device="cpu").manual_seed(0)
        low = torch.rand(1, 3, max(2, self.ph // 32), max(2, self.pw // 32), generator=g)
        img0 = F.interpolate(low, size=(self.ph, self.pw), mode="bicubic", align_corners=False).clamp(0, 1)
        img1 = torch.roll(img0, shifts=(3, 5), dims=(2, 3))
        return img0.to(self.device), img1.to(self.device)

    @torch.inference_mode()
    def enable_cuda_graph(self) -> bool:
        """Capture the network for the fixed padded shape. Keeps it only if it matches eager output."""
        if self.device.type != "cuda":
            return False
        import logging
        log = logging.getLogger(__name__)
        try:
            self._g_in0, self._g_in1 = self._smooth_pair()
            self._g_t = torch.full((1, 1, 1, 1), 0.5, device=self.device)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):  # warm-up: cudnn autotune, warplayer grid cache
                    self.run_padded(self._g_in0, self._g_in1, self._g_t)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._g_out = self.run_padded(self._g_in0, self._g_in1, self._g_t)
            eager_a = self.run_padded(self._g_in0, self._g_in1, self._g_t).clone()
            eager_b = self.run_padded(self._g_in0, self._g_in1, self._g_t).clone()
            graph.replay()
            ok, err, baseline = graph_matches(self._g_out, eager_a, eager_b)
            if not ok:
                log.warning("CUDA graph output differs from eager by %.2e (eager vs eager: %.2e); using eager",
                            err, baseline)
                return False
            self._graph = graph
            log.info("CUDA graph captured for %dx%d (max diff %.1e)", self.pw, self.ph, err)
            return True
        except Exception as exc:  # noqa: BLE001 - graphs are an optimisation only
            log.warning("CUDA graph capture failed (%s); using eager", exc)
            self._graph = None
            return False

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
        if self._graph is not None:
            self._g_in0.copy_(img0p)
            self._g_in1.copy_(img1p)
            self._g_t.fill_(float(t))
            self._graph.replay()
            out = self._g_out
        else:
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
