"""ONNX export of IFNet for TensorRT (PLAN.md Phase 6 step 2; replaces the broken export_onnx.py).

Inputs (img0, img1, timestep[1,1,1,1]) at one static padded size, the full scale_list for the chosen
scale (5 entries for v4.25), output merged[-1] as a tensor, opset 17 (GridSample is supported by TRT 8.6).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from .loader import count_blocks

log = logging.getLogger(__name__)


class IFNetForExport(nn.Module):
    def __init__(self, net: nn.Module, scale: float):
        super().__init__()
        self.net = net
        blocks = count_blocks(net) or 5
        self.scale_list = [2 ** k / scale for k in reversed(range(blocks))]

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        _, _, merged = self.net(torch.cat((img0, img1), 1), timestep, self.scale_list)
        return merged[-1]


def export_onnx(net: nn.Module, padded_h: int, padded_w: int, scale: float, path: Path,
                opset: int = 17) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = IFNetForExport(net, scale).eval()
    device = next(net.parameters()).device
    img0 = torch.rand(1, 3, padded_h, padded_w, device=device)
    img1 = torch.rand(1, 3, padded_h, padded_w, device=device)
    t = torch.full((1, 1, 1, 1), 0.5, device=device)
    with torch.inference_mode(False), torch.no_grad():
        torch.onnx.export(wrapper, (img0, img1, t), str(path), export_params=True, opset_version=opset,
                          do_constant_folding=True, input_names=["img0", "img1", "timestep"],
                          output_names=["output"], dynamo=False)
    log.info("exported %s (%dx%d, scale %s)", path, padded_w, padded_h, scale)
    return path


def error_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a.float() - b.float()).abs().flatten()
    k = max(1, int(0.999 * diff.numel()))
    mse = float((diff ** 2).mean())
    return {"max": float(diff.max()), "mean": float(diff.mean()), "p999": float(diff.kthvalue(k).values),
            "psnr": float("inf") if mse == 0 else float(10 * torch.log10(torch.tensor(1.0 / mse)))}


def stats_ok(stats: dict) -> bool:
    """Equivalent for video purposes: tiny average error and 99.9% of values within 4 of 255 levels.

    A single worst pixel isn't a fair test: CPU and GPU kernels round differently, and occlusion edges in the
    flow can turn that into a few large but invisible outliers.
    """
    return stats["mean"] < 1e-3 and stats["p999"] < 4 / 255


def verify_onnx(net: nn.Module, path: Path, padded_h: int, padded_w: int, scale: float,
                pairs: list[tuple[torch.Tensor, torch.Tensor]] | None = None) -> dict:
    """Compare ONNX Runtime (CPU) with PyTorch on image-like inputs (real frames when given).

    ``pairs`` are padded 1x3xHxW float tensors in [0, 1]. Returns the worst stats over pairs and timesteps.
    """
    import onnxruntime as ort

    from .rife import smooth_pair

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    wrapper = IFNetForExport(net, scale).eval()
    device = next(net.parameters()).device
    pairs = pairs or [smooth_pair(padded_h, padded_w)]
    worst: dict = {}
    for img0, img1 in pairs:
        img0, img1 = img0.cpu().contiguous(), img1.cpu().contiguous()
        for tv in (0.25, 0.5):
            t = torch.full((1, 1, 1, 1), tv)
            with torch.no_grad():
                ref = wrapper(img0.to(device), img1.to(device), t.to(device)).cpu()
            out = session.run(None, {"img0": img0.numpy(), "img1": img1.numpy(), "timestep": t.numpy()})[0]
            stats = error_stats(torch.from_numpy(out), ref)
            if not worst or stats["mean"] > worst["mean"]:
                worst = stats
    return worst
