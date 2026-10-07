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


def verify_onnx(net: nn.Module, path: Path, padded_h: int, padded_w: int, scale: float,
                trials: int = 2) -> float:
    """Max abs difference between ONNX Runtime (CPU) and PyTorch on random inputs. Plan: < 1e-3."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    wrapper = IFNetForExport(net, scale).eval()
    device = next(net.parameters()).device
    worst = 0.0
    g = torch.Generator().manual_seed(0)
    for k in range(trials):
        img0 = torch.rand(1, 3, padded_h, padded_w, generator=g)
        img1 = torch.rand(1, 3, padded_h, padded_w, generator=g)
        t = torch.full((1, 1, 1, 1), (k + 1) / (trials + 1))
        with torch.no_grad():
            ref = wrapper(img0.to(device), img1.to(device), t.to(device)).cpu()
        out = session.run(None, {"img0": img0.numpy(), "img1": img1.numpy(), "timestep": t.numpy()})[0]
        worst = max(worst, float((torch.from_numpy(out) - ref).abs().max()))
    return worst
