"""Optional TensorRT 8.6 backend (PLAN.md Phase 6). FP32 only (no fast FP16 on Pascal), static shapes.

Experimental and opt-in (profile ``backend = "tensorrt"``): keep it only if `trt compare` says GO
(>= 1.25x the PyTorch + CUDA graphs speed at PSNR >= 45 dB). Engines are cached per GPU, model, padded
size and scale under <cache_root>/engines/.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import torch

from .rife import RifeEngine

log = logging.getLogger(__name__)


def engine_path(cache_root, gpu: str, model: str, padded_w: int, padded_h: int, scale: float) -> Path:
    safe = re.sub(r"[^\w.-]+", "_", f"{gpu}_{model}")
    return Path(cache_root) / "engines" / f"{safe}_{padded_w}x{padded_h}_s{float(scale)}.engine"


def build_engine(onnx_path: Path, out_path: Path, workspace_gb: float = 2.0) -> Path:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 2**30))
    serialized = builder.build_serialized_network(network, config)  # FP32: no FP16 flag on Pascal
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (see the log above)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(serialized))
    log.info("built %s", out_path)
    return out_path


class TrtRifeEngine(RifeEngine):
    """RifeEngine whose network call runs a serialized TensorRT engine on torch tensors' memory.

    Padding, scene-cut and duplicate detection are inherited (they still use the PyTorch net's helpers).
    """

    def __init__(self, net, height, width, scale, device, engine_file: Path, **kw):
        super().__init__(net, height, width, scale, device, **kw)
        import tensorrt as trt

        self._logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self._logger)
        self._engine = runtime.deserialize_cuda_engine(Path(engine_file).read_bytes())
        if self._engine is None:
            raise RuntimeError(f"could not load TensorRT engine {engine_file}")
        self._context = self._engine.create_execution_context()
        self._out = torch.empty(1, 3, self.ph, self.pw, device=self.device)

    def enable_cuda_graph(self, tolerance: float = 1e-5) -> bool:
        return False  # TensorRT already runs as one fused engine

    @torch.inference_mode()
    def run_padded(self, img0p, img1p, t):
        img0p, img1p, t = img0p.contiguous(), img1p.contiguous(), t.contiguous()
        for name, tensor in (("img0", img0p), ("img1", img1p), ("timestep", t), ("output", self._out)):
            self._context.set_tensor_address(name, tensor.data_ptr())
        if not self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self._out


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()
