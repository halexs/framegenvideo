"""Model benchmark + calibration table used by scale=auto (PLAN.md Phase 2 item 5)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from .cache.store import atomic_write_json, read_json

log = logging.getLogger(__name__)
SCALES = (1.0, 0.5)


def calibration_path(cache_root: str | Path) -> Path:
    return Path(cache_root) / "calibration.json"


def entry_key(gpu: str, model: str, width: int, height: int, scale: float, graphs: bool) -> str:
    return f"{gpu}|{model}|{width}x{height}|s{float(scale)}|g{int(bool(graphs))}"


def load_calibration(cache_root) -> dict:
    return (read_json(calibration_path(cache_root)) or {}).get("entries", {})


def save_entries(cache_root, entries: dict) -> None:
    path = calibration_path(cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_json(path) or {"entries": {}}
    data["entries"].update(entries)
    atomic_write_json(path, data)


def estimate_ms(entries: dict, model: str, width: int, height: int, scale: float, graphs: bool,
                gpu: str | None = None) -> float | None:
    """Exact entry if present, else the nearest measured size scaled by pixel count."""
    best = None
    for key, value in entries.items():
        g, m, size, s, gr = key.split("|")
        if m != model or s != f"s{float(scale)}" or gr != f"g{int(bool(graphs))}" or (gpu and g != gpu):
            continue
        w, h = map(int, size.split("x"))
        ms = float(value["ms"]) * (width * height) / (w * h)
        dist = abs(w * h - width * height)
        if best is None or dist < best[0]:
            best = (dist, ms)
    return best[1] if best else None


def choose_scale(entries: dict, model: str, width: int, height: int, interps_per_sec: float, graphs: bool,
                 gpu: str | None = None, margin: float = 1.1) -> float | None:
    """Highest-quality scale fast enough for real time with ``margin``, else the fastest measured one."""
    measured = {s: estimate_ms(entries, model, width, height, s, graphs, gpu) for s in SCALES}
    measured = {s: ms for s, ms in measured.items() if ms}
    if not measured:
        return None
    for s in sorted(measured, reverse=True):
        if 1000.0 / measured[s] >= margin * interps_per_sec:
            return s
    return min(measured, key=measured.get)


def bench_engine(engine, warmup: int = 5, iters: int = 30) -> float:
    import torch

    shape = (1, 3, *engine.padded_shape)
    a = torch.rand(shape, device=engine.device)
    b = torch.rand(shape, device=engine.device)
    sync = torch.cuda.synchronize if engine.device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        engine.interpolate(a, b, 0.5)
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        engine.interpolate(a, b, 0.5)
    sync()
    return (time.perf_counter() - start) * 1000.0 / iters


def gpu_name(device: str) -> str:
    if str(device).startswith("cuda"):
        import torch
        return torch.cuda.get_device_name(0)
    return "cpu"


def run_bench(model_dir: Path, model: str, width: int, height: int, device: str = "cuda",
              scales=SCALES, graphs_options=(False, True), warmup: int = 5, iters: int = 30,
              net=None) -> dict:
    from .engine.loader import load_ifnet
    from .engine.rife import RifeEngine

    net = net or load_ifnet(model_dir, device)
    gpu = gpu_name(device)
    results = {}
    for scale in scales:
        for graphs in graphs_options:
            engine = RifeEngine(net, height, width, scale, device)
            if graphs and not engine.enable_cuda_graph():
                continue
            ms = bench_engine(engine, warmup, iters)
            results[entry_key(gpu, model, width, height, scale, graphs)] = {
                "ms": round(ms, 3), "measured_at": time.time()}
            log.info("%s %dx%d scale=%s graphs=%s: %.1f ms (%.1f interps/s)", model, width, height, scale,
                     graphs, ms, 1000 / ms)
    return results
