"""Build upstream IFNet from a model dir (IFNet_HDv3.py + flownet.pkl) without RIFE_HDv3.Model.

RIFE_HDv3.Model builds an AdamW optimizer and imports the training losses; we only need the network.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import re
import sys
from pathlib import Path

import torch

from ..config import REPO_ROOT

log = logging.getLogger(__name__)


def _import_ifnet_module(model_dir: Path):
    source = model_dir / "IFNet_HDv3.py"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found; download a RIFE model (see README)")
    # Upstream files import `model.warplayer` from the repo root, and sometimes siblings in their own dir.
    for p in (str(REPO_ROOT), str(model_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    name = "sf_ifnet_" + hashlib.sha1(str(model_dir).encode()).hexdigest()[:8]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def count_blocks(net: torch.nn.Module) -> int:
    return sum(1 for n, _ in net.named_children() if re.fullmatch(r"block\d+", n))


def load_ifnet(model_dir: str | Path, device: torch.device | str = "cuda") -> torch.nn.Module:
    model_dir = Path(model_dir).resolve()
    module = _import_ifnet_module(model_dir)
    net = module.IFNet()
    weights = model_dir / "flownet.pkl"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"{weights}: missing weights {missing[:5]}... (wrong IFNet_HDv3.py for this model?)")
    if unexpected:
        log.warning("%s: ignoring %d unexpected keys", weights, len(unexpected))
    net.to(device).eval().requires_grad_(False)
    log.info("loaded %s (%d flow blocks)", model_dir, count_blocks(net))
    return net
