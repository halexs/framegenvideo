"""A tiny stand-in for upstream IFNet with the same file layout and call contract.

``merged[-1]`` is a linear blend, so pipeline tests can check exact pixel values without real weights.
"""
from pathlib import Path

IFNET_SOURCE = '''
import torch
import torch.nn as nn
from model.warplayer import warp  # same import as upstream; proves the repo root is on sys.path


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        for k in range(5):
            setattr(self, f"block{k}", nn.Conv2d(1, 1, 1))
        self.calls = 0

    def forward(self, x, timestep=0.5, scale_list=None):
        self.calls += 1
        img0, img1 = x[:, :3], x[:, 3:6]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, x.shape[2], x.shape[3])
        assert len(scale_list) == 5
        merged = img0 * (1 - timestep) + img1 * timestep
        return [None], None, [merged]
'''


def write_fake_model(dest: Path, prefix: str = "module.") -> Path:
    import torch

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "IFNet_HDv3.py").write_text(IFNET_SOURCE)
    state = {}
    for k in range(5):
        state[f"{prefix}block{k}.weight"] = torch.ones(1, 1, 1, 1)
        state[f"{prefix}block{k}.bias"] = torch.zeros(1)
    torch.save(state, dest / "flownet.pkl")
    return dest
