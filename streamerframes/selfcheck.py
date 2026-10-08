"""Startup self-check (PLAN.md Phase 0 item 7): GPU, torch arch, ffmpeg, NVENC, model files."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from .config import Settings


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run_checks(settings: Settings, model: str = "default", need_gpu: bool = True) -> list[Check]:
    checks: list[Check] = []
    try:
        import torch
        cuda = torch.cuda.is_available()
        checks.append(Check("torch", True, torch.__version__))
        if cuda:
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"sm_{major}{minor}"
            archs = torch.cuda.get_arch_list()
            checks.append(Check("cuda", True, torch.cuda.get_device_name(0)))
            checks.append(Check("torch arch", arch in archs or f"compute_{major}{minor}" in archs,
                                f"{arch} {'in' if arch in archs else 'NOT in'} {archs}"
                                + ("" if arch in archs else " (install cu126 wheels for Pascal)")))
        else:
            checks.append(Check("cuda", False, "torch.cuda.is_available() is False", need_gpu))
    except ImportError as exc:
        checks.append(Check("torch", False, str(exc)))

    for tool in (settings.ffmpeg, settings.ffprobe):
        path = shutil.which(tool)
        checks.append(Check(tool, bool(path), path or "not found on PATH"))
    if shutil.which(settings.ffmpeg):
        out = subprocess.run([settings.ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
        nvenc = "h264_nvenc" in out.stdout
        checks.append(Check("h264_nvenc", nvenc, "available" if nvenc else "missing", need_gpu))

    model_dir = settings.model_path(model)
    for name in ("IFNet_HDv3.py", "flownet.pkl"):
        p = model_dir / name
        checks.append(Check(f"model {name}", p.exists(), str(p)))
    return checks


def format_checks(checks: list[Check]) -> str:
    return "\n".join(f"  [{'ok' if c.ok else ('FAIL' if c.required else 'warn')}] {c.name}: {c.detail}"
                     for c in checks)


def failed(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.required and not c.ok]
