"""Settings: streamerframes.toml + environment overrides (PLAN.md Phase 3 "Config")."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ENV = "STREAMERFRAMES_CONFIG"


@dataclass
class EncoderSettings:
    codec: str = "h264_nvenc"          # h264_nvenc | lossless (libx264 -qp 0, for tests)
    preset: str = "p4"
    cq: int = 20
    maxrate: str = "25M"


@dataclass
class Profile:
    name: str = "realtime"
    mode: str = "multi"                # multi | target
    multi: int = 2
    target_fps: int = 60
    scale: float | str = "auto"        # 1.0 | 0.5 | auto
    model: str = "default"             # "default" (model_dir), a directory under models_dir, or "auto"
    seg_seconds: float = 4.0
    scene_detect: bool = True
    scene_ssim: float = 0.2            # SSIM below this between neighbours = hard cut
    dup_mad: float = 0.0               # mean abs diff below this = duplicate frame (0 disables)
    letterbox_crop: bool = True
    cuda_graphs: bool = True
    encoder: EncoderSettings = field(default_factory=EncoderSettings)

    def pixel_settings(self) -> dict:
        """Everything that changes output pixels. Hosts, paths and names are excluded."""
        data = asdict(self)
        data.pop("name")
        data.pop("cuda_graphs")        # equivalent output, only speed differs
        if data["mode"] == "multi":
            data.pop("target_fps")
        else:
            data.pop("multi")
        return data

    def profile_id(self, resolved_scale: float | None = None) -> str:
        data = self.pixel_settings()
        if resolved_scale is not None:
            data["scale"] = float(resolved_scale)
        blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


DEFAULT_PROFILES = {
    "realtime": Profile(name="realtime", mode="multi", multi=2, scale="auto",
                        encoder=EncoderSettings(preset="p4", cq=21, maxrate="20M")),
    "quality": Profile(name="quality", mode="target", target_fps=60, scale=1.0,
                       encoder=EncoderSettings(preset="p6", cq=18, maxrate="40M")),
}


@dataclass
class Settings:
    movies_roots: list[str] = field(default_factory=lambda: ["E:/Movies"])
    cache_root: str = str(REPO_ROOT / "cache")
    export_dir: str = ""               # empty = inside the cache
    model_dir: str = str(REPO_ROOT / "train_log")
    models_dir: str = str(REPO_ROOT / "models")
    host: str = "127.0.0.1"            # set 0.0.0.0 for LAN access (there is no auth)
    port: int = 8000
    cache_max_gb: float = 200.0
    auto_resume: bool = True
    fill_gaps: bool = True
    min_free_vram_mb: int = 1500
    stream_policy: str = "newest_wins"  # newest_wins | queue
    watch_dirs: list[str] = field(default_factory=list)
    watch_profile: str = "quality"
    model_preference: list[str] = field(default_factory=list)  # best quality first, for model = "auto"
    hevc_encoder: str = "hevc_nvenc"   # for HEVC exports; libx265 works without NVENC
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    profiles: dict[str, Profile] = field(default_factory=lambda: copy.deepcopy(DEFAULT_PROFILES))

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_root)

    def profile(self, name: str) -> Profile:
        try:
            return self.profiles[name]
        except KeyError:
            raise ValueError(f"unknown profile {name!r}; have {sorted(self.profiles)}") from None

    def model_path(self, name: str) -> Path:
        if name in ("", "default"):
            return Path(self.model_dir)
        if name == "auto":
            raise ValueError("model 'auto' must be resolved first")
        return Path(self.models_dir) / name

    def available_models(self) -> list[str]:
        """Usable models, best quality first: model_preference, else default, full models, then lite ones."""
        def usable(p: Path) -> bool:
            return (p / "IFNet_HDv3.py").exists() and (p / "flownet.pkl").exists()

        found = ["default"] if usable(Path(self.model_dir)) else []
        root = Path(self.models_dir)
        if root.is_dir():
            names = sorted(p.name for p in root.iterdir() if p.is_dir() and usable(p))
            found += sorted(names, key=lambda n: ("lite" in n.lower(), n))
        if self.model_preference:
            ranked = [m for m in self.model_preference if m in found]
            found = ranked + [m for m in found if m not in ranked]
        return found


def _profile_from_dict(name: str, raw: dict, base: Profile | None) -> Profile:
    prof = copy.deepcopy(base) if base else Profile(name=name)
    prof.name = name
    for key, value in raw.items():
        if key == "encoder":
            for ek, ev in value.items():
                if not hasattr(prof.encoder, ek):
                    raise ValueError(f"profiles.{name}.encoder: unknown key {ek!r}")
                setattr(prof.encoder, ek, ev)
        elif hasattr(prof, key):
            setattr(prof, key, value)
        else:
            raise ValueError(f"profiles.{name}: unknown key {key!r}")
    if prof.mode not in ("multi", "target"):
        raise ValueError(f"profiles.{name}.mode must be 'multi' or 'target'")
    return prof


def load_settings(path: str | Path | None = None, env: dict | None = None) -> Settings:
    env = os.environ if env is None else env
    path = Path(path or env.get(CONFIG_ENV) or REPO_ROOT / "streamerframes.toml")
    settings = Settings()
    if path.exists():
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        for name, prof in (raw.pop("profiles", None) or {}).items():
            settings.profiles[name] = _profile_from_dict(name, prof, settings.profiles.get(name))
        for key, value in raw.items():
            if not hasattr(settings, key):
                raise ValueError(f"{path}: unknown setting {key!r}")
            setattr(settings, key, value)
    if env.get("MOVIES_DIR"):
        settings.movies_roots = [p for p in env["MOVIES_DIR"].split(os.pathsep) if p]
    for key, attr in (("CACHE_DIR", "cache_root"), ("MODEL_DIR", "model_dir"), ("HOST", "host")):
        if env.get(key):
            setattr(settings, attr, env[key])
    if env.get("PORT"):
        settings.port = int(env["PORT"])
    return settings
