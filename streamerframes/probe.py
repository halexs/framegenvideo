"""ffprobe JSON -> VideoInfo with exact Fraction frame rates (PLAN.md Phase 1)."""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

log = logging.getLogger(__name__)

BITMAP_SUBTITLE_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}


def parse_rate(value: str | None) -> Fraction | None:
    """'24000/1001' -> Fraction; None for missing or '0/0'."""
    if not value:
        return None
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class StreamInfo:
    index: int
    codec: str
    language: str | None
    default: bool


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    src_fps: Fraction
    n_src: int
    duration: float
    start_time: float
    pix_fmt: str
    bit_depth: int
    color_space: str
    color_primaries: str
    color_transfer: str
    color_range: str
    rotation: int = 0
    sar: Fraction = Fraction(1)
    audio: list[StreamInfo] = field(default_factory=list)
    subtitles: list[StreamInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_bitmap_subtitles(self) -> bool:
        return any(s.codec in BITMAP_SUBTITLE_CODECS for s in self.subtitles)

    def to_json(self) -> dict:
        data = asdict(self)
        data["src_fps"] = f"{self.src_fps.numerator}/{self.src_fps.denominator}"
        data["sar"] = f"{self.sar.numerator}/{self.sar.denominator}"
        return data


def _stream_info(s: dict) -> StreamInfo:
    return StreamInfo(
        index=int(s.get("index", 0)),
        codec=s.get("codec_name", "unknown"),
        language=(s.get("tags") or {}).get("language"),
        default=bool((s.get("disposition") or {}).get("default", 0)),
    )


def _rotation(s: dict) -> int:
    for sd in s.get("side_data_list") or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"]))) % 360
    rotate = (s.get("tags") or {}).get("rotate")
    return int(rotate) % 360 if rotate else 0


def _bit_depth(s: dict) -> int:
    raw = s.get("bits_per_raw_sample")
    if raw and str(raw).isdigit():
        return int(raw)
    pix = s.get("pix_fmt", "")
    for depth in (16, 12, 10):
        if f"p{depth}" in pix:
            return depth
    return 8


def parse_ffprobe(data: dict, path: str = "") -> VideoInfo:
    streams = data.get("streams") or []
    video = [s for s in streams if s.get("codec_type") == "video"
             and not (s.get("disposition") or {}).get("attached_pic")]
    if not video:
        raise ValueError(f"no video stream in {path or 'input'}")
    v = video[0]
    fmt = data.get("format") or {}
    warnings: list[str] = []

    src_fps = parse_rate(v.get("r_frame_rate")) or parse_rate(v.get("avg_frame_rate"))
    if src_fps is None:
        raise ValueError("could not determine source frame rate")
    avg = parse_rate(v.get("avg_frame_rate"))
    if avg and abs(float(avg) - float(src_fps)) > 0.01 * float(src_fps):
        warnings.append(f"possible VFR source (r={src_fps}, avg={avg}); decoding as CFR")

    duration = _float(v.get("duration")) or _float(fmt.get("duration")) or 0.0
    estimate = round(Fraction(duration) * src_fps) if duration else 0
    nb = v.get("nb_frames")
    n_src = int(nb) if nb and str(nb).isdigit() else 0
    # nb_frames is sometimes missing or nonsense; trust it only near the duration estimate.
    if n_src <= 0 or (estimate and abs(n_src - estimate) > max(2, estimate // 100)):
        n_src = estimate
    if n_src <= 0:
        raise ValueError("could not determine frame count")

    height = int(v["height"])
    hd = height >= 720
    sar = parse_rate(v.get("sample_aspect_ratio", "").replace(":", "/")) or Fraction(1)
    rotation = _rotation(v)
    bit_depth = _bit_depth(v)
    transfer = v.get("color_transfer") or ("bt709" if hd else "smpte170m")
    if sar != 1:
        warnings.append(f"non-square pixels (SAR {sar}) are not handled")
    if rotation:
        warnings.append(f"rotation {rotation} is not handled")
    if bit_depth > 8 or transfer in ("smpte2084", "arib-std-b67"):
        warnings.append("10-bit/HDR input is decoded to 8-bit SDR")

    return VideoInfo(
        path=path,
        width=int(v["width"]),
        height=height,
        src_fps=src_fps,
        n_src=n_src,
        duration=duration,
        start_time=_float(v.get("start_time")) or _float(fmt.get("start_time")) or 0.0,
        pix_fmt=v.get("pix_fmt", "unknown"),
        bit_depth=bit_depth,
        color_space=v.get("color_space") or ("bt709" if hd else "bt470bg"),
        color_primaries=v.get("color_primaries") or ("bt709" if hd else "smpte170m"),
        color_transfer=transfer,
        color_range=v.get("color_range") or "tv",
        rotation=rotation,
        sar=sar,
        audio=[_stream_info(s) for s in streams if s.get("codec_type") == "audio"],
        subtitles=[_stream_info(s) for s in streams if s.get("codec_type") == "subtitle"],
        warnings=warnings,
    )


def probe(path: str | Path, ffprobe: str = "ffprobe") -> VideoInfo:
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    info = parse_ffprobe(json.loads(result.stdout), str(path))
    for w in info.warnings:
        log.warning("%s: %s", path, w)
    return info
