"""Finalize: concatenate a complete segment cache into MP4/MKV with the source's audio and subtitles."""
from __future__ import annotations

import json
import logging
import re
import subprocess
from fractions import Fraction
from pathlib import Path

from ..cache.store import ProfileCache
from ..probe import VideoInfo
from . import ffmpeg
from .procs import FfmpegProcess

log = logging.getLogger(__name__)


def default_export_name(info: VideoInfo, manifest: dict, container: str, video_codec: str = "copy") -> str:
    stem = re.sub(r"[^\w.\- ]+", "_", Path(info.path).stem).strip() or "video"
    fps = Fraction(manifest["out_fps"])
    tag = ".hevc" if video_codec == "hevc" else ""
    return f"{stem}.{float(fps):.3f}fps{tag}.{container}".replace(".000fps", "fps")


def video_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    """Duration of the first video stream (the container duration also covers audio)."""
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=duration:stream_tags=DURATION:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    stream = (data.get("streams") or [{}])[0]
    if stream.get("duration"):
        return float(stream["duration"])
    tag = (stream.get("tags") or {}).get("DURATION")
    if tag:
        h, m, sec = tag.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    return float(data["format"]["duration"])


def export(cache: ProfileCache, info: VideoInfo, output: Path | None = None, container: str = "auto",
           export_dir: Path | None = None, video_codec: str = "copy", hevc_encoder: str = "hevc_nvenc") -> Path:
    manifest = cache.manifest() or {}
    if manifest.get("status") != "complete":
        raise RuntimeError(f"cache is {manifest.get('status', 'missing')}, not complete")
    container = ffmpeg.export_container(info, container)
    if output is None:
        out_dir = Path(export_dir) if export_dir else cache.export_dir
        output = out_dir / default_export_name(info, manifest, container, video_codec)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    segments = [cache.segment_path(k) for k in range(manifest["total_segments"])]
    concat = ffmpeg.write_concat_list(segments, cache.root / "segments.txt")
    tmp = output.with_name(output.stem + ".partial" + output.suffix)
    proc = FfmpegProcess(ffmpeg.finalize_cmd(info, concat, tmp, video_codec, hevc_encoder),
                         cache.root / "ffmpeg-export.log")
    rc = proc.wait()
    if rc != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"export failed ({rc}): " + " | ".join(proc.tail(5)))
    got = video_duration(tmp)
    expected = manifest["total_out_frames"] / Fraction(manifest["out_fps"])
    if abs(got - float(expected)) > 2 / float(Fraction(manifest["out_fps"])):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"export duration {got:.3f}s differs from expected {float(expected):.3f}s")
    tmp.replace(output)
    exports = sorted(set(manifest.get("exports", [])) | {str(output)})
    cache.update_manifest(exports=exports)
    log.info("exported %s", output)
    return output
