"""Pure ffmpeg command builders (PLAN.md Phase 1). Each returns a list[str] for subprocess."""
from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

from ..probe import VideoInfo
from ..timeline import Timeline

BASE = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "warning"]
AAC_PRIMING_SAMPLES = 1024


def frame_bytes(width: int, height: int) -> int:
    """Size of one packed yuv420p frame (chroma planes round up for odd sizes)."""
    return width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)


def rate(value: Fraction) -> str:
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def seconds(value: Fraction | float) -> str:
    return f"{float(value):.6f}"


def color_args(info: VideoInfo) -> list[str]:
    if info.height >= 720:
        space, primaries, trc = "bt709", "bt709", "bt709"
    else:
        space, primaries, trc = "bt470bg", "smpte170m", "smpte170m"
    return ["-colorspace", space, "-color_primaries", primaries, "-color_trc", trc, "-color_range", "tv"]


def decoder_cmd(info: VideoInfo, start_frame: int = 0, hwaccel: bool = True) -> list[str]:
    """Raw yuv420p frames on stdout, starting exactly at source frame ``start_frame``."""
    cmd = list(BASE)
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    if start_frame > 0:
        # Input seeking while transcoding is frame accurate; the -1/4 frame avoids rounding past it.
        cmd += ["-ss", seconds(info.start_time + (start_frame - Fraction(1, 4)) / info.src_fps)]
    cmd += [
        "-i", info.path,
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-fps_mode", "cfr", "-r", rate(info.src_fps),
        "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
    ]
    return cmd


def _double_rate(maxrate: str) -> str:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmMgG]?)", maxrate.strip())
    if not m:
        raise ValueError(f"bad bitrate {maxrate!r}")
    value = float(m.group(1)) * 2
    return f"{value:g}{m.group(2)}"


def encoder_cmd(info: VideoInfo, tl: Timeline, start_segment: int, run_id: int, out_dir: Path,
                encoder: str = "nvenc", preset: str = "p4", cq: int = 20, maxrate: str = "25M",
                width: int | None = None, height: int | None = None) -> list[str]:
    """Raw yuv420p on stdin -> fixed-length MPEG-TS segments, each starting with an IDR frame."""
    n = tl.seg_frames
    cmd = list(BASE) + [
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width or info.width}x{height or info.height}",
        "-framerate", rate(tl.out_fps), "-i", "pipe:0", "-an",
    ]
    if encoder == "nvenc":
        cmd += [
            "-c:v", "h264_nvenc", "-preset", preset, "-tune", "hq", "-profile:v", "high",
            "-rc", "vbr", "-cq", str(cq), "-b:v", "0", "-maxrate", maxrate, "-bufsize", _double_rate(maxrate),
            "-g", str(n), "-forced-idr", "1", "-no-scenecut", "1", "-strict_gop", "1",
            "-spatial-aq", "1",
        ]
    elif encoder == "lossless":
        cmd += ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-g", str(n),
                "-sc_threshold", "0", "-x264-params", "open-gop=0"]
    else:
        raise ValueError(f"unknown encoder {encoder!r}")
    cmd += ["-force_key_frames", f"expr:eq(mod(n,{n}),0)"]
    cmd += color_args(info)
    cmd += [
        "-output_ts_offset", seconds(tl.segment_start_time(start_segment)),
        "-f", "segment", "-segment_format", "mpegts",
        "-segment_time", seconds(Fraction(n) / tl.out_fps),
        "-segment_time_delta", seconds(Fraction(1, 2) / tl.out_fps),
        "-segment_start_number", str(start_segment),
        "-segment_list", str(Path(out_dir) / "index" / f"run_{run_id:04d}.csv"),
        "-segment_list_type", "csv",
        str(Path(out_dir) / "segments" / "v_%05d.ts"),
    ]
    return cmd


def audio_cmd(info: VideoInfo, audio_dir: Path, stream: int | None = None,
              segment_seconds: float = 4.004) -> list[str]:
    """AAC stereo HLS audio rendition, encoded once per video."""
    if stream is None:
        defaults = [n for n, s in enumerate(info.audio) if s.default]
        stream = defaults[0] if defaults else 0
    audio_dir = Path(audio_dir)
    return list(BASE) + [
        "-i", info.path, "-map", f"0:a:{stream}", "-vn",
        # The AAC encoder prepends 1024 priming samples, and MPEG-TS has no edit list to hide them, so a
        # player would hear everything ~21 ms late. Drop 1024 source samples to cancel them out.
        "-af", f"atrim=start_sample={AAC_PRIMING_SAMPLES},asetpts=PTS-STARTPTS",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-f", "segment", "-segment_format", "mpegts", "-segment_time", str(segment_seconds),
        "-segment_list", str(audio_dir / "audio.csv"), "-segment_list_type", "csv",
        str(audio_dir / "a_%05d.ts"),
    ]


def export_container(info: VideoInfo, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "mkv" if info.has_bitmap_subtitles else "mp4"


def finalize_cmd(info: VideoInfo, concat_list: Path, output: Path) -> list[str]:
    """Concatenate video segments and copy the source's audio and subtitles untouched."""
    output = Path(output)
    mp4 = output.suffix.lower() in (".mp4", ".m4v", ".mov")
    cmd = list(BASE) + [
        "-f", "concat", "-safe", "0", "-i", str(concat_list), "-i", info.path,
        "-map", "0:v:0", "-map", "1:a?",
    ]
    if not (mp4 and info.has_bitmap_subtitles):
        cmd += ["-map", "1:s?"]
    cmd += ["-c", "copy"]
    if mp4:
        cmd += ["-c:s", "mov_text", "-movflags", "+faststart"]
    cmd += ["-y", str(output)]
    return cmd


def write_concat_list(segment_paths: list[Path], dest: Path) -> Path:
    lines = []
    for p in segment_paths:
        escaped = str(Path(p).resolve()).replace("\\", "/").replace("'", r"'\''")
        lines.append(f"file '{escaped}'\n")
    Path(dest).write_text("".join(lines), encoding="utf-8")
    return Path(dest)
