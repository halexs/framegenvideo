"""HLS playlists built from the manifest + segment index (PLAN.md Phase 3 "Playlists")."""
from __future__ import annotations

import csv
import math
from fractions import Fraction
from pathlib import Path

from .store import ProfileCache

H264_HIGH = "avc1.640032"  # High@5.0: covers 1080p60; browsers take the real level from the stream
AAC_LC = "mp4a.40.2"


def _bitrate(maxrate: str) -> int:
    units = {"": 1, "k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    value = maxrate.strip().lower()
    unit = value[-1] if value[-1] in "kmg" else ""
    return int(float(value[: len(value) - len(unit)]) * units[unit])


def master_playlist(manifest: dict, has_audio: bool) -> str:
    out_fps = Fraction(manifest["out_fps"])
    encoder = manifest.get("profile", {}).get("encoder", {})
    bandwidth = _bitrate(encoder.get("maxrate", "25M")) + (192_000 if has_audio else 0)
    lines = ["#EXTM3U", "#EXT-X-VERSION:6", "#EXT-X-INDEPENDENT-SEGMENTS"]
    codecs = H264_HIGH
    stream_inf = (f"#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},RESOLUTION={manifest['width']}x{manifest['height']},"
                  f"FRAME-RATE={float(out_fps):.3f}")
    if has_audio:
        lines.append('#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="aud",NAME="Audio",DEFAULT=YES,AUTOSELECT=YES,'
                     'URI="../audio.m3u8"')
        codecs += f",{AAC_LC}"
        stream_inf += ',AUDIO="aud"'
    lines += [stream_inf + f',CODECS="{codecs}"', "video.m3u8", ""]
    return "\n".join(lines)


def segment_seconds(manifest: dict, k: int) -> Fraction:
    out_fps = Fraction(manifest["out_fps"])
    n = manifest["seg_frames"]
    if k == manifest["total_segments"] - 1:
        n = manifest["last_segment_frames"]
    return Fraction(n) / out_fps


def video_playlist(cache: ProfileCache, manifest: dict | None = None, full: bool = False) -> str:
    """Phase 3: an EVENT playlist of the contiguous completed prefix (VOD once complete).

    ``full=True`` (Phase 4) lists every segment up front so the player can seek anywhere.
    """
    manifest = manifest or cache.manifest()
    total = manifest["total_segments"]
    complete = manifest.get("status") == "complete"
    count = total if (full or complete) else cache.contiguous_prefix()
    target = math.ceil(float(Fraction(manifest["seg_frames"]) / Fraction(manifest["out_fps"])))
    lines = ["#EXTM3U", "#EXT-X-VERSION:6", f"#EXT-X-TARGETDURATION:{target}", "#EXT-X-MEDIA-SEQUENCE:0",
             "#EXT-X-INDEPENDENT-SEGMENTS",
             f"#EXT-X-PLAYLIST-TYPE:{'VOD' if (full or complete) else 'EVENT'}"]
    for k in range(count):
        lines += [f"#EXTINF:{float(segment_seconds(manifest, k)):.6f},", f"seg/{k}.ts"]
    if full or complete:
        lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"


def audio_segments(audio_dir: Path) -> list[tuple[str, float]]:
    rows = []
    try:
        with open(Path(audio_dir) / "audio.csv", newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) >= 3:
                    rows.append((row[0], float(row[2]) - float(row[1])))
    except FileNotFoundError:
        pass
    return rows


def audio_playlist(audio_dir: Path) -> str:
    rows = audio_segments(audio_dir)
    done = (Path(audio_dir) / "DONE").exists()
    target = math.ceil(max((d for _, d in rows), default=4.0))
    lines = ["#EXTM3U", "#EXT-X-VERSION:6", f"#EXT-X-TARGETDURATION:{target}", "#EXT-X-MEDIA-SEQUENCE:0",
             "#EXT-X-INDEPENDENT-SEGMENTS", f"#EXT-X-PLAYLIST-TYPE:{'VOD' if done else 'EVENT'}"]
    for n, (_, duration) in enumerate(rows):
        lines += [f"#EXTINF:{duration:.6f},", f"audio/{n}.ts"]
    if done:
        lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"


def audio_segment_path(audio_dir: Path, n: int) -> Path | None:
    rows = audio_segments(audio_dir)
    if 0 <= n < len(rows):
        return Path(audio_dir) / rows[n][0]
    return None
