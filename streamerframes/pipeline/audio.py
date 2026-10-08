"""AAC audio rendition for HLS, encoded once per video and independent of the GPU."""
from __future__ import annotations

import logging
from pathlib import Path

from ..probe import VideoInfo
from . import ffmpeg
from .procs import FfmpegProcess

log = logging.getLogger(__name__)


def audio_state(audio_dir: Path) -> str:
    """'done', 'none' (source has no audio), or 'missing'."""
    done = Path(audio_dir) / "DONE"
    if not done.exists():
        return "missing"
    return "none" if done.read_text().strip() == "none" else "done"


def ensure_audio(info: VideoInfo, audio_dir: Path) -> str:
    audio_dir = Path(audio_dir)
    state = audio_state(audio_dir)
    if state != "missing":
        return state
    audio_dir.mkdir(parents=True, exist_ok=True)
    if not info.audio:
        (audio_dir / "DONE").write_text("none")
        return "none"
    for old in audio_dir.glob("a_*.ts"):
        old.unlink()
    (audio_dir / "audio.csv").unlink(missing_ok=True)
    proc = FfmpegProcess(ffmpeg.audio_cmd(info, audio_dir), audio_dir / "ffmpeg-audio.log")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"audio encode failed ({rc}): " + " | ".join(proc.tail(5)))
    (audio_dir / "DONE").write_text("done")
    log.info("audio rendition ready in %s", audio_dir)
    return "done"
