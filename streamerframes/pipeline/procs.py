"""ffmpeg subprocess wrappers: exact-size frame reads, frame writes, stderr to log files (never DEVNULL)."""
from __future__ import annotations

import os
import subprocess
from collections import deque
from pathlib import Path

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def log_tail(path: Path, lines: int = 50) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\n") for line in deque(fh, maxlen=lines)]
    except FileNotFoundError:
        return []


class FfmpegProcess:
    def __init__(self, cmd: list[str], log_path: Path, stdin: bool = False, stdout: bool = False):
        self.cmd = cmd
        self.log_path = Path(log_path)
        self._log = open(self.log_path, "ab")
        self._log.write(("$ " + " ".join(cmd) + "\n").encode())
        self._log.flush()
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE if stdout else subprocess.DEVNULL, stderr=self._log,
            bufsize=0, creationflags=_NO_WINDOW)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def wait(self, timeout: float | None = None) -> int:
        try:
            return self.proc.wait(timeout)
        finally:
            if self.proc.returncode is not None:
                self._log.close()

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.wait()

    def tail(self, lines: int = 50) -> list[str]:
        return log_tail(self.log_path, lines)


class FrameReader(FfmpegProcess):
    """Reads fixed-size raw frames from ffmpeg's stdout."""

    def __init__(self, cmd: list[str], log_path: Path, frame_size: int):
        super().__init__(cmd, log_path, stdout=True)
        self.frame_size = frame_size
        self.frames_read = 0
        self.eof = False

    def read_into(self, buf: memoryview) -> bool:
        """Fill ``buf`` (exactly frame_size bytes). False on EOF; a trailing partial frame is dropped."""
        got = 0
        while got < self.frame_size:
            n = self.proc.stdout.readinto(buf[got:])
            if not n:
                self.eof = True
                return False
            got += n
        self.frames_read += 1
        return True

    def read(self) -> bytes | None:
        buf = bytearray(self.frame_size)
        return bytes(buf) if self.read_into(memoryview(buf)) else None

    def close(self, wait_timeout: float = 30) -> int:
        """Exit code; kills the decoder first if we stopped before EOF."""
        if not self.eof:
            self.kill()
            return self.proc.returncode
        self.proc.stdout.close()
        return self.wait(wait_timeout)


class FrameWriter(FfmpegProcess):
    def __init__(self, cmd: list[str], log_path: Path):
        super().__init__(cmd, log_path, stdin=True)
        self.frames_written = 0

    def write(self, data) -> None:
        view = memoryview(data)
        while view:
            n = self.proc.stdin.write(view)
            view = view[n:] if n else view
        self.frames_written += 1

    def close(self, wait_timeout: float = 120) -> int:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        return self.wait(wait_timeout)


def keep_awake(on: bool) -> None:
    """Stop Windows from sleeping during long renders (no-op elsewhere)."""
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
