"""Hold off system sleep while someone is watching. Windows only counts local input as activity, so a movie
played from another device could otherwise be cut off by the sleep timeout."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from ..pipeline.procs import keep_awake

log = logging.getLogger(__name__)

PLAYBACK_IDLE_SECONDS = 120.0  # players fetch a segment every few seconds; a paused one stops within ~a minute


class PlaybackKeepAwake:
    """`touch()` on every media fetch; sleep is blocked until no fetch has arrived for `idle_seconds`.

    The wake request belongs to the thread that set it, so one dedicated thread owns it."""

    def __init__(self, idle_seconds: float = PLAYBACK_IDLE_SECONDS, poll_seconds: float = 5.0,
                 set_state: Callable[[bool], None] = keep_awake, clock: Callable[[], float] = time.monotonic):
        self.idle_seconds, self.poll_seconds = idle_seconds, poll_seconds
        self._set_state, self._clock = set_state, clock
        self._last: float | None = None
        self.held = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def touch(self) -> None:
        self._last = self._clock()

    def tick(self) -> None:
        active = self._last is not None and self._clock() - self._last < self.idle_seconds
        if active != self.held:
            self._set_state(active)
            self.held = active
            log.info("playback %s; system sleep %s", "active" if active else "idle",
                     "blocked" if active else "allowed")

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="playback-keep-awake", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_seconds + 1)
            self._thread = None

    def _loop(self) -> None:
        try:
            while not self._stop.wait(self.poll_seconds):
                self.tick()
        finally:
            if self.held:
                self._set_state(False)
                self.held = False
