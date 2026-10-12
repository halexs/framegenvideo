from streamerframes.web.app import is_playback_request
from streamerframes.web.awake import PlaybackKeepAwake


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make(idle=120.0):
    clock, calls = Clock(), []
    return PlaybackKeepAwake(idle_seconds=idle, set_state=calls.append, clock=clock), clock, calls


def test_blocks_sleep_while_fetching_and_releases_when_idle():
    awake, clock, calls = make()
    awake.tick()
    assert calls == [] and not awake.held  # nothing played yet: sleep allowed, no call needed
    awake.touch()
    awake.tick()
    assert calls == [True]
    clock.now += 100
    awake.touch()
    clock.now += 100
    awake.tick()
    assert calls == [True]  # last fetch 100 s ago, still within the idle window
    clock.now += 21
    awake.tick()
    assert calls == [True, False] and not awake.held
    awake.tick()
    assert calls == [True, False]  # state changes only


def test_stop_releases_hold():
    awake, clock, calls = make()
    awake.poll_seconds = 0.01
    awake.start()
    awake.touch()
    awake.tick()
    awake.stop()
    assert calls[0] is True and calls[-1] is False and not awake.held


def test_playback_paths():
    assert is_playback_request("/hls/0123456789abcdef/abcdef012345/seg/12.ts")
    assert is_playback_request("/hls/0123456789abcdef/audio/3.ts")
    assert is_playback_request("/media/original/0123456789abcdef")
    assert not is_playback_request("/hls/0123456789abcdef/abcdef012345/video.m3u8")
    assert not is_playback_request("/api/status/0123456789abcdef/abcdef012345")
    assert not is_playback_request("/watch/0123456789abcdef")
