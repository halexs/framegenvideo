from streamerframes.cache import playlist
from streamerframes.cache.store import ProfileCache

MANIFEST = {"out_fps": "48000/1001", "seg_frames": 192, "total_segments": 3, "last_segment_frames": 50,
            "total_out_frames": 434, "width": 1920, "height": 800, "status": "running",
            "profile": {"encoder": {"maxrate": "20M"}}}


def _cache(tmp_path, done):
    pc = ProfileCache(tmp_path / "v" / "p")
    pc.ensure_dirs()
    rows = []
    for k in done:
        pc.segment_path(k).write_bytes(b"G")
        rows.append(f"v_{k:05d}.ts,0,1\n")
    (pc.index_dir / "run_0000.csv").write_text("".join(rows))
    return pc


def test_master():
    text = playlist.master_playlist(MANIFEST, has_audio=True)
    assert "BANDWIDTH=20192000,RESOLUTION=1920x800,FRAME-RATE=47.952" in text
    assert 'AUDIO="aud"' in text and 'URI="../audio.m3u8"' in text and text.strip().endswith("video.m3u8")
    silent = playlist.master_playlist(MANIFEST, has_audio=False)
    assert "EXT-X-MEDIA" not in silent and "mp4a" not in silent


def test_video_event_prefix_then_vod(tmp_path):
    pc = _cache(tmp_path, [0, 2])
    text = playlist.video_playlist(pc, MANIFEST)
    assert "#EXT-X-PLAYLIST-TYPE:EVENT" in text and "#EXT-X-ENDLIST" not in text
    assert text.count("#EXTINF") == 1 and "#EXTINF:4.004000," in text and "#EXT-X-TARGETDURATION:5" in text
    full = playlist.video_playlist(pc, MANIFEST, full=True)
    assert full.count("#EXTINF") == 3 and "#EXT-X-PLAYLIST-TYPE:VOD" in full and "#EXT-X-ENDLIST" in full
    assert f"#EXTINF:{50 * 1001 / 48000:.6f}," in full and "seg/2.ts" in full
    done = playlist.video_playlist(pc, {**MANIFEST, "status": "complete"})
    assert done.count("#EXTINF") == 3 and "#EXT-X-ENDLIST" in done


def test_audio_playlist(tmp_path):
    d = tmp_path / "audio"
    d.mkdir()
    (d / "audio.csv").write_text("a_00000.ts,0.000000,4.010667\na_00001.ts,4.010667,6.000000\n")
    for n in (0, 1):
        (d / f"a_{n:05d}.ts").write_bytes(b"G")
    text = playlist.audio_playlist(d)
    assert "#EXT-X-PLAYLIST-TYPE:EVENT" in text and "#EXTINF:4.010667," in text and "audio/1.ts" in text
    (d / "DONE").write_text("done")
    assert "#EXT-X-ENDLIST" in playlist.audio_playlist(d)
    assert playlist.audio_segment_path(d, 1).name == "a_00001.ts"
    assert playlist.audio_segment_path(d, 2) is None
