"""`report`: benchmark and generation tables from calibration.json and cache manifests."""
from streamerframes.__main__ import main
from streamerframes.bench import entry_key, save_entries
from streamerframes.cache.store import CacheStore
from streamerframes.report import BENCH_HINT, GEN_HINT, bench_rows, generation_rows, report

GPU = "NVIDIA GeForce GTX 1080 Ti"


def fill(root):
    save_entries(root, {
        entry_key(GPU, "default", 1920, 800, 1.0, True): {"ms": 40.0},    # 25/s: 2x 23.976 needs 26.4
        entry_key(GPU, "default", 1920, 800, 0.5, True): {"ms": 25.0},    # 40/s: enough for 23.976 and 29.97
        entry_key(GPU, "4.25.lite", 1280, 720, 1.0, False): {"ms": 36.0},  # 27.8/s: 23.976 only
    })
    store = CacheStore(root)
    for vid, name in (("a" * 16, "Batman Begins.mp4"), ("b" * 16, "Wreck-It Ralph.mp4")):
        store.write_source(vid, {"path": f"E:/Movies/{name}"})
    done = store.profile("a" * 16, "p1")
    done.ensure_dirs()
    done.write_manifest({"source": "E:/Movies/Batman Begins.mp4", "profile_name": "quality",
                         "out_fps": "60000/1001", "status": "complete", "total_segments": 2})
    for k in range(2):
        done.segment_path(k).write_bytes(b"x")
    (done.index_dir / "run_0.csv").write_text("v_00000.ts,0,4\nv_00001.ts,4,8\n")
    done.write_progress({"src_fps_measured": 21.25, "realtime_ratio": 0.886})
    running = store.profile("b" * 16, "p2")
    running.ensure_dirs()
    running.write_manifest({"source": "E:/Movies/Wreck-It Ralph.mp4", "profile_name": "realtime",
                            "out_fps": "48000/1001", "status": "paused", "total_segments": 4})
    running.segment_path(0).write_bytes(b"x")
    (running.index_dir / "run_0.csv").write_text("v_00000.ts,0,4\n")


def test_rows(tmp_path):
    fill(tmp_path)
    from streamerframes.bench import load_calibration
    assert bench_rows(load_calibration(tmp_path)) == [
        ["1920x800", "default", "1", "yes", "40.0", "25.0", "no", "no"],
        ["1920x800", "default", "0.5", "yes", "25.0", "40.0", "yes", "yes"],
        ["1280x720", "4.25.lite", "1", "no", "36.0", "27.8", "yes", "no"],
    ]
    assert generation_rows(CacheStore(tmp_path)) == [
        ["Batman Begins.mp4", "quality", "59.940", "complete", "100%", "21.2", "0.89"],
        ["Wreck-It Ralph.mp4", "realtime", "47.952", "paused", "25%", "", ""],
    ]


def test_empty(tmp_path):
    out = report(tmp_path)
    assert BENCH_HINT in out and GEN_HINT in out


def test_markdown_tables_parse(tmp_path, capsys):
    fill(tmp_path)
    cfg = tmp_path / "sf.toml"
    cfg.write_text(f'cache_root = "{tmp_path.as_posix()}"\n')
    assert main(["--config", str(cfg), "report", "--markdown"]) == 0
    out = capsys.readouterr().out
    tables, current = [], None
    for line in out.splitlines():
        if line.startswith("|"):
            if current is None:
                current = []
                tables.append(current)
            current.append([c.strip() for c in line.strip().strip("|").split("|")])
        else:
            current = None
    assert len(tables) == 2
    for table in tables:
        header, sep, *rows = table
        assert all(c == "---" for c in sep) and len(sep) == len(header)
        assert rows and all(len(r) == len(header) for r in rows)
    assert tables[0][0][:2] == ["Size", "Model"] and len(tables[0]) == 5
    assert tables[1][2][0] == "Batman Begins.mp4" and len(tables[1]) == 4
