"""Stand-in for streamerframes.pipeline.worker used by job manager tests.

Behaviour is chosen by the video file name: ok, slow (runs until stopped), fail, hang (ignores stop).
"""
import json
import os
import sys
import time
from pathlib import Path


def write(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    cache = Path(job["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    mode = Path(job["video_path"]).stem
    pid = os.getpid()
    import psutil
    write(cache / "lock", {"pid": pid, "create_time": psutil.Process(pid).create_time(), "run": 0})
    write(cache / "manifest.json", {"status": "running", "total_segments": 10, "seg_frames": 24,
                                    "out_fps": "48/1", "total_out_frames": 240})
    log = cache / "calls.log"
    with open(log, "a") as fh:
        fh.write(f"{job['kind']} start={job['start_segment']}\n")
    status, code = "complete", 0
    if mode == "fail":
        status, code = "failed", 1
    elif mode == "partial":  # a stream run that reached the end of its stretch, leaving holes
        status, code = "paused", 0
    elif mode in ("slow", "hang"):
        while True:
            if mode == "slow":
                write(cache / "progress.json", {"run": 0, "updated_at": time.time()})
            if mode == "slow" and (cache / "stop").exists():
                status, code = "paused", 3
                break
            if (cache / "finish").exists():
                break
            time.sleep(0.05)
    m = json.loads((cache / "manifest.json").read_text())
    m.update(status=status, error="boom" if mode == "fail" else None)
    write(cache / "manifest.json", m)
    (cache / "lock").unlink()
    sys.exit(code)


if __name__ == "__main__":
    main()
