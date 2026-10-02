import os
import re
import sys
import subprocess
import threading
import time
import json
from collections import deque
from pathlib import Path
from urllib.parse import quote, unquote
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

SERVER_LOG_PATH = Path(__file__).resolve().parent / "server.log"


def write_server_log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    try:
        with SERVER_LOG_PATH.open("a", encoding="utf-8", errors="ignore") as handle:
            handle.write(line)
    except Exception:
        pass

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}
MOVIES_ROOT = Path("E:/Movies")
MOVIES_SHORTCUT = Path(__file__).resolve().parent / "Movies - Shortcut.lnk"
HLS_ROOT = Path(__file__).resolve().parent / "hls"

app = FastAPI()

HLS_ROOT.mkdir(exist_ok=True)
app.mount("/hls", StaticFiles(directory=HLS_ROOT), name="hls")
write_server_log("StreamerFrames server initialized")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    write_server_log(f"Request {request.method} {request.url.path}?{request.url.query}")
    try:
        response = await call_next(request)
        write_server_log(f"Response {request.method} {request.url.path} status={response.status_code}")
        return response
    except Exception as exc:
        write_server_log(f"Exception during request {request.method} {request.url.path}: {exc}")
        raise


def resolve_shortcut_target(shortcut_path: Path) -> Path | None:
    if not shortcut_path.exists():
        return None
    try:
        shortcut_literal = str(shortcut_path).replace("'", "''")
        result = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{shortcut_literal}'); Write-Output $s.TargetPath",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if result:
            return Path(result)
    except Exception:
        pass
    return None


def get_movies_root() -> Path:
    if MOVIES_ROOT.exists() and MOVIES_ROOT.is_dir():
        return MOVIES_ROOT
    target = resolve_shortcut_target(MOVIES_SHORTCUT)
    if target and target.exists() and target.is_dir():
        write_server_log(f"Resolved movies root via shortcut: {target}")
        return target
    raise FileNotFoundError(
        "Could not locate the movies directory. "
        "Create E:/Movies or a Movies - Shortcut.lnk shortcut pointing to it."
    )


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def list_video_paths() -> list[Path]:
    root = get_movies_root()
    entries = []
    for ext in VIDEO_EXTENSIONS:
        entries.extend(root.rglob(f"*{ext}"))
    return sorted(p for p in entries if p.is_file())


def format_display_name(path: Path) -> str:
    root = get_movies_root()
    return str(path.relative_to(root)).replace("\\", "/")


def get_video_path(video_name: str) -> Path:
    root = get_movies_root()
    rel = Path(unquote(video_name))
    candidate = (root / rel).resolve()
    if not candidate.exists() or not candidate.is_file() or not candidate.is_relative_to(root):
        raise HTTPException(status_code=404, detail="Video not found")
    return candidate


def get_hls_dir(video_path: Path) -> Path:
    safe = sanitize_name(str(video_path.relative_to(get_movies_root())))
    return HLS_ROOT / safe


def is_hls_ready(video_path: Path) -> bool:
    hls_dir = get_hls_dir(video_path)
    if not (hls_dir / "stream.m3u8").exists():
        return False

    state_path = hls_dir / "resume_state.json"
    if state_path.exists():
        try:
            import json

            data = json.loads(state_path.read_text(encoding="utf-8"))
            return bool(data.get("complete", False))
        except Exception:
            return False
    return True


generation_jobs: dict[str, dict] = {}


def _tail_writer(deque_obj: deque, line: str, maxlen: int = 1000):
    if line is None:
        return
    deque_obj.append(line)


def run_framegen(video_path: Path) -> None:
    hls_dir = get_hls_dir(video_path)
    hls_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "rife_server.py"),
        "--input",
        str(video_path),
        "--hls-dir",
        str(hls_dir),
        "--generate-only",
    ]

    write_server_log(f"Starting framegen for {video_path} into {hls_dir}")
    write_server_log(f"Command: {' '.join(command)}")

    # prepare log file
    log_path = hls_dir / "generation.log"
    logfile = open(log_path, "a", encoding="utf-8", buffering=1)
    logfile.write(f"[SERVER] {time.strftime('%Y-%m-%d %H:%M:%S')} Starting generation for {video_path}\n")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    key = sanitize_name(str(video_path.relative_to(get_movies_root())))
    tail = deque(maxlen=500)

    generation_jobs[key] = {
        "process": process,
        "pid": process.pid,
        "started_at": time.time(),
        "status": "starting",
        "success": None,
        "log_path": str(log_path),
        "tail": tail,
        "last_output_line": None,
        "last_output_time": None,
    }

    def _reader(pipe, name: str):
        try:
            for line in iter(pipe.readline, ""):
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                out = f"[{name}] {ts} {line}"
                try:
                    logfile.write(out)
                except Exception:
                    pass
                _tail_writer(tail, out)
                generation_jobs[key]["last_output_line"] = out.strip()
                generation_jobs[key]["last_output_time"] = time.time()
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _monitor():
        generation_jobs[key]["status"] = "running"
        write_server_log(f"Framegen process started for {video_path} pid={process.pid}")
        stdout_thr = threading.Thread(target=_reader, args=(process.stdout, "OUT"), daemon=True)
        stderr_thr = threading.Thread(target=_reader, args=(process.stderr, "ERR"), daemon=True)
        stdout_thr.start()
        stderr_thr.start()

        rc = process.wait()
        stdout_thr.join(timeout=1)
        stderr_thr.join(timeout=1)

        generation_jobs[key]["success"] = rc == 0
        generation_jobs[key]["status"] = "done" if rc == 0 else "failed"
        generation_jobs[key]["ended_at"] = time.time()
        generation_jobs[key]["last_output_line"] = (
            generation_jobs[key].get("last_output_line") or f"Process exited with {rc}"
        )
        generation_jobs[key]["last_output_time"] = time.time()

        write_server_log(
            f"Framegen process ended for {video_path} pid={process.pid} returncode={rc}"
        )
        try:
            logfile.write(f"[MONITOR] returncode={rc}\n")
        except Exception:
            pass
        try:
            logfile.close()
        except Exception:
            pass

    threading.Thread(target=_monitor, daemon=True).start()


@app.get("/")
def home(request: Request):
    try:
        videos = list_video_paths()
    except FileNotFoundError as exc:
        return HTMLResponse(f"<h1>Error</h1><p>{exc}</p>", status_code=500)

    links = []
    for video_path in videos:
        display_name = format_display_name(video_path)
        encoded = quote(display_name, safe="")
        links.append(
            f"<li>{display_name} — "
            f"<a href=\"/watch?video={encoded}&mode=normal\">Normal</a> | "
            f"<a href=\"/watch?video={encoded}&mode=framegen\">Framegen</a></li>"
        )

    body = """
<!DOCTYPE html>
<html>
<head>
    <title>StreamerFrames Movie Selection</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        h1 { margin-bottom: 0.25em; }
        ul { list-style: none; padding: 0; }
        li { margin: 0.5em 0; }
        a { color: #38bdf8; }
    </style>
</head>
<body>
<h1>StreamerFrames</h1>
<p>Select a movie and choose a playback mode.</p>
<ul>
""" + "\n".join(links) + """
</ul>
</body>
</html>
"""
    return HTMLResponse(body)


@app.get("/video")
def stream_video(name: str):
    video_path = get_video_path(name)
    ext = video_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(video_path, media_type=media_type, filename=video_path.name)


@app.get("/generation-status")
def generation_status(name: str):
    """Return status and tail logs for a generation job for the given video name (same encoding as /video name param)."""
    video_path = get_video_path(name)
    key = sanitize_name(str(video_path.relative_to(get_movies_root())))
    job = generation_jobs.get(key)
    if not job:
        write_server_log(f"generation_status request for {video_path} with no active job")
        return {
            "status": "none",
            "hls_ready": is_hls_ready(video_path),
            "complete": False,
            "last_output_line": None,
            "last_output_time": None,
        }
    tail = list(job.get("tail", []))
    state_path = get_hls_dir(video_path) / "resume_state.json"
    complete_state = False
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            complete_state = bool(data.get("complete", False))
        except Exception:
            pass

    info = {
        "status": job.get("status"),
        "pid": job.get("pid"),
        "started_at": job.get("started_at"),
        "ended_at": job.get("ended_at", None),
        "success": job.get("success"),
        "log_path": job.get("log_path"),
        "last_output_line": job.get("last_output_line"),
        "last_output_time": job.get("last_output_time"),
        "tail": tail,
        "hls_ready": is_hls_ready(video_path),
        "complete": complete_state,
    }
    return info


@app.get("/server-log")
def server_log(tail: int = 200):
    if not SERVER_LOG_PATH.exists():
        return PlainTextResponse("No server log yet.", status_code=404)
    lines = SERVER_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    response = "\n".join(lines[-tail:])
    return PlainTextResponse(response, media_type="text/plain")


@app.get("/generation-log")
def generation_log(name: str, tail: int = 200):
    video_path = get_video_path(name)
    log_path = get_hls_dir(video_path) / "generation.log"
    if not log_path.exists():
        return PlainTextResponse("No generation log for this video.", status_code=404)
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    response = "\n".join(lines[-tail:])
    return PlainTextResponse(response, media_type="text/plain")


@app.get("/watch")
def watch(video: str, mode: str = "normal"):
    if mode not in {"normal", "framegen"}:
        raise HTTPException(status_code=400, detail="Invalid mode")

    video_path = get_video_path(video)
    display_name = format_display_name(video_path)
    hls_dir = get_hls_dir(video_path)
    hls_url = f"/hls/{quote(str(hls_dir.name))}/stream.m3u8"

    if mode == "framegen":
        write_server_log(f"Watch request for framegen {video_path}")
        if not is_hls_ready(video_path):
            key = sanitize_name(str(video_path.relative_to(get_movies_root())))
            if key not in generation_jobs or generation_jobs[key]["process"].poll() is not None:
                write_server_log(f"No active generation job found for {video_path}, starting a new one")
                run_framegen(video_path)
            else:
                write_server_log(f"Found active generation job for {video_path} pid={generation_jobs[key]['pid']}")
            status = generation_jobs.get(key, {})
            status_text = (
                "Generating"
                if status.get("success") is None
                else ("Ready" if status.get("success") else "Failed")
            )
            return HTMLResponse(
                f"""
<!DOCTYPE html>
<html>
<head>
    <title>Preparing Framegen</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }}
        a {{ color: #38bdf8; }}
    </style>
</head>
<body>
    <h1>Preparing Framegen for {display_name}</h1>
    <p>Your framegen stream is being prepared. Refresh this page in a few seconds.</p>
    <p>Status: {status_text}</p>
    <p><a href='/'>Back to list</a></p>
</body>
</html>
"""
            )
        source_url = hls_url
        player_script = """
    <script src='https://cdn.jsdelivr.net/npm/hls.js@latest'></script>
    <script>
        const videoEl = document.getElementById('video');
        const hlsUrl = '%s';
        if (Hls.isSupported()) {
            const hls = new Hls();
            hls.loadSource(hlsUrl);
            hls.attachMedia(videoEl);
        } else if (videoEl.canPlayType('application/vnd.apple.mpegurl')) {
            videoEl.src = hlsUrl;
        }
    </script>
""" % hls_url
        content_type = "application/vnd.apple.mpegurl"
    else:
        source_url = f"/video?name={quote(str(video))}"
        player_script = ""
        content_type = "video/mp4"

    return HTMLResponse(
        f"""
<!DOCTYPE html>
<html>
<head>
    <title>Playing {display_name}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
        a {{ color: #38bdf8; }}
        .wrapper {{ max-width: 1000px; margin: auto; padding: 20px; }}
        video {{ width: 100%; max-width: 920px; border-radius: 12px; background: #000; }}
    </style>
</head>
<body>
    <div class='wrapper'>
        <h1>{display_name}</h1>
        <p>Mode: {mode}</p>
        <video id='video' controls autoplay>
            <source src='{source_url}' type='{content_type}'>
            Your browser does not support this video format.
        </video>
        <p><a href='/'>Back to list</a></p>
    </div>
    {player_script}
</body>
</html>
"""
    )
