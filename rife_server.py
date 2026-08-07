import os
import queue
import subprocess
import threading

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch

from train_log.RIFE_HDv3 import Model

INPUT = "input.mp4"
HLS_DIR = "hls"

os.makedirs(HLS_DIR, exist_ok=True)

app = FastAPI()

print("Loading RIFE...")
model = Model()
model.load_model("train_log", -1)
model.eval()
model.device()

# Enable PyTorch CUDNN Benchmarking for static shape optimization
torch.backends.cudnn.benchmark = True

print("GPU:", torch.cuda.get_device_name(0))


def pad_tensor(img):
    """Pads a torch tensor (1, C, H, W) so H and W are multiples of 64."""
    _, _, h, w = img.shape
    ph = ((h + 63) // 64) * 64 - h
    pw = ((w + 63) // 64) * 64 - w
    if ph > 0 or pw > 0:
        img = torch.nn.functional.pad(img, (0, pw, 0, ph), mode="reflect")
    return img, h, w


def interpolate(frame1, frame2, scale_factor=0.5):
    # Direct CUDA allocation without non-blocking wrapper overhead
    img0 = (
        torch.from_numpy(frame1)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .cuda()
        .float()
        / 255.0
    )
    img1 = (
        torch.from_numpy(frame2)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .cuda()
        .float()
        / 255.0
    )

    img0, h, w = pad_tensor(img0)
    img1, _, _ = pad_tensor(img1)

    with torch.inference_mode():
        mid = model.inference(img0, img1, 0.5, scale_factor)

    mid = mid[:, :, :h, :w]

    mid = (
        mid[0]
        .clamp(0, 1)
        .mul(255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    return mid


def generate():
    probe = (
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                INPUT,
            ]
        )
        .decode()
        .strip()
    )

    width, height = map(int, probe.split("x"))
    print(f"Input resolution: {width}x{height}")

    # Aggressive downscaling for flow computation on high resolutions
    if width >= 2560 or height >= 1440:
        scale_factor = 0.5
        print("High resolution detected: Using scale_factor = 0.5")
    else:
        scale_factor = 1.0
        print("Standard resolution detected: Using scale_factor = 1.0")

    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-i",
            INPUT,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
    )

    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "60",
            "-i",
            "-",
            "-i",
            INPUT,
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "p1",  # Fastest NVENC Encoding Preset
            "-tune",
            "ll",  # Low Latency Tuning
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-f",
            "hls",
            "-hls_time",
            "4",
            "-hls_list_size",
            "0",
            f"{HLS_DIR}/stream.m3u8",
        ],
        stdin=subprocess.PIPE,
        bufsize=10**8,
    )

    frame_size = width * height * 3

    # Thread-safe pipeline queues
    raw_queue = queue.Queue(maxsize=4)
    out_queue = queue.Queue(maxsize=8)

    def read_frames():
        while True:
            raw = decoder.stdout.read(frame_size)
            if len(raw) != frame_size:
                raw_queue.put(None)
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            raw_queue.put(frame)

    def write_frames():
        while True:
            data = out_queue.get()
            if data is None:
                break
            encoder.stdin.write(data)
            out_queue.task_done()

    # Launch background I/O threads
    reader_thread = threading.Thread(target=read_frames, daemon=True)
    writer_thread = threading.Thread(target=write_frames, daemon=True)
    reader_thread.start()
    writer_thread.start()

    previous = None

    while True:
        frame = raw_queue.get()
        if frame is None:
            break

        if previous is not None:
            middle = interpolate(previous, frame, scale_factor=scale_factor)

            out_queue.put(previous.tobytes())
            out_queue.put(middle.tobytes())

        previous = frame

    if previous is not None:
        out_queue.put(previous.tobytes())

    out_queue.put(None)
    writer_thread.join()

    encoder.stdin.close()
    encoder.wait()


@app.on_event("startup")
def startup():
    threading.Thread(target=generate, daemon=True).start()


app.mount("/hls", StaticFiles(directory=HLS_DIR), name="hls")


@app.get("/original")
def get_original():
    return FileResponse(INPUT, media_type="video/mp4")


@app.get("/compare")
def compare_page():
    return HTMLResponse(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Synchronized Comparison</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; text-align: center; margin: 0; padding: 20px; }
        h1 { margin-bottom: 10px; }
        .controls { margin: 20px 0; }
        button {
            background: #2563eb; color: #fff; border: none; padding: 12px 24px;
            font-size: 16px; font-weight: bold; border-radius: 6px; cursor: pointer;
            transition: background 0.2s ease;
        }
        button:hover { background: #1d4ed8; }
        .grid { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
        .card { background: #1e293b; padding: 16px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
        video { width: 560px; border-radius: 8px; background: #000; }
    </style>
</head>
<body>
    <h1>RIFE Synchronized Comparison</h1>
    <div class="controls">
        <button onclick="playBoth()">▶ Play Simultaneously from Start</button>
        <button onclick="pauseBoth()">⏸ Pause Both</button>
    </div>
    <div class="grid">
        <div class="card">
            <h3>Original Input</h3>
            <video id="v1" controls src="/original" muted></video>
        </div>
        <div class="card">
            <h3>RIFE 60 FPS Interpolated</h3>
            <video id="v2" controls>
                <source src="/hls/stream.m3u8" type="application/x-mpegURL">
            </video>
        </div>
    </div>

    <script>
        const v1 = document.getElementById('v1');
        const v2 = document.getElementById('v2');

        function playBoth() {
            v1.currentTime = 0;
            v2.currentTime = 0;
            v1.play();
            v2.play();
        }

        function pauseBoth() {
            v1.pause();
            v2.pause();
        }

        v1.addEventListener('seeked', () => { v2.currentTime = v1.currentTime; });
        v2.addEventListener('seeked', () => { v1.currentTime = v2.currentTime; });
    </script>
</body>
</html>
"""
    )


@app.get("/")
def home():
    return HTMLResponse(
        """
<!DOCTYPE html>
<html>
<head>
    <title>RIFE Frame Generation</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; text-align: center; padding: 40px; }
        a { color: #38bdf8; font-size: 18px; text-decoration: none; margin: 0 10px; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>RIFE Live Stream Server</h1>
    <p>Select a route to inspect output:</p>
    <a href="/compare">Go to Synchronized Comparison Page (/compare)</a>
</body>
</html>
"""
    )