import os
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

print("GPU:", torch.cuda.get_device_name(0))


def pad_tensor(img):
    """Pads a torch tensor (1, C, H, W) so H and W are multiples of 64 for multi-scale RIFE blocks."""
    _, _, h, w = img.shape
    ph = ((h + 63) // 64) * 64 - h
    pw = ((w + 63) // 64) * 64 - w
    if ph > 0 or pw > 0:
        img = torch.nn.functional.pad(img, (0, pw, 0, ph), mode="reflect")
    return img, h, w


def interpolate(frame1, frame2):
    img0 = (
        torch.from_numpy(frame1.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .cuda()
        / 255.0
    )
    img1 = (
        torch.from_numpy(frame2.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .cuda()
        / 255.0
    )

    img0, h, w = pad_tensor(img0)
    img1, _, _ = pad_tensor(img1)

    with torch.no_grad():
        mid = model.inference(img0, img1, 0.5, 1.0)

    mid = mid[:, :, :h, :w]

    mid = (
        mid[0]
        .clamp(0, 1)
        .mul(255)
        .byte()
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
    print("Input resolution:", width, height)

    # GPU-Accelerated Decoder
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
    )

    # GPU-Accelerated Encoder with Audio passthrough/transcode
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
            "p5",
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
    )

    frame_size = width * height * 3
    previous = None

    while True:
        raw = decoder.stdout.read(frame_size)
        if len(raw) != frame_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()

        if previous is not None:
            middle = interpolate(previous, frame)

            encoder.stdin.write(previous.tobytes())
            encoder.stdin.write(middle.tobytes())

        previous = frame

    if previous is not None:
        encoder.stdin.write(previous.tobytes())

    encoder.stdin.close()
    encoder.wait()


@app.on_event("startup")
def startup():
    threading.Thread(target=generate, daemon=True).start()


app.mount("/hls", StaticFiles(directory=HLS_DIR), name="hls")


@app.get("/original")
def get_original():
    """Serves the original un-interpolated video file."""
    return FileResponse(INPUT, media_type="video/mp4")


@app.get("/")
def home():
    return HTMLResponse(
        """
<!DOCTYPE html>
<html>
<head>
    <title>RIFE Frame Generation Comparison</title>
    <style>
        body { font-family: sans-serif; background: #121212; color: #fff; text-align: center; margin: 20px; }
        .container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 20px; }
        .card { background: #1e1e1e; padding: 15px; border-radius: 8px; }
        video { width: 600px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>RIFE Frame Generation Comparison</h1>
    <div class="container">
        <div class="card">
            <h3>Original Video</h3>
            <video controls autoplay muted loop src="/original"></video>
        </div>
        <div class="card">
            <h3>RIFE Interpolated Stream (60 FPS)</h3>
            <video controls autoplay width="600">
                <source src="/hls/stream.m3u8" type="application/x-mpegURL">
            </video>
        </div>
    </div>
</body>
</html>
"""
    )