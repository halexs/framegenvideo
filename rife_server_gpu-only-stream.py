import os
import subprocess
import threading
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
import torchvision.io as io

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

torch.backends.cudnn.benchmark = True
print("GPU:", torch.cuda.get_device_name(0))


def pad_tensor(img):
    _, _, h, w = img.shape
    ph = ((h + 63) // 64) * 64 - h
    pw = ((w + 63) // 64) * 64 - w
    if ph > 0 or pw > 0:
        img = torch.nn.functional.pad(img, (0, pw, 0, ph), mode="reflect")
    return img, h, w


def interpolate_gpu(img0, img1, scale_factor=0.5):
    img0, h, w = pad_tensor(img0)
    img1, _, _ = pad_tensor(img1)

    with torch.inference_mode():
        mid = model.inference(img0, img1, 0.5, scale_factor)

    mid = mid[:, :, :h, :w]
    return mid.clamp(0, 1)


def generate():
    # 1. Read Video directly into 1080 Ti VRAM using TorchVision NVDEC
    # Set video backend to CUDA hardware accelerated decoder
    io.set_video_backend("cuda")
    
    # Read frames directly as GPU tensors
    # video_frames shape: [T, C, H, W], dtype: torch.uint8 on cuda:0
    video_frames, _, info = io.read_video(INPUT, pts_unit="sec", output_format="TCHW")
    video_frames = video_frames.to("cuda", non_blocking=True)

    num_frames, _, height, width = video_frames.shape
    fps = info.get("video_fps", 30)

    print(f"Input resolution: {width}x{height} @ {fps} FPS across {num_frames} frames")

    scale_factor = 0.5 if (width >= 2560 or height >= 1440) else 1.0

    # 2. Setup FFmpeg Encoder for NVENC output
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps * 2),
            "-i", "-",
            "-i", INPUT,
            "-map", "0:v",
            "-map", "1:a?",
            "-c:v", "h264_nvenc",
            "-pix_fmt", "yuv420p",
            "-preset", "p1",
            "-tune", "ll",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-f", "hls",
            "-hls_time", "4",
            "-hls_list_size", "0",
            f"{HLS_DIR}/stream.m3u8",
        ],
        stdin=subprocess.PIPE,
        bufsize=10**8,
    )

    def write_tensor_to_encoder(tensor):
        """Converts float32 GPU tensor [1, 3, H, W] to uint8 bytes for NVENC stdin."""
        frame_uint8 = (tensor.squeeze(0).permute(1, 2, 0) * 255.0).to(torch.uint8)
        # Direct GPU pinned memory copy to stdin stream
        encoder.stdin.write(frame_uint8.cpu().numpy().tobytes())

    # 3. Process interpolation loop entirely in GPU VRAM
    for i in range(num_frames - 1):
        frame0 = (video_frames[i : i + 1].float() / 255.0)
        frame1 = (video_frames[i + 1 : i + 2].float() / 255.0)

        mid_frame = interpolate_gpu(frame0, frame1, scale_factor=scale_factor)

        write_tensor_to_encoder(frame0)
        write_tensor_to_encoder(mid_frame)

    # Write final frame
    last_frame = (video_frames[-1:] .float() / 255.0)
    write_tensor_to_encoder(last_frame)

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
        .controls { margin: 20px 0; }
        button { background: #2563eb; color: #fff; border: none; padding: 12px 24px; font-size: 16px; font-weight: bold; border-radius: 6px; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        .grid { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
        .card { background: #1e293b; padding: 16px; border-radius: 12px; }
        video { width: 560px; border-radius: 8px; background: #000; }
    </style>
</head>
<body>
    <h1>RIFE Synchronized Comparison</h1>
    <div class="controls">
        <button onclick="playBoth()">▶ Play Simultaneously</button>
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
        function playBoth() { v1.currentTime = 0; v2.currentTime = 0; v1.play(); v2.play(); }
        function pauseBoth() { v1.pause(); v2.pause(); }
    </script>
</body>
</html>
"""
    )


@app.get("/")
def home():
    return HTMLResponse('<a href="/compare">Go to Synchronized Comparison Page (/compare)</a>')