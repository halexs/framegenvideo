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

# Benchmark mode for static shape optimization on GTX 1080 Ti
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


def interpolate_gpu(img0, img1, scale_factor=0.5):
    """Runs RIFE frame interpolation entirely inside 1080 Ti VRAM."""
    img0, h, w = pad_tensor(img0)
    img1, _, _ = pad_tensor(img1)

    with torch.inference_mode():
        mid = model.inference(img0, img1, 0.5, scale_factor)

    return mid[:, :, :h, :w].clamp(0, 1)


def generate():
    # Probe input dimensions and framerate
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=s=x:p=0",
            INPUT,
        ]
    ).decode().strip()

    parts = probe.split("x")
    width = int(parts[0])
    height = int(parts[1])
    
    # Calculate FPS from fraction (e.g. "60/1" or "30000/1001")
    if "/" in parts[2]:
        num, den = map(int, parts[2].split("/"))
        fps = num / den if den != 0 else 30
    else:
        fps = float(parts[2])

    print(f"Input resolution: {width}x{height} @ {fps:.2f} FPS")

    scale_factor = 0.5 if (width >= 2560 or height >= 1440) else 1.0

    # 1. FFmpeg Decoder: Accelerated by CUDA NVDEC
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-i", INPUT,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
    )

    # 2. FFmpeg Encoder: Accelerated by CUDA NVENC
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(int(fps * 2)),
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

    frame_size = width * height * 3
    raw_queue = queue.Queue(maxsize=4)
    out_queue = queue.Queue(maxsize=8)

    def read_frames():
        """Reads raw bytes into CUDA-Pinned CPU memory for direct DMA transfers."""
        while True:
            raw = decoder.stdout.read(frame_size)
            if len(raw) != frame_size:
                raw_queue.put(None)
                break
            
            # Wrap bytes in a numpy array and pin memory for rapid PCIe DMA copy
            frame_np = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            tensor_pinned = torch.from_numpy(frame_np).pin_memory()
            raw_queue.put(tensor_pinned)

    def write_frames():
        """Writes output frames sent back from GPU memory."""
        while True:
            data = out_queue.get()
            if data is None:
                break
            encoder.stdin.write(data)
            out_queue.task_done()

    # Launch threaded background I/O
    threading.Thread(target=read_frames, daemon=True).start()
    threading.Thread(target=write_frames, daemon=True).start()

    prev_gpu_tensor = None

    while True:
        frame_pinned = raw_queue.get()
        if frame_pinned is None:
            break

        # Fast non-blocking DMA transfer directly into 1080 Ti CUDA memory
        curr_gpu_tensor = (
            frame_pinned.to("cuda", non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )

        if prev_gpu_tensor is not None:
            # 3. RIFE Inference entirely in GPU VRAM
            mid_gpu_tensor = interpolate_gpu(
                prev_gpu_tensor, curr_gpu_tensor, scale_factor=scale_factor
            )

            # Convert GPU float tensors back to uint8 byte streams for encoder
            prev_bytes = (
                (prev_gpu_tensor[0].permute(1, 2, 0) * 255.0)
                .to(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
            mid_bytes = (
                (mid_gpu_tensor[0].permute(1, 2, 0) * 255.0)
                .to(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )

            out_queue.put(prev_bytes)
            out_queue.put(mid_bytes)

        prev_gpu_tensor = curr_gpu_tensor

    # Write the final frame
    if prev_gpu_tensor is not None:
        last_bytes = (
            (prev_gpu_tensor[0].permute(1, 2, 0) * 255.0)
            .to(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
        out_queue.put(last_bytes)

    out_queue.put(None)

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