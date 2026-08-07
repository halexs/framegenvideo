#!E:\StreamerFrames\.framegen\Scripts\python.exe
import argparse
import os
import queue
import subprocess
import threading
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch

try:
    import torchvision.io as tvio
except ImportError:  # pragma: no cover - dependency may be absent in some environments
    tvio = None

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

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this server")

print("GPU:", torch.cuda.get_device_name(0))


def pad_tensor(img):
    """Pads a torch tensor (1, C, H, W) so H and W are multiples of 64."""
    _, _, h, w = img.shape
    ph = ((h + 63) // 64) * 64 - h
    pw = ((w + 63) // 64) * 64 - w
    if ph > 0 or pw > 0:
        img = torch.nn.functional.pad(img, (0, pw, 0, ph), mode="reflect")
    return img, h, w


def prepare_frame_tensor(frame):
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame).permute(2, 0, 1)

    if frame.dim() == 3:
        frame = frame.unsqueeze(0)

    if frame.device.type != "cuda":
        frame = frame.to("cuda", non_blocking=True)

    if frame.dtype != torch.float32:
        frame = frame.float() / 255.0

    return frame


def interpolate(frame1, frame2, scale_factor=0.5):
    img0 = prepare_frame_tensor(frame1)
    img1 = prepare_frame_tensor(frame2)

    img0, h, w = pad_tensor(img0)
    img1, _, _ = pad_tensor(img1)

    with torch.inference_mode():
        mid = model.inference(img0, img1, 0.5, scale_factor)

    return mid[:, :, :h, :w].clamp(0, 1)


def serialize_frame(frame, pinned_buffer=None):
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame).permute(2, 0, 1)

    if frame.dim() == 4:
        frame = frame.squeeze(0)

    if frame.dim() != 3:
        raise ValueError(f"Expected a 3D frame tensor, got shape {tuple(frame.shape)}")

    if frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = frame.permute(1, 2, 0)
    elif frame.shape[-1] != 3:
        raise ValueError(f"Unsupported frame tensor shape {tuple(frame.shape)}")

    if frame.dtype != torch.uint8:
        frame = frame.mul(255.0).clamp(0, 255).to(torch.uint8)

    frame = frame.contiguous()
    if frame.device.type == "cuda":
        if pinned_buffer is None or pinned_buffer.shape != frame.shape or pinned_buffer.dtype != torch.uint8:
            pinned_buffer = torch.empty(frame.shape, dtype=torch.uint8, device="cpu", pin_memory=True)
        pinned_buffer.copy_(frame, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return pinned_buffer.numpy().tobytes()

    if pinned_buffer is not None and pinned_buffer.shape == frame.shape and pinned_buffer.dtype == torch.uint8:
        pinned_buffer.copy_(frame)
        return pinned_buffer.numpy().tobytes()

    return frame.numpy().tobytes()


def probe_video_metadata(path):
    probe = (
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "csv=s=x:p=0",
                path,
            ]
        )
        .decode()
        .strip()
    )

    parts = probe.split("x")
    width = int(parts[0])
    height = int(parts[1])

    if len(parts) < 3:
        fps = 30.0
    elif "/" in parts[2]:
        num, den = map(int, parts[2].split("/"))
        fps = num / den if den != 0 else 30.0
    else:
        fps = float(parts[2])

    return width, height, fps


def generate(input_path=None, hls_dir=None):
    input_path = input_path or INPUT
    hls_dir = hls_dir or HLS_DIR
    os.makedirs(hls_dir, exist_ok=True)

    width, height, fps = probe_video_metadata(input_path)
    print(f"Input resolution: {width}x{height} @ {fps:.2f} FPS")

    if width >= 2560 or height >= 1440:
        scale_factor = 0.5
        print("High resolution detected: Using scale_factor = 0.5")
    else:
        scale_factor = 1.0
        print("Standard resolution detected: Using scale_factor = 1.0")

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
            str(int(round(fps * 2))),
            "-i",
            "-",
            "-i",
            input_path,
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "p4",
            "-tune",
            "ll",
            "-rc",
            "cbr",
            "-bf",
            "2",
            "-maxrate",
            "20M",
            "-bufsize",
            "20M",
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
            f"{hls_dir}/stream.m3u8",
        ],
        stdin=subprocess.PIPE,
        bufsize=10**8,
    )

    use_cuda_decode = False
    if tvio is not None:
        try:
            tvio.set_video_backend("cuda")
            video_frames, _, info = tvio.read_video(
                input_path,
                pts_unit="sec",
                output_format="TCHW",
            )
            if video_frames.numel() > 0:
                video_frames = video_frames.to("cuda", non_blocking=True)
                fps = info.get("video_fps", fps)
                use_cuda_decode = True
                print(
                    f"Using TorchVision CUDA decode for {video_frames.shape[0]} frames"
                )
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"Falling back to ffmpeg decode path: {exc}")

    if use_cuda_decode:
        prev_frame_gpu = None
        prev_frame_uint8 = None
        out_buffer = torch.empty((height, width, 3), dtype=torch.uint8, device="cpu", pin_memory=True)

        for idx in range(video_frames.shape[0]):
            curr_frame_uint8 = video_frames[idx]
            curr_frame_gpu = curr_frame_uint8.float() / 255.0

            if prev_frame_gpu is not None:
                mid = interpolate(prev_frame_gpu, curr_frame_gpu, scale_factor=scale_factor)
                encoder.stdin.write(serialize_frame(prev_frame_uint8, pinned_buffer=out_buffer))
                encoder.stdin.write(serialize_frame(mid, pinned_buffer=out_buffer))

            prev_frame_gpu = curr_frame_gpu
            prev_frame_uint8 = curr_frame_uint8

        if prev_frame_uint8 is not None:
            encoder.stdin.write(serialize_frame(prev_frame_uint8, pinned_buffer=out_buffer))

        encoder.stdin.close()
        encoder.wait()
        return

    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-i",
            input_path,
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

    frame_size = width * height * 3

    raw_queue = queue.Queue(maxsize=4)
    out_queue = queue.Queue(maxsize=8)
    out_buffer = torch.empty((height, width, 3), dtype=torch.uint8, device="cpu", pin_memory=True)

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
            out_queue.put(serialize_frame(previous, pinned_buffer=out_buffer))
            out_queue.put(serialize_frame(middle, pinned_buffer=out_buffer))

        previous = frame

    if previous is not None:
        out_queue.put(serialize_frame(previous, pinned_buffer=out_buffer))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RIFE HLS output or run the FastAPI app.")
    parser.add_argument("--input", default=INPUT, help="Input video path")
    parser.add_argument("--hls-dir", default=HLS_DIR, help="Output HLS directory")
    parser.add_argument("--generate-only", action="store_true", help="Generate HLS for the input and exit")
    args = parser.parse_args()

    if args.generate_only:
        generate(args.input, args.hls_dir)
    else:
        try:
            import uvicorn
        except ImportError:
            raise RuntimeError("uvicorn is required to run the app directly. Use `uvicorn rife_server:app` instead.")
        uvicorn.run("rife_server:app", host="0.0.0.0", port=8000)
