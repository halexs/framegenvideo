import subprocess
import sys
import os
import tempfile
import shutil
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available!")

print("GPU:", torch.cuda.get_device_name(0))


INPUT = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "output_60fps.mp4"


if not os.path.exists(INPUT):
    print(f"Input file not found: {INPUT}")
    sys.exit(1)

# Temporary directory for extracted frames
temp_dir = tempfile.mkdtemp(prefix="rife_")

try:
    frames_dir = os.path.join(temp_dir, "frames")
    output_dir = os.path.join(temp_dir, "output")

    os.makedirs(frames_dir)
    os.makedirs(output_dir)

    print("Extracting frames...")

    # cpu only
    # subprocess.run([
    #     "ffmpeg",
    #     "-y",
    #     "-i", INPUT,
    #     "-qscale:v", "2",
    #     os.path.join(frames_dir, "%08d.png")
    # ], check=True)

    # # gpu hardware accel
    # subprocess.run([
    #     "ffmpeg",
    #     "-y",
    #     "-hwaccel", "cuda",
    #     "-hwaccel_output_format", "cuda",
    #     "-i", INPUT,
    #     "-qscale:v", "2",
    #     os.path.join(frames_dir, "%08d.png")
    # ], check=True)

    # recommended for some reason
    subprocess.run([
        "ffmpeg",
        "-y",
        "-hwaccel", "cuda",
        "-i", INPUT,
        "-qscale:v", "2",
        os.path.join(frames_dir, "%08d.png")
    ], check=True)

    print("Running RIFE...")

    # Practical-RIFE's inference script.
    #
    # The exact command can change between RIFE versions,
    # so adjust this command to match the repository's
    # current inference interface.
    subprocess.run([
        sys.executable,
        "inference_video.py",
        "--video", frames_dir,
        "--output", output_dir,
        "--exp", "1"
    ], check=True)

    print("Encoding 60 FPS video...")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-framerate", "60",
        "-i", os.path.join(output_dir, "%08d.png"),
        "-i", INPUT,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-cq", "20",
        "-c:a", "copy",
        "-r", "60",
        OUTPUT
    ], check=True)

    print()
    print("Done!")
    print(f"Output: {OUTPUT}")

finally:
    shutil.rmtree(temp_dir, ignore_errors=True)