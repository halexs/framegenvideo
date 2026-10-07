import subprocess
import sys
import os
import torch


# ==============================
# Configuration
# ==============================

INPUT = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "output_60fps.mp4"

EXP = "1"   # 2x interpolation (24fps -> 48fps, 30fps -> 60fps)
# EXP=2 would be 4x interpolation


# ==============================
# Check files
# ==============================

if not os.path.exists(INPUT):
    print(f"Input video not found: {INPUT}")
    sys.exit(1)


# ==============================
# Check CUDA
# ==============================

print("Checking CUDA...")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "Install CUDA-enabled PyTorch first."
    )

print("GPU:", torch.cuda.get_device_name(0))


# ==============================
# Run Practical-RIFE
# ==============================

cmd = [
    sys.executable,
    "inference_video.py",
    "--video",
    INPUT,
    "--output",
    OUTPUT,
    "--exp",
    EXP,
]


print("\nRunning:")
print(" ".join(cmd))
print()


subprocess.run(
    cmd,
    check=True
)


print("\nDone!")
print("Output:", OUTPUT)