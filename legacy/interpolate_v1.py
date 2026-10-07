import subprocess
import numpy as np
import torch
import cv2
import time

# from model.RIFE import Model

INPUT = "input.mp4"
OUTPUT = "output.mp4"

WIDTH = 1920
HEIGHT = 1080

FRAME_SIZE = WIDTH * HEIGHT * 3


####################################################
# CUDA
####################################################

assert torch.cuda.is_available()

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))

####################################################
# Load RIFE
####################################################

from train_log.RIFE_HDv3 import Model

model = Model()

if not hasattr(model, 'version'):
    model.version = 0

model.load_model("train_log", -1)

model.eval()
model.device()

print("RIFE model loaded")


####################################################
# FFmpeg Decoder
####################################################

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
        "-"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)


####################################################
# FFmpeg Encoder
####################################################

encoder = subprocess.Popen(
    [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        "60",
        "-i",
        "-",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        OUTPUT
    ],
    stdin=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)


####################################################
# Read first frame
####################################################

raw = decoder.stdout.read(FRAME_SIZE)

frame0 = np.frombuffer(raw, np.uint8)

frame0 = frame0.reshape((HEIGHT, WIDTH, 3))


frame_count = 0
start = time.time()


####################################################
# Main Loop
####################################################

while True:

    raw = decoder.stdout.read(FRAME_SIZE)

    if len(raw) != FRAME_SIZE:
        break

    frame1 = np.frombuffer(raw, np.uint8)

    frame1 = frame1.reshape((HEIGHT, WIDTH, 3))


    #############################################
    # Convert to Torch
    #############################################

    t0 = (
        torch.from_numpy(frame0)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    ).cuda(non_blocking=True)

    t1 = (
        torch.from_numpy(frame1)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    ).cuda(non_blocking=True)


    #############################################
    # RIFE
    #############################################

    #
    # THIS IS THE ONLY PART THAT WILL NEED
    # TO MATCH YOUR VERSION OF PRACTICAL-RIFE
    #

    # middle = model.inference(t0, t1)
    # middle = model.inference(t0, t1, 0.5, 1.0)

    with torch.no_grad():
        middle = model.inference(
            t0,
            t1,
            0.5,
            1.0
        )


    #############################################
    # Write frame 0
    #############################################

    encoder.stdin.write(frame0.tobytes())


    #############################################
    # Write interpolated frame
    #############################################

    out = (
        middle.squeeze(0)
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    encoder.stdin.write(out.tobytes())


    frame0 = frame1

    frame_count += 1

    if frame_count % 60 == 0:

        elapsed = time.time() - start

        fps = frame_count / elapsed

        print(
            f"{fps:.1f} source fps   "
            f"{fps*2:.1f} output fps   "
            f"{torch.cuda.memory_allocated()/1024**2:.0f} MB"
        )


####################################################
# Last frame
####################################################

encoder.stdin.write(frame0.tobytes())

encoder.stdin.close()

decoder.wait()

encoder.wait()

print("Done.")