# StreamerFrames

Raise low-frame-rate video (23.976/24/25/30 fps films) to a higher frame rate with RIFE on a local NVIDIA GPU,
either offline (render a file) or streamed to a browser while it generates. Built on Practical-RIFE (see
`README.md`); the design and its rationale are in `PLAN.md`.

## Setup (Windows, GTX 1080 Ti / Pascal)

1. Python 3.11 venv (the launchers expect `..\.framegen`).
2. Torch with CUDA 12.6 wheels (cu128 and newer dropped Pascal):
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
3. `pip install -r requirements-streamerframes.txt`
4. ffmpeg with `h264_nvenc` on PATH (e.g. the gyan.dev full build).
5. A RIFE model: put `IFNet_HDv3.py` and `flownet.pkl` in `train_log/` (4.25 recommended; links in `README.md`).
6. Optional: copy `streamerframes.example.toml` to `streamerframes.toml` and edit paths.
7. `python -m streamerframes check` must pass.

## Command line

```
python -m streamerframes check                      # GPU, torch arch, ffmpeg/NVENC, model files
python -m streamerframes probe <video>              # stream info as JSON
python -m streamerframes plan <video> --target-fps 60
python -m streamerframes bench [<video> --crop | --size 1920x800]   # calibrate scale=auto
python -m streamerframes render <video> [--profile quality|realtime] [--target-fps 60 | --multi 2]
                                [--scale auto|1.0|0.5] [--export out.mp4 | --no-export]
python -m streamerframes cache ls | rm <video_id|path> | gc [--max-gb N]
```

`render` is resumable: Ctrl+C stops after the current segment, killing it loses at most one segment, and
running the same command again continues. It exports MP4 (MKV when the source has bitmap subtitles) with the
source's audio and subtitles copied.

## How it works

- **Exact timing.** Frame rates are fractions (`24000/1001 × 2 = 48000/1001`, `60` → `60000/1001` for NTSC
  sources). Output frame `j` maps to source position `j × src/out`; integer positions are the source frame
  byte-for-byte, the rest are RIFE at that `t`. Hard cuts show the nearer source frame instead of a blend.
- **Segment cache.** Output is fixed-length MPEG-TS segments (about 4 s, a multiple of the frame-rate
  period so each one starts on a source frame, each starting with an IDR). A segment counts only once ffmpeg
  lists it in `index/run_*.csv`. A run always starts at the first missing segment, decoding from its exact
  source frame, so resume and seek are frame-exact and never redo finished work.
- **Nothing is "complete" unless it is.** Both ffmpeg exit codes, the decoded frame count, and every
  segment's end time are checked; failures keep the last 50 lines of the ffmpeg log in `manifest.json`.
- **Throughput.** Reader thread → GPU → writer thread over pinned host buffers; the GPU thread never calls
  `synchronize()` (the writer waits on a CUDA event per frame). Source frames are uploaded and converted once
  (YUV420 ↔ RGB on the GPU with the right BT.709/601 matrix). CUDA graphs replay the network for the fixed
  frame size. Letterboxed films are detected once (`cropdetect` at 8 points) and only the picture is inferred.
- **scale=auto.** `bench` records ms/frame per (GPU, model, size, scale, graphs) in `cache/calibration.json`.
  `realtime` picks the highest-quality scale that sustains 1.1× the needed interpolation rate. The choice is
  remembered per video, so the cache id never changes under you.

## Measured performance

To be filled in on the 1080 Ti (PLAN.md section 2 has the model-only numbers): `bench` at 1920×800 for
scale 1.0/0.5 with and without CUDA graphs, and end-to-end `render` src fps for `Batman Begins.mp4`.

## Tests

`pip install pytest httpx fastapi && pytest`. Everything runs on CPU with real ffmpeg and a fake IFNet
(`tests/fake_model.py`), so CI needs no GPU. GPU-only behaviour (NVENC options, `-hwaccel cuda` seek accuracy,
CUDA graphs, real-model quality and speed) needs checking on the target machine.
