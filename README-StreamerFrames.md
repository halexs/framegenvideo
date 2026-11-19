# StreamerFrames

Raise low-frame-rate video (23.976/24/25/30 fps films) to a higher frame rate with RIFE on a local NVIDIA GPU,
either offline (render a file) or streamed to a browser while it generates. Built on Practical-RIFE (see
`README.md`); the design and its rationale are in `PLAN.md`.

What to do next on the Windows PC (benchmarks, real-GPU checks, Jellyfin) is in `NEXT_STEPS.md`.

## Setup (Windows, GTX 1080 Ti / Pascal)

1. Python 3.11 venv (the launchers expect `..\.framegen`).
2. Torch with CUDA 12.6 wheels (cu128 and newer dropped Pascal):
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
3. `pip install -r requirements-streamerframes.txt`
4. ffmpeg with `h264_nvenc` on PATH (e.g. the gyan.dev full build).
5. A RIFE model: put `IFNet_HDv3.py` and `flownet.pkl` in `train_log/` (4.25 recommended; links in `README.md`).
6. Optional: copy `streamerframes.example.toml` to `streamerframes.toml` and edit paths.
7. `python -m streamerframes check` must pass.

## Web UI

`start_server.bat` starts the UI in the background (no window) and opens http://localhost:8000; `stop_server.bat`
stops it. `start_server.bat -Status` lists the addresses it's reachable on, `-NoBrowser` skips the browser and
`-Foreground` runs it in the window with live output (same as `python -m streamerframes serve`). Startup errors go
to `server-console.log`. `create_schtask.bat` starts it in the background at every logon.

An idle server doesn't keep the PC awake, so normal sleep settings apply. A running generation job holds off
sleep until it finishes. Playback from any device (this PC, a phone, the TV) also holds it off: every fetch of a
video segment or of the original file counts, and sleep is allowed again about two minutes after the last one.
A paused player stops fetching, so pausing releases the PC; an open page that only polls status does not keep it
awake. This is covered by unit tests (`tests/test_awake.py`) but not yet confirmed on the Windows machine; see
`NEXT_STEPS.md` step 4.

- **Library:** every video under `movies_roots`, with its generation status and buttons: Original,
  Framegen (stream now), Generate in background (offline `quality` job), Export, Download, Delete cache.
- **Framegen** starts a `stream` job and plays the HLS output while it is generated. The status line shows
  how much is ready, speed (× real time), and whether playing now will run without stalling
  (lead `B ≥ remaining × (1 − speed)`); otherwise how long to wait. Playback starts by itself once it's safe.
- **Seeking works anywhere**, even far ahead of what's generated: the playlist lists the whole movie up
  front, and asking for a segment that doesn't exist yet either waits for it (if it's within ~3 segments of
  the current run) or restarts generation right there (debounced, so scrubbing doesn't thrash). Holes left
  behind are filled afterwards by a background job (`fill_gaps`), so the cache still ends up complete.
- **Compare** shows the original and the framegen version side by side, synced.
- **One GPU worker at a time.** Streaming outranks offline jobs (offline work pauses after its current
  segment and resumes afterwards); by default a new stream replaces an older one (`stream_policy`). Jobs
  survive server restarts: running workers are adopted, interrupted offline jobs resume (`auto_resume`).
- Host is `127.0.0.1` by default. Set `host = "0.0.0.0"` to use it from other devices; there's no login.
  If Windows Firewall blocks other devices (the page loads on this PC but not elsewhere), allow the port for the
  local network only, from an admin PowerShell (change `8000` if you set a different `port`):
  `New-NetFirewallRule -DisplayName "StreamerFrames (TCP 8000, LAN only)" -Direction Inbound -Protocol TCP -LocalPort 8000 -RemoteAddress LocalSubnet -Action Allow -Profile Any`
- Logs: `cache/server.log` (rotated), and per generation `worker.log` / `ffmpeg-*.log` in its cache folder.

Run `python -m streamerframes bench <movie> --crop` once before streaming so `scale=auto` can choose from
real measurements (without calibration it falls back to: full scale up to 720p, half scale above).

## Models, quality and extras

- **More models:** put each extra model in `models/<name>/` (`IFNet_HDv3.py` + `flownet.pkl`, e.g.
  `models/4.25.lite/` from the links in `README.md`) and use `model = "<name>"` in a profile, or
  `model = "auto"`: with calibration (`bench --model all`), it picks the best model and scale that keep up
  with real time, preferring full models over `lite` ones (override with `model_preference`).
- **Scene cuts and duplicates:** `scene_ssim` (default 0.2) decides when neighbouring frames are a hard cut;
  those show the nearer real frame instead of a blend. `dup_mad` (off by default; ~0.002 suits anime drawn on
  twos) copies duplicated frames instead of interpolating between them.
- **HEVC export:** `render ... --export-codec hevc`, or the HEVC button in the library, re-encodes the export
  with `hevc_encoder` (NVENC on Pascal has no HEVC B-frames). Streaming stays H.264 for browsers.
- **Watch folders:** `watch_dirs = ["E:/Movies/incoming"]` queues an offline `watch_profile` job for each new
  file once it has finished copying, and exports it to `export_dir` when it finishes (`watch_export = false`
  turns that off). Files already there when a folder is first watched are left alone.
- **Export names:** by default an export is `<export_dir>/<title>.59.940fps.mp4`. For a media server such as
  Jellyfin set `export_layout = "library"`: the export becomes `<export_dir>/<title>/<title> - 60fps.mp4`
  (` HEVC` added for HEVC exports), which Jellyfin shows as a "60fps" version of the film and still matches
  metadata on `Title (Year)`. Only characters Windows forbids in file names are replaced. `--export <path>`
  always wins.

## TensorRT (experimental, PLAN.md Phase 6)

Off by default. A go/no-go experiment for the 1080 Ti: TensorRT 8.6 is the last version that supports Pascal, and
only FP32 makes sense there.

```
pip install onnx onnxruntime onnxscript  (+ the TensorRT 8.6 wheel; see requirements-streamerframes.txt)
python -m streamerframes trt export  "E:\Movies\Batman Begins.mp4" --crop   # ONNX + check vs PyTorch (< 1e-3)
python -m streamerframes trt build   "E:\Movies\Batman Begins.mp4" --crop   # FP32 engine in cache/engines/
python -m streamerframes trt compare "E:\Movies\Batman Begins.mp4" --crop   # speed + PSNR -> GO / NO-GO
```

GO means at least 1.25× faster than PyTorch + CUDA graphs with PSNR ≥ 45 dB. Only then set
`backend = "tensorrt"` in a profile. Missing TensorRT or a missing engine for a size falls back to PyTorch with
a warning. If it's NO-GO, record the numbers below and leave the backend off.

On Windows, TensorRT's Python package needs three DLL folders on `PATH`: `TensorRT-8.6.1.6\lib`, the venv's
`Lib\site-packages\torch\lib` (cuBLAS 12) and a folder with cuDNN 8 (`cudnn64_8.dll`; torch bundles cuDNN 9).
cuDNN 8 can be taken from the `nvidia-cudnn-cu12==8.9.7.29` wheel's `nvidia/cudnn/bin` without installing it.

**Result on the 1080 Ti (2026-11-19): NO-GO.** 1920×800 scale 1.0: PyTorch + CUDA graphs 46.0 ms, TensorRT FP32
56.2 ms (0.82×, slower), PSNR 124.8 dB. The backend stays off.

## Command line

```
python -m streamerframes serve [--host H --port P]   # web UI + job manager
python -m streamerframes check                      # GPU, torch arch, ffmpeg/NVENC, model files
python -m streamerframes probe <video>              # stream info as JSON
python -m streamerframes plan <video> --target-fps 60
python -m streamerframes bench [<video> --crop | --size 1920x800] [--model all]   # calibrate scale/model=auto
python -m streamerframes render <video> [--profile quality|realtime] [--target-fps 60 | --multi 2]
                                [--scale auto|1.0|0.5] [--model NAME|auto] [--export out.mp4 | --no-export]
                                [--export-codec copy|hevc]
python -m streamerframes report [--markdown]          # benchmark + generation-speed tables (for this README)
python -m streamerframes cache ls | rm <video_id|path> | gc [--max-gb N]
python -m streamerframes trt export|build|compare [<video> --crop | --size WxH] [--scale 1.0]
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
- **Audio** for streaming is a separate AAC HLS rendition encoded alongside the video. The encoder's 1024
  priming samples are trimmed so audio isn't 21 ms late; a beep/flash test checks sync through the real
  HLS output, across a resume.
- **scale=auto.** `bench` records ms/frame per (GPU, model, size, scale, graphs) in `cache/calibration.json`.
  `realtime` picks the highest-quality scale that sustains 1.1× the needed interpolation rate. The choice is
  remembered per video, so the cache id never changes under you.

## Measured performance

Measured 2026-11-19 on the GTX 1080 Ti PC (Ryzen 5 5600G, driver 582.28, torch 2.13.0+cu126, RIFE 4.25),
nothing else on the GPU. `python -m streamerframes report --markdown` prints these tables from the cache.

**Model only** (`bench`; `Batman Begins.mp4` is 1920×800 with no letterbox to crop). A 2× stream of a 23.976 fps
film needs 24 interpolations/s, and `scale = "auto"` wants 1.1× that (26.4/s).

| Size | Scale | ms/frame, no graphs | ms/frame, CUDA graphs | Interpolations/s | Real time for 24p 2×? |
|---|---|---|---|---|---|
| 1920×800 | 1.0 | 46.8 | 46.2 | 21.6 | no |
| 1920×800 | 0.5 | 33.8 | 33.5 | 29.9 | yes (auto picks this) |
| 1920×1080 | 1.0 | 61.4 | 60.6 | 16.5 | no |
| 1920×1080 | 0.5 | 43.1 | 42.4 | 23.6 | no |
| 1280×720 | 1.0 | 30.0 | 30.0 | 33.3 | yes |
| 1280×720 | 0.5 | 21.1 | 20.4 | 49.1 | yes |

In line with PLAN.md section 2 (49.5 / 35.4 ms at 1920×832). CUDA graphs gain only 1–3% here.

**End to end:**

| What | Result |
|---|---|
| Offline `render --profile quality` (23.976 → 59.94), 10 min of `Batman Begins.mp4` | 8.8–9.1 source fps, 0.37× real time (27 min for 10 min; a full film ≈ 6 h) |
| Export size, `quality` (H.264, cq 18) | 1.84 GB per 10 min (24.5 Mbit/s; ≈ 26 GB per film) |
| Export size, `--export-codec hevc` | 1.0 GB per 10 min (13.2 Mbit/s); the re-encode takes 2 min per 10 min |
| Stream (`realtime`, 2×, auto → scale 0.5), time to first segment | 14 s |
| Stream speed | 0.81–0.84× real time (19.5–20 source fps) |
| Seek 1 h ahead of the generated part | segment served after 14 s |
| Server restart mid-stream | running worker adopted, generation continued |

Streaming 1920×800 is **not** real time on this PC: the model alone would manage 1.25×, but end to end the GPU is
only ~71% busy while the worker process uses ~5 CPU cores and the NVENC encoder's ffmpeg ~2.4 (the 6-core CPU is
at 100%). The page still plays without stalling, after waiting for a lead of about 18% of the remaining film.
Reducing the worker's CPU work is the next speed-up to look at; 720p sources stream in real time.

## Tests

`pip install pytest httpx fastapi && pytest`. Everything runs on CPU with real ffmpeg and a fake IFNet
(`tests/fake_model.py`), so CI needs no GPU. GPU-only behaviour (NVENC options, `-hwaccel cuda` seek accuracy,
CUDA graphs, real-model quality and speed) needs checking on the target machine.
