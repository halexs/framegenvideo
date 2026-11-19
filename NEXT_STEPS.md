# Next steps (run on the Windows PC)

Written 2026-11-16. This is a runbook for the Windows PC with the GTX 1080 Ti. StreamerFrames is built (`PLAN.md` phases 0–6), but everything that needs the real GPU is still unmeasured: `README-StreamerFrames.md` has an empty "Measured performance" section, and its Tests section lists GPU-only behaviour that CI cannot check. Steps 1–5 close that gap. Steps 6–8 turn the PC into a general streaming box.

Work top to bottom. Tick the boxes and fill in the tables in this file, then commit it.

**Code first, PC second.** The three small code tasks in `AGENT_TASKS.md` are done: Jellyfin-friendly export names (`export_layout = "library"`, step 8), a `report` command that prints the tables for steps 2–4, and the README's description of sleep (step 4). Watch-folder jobs now also export when they finish (step 8).

All commands are PowerShell, run from the repo root. The launchers expect the venv one level up, at `..\.framegen`.

```powershell
git fetch origin
git switch plan/windows-next-steps
..\.framegen\Scripts\Activate.ps1
```

## Status

| # | Step | Needs new code | State |
|---|---|---|---|
| 1 | Environment and self-check | no | not started |
| 2 | Benchmark and calibrate | no | not started |
| 3 | Offline render, end to end | no | not started |
| 4 | Streaming in the browser | no | not started |
| 5 | TensorRT go/no-go | no | not started |
| 6 | Settle the open questions in `PLAN.md` | config only | not started |
| 7 | Jellyfin as the general streaming box | no | not started |
| 8 | Overnight renders into Jellyfin | small | not started |

Pick one test film and use it throughout so the numbers are comparable. `PLAN.md` uses `E:\Movies\Batman Begins.mp4` (1920×800, 23.976 fps); the commands below assume it as `$movie`.

```powershell
$movie = "E:\Movies\Batman Begins.mp4"
```

---

## 1. Environment and self-check

- [ ] If `..\.framegen` does not exist, create it (Python 3.11) and install, exactly as in `README-StreamerFrames.md` "Setup":
  ```powershell
  py -3.11 -m venv ..\.framegen
  ..\.framegen\Scripts\Activate.ps1
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  pip install -r requirements-streamerframes.txt
  ```
  The `cu126` wheels matter: newer CUDA builds dropped the 1080 Ti.
- [ ] `ffmpeg -hide_banner -encoders | Select-String nvenc` lists `h264_nvenc`.
- [ ] The RIFE model is in `train_log\` (`IFNet_HDv3.py` and `flownet.pkl`, version 4.25). Optional: put `4.25.lite` in `models\4.25.lite\` so step 2 can compare it.
- [ ] `copy streamerframes.example.toml streamerframes.toml`, then edit `movies_roots`, `cache_root`, `model_dir` and `models_dir` to the real paths.
- [ ] `python -m streamerframes check` passes.
- [ ] `pytest` passes (CPU only, about a minute).
- [ ] Close other GPU programs (Ollama, LM Studio, games) before measuring anything below. Check with `nvidia-smi`.

**Done when:** `check` prints no failures and `pytest` is green.

## 2. Benchmark and calibrate

`scale = "auto"` and `model = "auto"` choose from `cache\calibration.json`; without it they fall back to a guess.

- [ ] Run:
  ```powershell
  python -m streamerframes bench $movie --crop --model all
  python -m streamerframes bench --size 1920x1080 --model all
  python -m streamerframes bench --size 1280x720 --model all
  ```
- [ ] Copy the numbers into this table (`python -m streamerframes report` prints them as a table). A 23.976 fps film at 2× needs 24 interpolations per second for real time; the `realtime` profile wants 1.1× that, so 26.4.

| Size | Model | Scale | CUDA graphs | ms/frame | Interpolations/s | Real time for 24p 2×? |
|---|---|---|---|---|---|---|
| 1920×800 | default | 1.0 | | | | |
| 1920×800 | default | 0.5 | | | | |
| 1920×1080 | default | 1.0 | | | | |
| 1920×1080 | default | 0.5 | | | | |
| 1280×720 | default | 1.0 | | | | |

Compare with the model-only numbers in `PLAN.md` section 2 (49.5 ms at 1920×832 scale 1.0, 35.4 ms at scale 0.5, measured without CUDA graphs). Much slower than that means something else is using the GPU.

**Done when:** the table is filled in and `cache\calibration.json` exists.

## 3. Offline render, end to end

This is the "watch at 60 fps" path: a true 59.94 fps file, generated unattended.

- [ ] Preview what will be produced: `python -m streamerframes plan $movie --target-fps 60`
- [ ] Render a short film or clip first (10 minutes is enough), then the full test film:
  ```powershell
  python -m streamerframes render $movie --profile quality
  ```
  The progress line shows source fps and the ratio to real time. Note both.
- [ ] Resume test: press Ctrl+C once mid-render, wait for it to stop after the current segment, run the same command again. It must continue, not restart.
- [ ] Check the export: plays in a normal player, `python -m streamerframes probe <export>` shows 60000/1001 fps and the same duration as the source, audio is in sync at the start, the middle and the end, subtitles are present.
- [ ] `python -m streamerframes cache ls` shows the generation as `complete`.

| Film | Profile | Source fps processed | × real time | Total render time | Export size |
|---|---|---|---|---|---|
| | quality (60) | | | | |

**Done when:** a full film exports, survives a resume, and plays in sync.

## 4. Streaming in the browser

- [ ] `start_server.bat`, open http://localhost:8000, find the test film in the library.
- [ ] Click **Framegen**. Note the seconds until playback starts, and the speed (× real time) in the status line. `PLAN.md` section 8 expects a start within about 20 s and at least 1.0× for a 1920×800 film.
- [ ] Seek far ahead of what has been generated. Playback should resume there after a short wait; afterwards the holes get filled in the background.
- [ ] **Compare** shows the original and the framegen version side by side, in sync.
- [ ] Restart the server mid-generation (`stop_server.bat`, `start_server.bat`). The job should be adopted or resumed.
- [ ] From another device: set `host = "0.0.0.0"` in `streamerframes.toml`, add the LAN-only firewall rule from `README-StreamerFrames.md`, and open `http://<pc-ip>:8000` on a phone or the TV's browser. There is no login, so keep it on the home network.
- [ ] Sleep check: confirm the README's description ("Web UI") on the real machine. Set the sleep timeout to 5 minutes, play a finished film from another device for 15 minutes (the PC must stay awake), then pause it (it should sleep about 2 minutes plus the timeout later). Try it with **Original** too, since a browser may buffer a whole file and stop fetching. Restore your sleep timeout afterwards.

| Check | Result |
|---|---|
| Seconds to first frame | |
| Stream speed, × real time | |
| Scale and model chosen by `auto` | |
| Seek ahead works | |
| Stays awake during remote playback | |

- [ ] Write the numbers from steps 2–4 into the "Measured performance" section of `README-StreamerFrames.md` (`python -m streamerframes report --markdown` prints the tables).

**Done when:** a film streams without stalling on another device, and the README's performance section is no longer empty.

## 5. TensorRT go/no-go

One experiment, then a decision. TensorRT 8.6 is the last version that supports the 1080 Ti.

- [ ] Install the optional packages (`requirements-streamerframes.txt` has the exact lines, including the TensorRT 8.6 wheel).
- [ ] Run:
  ```powershell
  python -m streamerframes trt export  $movie --crop
  python -m streamerframes trt build   $movie --crop
  python -m streamerframes trt compare $movie --crop
  ```
- [ ] `compare` prints GO or NO-GO. GO means at least 1.25× faster than PyTorch with CUDA graphs, at 45 dB PSNR or better.

| PyTorch ms | TensorRT ms | Speedup | PSNR | Verdict |
|---|---|---|---|---|
| | | | | |

- [ ] If GO: set `backend = "tensorrt"` in the `realtime` profile and repeat step 4's speed measurement. If NO-GO: leave it off and record the numbers in `README-StreamerFrames.md` under "TensorRT".

**Done when:** the verdict and numbers are written down. Either outcome closes this step.

## 6. Settle the open questions in `PLAN.md`

`PLAN.md` section 7 ends with four questions that were answered by default. Decide them now that there are real numbers, and set the config to match.

- [ ] **2× (47.95 fps) for streaming, true 60 only offline?** Watch ten minutes of each on your actual TV. If the 47.95 cadence bothers you on a 60 Hz screen, plan to watch pre-rendered `quality` files (step 8) and use streaming only for previews.
- [ ] **Where exports go.** For step 7, set `export_dir` to a folder of its own that is not inside `movies_roots`, for example `E:/Media/Framegen`.
- [ ] **LAN access.** `host = "0.0.0.0"` if step 4 worked from other devices.
- [ ] **Pre-generate overnight?** Decided in step 8.
- [ ] Disk: a 1080p film takes roughly 5–10 GB of cache. Set `cache_max_gb` to what the drive can spare.

**Done when:** `streamerframes.toml` reflects the answers, and they are noted here.

## 7. Jellyfin as the general streaming box

StreamerFrames plays films in a browser. A media server adds what it lacks: TV and phone apps, watch progress, shows and seasons, metadata. Jellyfin is free and open source; Plex and Emby charge for some features.

It replaces a personal library only. Netflix, Crunchyroll and YouTube content still needs their own apps.

- [ ] Install Jellyfin Server (installer from <https://jellyfin.org/downloads/server>, or `winget search jellyfin` and install the server package). Choose "install as a service" so it runs without a login.
- [ ] Open http://localhost:8096 and finish the setup wizard. Create two libraries:
  - **Movies** → `E:\Movies`
  - **Movies 60fps** → the `export_dir` from step 6
- [ ] Dashboard → Playback → Transcoding: hardware acceleration **NVENC**. Leave HEVC encoding off unless a test proves it works; the 1080 Ti's NVENC is limited there.
- [ ] Firewall: allow TCP 8096 for the local subnet only (same pattern as the StreamerFrames rule).
- [ ] Install the Jellyfin app on the TV and phone and sign in.
- [ ] Play a 60 fps export on the TV. Open the playback info overlay and confirm it is **direct playing**, not transcoding, and that the TV shows 59.94 fps. If Jellyfin transcodes it, the frame rate or bitrate may be reduced and the GPU is being used twice; fix the client's quality setting or the export codec before going further.
- [ ] Away from home: Tailscale on the PC and the phone. Do not forward port 8096 on the router.
- [ ] GPU sharing: a Jellyfin transcode and a StreamerFrames render compete for the same card. Note what happens to render speed while a transcode runs.

**Done when:** the TV app plays a 60 fps export by direct play, and remembers where you stopped.

## 8. Overnight renders into Jellyfin

Goal: drop a film in, find the 60 fps version in Jellyfin the next day.

- [ ] In `streamerframes.toml`:
  ```toml
  export_dir    = "E:/Media/Framegen"
  export_layout = "library"
  watch_dirs    = ["E:/Movies/incoming"]
  watch_profile = "quality"
  ```
  New files copied into a watch folder are queued automatically and exported when they finish (`watch_export`, on by default). Files already there are left alone.
- [ ] `create_schtask.bat` so the server starts at every logon.
- [ ] Test with one film: copy it into the watch folder, confirm a job appears in the UI, and that the export lands in `export_dir` when it finishes.
- [ ] Check the export's file name: with `export_layout = "library"` it is `E:\Media\Framegen\Title (Year)\Title (Year) - 60fps.mp4`. Confirm Jellyfin matches the film's metadata (name source files `Title (Year).mp4` for reliable matches).
- [ ] Make Jellyfin notice new files: enable real-time monitoring on the **Movies 60fps** library, or leave the scheduled library scan on.
- [ ] Existing library: decide whether to queue everything. At the speed measured in step 3, estimate the total (hours per film × number of films) before starting; this may be weeks of nights.
- [ ] Waking the PC for overnight work. Two options:
  - A Windows scheduled task with "wake the computer to run this task" that runs `start_server.ps1 -NoBrowser`; interrupted jobs resume because `auto_resume = true`, and a running generation holds off sleep until it finishes.
  - The Raspberry Pi wakes the PC over the network, starts the job over SSH and hibernates it afterwards. That is tracked outside this repo; this PC needs Wake-on-LAN enabled in BIOS and on the network adapter, the OpenSSH server, and `powercfg /hibernate on`.
- [ ] The local calendar assistant uses the same GPU each morning. Make sure renders are finished or paused by then.

**Done when:** a film dropped into the watch folder in the evening is playable at 60 fps from the Jellyfin TV app the next day, with no manual steps.

---

## If something goes wrong

| Symptom | Check |
|---|---|
| `check` fails on CUDA or architecture | Torch was installed from the wrong index. Reinstall with the `cu126` line in step 1 |
| `check` fails on NVENC | ffmpeg build lacks `h264_nvenc`, or the NVIDIA driver is too old for that ffmpeg build |
| Server will not start | `server-console.log` in the repo root |
| A generation failed | `worker.log` and `ffmpeg-*.log` in that generation's cache folder; the last 50 ffmpeg lines are also in its `manifest.json` |
| Much slower than the step 2 numbers | `nvidia-smi`: another program is on the GPU, or VRAM is full |
| Page loads on the PC but not on other devices | `host` is still `127.0.0.1`, or the firewall rule is missing |
| Audio drifts | Note the film and timestamp, and run `pytest tests/test_av_sync.py`; then compare against a `render` of the same film |
