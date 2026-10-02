# StreamerFrames: Implementation Plan

> Audience: a coding agent implementing this in `E:\StreamerFrames\Practical-RIFE`.
> Written 2026-10-02 from an audit of the repo, its logs (`server.log`,
> `hls/Batman_Begins.mp4/generation.log`), and benchmarks run on the target GPU.
> Read sections 1–3 before writing code. Do the phases in order; each one has acceptance criteria.

---

## 0. Goal

Use the local **GTX 1080 Ti (Pascal, sm_61, 11 GB)** to raise low‑frame‑rate video (≤30 fps, mostly
23.976 fps films from `E:\Movies`) to a higher frame rate (60 fps target, or a 2× multiplier) using RIFE.
It should work in two modes:

1. **Streaming mode.** The user opens the web UI, picks a movie, and it plays in the browser while the GPU
   generates frames in the background (HLS that grows as generation runs). Real‑time is the ideal; when the GPU
   can't keep up, the UI says how long to wait before playback will run without stalling.
2. **Offline mode.** The user gives it an input file from the CLI or queues it in the web UI. It generates
   unattended and can be resumed. The finished result stays available: it streams from the web UI and can be
   exported to MP4/MKV.

Both modes share **one generator and one on‑disk segment cache**, so a movie that was rendered offline streams
instantly, and a partly streamed movie can be finished offline.

**Decision: keep FastAPI.** The user said "flask webserver", but the existing server is FastAPI + uvicorn,
which is already installed and handles range requests and async status polling well. Do not migrate to Flask
unless the user asks.

---

## 1. Current state audit

### 1.1 What exists

| File | Role | Verdict |
|---|---|---|
| `server.py` | FastAPI movie browser on :8000: lists `E:\Movies`, plays originals, launches `rife_server.py --generate-only` as a subprocess per movie, serves `hls/` | Keep the idea; rewrite |
| `rife_server.py` | Generator: ffmpeg decode → RIFE 2× → ffmpeg NVENC → HLS. Also a stale standalone FastAPI app | Rewrite as a package |
| `rife_server_gpu-only.py`, `rife_server_gpu-only-stream.py`, `interpolate_v0/1/2.py`, `test_framegen.py` | Earlier experiments | Move to `legacy/` |
| `export_onnx.py` | ONNX export attempt | Broken (see 1.2); replace in Phase 6 |
| `inference_*.py`, `model/`, `train_log/` | Upstream Practical‑RIFE. `train_log/` holds **RIFE v4.25** (`IFNet_HDv3.py`, 5 IFBlocks, `flownet.pkl`) | Keep; don't edit upstream files except where noted |
| `start_server.bat/.ps1`, `create_schtask.bat` | Launchers | Fix |
| `hls/Batman_Begins.mp4/` | Output of the last run | **Invalid cache** (see 1.2): delete |
| `tmp_resume_test*/`, `tmppqyre2r0/`, `server.log`, `__pycache__/` | Junk | Delete |
| `input.mp4` (263 MB), `thriller_*.mp4` | Test media (gitignored) | Keep; useful for tests |

Environment (verified):
- venv: `E:\StreamerFrames\.framegen` (Python 3.11.9), **torch 2.13.0+cu126**, torchvision 0.28.0+cu126.
  `torch.cuda.get_arch_list()` includes `sm_61`. **Never upgrade to cu128/cu129/cu13 wheels: they drop Pascal.**
- venv also contains **TensorRT 11.2 (cu13)**. CUDA 13 and TRT ≥10 don't support Pascal, so it's useless here.
  `E:\StreamerFrames\TensorRT-8.6.1.6\python\tensorrt-8.6.1-cp311-none-win_amd64.whl` is the only TRT that can work (Phase 6).
- ffmpeg 7.1.1 full build (gyan.dev) on PATH, with `h264_nvenc`, `hevc_nvenc`, `*_cuvid`, `-hwaccel cuda`.
  (Pascal NVENC: H.264 with B‑frames, HEVC without B‑frames, no AV1 encode.)
- Driver 582.28. Other GPU users are present (LM Studio, browsers), so don't assume the full 11 GB is free.
- Movies: `E:\Movies\*.mp4` plus a subfolder. Example: `Batman Begins.mp4` is h264 1920×800,
  24000/1001 fps, 201,471 frames, 8403 s, HE‑AAC stereo.
- Git: `main` tracks `git@github.com:halexs/framegenvideo.git`. `server.py` and `rife_server.py` have uncommitted WIP.

### 1.2 Bugs found (with evidence)

Generator (`rife_server.py`):
1. **No keyframe or GOP control, so HLS never segments.** The Batman playlist is one `#EXTINF:1350.208` segment
   (`stream0.ts`, 388 MB). Streaming during generation can't work: the only segment closes when encoding ends.
   (encoder args at `rife_server.py:231-284`)
2. **Bitrate is effectively 2 Mbps.** `-rc cbr` sets `-maxrate 20M` but no `-b:v`, so ffmpeg's 2 Mbps default
   applies (the log shows `bitrate max/min/avg: 20000000/0/2000000`). Visibly poor quality at 1080p48.
   `-tune ll` plus `-bf 2` is also contradictory.
3. **Wrong output frame rate, so A/V drift.** `output_fps = int(round(fps * 2))` (`:225`) turns 47.952 into 48.
   Over a 140‑minute film the video runs 8.4 s short against the audio. Use exact rationals (`48000/1001`).
4. **"Complete" is set unconditionally** (`:435`, `:347`). The decoder's exit code is never checked and its stderr goes to
   `DEVNULL` (`:370`). In the logged resume run the decoder stopped after ~4,380 of 201,471 frames (cause
   unknown because stderr was discarded), and the state was still written as `{"complete": true}`. The server then served
   a 22‑minute "finished" movie.
5. **Resume is broken several ways.**
   - It re‑decodes from frame 0 **and still runs RIFE on every skipped frame** (`interpolate()` is called before the
     skip check, `:405` vs `:194-198`). Resuming at 22 minutes costs about 37 minutes of wasted GPU time.
   - The new encoder starts with timestamps at 0 and `-y` and `append_list`, and the audio restarts at 0, so the
     resumed part is desynced by the resume offset.
   - Inferring the resume position from the segment count (`:160-168`) assumes 4 s segments, which item 1 shows never happens.
   - The resume test (`tmp_resume_test/hls`) shows `#EXT-X-DISCONTINUITY` and a 0‑byte `stream1.ts`.
6. **`resume_state.json` is rewritten after every output frame** (`:205`): about 48 synchronous file writes per second
   of video, and log spam.
7. **The torchvision CUDA decode path is dead and dangerous** (`:286-303`). `torchvision.io.set_video_backend` doesn't
   exist in 0.28, so it always falls back. If it ever worked, it would load the **whole movie into VRAM**
   (201k × 4.6 MB). Delete it.
8. **Each source frame is uploaded and converted twice**: `prepare_frame_tensor` runs on both `previous` and
   `frame` every iteration (`:80-81`).
9. **Fully serial pipeline.** Inference, a blocking `synchronize()`, `.tobytes()` copies, and the blocking pipe write
   all run on one thread. Measured end to end: **14.7 source fps** (64,831 output frames in 36m48s), against **20.2**
   for the model alone at the same resolution. Roughly 27% overhead.
10. **Padding ignores scale.** `pad_tensor` always aligns to 64 (`:53-60`). RIFE v4.25 needs `64/scale`
    (128 at `scale=0.5`). It works by luck for some sizes and breaks for others.
11. RGB24 over pipes: the decoder makes ffmpeg's CPU `swscale` convert YUV→RGB (and RGB→YUV on encode) with
    implicit BT.601 and untagged output. That's twice the pipe bandwidth of YUV420 and the colors are subtly wrong.
12. Model loading at import time uses cwd‑relative paths (`"train_log"`, `:38-42`), so it breaks when launched from another cwd.
13. No scene‑cut handling, so hard cuts produce ghosted blend frames.
14. 24→60 isn't supported at all. 2× of 23.976 gives 47.95 fps, not 60.

Server (`server.py`):
15. No GPU concurrency control: every click on a different movie spawns another generator process (`:143`,
    `:372`), each loading the model and contending for VRAM and NVENC.
16. `is_hls_ready` requires `complete` (`:117-131`), so the "Framegen" page only auto‑refreshes until the
    **entire movie** is done. There's no actual streaming. It also trusts the bogus `complete` flag (bug 4).
17. Job state lives only in memory. A server restart orphans running generators, and the next click starts a duplicate.
18. Cache key `sanitize_name()` (`:86`) can collide ("a b" and "a_b") and doesn't include generation settings.
19. `hls.js@latest` comes from a CDN (`:406`), so versions are unpinned and it needs internet. Display names are put into HTML unescaped (`:252`).
20. Every request, including every `.ts` segment fetch, is logged with a synchronous file append (`:39-48`).
21. `get_movies_root()` may shell out to PowerShell on every call when `E:\Movies` is missing; there's no caching or config.

Misc:
22. `start_server.bat` is broken: the lines read `n"%VENV_PY%" -m uvicorn …` and `necho Server exited…`
    (a stray `n` from a bad `\n` substitution).
23. `export_onnx.py` passes a 4‑entry `scale_list` to a 5‑block network (IndexError) and returns `merged[3]`
    (a tuple), so it can't work.
24. fp16 is broken (`model/warplayer.py` caches an fp32 grid, giving "expected scalar type Half but found Float"), and it's
    **slower on Pascal anyway** (see section 2). Don't pursue fp16.
25. `README.md` ends with UTF‑16 garbage (`#\0 \0f\0r\0a\0m\0e\0…`). `requirements.txt` lacks fastapi/uvicorn.

---

## 2. Measured performance and feasibility

Model‑only benchmark (IFNet v4.25, batch 1, `cudnn.benchmark=True`, fp32, timestep 0.5, LM Studio idle in
the background). "src fps" is the source frame rate that 2× interpolation can sustain (one RIFE call per source frame).

| Inference size | scale=1.0 | scale=0.5 | fp16 autocast (scale 1.0) |
|---|---|---|---|
| 1920×832 (1920×800 padded) | 49.5 ms → **20.2 src fps** | 35.4 ms → **28.2** | 57.7 ms → 17.3 |
| 1920×1088 (1080p) | 63.5 ms → 15.8 | 45.4 ms → **22.0** | 73.5 ms → 13.6 |
| 1280×704 (720p) | 29.0 ms → **34.5** | 21.9 ms → **45.7** | 34.8 ms → 28.8 |

GPU‑side I/O (upload, uint8↔float, download) costs 1–2 ms per frame at 1080p, which is negligible if overlapped.

**What this means:**
- **24p film to 2× (47.95 fps), real time:** feasible at 1920×800 with `scale=0.5` (28 > 24 with about 15% headroom).
  It isn't feasible at `scale=1.0`. Full‑frame 1080p24 is about 0.9× real time at `scale=0.5`, so real time there needs
  letterbox cropping (most films are letterboxed), CUDA graphs, TRT, or a lite model.
- **30p to 60p at 1080p:** not real time (needs 33 ms; best is 45 ms). 720p30 to 60 is real time.
- **24p to 60p (2.5×):** needs **two** RIFE calls per source frame (5 outputs per 2 inputs; only 1 of 5 is an original),
  which halves throughput. **Offline only** at 1080p. Real time only at ≤720p.
- fp16 is slower (Pascal has a 1/64‑rate FP16 path and no tensor cores). `torch.compile`/Triton needs sm_70+.
  Neither is an option. Speedups have to come from pipeline overlap, smaller inference area, CUDA graphs,
  TensorRT 8.6 (fp32), or a lighter model.

**Default policy, which follows from the numbers:**
- Streaming profile `realtime`: 2× multiplier (23.976→47.95, 29.97→59.94), `scale` chosen per resolution from a
  calibration table so throughput is at least 1.1× real time where possible.
- Offline profile `quality`: target 60 fps (snapped to 60000/1001 for NTSC‑rate sources), `scale=1.0`
  (`0.5` for ≥1440p), with higher encoder quality.
- Both are configurable. The UI must show measured speed against real time and an ETA.

---

## 3. Target architecture

### 3.1 Layout

```
Practical-RIFE/
  streamerframes/
    __init__.py
    __main__.py            # CLI: python -m streamerframes {serve,render,bench,probe,cache}
    config.py              # Settings dataclass; loads streamerframes.toml + env overrides
    probe.py               # ffprobe JSON → VideoInfo (exact Fractions)
    timeline.py            # pure math: output schedule, segment ↔ frame mapping (NO torch import)
    engine/
      loader.py            # build IFNet + load flownet.pkl by absolute path; no optimizer/loss imports
      rife.py              # RifeEngine: pad/unpad, inference(t), CUDA-graph cache, scene detection
      color.py             # GPU YUV420 ↔ RGB (BT.709/BT.601, limited/full range)
      trt_backend.py       # Phase 6, optional
    pipeline/
      ffmpeg.py            # command builders (decoder, encoder, audio, finalize); pure functions → list[str]
      decoder.py           # ffmpeg reader process + thread into pinned ring buffers
      encoder.py           # ffmpeg writer process + thread
      generator.py         # orchestrates one run: [start_segment, end) → segments on disk
      worker.py            # `python -m streamerframes.pipeline.worker <job.json>` subprocess entry
    cache/
      store.py             # cache dir layout, manifest, segment index, atomic JSON, locks, eviction
      playlist.py          # builds m3u8 text from manifest + segment index
    jobs/
      manager.py           # single-GPU scheduler, priorities, preemption, persistence, adoption on restart
    web/
      app.py               # FastAPI app factory + routes
      static/              # index.html, watch.html, compare.html, app.js, style.css, hls.min.js (vendored, pinned)
  tools/
    bench.py               # model/pipeline benchmark → writes calibration table
  tests/                   # pytest; GPU tests marked @pytest.mark.gpu
  legacy/                  # old scripts moved here untouched
  streamerframes.toml      # user config (gitignored); streamerframes.example.toml committed
  start_server.bat / start_server.ps1 / create_schtask.bat
  PLAN.md (this file), README-StreamerFrames.md
```

### 3.2 Data flow for one generation run

```
                ┌─────────── worker subprocess (one at a time, owns the GPU) ───────────┐
source.mp4 ──► ffmpeg decode (NVDEC, -ss start, yuv420p raw) ──pipe──► reader thread ──► pinned ring (K=6)
                                                                                     │
                                                       GPU thread: H2D → YUV→RGB → RIFE(t…) → RGB→YUV → D2H
                                                                                     │ (CUDA events)
           segments/v_00042.ts … ◄── ffmpeg encode (NVENC, forced IDR every N frames, segment muxer) ◄── writer thread
           index/run_0003.csv   ◄── (segment muxer appends a line when each segment closes)
           progress.json        ◄── worker, every ~2 s (atomic replace)
                └──────────────────────────────────────────────────────────────────────────┘
server (FastAPI) reads manifest/index/progress → builds playlists → hls.js plays video + separate audio rendition
```

### 3.3 Cache layout (the contract between worker, server, and finalize)

```
<cache_root>/                                  default: E:\StreamerFrames\cache  (configurable; gitignored)
  jobs.json                                    persistent job queue
  <video_id>/                                  video_id = sha1(abs_path_lower + size + mtime_ns)[:16]
    source.json                                VideoInfo snapshot + original path
    audio/                                     audio rendition, encoded once per video
      a_00000.ts …, audio.csv, DONE
    <profile_id>/                              profile_id = sha1(canonical JSON of everything that affects pixels)[:12]
      manifest.json                            profile, out_fps, seg_frames N, total_segments, total_out_frames, status
      segments/v_00000.ts …
      index/run_0000.csv, run_0001.csv …       ffmpeg segment_list output, one per run (union = completed segments)
      progress.json                            live run stats
      lock                                     {"pid":…, "started":…, "run":…}; present only while a worker runs
      stop                                     touched by server to request graceful stop
      worker.log, ffmpeg-decode.log, ffmpeg-encode.log
      export/<name>.mp4                        finalize output (or a configured library folder)
```

Rules:
- A segment counts as **complete only if it appears in some `index/run_*.csv`** (the segment muxer writes the line
  after closing the file). Files without an index line are partial and get overwritten on resume.
- All JSON writes are atomic: write `*.tmp`, then `os.replace`.
- `status` in `manifest.json` is one of `new | running | paused | complete | failed`. It's `complete` only when
  **every** segment index `0..total_segments-1` is present **and** the decoder and encoder both exited 0 **and** the
  frame count matches `total_out_frames` (bug 4 must never recur).

---

## 4. Core math (`timeline.py`, pure Python with `fractions.Fraction`)

Everything is exact rational arithmetic. No floats until you format ffmpeg arguments.

- `src_fps`: from ffprobe `r_frame_rate` (fall back to `avg_frame_rate`), as a `Fraction`. VFR sources are made
  CFR at decode with `-fps_mode cfr -r <src_fps>`.
- `n_src`: from `nb_frames` if present and plausible, else `round(duration * src_fps)`. Treat it as an estimate.
  The true count is whatever the decoder delivers, and the last segment is closed against that (see below).
- `out_fps`:
  - multiplier mode `m`: `out_fps = src_fps * m`
  - target mode `F`: if `src_fps.denominator == 1001`, use `Fraction(F*1000, 1001)`, else `Fraction(F)`.
    If `out_fps <= src_fps`, refuse (no‑op).
- `ratio = src_fps / out_fps`. Output frame `j` maps to source position `p_j = j * ratio`:
  `i = floor(p_j)`, `t = p_j - i`. If `t == 0`, emit source frame `i` unchanged. Otherwise emit `RIFE(frame_i, frame_{i+1}, t)`.
- Period: `P = (out_fps / src_fps).numerator` output frames per `(out_fps / src_fps).denominator` source frames
  (2× gives P=2/1; 23.976→59.94 gives 5/2).
- Segment length `N` output frames: the multiple of `P` closest to `seg_target_seconds * out_fps`
  (default 4 s). Examples: 47.952 fps gives `N=192` (4.004 s); 59.94 fps gives `N=240` (4.004 s).
  Because `N` is a multiple of `P`, **every segment starts on an original source frame**, which makes resume and seek
  frame‑exact with no previous‑frame dependency.
- Segment `k` covers output frames `[kN, (k+1)N)`, starts at source frame `s_k = kN * ratio` (an integer),
  and starts at time `kN / out_fps`.
- End of stream: output frames continue while `p_j <= n_src_actual - 1`, then the final segment is padded by
  repeating the last source frame until the output duration is at least the source duration
  (`ceil(n_src * out_fps / src_fps)` frames total). The last segment may be shorter than N. Record its real
  frame count in the manifest when the run ends.
- Scene cut between `i` and `i+1`: emit frame `i` for `t < 0.5` and frame `i+1` for `t >= 0.5`.

Unit test these exhaustively: 2×, 2.5×, 59.94/30, 25→60 (P=12/5), 30→60, segment starts are integers, and the
union of segments covers every output frame exactly once.

---

## 5. Phased implementation

### Phase 0: Baseline, cleanup, environment (small)

1. Commit the current WIP (`server.py`, `rife_server.py`) on a branch `wip-baseline` so nothing is lost. Do the new work on
   `streamerframes-v2`. (Ask the user before pushing.)
2. Move `rife_server_gpu-only*.py`, `interpolate_v*.py`, `test_framegen.py`, `rife_server.py` (after Phase 3 replaces it) into `legacy/`.
3. Delete `hls/` (invalid cache), `tmp_resume_test/`, `tmp_resume_test2/`, `tmppqyre2r0/`, `server.log`.
   Keep `input.mp4` and `thriller_*.mp4`.
4. Fix `README.md`: strip the UTF‑16 tail and add `README-StreamerFrames.md` for this project (replace the
   "Running the StreamerFrames server" section with a pointer to it).
5. `.gitignore`: add `cache/`, `streamerframes.toml`, `*.log`, `*.onnx`, `*.engine`, `legacy/__pycache__/`.
6. Add `requirements-streamerframes.txt`: `fastapi`, `uvicorn`, `numpy`, `psutil`, `pytest`, plus a comment with the
   exact torch install line (`--index-url https://download.pytorch.org/whl/cu126`) and the warning about cu128+ dropping Pascal.
   Leave upstream `requirements.txt` as is.
7. Add a startup self‑check (used by `serve`, `render`, and `bench`): CUDA available, `sm_61` (or the device's arch) in
   `torch.cuda.get_arch_list()`, `ffmpeg`/`ffprobe` found, `h264_nvenc` listed, model file present. Fail with a clear message.
8. Optionally uninstall the unusable `tensorrt` 11 packages from `.framegen` (ask the user first).

**Accept:** repo clean, `python -m streamerframes --help` runs, self‑check passes on this machine.

### Phase 1: Engine and offline render (correctness first)

Build `probe.py`, `timeline.py`, `engine/loader.py`, `engine/rife.py`, `engine/color.py`, `pipeline/*`,
and `cache/store.py`, plus the CLI `render`. A **simple serial loop** is fine in this phase; Phase 2 makes it fast.

**probe.py:** `ffprobe -v error -print_format json -show_streams -show_format -select_streams v:0` (plus a second
call for audio and subtitle streams). Return a `VideoInfo` with width, height, `src_fps` (Fraction), `n_src`, duration,
`start_time`, `pix_fmt`, `color_space/primaries/transfer/range` (default to BT.709 limited for ≥720p, else BT.601),
rotation and SAR (warn if non‑square or rotated; out of scope), and bit depth (10‑bit or HDR: warn, decode to 8‑bit
yuv420p, and tag SDR; HDR passthrough is out of scope). Also list audio and subtitle streams (codec, language, default flag).

**engine/loader.py:** construct `train_log.IFNet_HDv3.IFNet` directly. Don't use `RIFE_HDv3.Model`, which builds AdamW
and imports losses. Load `flownet.pkl` from an **absolute** model dir (config `model_dir`, default `<repo>/train_log`),
strip `module.` prefixes, `map_location="cuda"`, `eval()`, `requires_grad_(False)`. Support multiple model dirs
(`models/4.25`, `models/4.25.lite`…), each containing `IFNet_HDv3.py` and `flownet.pkl`, imported via `importlib` from
that path. `model/warplayer.py` is shared.

**engine/rife.py `RifeEngine`:**
- `__init__(model_dir, scale, height, width)`: `align = int(64 / scale)`. Pad H and W up to multiples of `align`
  (use `F.pad(..., mode="replicate")`). Store crop dims. `scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]`.
- `interpolate(img0, img1, t: float) -> Tensor`: inputs are padded `1×3×H×W` float32 in [0,1] that are already on
  the GPU. `t` is passed as a tensor of shape `(1,1,1,1)` so the call graph is static (needed for CUDA graphs later).
  Returns the clamped, unpadded result.
- `is_scene_cut(img0, img1) -> bool`: downsample both to ~1/8 size luma and compute mean absolute difference. Also
  compute the upstream `model.pytorch_msssim.ssim_matlab` at 32×32 like `inference_video.py` does. Threshold from config
  (start with SSIM < 0.2). Log every detected cut with its timestamp.
- Optional `is_duplicate(img0, img1)` (MAD < tiny epsilon) skips inference and copies the frame (anime, static scenes).
- Wrap everything in `torch.inference_mode()`. Set `torch.backends.cudnn.benchmark = True`.

**engine/color.py:** GPU conversion between planar YUV420 uint8 (as delivered by `-pix_fmt yuv420p`) and RGB float.
- YUV→RGB: Y full res; U and V upsampled 2× (bilinear, or nearest with chroma siting correction; bilinear is fine);
  limited‑range scaling; BT.709 or BT.601 matrix chosen from `VideoInfo`.
- RGB→YUV: matrix, then 2×2 average‑pool for chroma, round, clamp, and pack Y, U, V contiguous in one uint8 buffer.
- Test: round trip on a natural image at PSNR ≥ 45 dB, and solid colour bars land within ±1 code value of ffmpeg's
  `-vf colorspace`/`scale` reference.

**pipeline/ffmpeg.py:** pure command builders, unit tested as lists. Templates (fill in with exact values):

Decoder (one per run; `S` = start source frame, `start_time` from probe):
```
ffmpeg -hide_banner -nostdin -loglevel warning
  -hwaccel cuda
  -ss <(start_time + (S - 0.25) / src_fps) if S > 0 else omit>
  -i <source>
  -map 0:v:0 -an -sn -dn
  -fps_mode cfr -r <src_fps as num/den>
  -pix_fmt yuv420p -f rawvideo pipe:1
```
(`-ss` before `-i` with transcoding is frame‑accurate. The −0.25‑frame offset avoids rounding off the wanted frame.
Phase 1 tests must verify that the first delivered frame equals source frame S, using framemd5.)
Stderr goes to `ffmpeg-decode.log` (a file handle, **not** a pipe and **not** DEVNULL).

Video encoder (one per run, starting at segment `K`):
```
ffmpeg -hide_banner -nostdin -loglevel warning
  -f rawvideo -pix_fmt yuv420p -s <W>x<H> -framerate <out_num>/<out_den> -i pipe:0
  -an
  -c:v h264_nvenc -preset <p4|p6> -tune hq -profile:v high
  -rc vbr -cq <20|18> -b:v 0 -maxrate <25M> -bufsize <50M>
  -g <N> -forced-idr 1 -force_key_frames "expr:eq(mod(n,<N>),0)" -no-scenecut 1 -strict_gop 1
  -spatial-aq 1
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv     (or bt470bg/smpte170m for SD)
  -output_ts_offset <K*N/out_fps>
  -f segment -segment_format mpegts
  -segment_time <N/out_fps> -segment_time_delta <0.5/out_fps>
  -segment_start_number <K>
  -segment_list index/run_<R>.csv -segment_list_type csv
  segments/v_%05d.ts
```
Verify each NVENC option against `ffmpeg -h encoder=h264_nvenc` on this machine and drop any that Pascal rejects
(e.g. `-b_ref_mode` isn't available on Pascal, so don't add it). Test that **every non‑final segment contains exactly N
frames and starts with an IDR** (`ffprobe -count_frames`, `-show_frames -read_intervals`).

Audio rendition (once per video, independent of the GPU; runs before or alongside the first video run):
```
ffmpeg -hide_banner -nostdin -loglevel warning -i <source> -map 0:a:<chosen> -vn
  -c:a aac -b:a 192k -ac 2
  -f segment -segment_format mpegts -segment_time 4.004 -segment_list audio/audio.csv -segment_list_type csv
  audio/a_%05d.ts
```
Pick the default or first audio stream. Language selection is a later nicety. Write `audio/DONE` on exit code 0.

Finalize/export (offline mode, after `status == complete`):
```
ffmpeg -f concat -safe 0 -i segments.txt -i <source>
  -map 0:v:0 -map 1:a? -map 1:s? -c copy -c:s <mov_text for mp4 | copy for mkv>
  -movflags +faststart <output>
```
Container `auto`: MP4 unless the source has bitmap subtitles (PGS/VobSub), in which case use MKV. Source audio is copied
untouched here, not the AAC rendition. Validate that output duration matches source duration within 2 output frames.

**pipeline/generator.py `run(job)`:**
1. Load or create `manifest.json` (from probe + profile + timeline). Take the `lock` (fail if a live PID holds it,
   steal it if the PID is dead; check with `psutil.pid_exists` plus the process create‑time).
2. Completed set = union of `index/run_*.csv`. Pick `K` = the first missing segment index ≥ `job.start_segment`
   (default 0). If none are missing, finalize the status and exit.
3. Determine `end_segment` = the next already‑completed segment after K (or total). The run fills exactly that gap, so a
   seek in Phase 4 can create holes that later runs fill.
4. Start the decoder at `s_K`, the encoder at K with a new run id R, and loop over output frames in `[K*N, end*N)` using
   the timeline schedule. Keep the current pair of source frames on the GPU and advance through the decoder as `i` increases.
5. Every ~2 s, write `progress.json`: `{run, segment_frontier, out_frames_done, src_fps_measured, realtime_ratio,
   eta_seconds, gpu_mem_mb, started_at, updated_at}`. Check the `stop` file **between segments**. On stop, finish the current
   segment, close the encoder stdin, wait, and exit with code 3 ("paused").
6. On decoder EOF: close the last (short) segment as described in section 4. Compare delivered source frames against
   the expectation, and record `n_src_actual` and the real final frame count in the manifest.
7. Check `returncode` for **both** ffmpeg processes. Any non‑zero code means `status=failed`, `error` is set, and the
   last 50 lines of the relevant ffmpeg log are copied into `manifest.error_tail`.
8. Set `complete` only per the rule in section 3.3. Release the lock.
9. Keep Windows awake during the run: `ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`,
   reset on exit.

**CLI:**
```
python -m streamerframes render <input> [--profile quality|realtime] [--target-fps 60 | --multi 2]
       [--scale auto|1.0|0.5] [--model 4.25] [--export out.mp4] [--no-export]
python -m streamerframes probe <input>
python -m streamerframes cache {ls,rm <video_id|path>,gc}
```
`render` runs in the foreground with a progress line (src fps, ×real time, ETA), is resumable (rerunning the same
command continues where it left off), and exports at the end unless `--no-export`. Ctrl+C acts as a graceful stop.

**Accept (Phase 1):**
- `render thriller_input.mp4 --multi 2` produces an MP4 with exactly `ceil(n_src*2)` (±1) frames at the exact rational
  fps, with A/V duration difference under 1 output frame.
- `--target-fps 60` on a 23.976 clip gives 59.94 fps with the expected frame count.
- **Lossless test mode** (`--encoder lossless`, which uses `-c:v libx264 -qp 0 -pix_fmt yuv420p` or `ffv1`): every output frame
  at an integer source position matches the source frame by framemd5 (after the same yuv420p conversion).
- Kill the process mid‑run (Task Manager) and rerun: there are no duplicate or missing frames (framemd5 sequence equals an
  uninterrupted run), and the resume costs at most one segment of rework.
- A deliberately corrupted or truncated input yields `status=failed` with an error tail, never `complete`.

### Phase 2: Throughput (make streaming viable)

Goal: end‑to‑end throughput at least 90% of the model‑only numbers in section 2.

1. **Three‑stage pipeline** inside the worker:
   - Reader thread: `readinto()` exact frame byte counts (`W*H*3/2`) from the decoder stdout into a ring of K=6
     **pinned** host buffers (`torch.empty(..., pin_memory=True)` viewed as numpy), then push to a `queue.Queue(maxsize=K)`.
     Return buffers through a free queue. No `np.frombuffer(bytes)` copies.
   - GPU thread (main): H2D with `non_blocking=True` on a dedicated copy stream, then YUV→RGB, then inference on the compute
     stream, then RGB→YUV, then D2H into a pinned output ring (K=8). Record a `torch.cuda.Event` per output and push
     `(buffer, event)` to the writer queue. **Never call `torch.cuda.synchronize()` in the loop.**
   - Writer thread: `event.synchronize()`, then `encoder.stdin.write(memoryview(np_view))` (no `.tobytes()`), then
     return the buffer to the free ring.
   - Pipes: `bufsize=0` on Popen and large writes. Measure pipe throughput. If Windows pipes become the limit (unlikely
     at ~110 MB/s in and ~220 MB/s out), fall back to `-f rawvideo` over a named pipe or TCP localhost.
2. **Upload each source frame exactly once.** Keep `cur` and `next` GPU tensors and rotate them.
3. **CUDA graphs:** capture `RifeEngine.interpolate` for the fixed padded shape with static input and output tensors and
   a static `t` tensor (copy `t` in before replay). Benchmark with and without; keep it if it's ≥3% faster and outputs
   match eager within 1e‑5. Note that `model/warplayer.py` caches grids in a dict keyed by size, which is fine for capture
   once warmed.
4. **Letterbox crop:** at job start, run `ffmpeg -ss <x> -i src -vf cropdetect=24:2:0 -frames:v 60 -f null -` at
   ~8 points across the file. If all samples agree on bars of ≥32 px total, infer only on the active area (decode the
   full frame, interpolate the crop, write it back into a full‑size frame whose bars are copied from the source frame).
   This is the main lever for 1080p24 real time (1920×1080 becomes about 1920×800, which is 28 src fps at scale 0.5).
5. **Calibration:** `tools/bench.py` (and `python -m streamerframes bench`) measures model ms per call for the
   current GPU at the movie's inference size for `scale ∈ {1.0, 0.5}` and writes
   `cache/calibration.json` keyed by `(gpu_name, model, W, H, scale, graphs)`. Profile `realtime` with `scale=auto` picks
   the highest‑quality scale whose predicted throughput is at least 1.1× the required interp rate
   (`out_fps - src_fps` interpolations per second of video), otherwise the fastest one. Reuse the scratch benchmark logic
   from the audit (two random tensors, 5 warmups, 30 timed iterations, `cuda.synchronize` around the timed block).
6. Profile with `torch.profiler` (or NVTX ranges plus Nsight Systems) once, confirm the GPU stays busy, and record findings in
   `README-StreamerFrames.md`.

**Accept:** on `Batman Begins.mp4` (1920×800, 23.976), the `realtime` profile sustains **≥ 25 src fps (≥1.04× real time)**
for 10 minutes, with GPU utilisation ≥ 90% in `nvidia-smi dmon`. At `scale=1.0` it reaches ≥ 18 src fps (up from 14.7).

### Phase 3: Job manager and web server (both modes in the UI)

**jobs/manager.py:**
- One GPU slot. A priority queue holds jobs `{id, video_id, profile_id, kind: "stream"|"offline", start_segment,
  created, state}`, persisted to `cache/jobs.json` atomically on every change.
- Each job runs in a **subprocess**: `python -m streamerframes.pipeline.worker <job.json>`, started with
  `creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`, cwd set to the repo, and stdout/stderr to `worker.log`. That
  gives isolation from CUDA crashes, clean VRAM release, and killability.
- **Preemption:** a `stream` job outranks `offline`. To preempt, touch `stop` and wait up to 30 s for a graceful exit
  (at most one segment of work is lost). After that, kill the worker **and its ffmpeg children** with
  `psutil.Process(pid).children(recursive=True)` and terminate them. Re‑queue the preempted offline job so it resumes automatically.
- **Adoption on server start:** scan `cache/*/*/lock`. If the PID is alive (and its create‑time matches), adopt it and monitor
  via `progress.json`. If it's dead, mark the job `paused` and re‑queue offline jobs (configurable `auto_resume=true`).
- Watchdog: if `progress.json.updated_at` is stale for more than 120 s while the PID is alive, kill it, mark it `failed`, and log.
- Before launching, check free VRAM with `torch.cuda.mem_get_info` inside the worker. Fail fast with a clear error if it's
  under ~1.5 GB, since other apps such as LM Studio use the GPU.
- Optional `idle_only` setting for offline jobs: run only when no stream job is active (always true), plus an optional
  time window, e.g. overnight.

**web/app.py routes** (JSON API plus static pages; IDs are opaque, so **never** accept file paths from clients):
```
GET  /                                  static index.html (library)
GET  /watch/{video_id}?mode=original|framegen&profile=…   static watch.html
GET  /compare/{video_id}                side-by-side original vs framegen, synced (port the old /compare page)
GET  /api/library                       [{video_id, title, duration, w, h, fps, caches:[{profile_id, status, pct, has_export}]}]
GET  /api/videos/{video_id}             VideoInfo + caches + current job
POST /api/jobs                          {video_id, kind, profile, start_seconds?} → job (dedupes: returns the existing job)
GET  /api/jobs                          queue + running job with progress
GET  /api/jobs/{job_id}                 progress.json + derived fields (see below)
DELETE /api/jobs/{job_id}               cancel (graceful stop)
POST /api/videos/{video_id}/export      queue a finalize for a complete cache
DELETE /api/cache/{video_id}[/{profile_id}]
GET  /media/original/{video_id}         FileResponse (range requests); 415 + message for mkv in browsers that can't play it
GET  /hls/{video_id}/{profile_id}/master.m3u8
GET  /hls/{video_id}/{profile_id}/video.m3u8
GET  /hls/{video_id}/audio.m3u8
GET  /hls/{video_id}/{profile_id}/seg/{n}.ts   and   /hls/{video_id}/audio/{n}.ts
GET  /exports/{video_id}/{profile_id}   download the exported file
```

**Playlists (`cache/playlist.py`):**
- `master.m3u8`:
  ```
  #EXTM3U
  #EXT-X-VERSION:6
  #EXT-X-INDEPENDENT-SEGMENTS
  #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="aud",NAME="Audio",DEFAULT=YES,AUTOSELECT=YES,URI="../audio.m3u8"
  #EXT-X-STREAM-INF:BANDWIDTH=<maxrate>,RESOLUTION=<W>x<H>,FRAME-RATE=<out_fps 3dp>,CODECS="avc1.640028,mp4a.40.2",AUDIO="aud"
  video.m3u8
  ```
- `video.m3u8` (Phase 3 behaviour): `#EXT-X-PLAYLIST-TYPE:EVENT`, `#EXT-X-TARGETDURATION:<ceil(seg secs)>`, and
  the **contiguous completed prefix** of segments with exact `#EXTINF` (`N/out_fps`, 6 dp). Add `#EXT-X-ENDLIST` and
  switch to `VOD` once status is complete. Send `Cache-Control: no-cache` for playlists and long caching for segments.
- `audio.m3u8`: the same idea from `audio/audio.csv`. Because audio encodes far faster than real time, it's normally complete.
- A/V sync: video segments get `-output_ts_offset` and both renditions use mpegts with the same muxer defaults.
  **Write an integration test** (`tests/test_av_sync.py`): synthetic source = `testsrc2` plus a 1 kHz beep on every
  24th frame with a white flash on that frame. After generation, decode the HLS output (`ffmpeg -i master.m3u8`) and check
  flash frame time against beep onset is within 1 output frame, including across a forced resume boundary. If demuxed
  audio can't be made to pass, fall back to muxing audio into each video segment run (`-ss`‑aligned second input,
  `-c:a aac`) and drop the audio rendition.

**Streaming UX (watch.html + app.js):**
- Vendor **hls.js** pinned (e.g. 1.5.x `hls.min.js` from jsdelivr, committed to `web/static/`). No CDN at runtime.
- On "Framegen": `POST /api/jobs {kind:"stream"}` (deduped). Poll `/api/jobs/{id}` every 2 s and show segments ready,
  generated time against duration, speed `×real time`, and ETA.
- **Safe‑start estimate**: with real‑time ratio `r` (generated video seconds per wall second, smoothed over 60 s),
  remaining duration `D_rem` from the current play position, and already‑buffered lead `B`, playback won't stall if
  `B ≥ D_rem·(1 − r)/r` when `r < 1`. Show "Playing now will run without stalling" if that holds, else "Wait ~X min for
  stall‑free playback (or play now)". Start automatically once 3 segments exist **and** that condition holds; otherwise wait
  for the user to click.
- hls.js config: `liveSyncDurationCount` large, or treat the EVENT playlist as VOD‑like with `startPosition: 0`. Make sure
  the player doesn't jump to the live edge. On a stall, show "Generating… (frontier at mm:ss)" instead of a spinner‑only state.
- Library page: one row per movie with badges (`—`, `Generating 43% · 0.92× · ETA 1h12m`, `Paused 60%`, `Ready`,
  `Exported`) and buttons: Watch original · Watch framegen · Generate in background · Export · Delete cache.
  HTML‑escape everything (use `textContent`).
- Logging: Python `logging` with `RotatingFileHandler` (`cache/server.log`, 5 MB × 3). Log API calls at INFO and
  segment and playlist fetches at DEBUG only.

**Config (`streamerframes.toml`, see the example file):**
```toml
movies_roots = ["E:/Movies"]          # the .lnk resolution is dropped; use a config path (resolve once at startup if kept)
cache_root   = "E:/StreamerFrames/cache"
export_dir   = ""                     # empty = inside the cache; else e.g. "E:/Movies/_framegen"
host = "0.0.0.0"; port = 8000         # LAN access is intended; there's no auth, so document that
cache_max_gb = 200                    # LRU eviction of non-exported, non-running profiles
auto_resume = true

[profiles.realtime]
mode = "multi"; multi = 2; scale = "auto"; model = "4.25"
encoder = { codec = "h264_nvenc", preset = "p4", cq = 21, maxrate = "20M" }
scene_detect = true; letterbox_crop = true; cuda_graphs = true

[profiles.quality]
mode = "target"; target_fps = 60; scale = 1.0; model = "4.25"
encoder = { codec = "h264_nvenc", preset = "p6", cq = 18, maxrate = "40M" }
scene_detect = true; letterbox_crop = true; cuda_graphs = true
```
The profile hash includes mode, multi or target, scale, model, scene and crop settings, and encoder settings, but **not** host or paths.

**Launchers:** fix `start_server.bat` to run `"%VENV_PY%" -m streamerframes serve` with cwd = `%~dp0`.
Make `start_server.ps1` do the same. Keep `create_schtask.bat`, but run it with `/RL LIMITED` (HIGHEST isn't needed).

**Accept (Phase 3):**
- Clicking Framegen on Batman Begins starts playback within about 20 s (3 segments), plays continuously at 47.95 fps
  with in‑sync audio, while the status shows ≥1.0× real time.
- Clicking a second movie while the first is generating doesn't start a second GPU process. It queues, or replaces
  the first stream job (configurable; default: the newest stream job wins and the older one is paused).
- Queue an offline job, start streaming another movie, and confirm the offline job pauses and auto‑resumes after
  the stream job finishes or is cancelled.
- Restart the server mid‑generation and the running worker is adopted (no duplicate). Kill the worker and it is re‑queued
  and resumes from the last complete segment.
- An offline job for `thriller_input.mp4` with the `quality` profile completes, and the export plays in the browser and in VLC.

### Phase 4: Seek‑aware on‑demand generation (streaming polish)

Because segment boundaries and durations are deterministic (section 4), the server can publish the **whole timeline up front**.
1. `video.m3u8` becomes `#EXT-X-PLAYLIST-TYPE:VOD` with **all** `total_segments` entries and `#EXT-X-ENDLIST`
   from the start, so the player shows the full duration and allows seeking anywhere.
2. Requesting segment `n` that isn't complete:
   - If `n` is within the current run's frontier plus 3 segments, long‑poll (await up to 25 s, checking the index file) and serve it when ready.
   - Otherwise, tell the job manager to **retarget**: gracefully stop the current stream run and start a new run at
     `start_segment = n` (the generator fills the gap `[n, next completed)`), then long‑poll. Return 503 with `Retry-After: 2`
     if the 25 s pass. Configure hls.js `fragLoadingMaxRetry`/`fragLoadingRetryDelay` to tolerate this.
   - Debounce retargets (e.g. at most one per 3 s) so scrubbing doesn't thrash ffmpeg start‑up (~1–2 s each).
3. After the user's run reaches the end, or the stream job ends, an offline continuation job (optional, `fill_gaps=true`)
   fills holes left by seeks so the cache eventually becomes `complete`.
4. Playlist `#EXTINF` for the final short segment uses the manifest's real frame count once known. Before that, use the
   predicted value (the last segment is the only one that can differ).

**Accept:** seek to 1:30:00 in a fresh movie and playback resumes within ~10 s at the right picture and audio.
Seeking back into already generated content is instant, and holes are filled afterwards.

### Phase 5: Quality and model options

1. Scene‑cut and duplicate‑frame handling (from Phase 1) is tuned on real films. Add `tests/data/` clips with cuts and
   check for no blended frames across cuts. Expose thresholds in the profile.
2. **Lite models:** support `models/<name>/` dirs. The user downloads 4.25.lite / 4.22.lite from the upstream
   README links (Google Drive; an agent can't fetch these, so ask the user). Benchmark with `tools/bench.py` and add the
   results to calibration so `scale=auto` can also choose `model=auto` for `realtime`.
3. Optional ensemble: none (RIFE ≥4.21 dropped it). Don't add it.
4. Optional HEVC export (`hevc_nvenc`, Pascal: no B‑frames) for smaller offline files. H.264 stays the streaming
   codec for browser compatibility.
5. Optional "watch folder" (`watch_dirs` in config): new files get an offline `quality` job queued automatically.

### Phase 6 (optional): TensorRT 8.6 backend, go/no‑go

Only after Phases 1–3 are done. Time‑box it to one session.
1. Create a **separate** venv (`.framegen-trt`) or uninstall TRT 11 from `.framegen`, then install
   `TensorRT-8.6.1.6\python\tensorrt-8.6.1-cp311-none-win_amd64.whl`. Check which CUDA major version the zip targets
   (`dumpbin /dependents TensorRT-8.6.1.6\lib\nvinfer.dll` or the Readme). If it's CUDA 12.x, the CUDA 12 DLLs that ship
   in `torch\lib` may satisfy it via `os.add_dll_directory`. If it's 11.x, a CUDA 11.8 runtime is needed (ask the user).
2. Rewrite the export: wrap `IFNet` with inputs `(img0, img1, timestep[1,1,1,1])`, the correct **5‑entry** `scale_list`
   for the chosen scale, output `merged[-1]` (a tensor), opset 17 (GridSample is supported by TRT 8.6), **static**
   shapes per padded resolution. Verify ONNX Runtime CPU output matches PyTorch (max abs error < 1e‑3).
3. Build an **FP32** engine (no FP16 on Pascal), workspace about 2 GB, and cache it as
   `cache/engines/<gpu>_<model>_<W>x<H>_s<scale>.engine`.
4. `engine/trt_backend.py` exposes the same `interpolate()` contract, with I/O bound to torch tensors' data pointers on the same stream.
5. **Go/no‑go:** keep it only if it's ≥1.25× faster than the PyTorch+CUDA‑graphs path at 1920×832 with output PSNR ≥ 45 dB
   against PyTorch. Otherwise delete the branch and record the numbers in the README.

---

## 6. Testing strategy

- `pytest -m "not gpu"` runs fast everywhere: timeline math (property tests over many fps pairs), ffmpeg command builders
  (exact arg lists), probe parsing (fixture JSON from the real Batman probe and from `input.mp4`), playlist rendering, cache
  index union and hole detection, profile hashing stability, atomic JSON writes, and the job manager state machine (with a fake worker).
- `pytest -m gpu` (this machine): engine output shape and padding for {640×360, 1280×720, 1920×800, 1920×1080, 3840×2160}
  × scale {1.0, 0.5}. Colour round trip. CUDA‑graph and eager equivalence. Lossless end‑to‑end framemd5 test. Resume
  equivalence test. A/V sync beep‑flash test. Segment frame counts and IDR checks. Generate fixtures with ffmpeg
  `testsrc2`/`sine` at 24000/1001 and 30000/1001, 5–20 s, small resolutions, to keep it quick.
- Manual acceptance per phase, using `Batman Begins.mp4` and `thriller_input.mp4`.
- Every phase ends with `README-StreamerFrames.md` updated (how to run, config, measured numbers).

---

## 7. Risks and open questions

| Risk | Mitigation |
|---|---|
| `-segment_time` rounding cuts at the wrong keyframe | `-segment_time_delta`, plus a test asserting exact N frames per segment. Fallback: `-segment_frames` list or the hls muxer with `-hls_time` |
| Demuxed audio rendition drifts or offsets in hls.js | Beep‑flash test. Fallback to muxed audio per run (section 5, Phase 3) |
| Frame‑accurate `-ss` resume off by one | framemd5 resume test. If flaky, decode from the previous keyframe and discard frames by counted pts in Python |
| Real time not reached at full‑frame 1080p | Letterbox crop, `scale=0.5`, CUDA graphs, lite model, TRT (Phase 6). The UI's safe‑start estimate keeps playback smooth regardless |
| 48 fps on a 60 Hz display shows a mild cadence | Offer the `quality` profile (true 59.94) for offline. Document the trade‑off |
| Other GPU apps (LM Studio) take VRAM or compute | VRAM pre‑check, clear error, throughput shown in UI |
| Disk usage (~5–10 GB per film at 1080p) | `cache_max_gb` LRU eviction, delete buttons, export then purge |
| Server exposed on LAN without auth | Intended for home LAN. Document it. Optional `host="127.0.0.1"` |
| 10‑bit/HDR, rotated, anamorphic, VFR sources | Detect and warn. VFR is CFR‑ized. HDR is decoded to 8‑bit SDR (washed out). Out of scope beyond warnings |

Questions for the user (defaults assumed if unanswered):
1. 24p films: is the default streaming output of **2× (47.95 fps)** acceptable, with true 60 only offline? (Assumed yes.)
2. Where should exports go: inside the cache (default) or next to the movies (e.g. `E:\Movies\_framegen\`)?
3. Should the server be reachable from other devices on the LAN (default `0.0.0.0`)?
4. Should it pre‑generate the whole library overnight (watch‑folder or "queue all")? (Default: no.)

---

## 8. Definition of done

- `python -m streamerframes render "E:\Movies\Some.mp4"` produces a correct, resumable, exported higher‑fps file with
  in‑sync audio and subtitles.
- `start_server.bat` brings up the UI. Framegen streaming of a 1920×800 24p film starts within ~20 s and plays without
  stalls at ≥1.0× real time. Other content shows an accurate safe‑start ETA.
- Only one GPU worker ever runs. Jobs survive server restarts and are resumable, and nothing is marked complete unless it is.
- No legacy scripts are needed. All tests pass. `README-StreamerFrames.md` documents setup, config, and measured performance.
