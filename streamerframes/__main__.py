"""CLI: python -m streamerframes {serve,check,probe,plan,render,bench,cache,trt}."""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from fractions import Fraction
from pathlib import Path

from .config import load_settings
from .probe import probe
from .selfcheck import failed, format_checks, run_checks
from .timeline import Timeline, out_fps_for_multiplier, out_fps_for_target


def _out_fps(args, src_fps: Fraction) -> Fraction:
    if args.target_fps:
        return out_fps_for_target(src_fps, args.target_fps)
    return out_fps_for_multiplier(src_fps, Fraction(args.multi))


def _fmt_secs(seconds) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def cmd_plan(args) -> int:
    info = probe(args.input)
    tl = Timeline.create(info.src_fps, _out_fps(args, info.src_fps), info.n_src, Fraction(args.segment_seconds))
    print(json.dumps({
        "src_fps": f"{tl.src_fps} ({float(tl.src_fps):.3f})",
        "out_fps": f"{tl.out_fps} ({float(tl.out_fps):.3f})",
        "n_src": tl.n_src,
        "total_out_frames": tl.total_out_frames,
        "seg_frames": tl.seg_frames,
        "seg_seconds": round(float(tl.seg_frames / tl.out_fps), 4),
        "total_segments": tl.total_segments,
    }, indent=2))
    return 0


def cmd_render(args) -> int:
    from .pipeline.worker import Job, run_job

    settings = load_settings(args.config)
    overrides: dict = {}
    if args.multi:
        overrides.update(mode="multi", multi=int(args.multi))
    if args.target_fps:
        overrides.update(mode="target", target_fps=args.target_fps)
    if args.scale:
        overrides["scale"] = args.scale if args.scale == "auto" else float(args.scale)
    if args.model:
        overrides["model"] = args.model
    if args.encoder:
        overrides["encoder"] = {"codec": "lossless" if args.encoder == "lossless" else "h264_nvenc"}
    job = Job(video_path=str(Path(args.input).resolve()), profile=args.profile, overrides=overrides or None,
              start_segment=args.start_segment, export=not args.no_export, export_path=args.export,
              container=args.container, export_codec=args.export_codec, device=args.device,
              hwaccel=not args.no_hwaccel, config=args.config)

    if not args.skip_check:
        checks = run_checks(settings, (overrides.get("model") or settings.profile(args.profile).model),
                            need_gpu=args.device.startswith("cuda"))
        if failed(checks):
            print("self-check failed:\n" + format_checks(checks), file=sys.stderr)
            return 1

    stop = threading.Event()

    def on_sigint(*_):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        print("\nstopping after the current segment (Ctrl+C again to abort)...", file=sys.stderr)

    signal.signal(signal.SIGINT, on_sigint)

    def on_progress(p: dict) -> None:
        print(f"\rframe {p['out_frame']}/{p['total_out_frames']}  src {p['src_fps_measured']:.1f} fps  "
              f"{p['realtime_ratio']:.2f}x real time  ETA {_fmt_secs(p['eta_seconds'])}   ",
              end="", file=sys.stderr, flush=True)

    result = run_job(job, settings, stop_event=stop, on_progress=on_progress)
    print(file=sys.stderr)
    if result.status == "failed":
        print(f"failed: {result.error}", file=sys.stderr)
    elif result.status == "paused":
        print("paused; run the same command again to resume", file=sys.stderr)
    else:
        print(f"{result.status}", file=sys.stderr)
    return result.exit_code


def cmd_serve(args) -> int:
    import uvicorn

    from .web.app import create_app, setup_logging

    settings = load_settings(args.config)
    host = args.host or settings.host
    port = args.port or settings.port
    if not args.skip_check:
        checks = run_checks(settings, need_gpu=True)
        if failed(checks):
            print("warning: self-check failed; generation will not work until this is fixed:\n"
                  + format_checks(checks), file=sys.stderr)
    setup_logging(settings.cache_path, logging.DEBUG if args.verbose else logging.INFO)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"note: serving on {host} without authentication; anyone on your network can use it",
              file=sys.stderr)
    print(f"StreamerFrames on http://{'localhost' if host in ('0.0.0.0', '::') else host}:{port}/", file=sys.stderr)
    uvicorn.run(create_app(settings, config_path=args.config), host=host, port=port, log_level="warning",
                access_log=False)
    return 0


def cmd_bench(args) -> int:
    from .bench import run_bench, save_entries

    settings = load_settings(args.config)
    if args.input:
        info = probe(args.input)
        width, height = info.width, info.height
        if args.crop:
            from .pipeline.letterbox import detect_letterbox
            crop = detect_letterbox(info, ffmpeg=settings.ffmpeg)
            if crop:
                width, height = crop.w, crop.h
    else:
        width, height = map(int, args.size.lower().split("x"))
    scales = [float(s) for s in args.scales.split(",")]
    models = settings.available_models() if args.model in ("all", "auto") else [args.model]
    if not models:
        raise RuntimeError("no usable models found (see `check`)")
    results = {}
    for model in models:
        results.update(run_bench(settings.model_path(model), model, width, height, args.device, scales,
                                 warmup=args.warmup, iters=args.iters))
    save_entries(settings.cache_root, results)
    for key, value in results.items():
        print(f"{key}: {value['ms']:.2f} ms/frame ({1000 / value['ms']:.1f} interpolations/s)")
    return 0


def cmd_trt(args) -> int:
    """Phase 6 go/no-go: export ONNX (+ verify), build an FP32 engine, compare against PyTorch."""
    from .engine.loader import load_ifnet
    from .engine.onnx_export import export_onnx, verify_onnx
    from .engine.rife import RifeEngine

    settings = load_settings(args.config)
    info = crop = None
    if args.input:
        info = probe(args.input)
        width, height = info.width, info.height
        if args.crop:
            from .pipeline.letterbox import detect_letterbox
            crop = detect_letterbox(info, ffmpeg=settings.ffmpeg)
            if crop:
                width, height = crop.w, crop.h
    else:
        width, height = map(int, args.size.lower().split("x"))
    device = args.device
    # Upstream model/warplayer.py builds its grids on CUDA whenever a GPU exists, so the network must run
    # on that device for export too (a CPU export fails with "found at least two devices").
    net = load_ifnet(settings.model_path(args.model), device)
    padder = RifeEngine(net, height, width, args.scale, "cpu")
    ph, pw = padder.padded_shape
    onnx_path = settings.cache_path / "engines" / f"{args.model}_{pw}x{ph}_s{args.scale}.onnx"

    # Check on real frames when a video is given (random noise makes optical flow chaotic, so harmless
    # CPU-vs-GPU rounding differences would look like large errors).
    from .engine.rife import smooth_pair
    pairs = []
    if info is not None:
        from .pipeline.samples import sample_frame_pairs
        pairs = [(padder.pad(a), padder.pad(b)) for a, b in sample_frame_pairs(info, crop, ffmpeg=settings.ffmpeg)]
    source = f"{len(pairs)} real frame pairs" if pairs else "a synthetic image pair"
    if not pairs:
        pairs = [smooth_pair(ph, pw)]

    if args.trt_cmd == "export":
        from .engine.onnx_export import stats_ok
        export_onnx(net, ph, pw, args.scale, onnx_path)
        st = verify_onnx(net, onnx_path, ph, pw, args.scale, pairs)
        ok = stats_ok(st)
        print(f"{onnx_path}\nONNX Runtime vs PyTorch on {source}: mean abs error {st['mean']:.1e}, "
              f"99.9th percentile {st['p999']:.1e}, max {st['max']:.1e}, PSNR {st['psnr']:.1f} dB "
              f"-> {'ok' if ok else 'TOO DIFFERENT (need mean < 1e-3 and 99.9th pct < 4/255)'}")
        return 0 if ok else 1

    from .bench import bench_engine, gpu_name
    from .engine.trt_backend import TrtRifeEngine, build_engine, engine_path
    path = engine_path(settings.cache_root, gpu_name(device), args.model, pw, ph, args.scale)
    if args.trt_cmd == "build":
        if not onnx_path.exists():
            print(f"error: {onnx_path} not found; run `trt export` first", file=sys.stderr)
            return 1
        build_engine(onnx_path, path, args.workspace_gb)
        print(path)
        return 0

    # compare
    torch_engine = RifeEngine(net, height, width, args.scale, device)
    graphs = torch_engine.enable_cuda_graph()
    trt_engine = TrtRifeEngine(net, height, width, args.scale, device, path)
    torch_ms = bench_engine(torch_engine, args.warmup, args.iters)
    trt_ms = bench_engine(trt_engine, args.warmup, args.iters)
    from .engine.onnx_export import error_stats
    psnrs = []
    for a, b in pairs:
        a, b = a.to(device), b.to(device)
        ref = torch_engine.interpolate(a, b, 0.5).clone()
        out = trt_engine.interpolate(a, b, 0.5)
        psnrs.append(error_stats(out, ref)["psnr"])
    psnr = min(psnrs)  # worst pair
    print(f"PSNR measured on {source}")
    speedup = torch_ms / trt_ms
    go = speedup >= 1.25 and psnr >= 45
    print(f"{width}x{height} scale {args.scale}: PyTorch{' + CUDA graphs' if graphs else ''} {torch_ms:.1f} ms, "
          f"TensorRT {trt_ms:.1f} ms ({speedup:.2f}x), PSNR {psnr:.1f} dB -> {'GO' if go else 'NO-GO'}"
          " (plan: >= 1.25x and >= 45 dB)")
    return 0


def cmd_cache(args) -> int:
    from .cache.store import CacheStore, LockHeld, video_id_for

    settings = load_settings(args.config)
    store = CacheStore(settings.cache_root)
    if args.cache_cmd == "ls":
        for vid in store.video_ids():
            src = store.source(vid) or {}
            print(f"{vid}  {src.get('path', '?')}")
            for pc in store.profiles(vid):
                m = pc.manifest() or {}
                done = len(pc.completed_segments())
                total = m.get("total_segments") or 1
                print(f"    {pc.profile_id}  {m.get('profile_name', '?'):<16} {m.get('status', '?'):<9} "
                      f"{100 * done / total:5.1f}%  {pc.size_bytes() / 2**30:6.2f} GB"
                      + ("  exported" if m.get("exports") else ""))
        return 0
    if args.cache_cmd == "rm":
        target = args.target
        vid = video_id_for(target) if Path(target).exists() else target
        try:
            store.remove(vid, args.profile)
        except LockHeld as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"removed {vid}{'/' + args.profile if args.profile else ''}")
        return 0
    if args.cache_cmd == "gc":
        max_gb = args.max_gb if args.max_gb is not None else settings.cache_max_gb
        for pc in store.gc(int(max_gb * 2**30)):
            print(f"evicted {pc.root}")
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="streamerframes")
    parser.add_argument("--config", help="path to streamerframes.toml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the web UI and job manager")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--skip-check", action="store_true")

    p = sub.add_parser("check", help="verify GPU, torch, ffmpeg/NVENC and model files")
    p.add_argument("--model", default="default")
    p.add_argument("--cpu", action="store_true", help="don't require CUDA/NVENC")

    p = sub.add_parser("probe", help="print stream info for a video")
    p.add_argument("input")

    p = sub.add_parser("plan", help="show the output frame rate and segment layout for a video")
    p.add_argument("input")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--multi", default="2", help="frame-rate multiplier (default 2)")
    group.add_argument("--target-fps", type=int, help="target fps, e.g. 60 (59.94 for x/1001 sources)")
    p.add_argument("--segment-seconds", type=float, default=4.0)

    p = sub.add_parser("render", help="generate (resumable) and export a higher-fps file")
    p.add_argument("input")
    p.add_argument("--profile", default="quality")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--multi", type=int)
    group.add_argument("--target-fps", type=int)
    p.add_argument("--scale", help="auto, 1.0 or 0.5")
    p.add_argument("--model", help="model name under models_dir, or 'default'")
    p.add_argument("--encoder", choices=["nvenc", "lossless"], help="lossless = libx264 -qp 0 (testing)")
    p.add_argument("--export", help="output path (default: inside the cache or export_dir)")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--container", default="auto", choices=["auto", "mp4", "mkv"])
    p.add_argument("--export-codec", default="copy", choices=["copy", "hevc"],
                   help="hevc re-encodes the export (hevc_encoder in config) for a smaller file")
    p.add_argument("--start-segment", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-hwaccel", action="store_true", help="decode on the CPU")
    p.add_argument("--skip-check", action="store_true")

    p = sub.add_parser("bench", help="time the model and record calibration for scale=auto")
    p.add_argument("input", nargs="?", help="video to take the resolution from")
    p.add_argument("--size", default="1920x800", help="WxH when no input is given")
    p.add_argument("--crop", action="store_true", help="benchmark the letterbox-cropped size")
    p.add_argument("--model", default="all", help="a model name, or 'all' installed models")
    p.add_argument("--scales", default="1.0,0.5")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)

    p = sub.add_parser("trt", help="TensorRT backend (experimental): export, build, compare")
    tsub = p.add_subparsers(dest="trt_cmd", required=True)
    for name, text in (("export", "export ONNX and verify it against PyTorch with ONNX Runtime"),
                       ("build", "build an FP32 TensorRT engine from the exported ONNX"),
                       ("compare", "speed and PSNR of TensorRT vs PyTorch + CUDA graphs (go/no-go)")):
        t = tsub.add_parser(name, help=text)
        t.add_argument("input", nargs="?", help="video to take the resolution from")
        t.add_argument("--size", default="1920x800")
        t.add_argument("--crop", action="store_true")
        t.add_argument("--model", default="default")
        t.add_argument("--scale", type=float, default=1.0)
        t.add_argument("--device", default="cuda")
        t.add_argument("--workspace-gb", type=float, default=2.0)
        t.add_argument("--warmup", type=int, default=5)
        t.add_argument("--iters", type=int, default=30)

    p = sub.add_parser("cache", help="list, remove or evict cached generations")
    csub = p.add_subparsers(dest="cache_cmd", required=True)
    csub.add_parser("ls")
    rm = csub.add_parser("rm")
    rm.add_argument("target", help="video_id or video path")
    rm.add_argument("--profile", help="only this profile_id")
    gc = csub.add_parser("gc")
    gc.add_argument("--max-gb", type=float)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "check":
        checks = run_checks(load_settings(args.config), args.model, need_gpu=not args.cpu)
        print(format_checks(checks))
        return 1 if failed(checks) else 0
    try:
        if args.command == "probe":
            print(json.dumps(probe(args.input).to_json(), indent=2))
            return 0
        if args.command == "plan":
            return cmd_plan(args)
        if args.command == "serve":
            return cmd_serve(args)
        if args.command == "render":
            return cmd_render(args)
        if args.command == "bench":
            return cmd_bench(args)
        if args.command == "cache":
            return cmd_cache(args)
        if args.command == "trt":
            return cmd_trt(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
