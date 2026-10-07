"""CLI: python -m streamerframes {check,probe,plan}. render/serve arrive in later phases (PLAN.md)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from fractions import Fraction

from .config import load_settings
from .probe import probe
from .selfcheck import failed, format_checks, run_checks
from .timeline import Timeline, out_fps_for_multiplier, out_fps_for_target


def _out_fps(args, src_fps: Fraction) -> Fraction:
    if args.target_fps:
        return out_fps_for_target(src_fps, args.target_fps)
    return out_fps_for_multiplier(src_fps, Fraction(args.multi))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="streamerframes")
    sub = parser.add_subparsers(dest="command", required=True)

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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.command == "check":
        checks = run_checks(load_settings(), args.model, need_gpu=not args.cpu)
        print(format_checks(checks))
        return 1 if failed(checks) else 0

    try:
        info = probe(args.input)
        if args.command == "probe":
            print(json.dumps(info.to_json(), indent=2))
            return 0
        tl = Timeline.create(info.src_fps, _out_fps(args, info.src_fps), info.n_src,
                             Fraction(args.segment_seconds))
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
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


if __name__ == "__main__":
    sys.exit(main())
