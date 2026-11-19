"""`report`: the measured numbers (benchmark calibration and generation speed) as tables for the README."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from .bench import REALTIME_MARGIN, load_calibration, parse_key
from .cache.store import CacheStore

# Source rates the realtime (2x) check is shown for: 23.976 fps films and 29.97 fps video.
REALTIME_SOURCES = (Fraction(24000, 1001), Fraction(30000, 1001))
MULTI = 2

BENCH_HINT = "No benchmark results yet: run `python -m streamerframes bench <movie> --crop --model all`."
GEN_HINT = "No generations yet: run `python -m streamerframes render <movie>` or use the web UI."


def sustains(ms: float, src_fps: Fraction, multi: int = MULTI, margin: float = REALTIME_MARGIN) -> bool:
    """Whether ``ms`` per interpolation keeps up with ``multi``x of ``src_fps`` at the realtime margin."""
    return 1000.0 / ms >= margin * float(src_fps) * (multi - 1)


def bench_rows(entries: dict) -> list[list[str]]:
    parsed = sorted(((parse_key(k), v) for k, v in entries.items()),
                    key=lambda e: (e[0]["gpu"], -e[0]["width"] * e[0]["height"], e[0]["model"], -e[0]["scale"],
                                   e[0]["graphs"]))
    rows = []
    for key, value in parsed:
        ms = float(value["ms"])
        rows.append([f"{key['width']}×{key['height']}", key["model"], f"{key['scale']:g}",
                     "yes" if key["graphs"] else "no", f"{ms:.1f}", f"{1000 / ms:.1f}"]
                    + ["yes" if sustains(ms, fps) else "no" for fps in REALTIME_SOURCES])
    return rows


def bench_header() -> list[str]:
    return (["Size", "Model", "Scale", "CUDA graphs", "ms/frame", "Interpolations/s"]
            + [f"2× {float(fps):.3f} fps" for fps in REALTIME_SOURCES])


GEN_HEADER = ["Film", "Profile", "Output fps", "Status", "Done", "Source fps", "× real time"]


def generation_rows(store: CacheStore) -> list[list[str]]:
    rows = []
    for vid in store.video_ids():
        source = store.source(vid) or {}
        for pc in store.profiles(vid):
            m = pc.manifest() or {}
            progress = pc.progress() or {}
            film = Path(m.get("source") or source.get("path") or vid).name
            out_fps = f"{float(Fraction(m['out_fps'])):.3f}" if m.get("out_fps") else "?"
            total = m.get("total_segments") or 0
            done = f"{100 * len(pc.completed_segments()) / total:.0f}%" if total else "?"
            src_fps = progress.get("src_fps_measured")
            ratio = progress.get("realtime_ratio")
            rows.append([film, m.get("profile_name", "?"), out_fps, m.get("status", "?"), done,
                         f"{src_fps:.1f}" if src_fps else "", f"{ratio:.2f}" if ratio else ""])
    return sorted(rows)


def format_table(header: list[str], rows: list[list[str]], markdown: bool) -> str:
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in [header, *rows])


def report(cache_root: str | Path, markdown: bool = False) -> str:
    entries = load_calibration(cache_root)
    gens = generation_rows(CacheStore(cache_root))
    out = []
    title = "### {}\n" if markdown else "{}\n"
    out.append(title.format("Benchmark") + (format_table(bench_header(), bench_rows(entries), markdown)
                                            if entries else BENCH_HINT))
    if entries:
        out.append(f"Real time = at least {REALTIME_MARGIN:g}× the interpolations a 2× stream needs "
                   "(the margin `scale = \"auto\"` uses).")
    out.append(title.format("Generations") + (format_table(GEN_HEADER, gens, markdown) if gens else GEN_HINT))
    if gens:
        out.append("Source fps and × real time are the most recent ~60 s of the last run.")
    return "\n\n".join(out) + "\n"
