"""Benchmark RIFE at a resolution and record calibration (same as `python -m streamerframes bench`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamerframes.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["bench", *sys.argv[1:]]))
