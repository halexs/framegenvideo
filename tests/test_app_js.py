"""Unit tests for the browser-side math in app.js, run with node when available."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parent.parent / "streamerframes" / "web" / "static" / "app.js"
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")

HARNESS = """
const window = {}; const document = {}; const location = {search: "", pathname: "/"};
class URLSearchParams { get() { return null; } }
const performance = {now: () => 0};
%s
const SF = window.SF;
const out = {
  ahead: SF.safeStart(100, 1000, 0, 1.2),
  slow_ok: SF.safeStart(500, 1000, 0, 0.5),
  slow_wait: SF.safeStart(100, 1000, 0, 0.5),
  stalled: SF.safeStart(10, 1000, 0, 0),
  done: SF.safeStart(1000, 1000, 990, 0),
  short_lead: SF.safeStart(3, 1000, 0, 2),
};
console.log(JSON.stringify(out));
"""


def test_safe_start():
    script = HARNESS % APP_JS.read_text()
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert out["ahead"] == {"ok": True, "wait": 0}
    # r=0.5 with 10% margin -> 0.45: need B >= 1000*0.55 = 550; 500 isn't enough...
    assert out["slow_ok"]["ok"] is False and abs(out["slow_ok"]["wait"] - 50 / 0.45) < 1e-6
    # ...and from B=100 the wait is (550-100)/0.45 = 1000 s.
    assert abs(out["slow_wait"]["wait"] - 1000) < 1e-6
    assert out["stalled"]["ok"] is False and out["stalled"]["wait"] is None  # Infinity -> null in JSON
    assert out["done"] == {"ok": True, "wait": 0}
    assert out["short_lead"]["ok"] is False  # always want ~8 s of lead before starting
