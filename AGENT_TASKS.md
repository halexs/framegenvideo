# Agent tasks (code that needs no GPU)

Written 2026-11-16 for a coding agent. `NEXT_STEPS.md` is the runbook a
person follows on the Windows PC with the GTX 1080 Ti. Most of what is
left in this project is measurement on that machine, which an agent
cannot do. This file is the small amount of **code** that makes those
steps easier, specified so an agent can build it on any machine in one
run.

Do the tasks in order, one commit each, on a branch off `main`, then open
one PR. Do not ask questions; take the defaults written here.

## Ground rules

- Read `README-StreamerFrames.md` and `PLAN.md` sections 3 and 6 first.
- Tests run on CPU with real ffmpeg and the fake model
  (`tests/fake_model.py`); see `README-StreamerFrames.md` "Tests". Keep
  it that way: no test may need CUDA, NVENC or a real RIFE model.
- No new dependencies.
- Do not change generation, timing or cache-format code. If a task seems
  to need that, stop and say so in the PR.
- Finish with `pytest` green, and update `README-StreamerFrames.md`,
  `streamerframes.example.toml` and `NEXT_STEPS.md` wherever a task
  changes what they describe.

---

## Task A: media-server-friendly export names (`NEXT_STEPS.md` step 8)

Today `default_export_name()` in `streamerframes/pipeline/export.py`
produces `<stem>.59.940fps.mp4`. Jellyfin and similar servers match
metadata on `Title (Year).ext`; the extra `.59.940fps` segment can break
that, and two versions of one film in one folder collide.

- New setting `export_layout` (`streamerframes/config.py`, default
  `"flat"` so nothing changes for existing users):
  - `"flat"`: current behaviour.
  - `"library"`: the export is written to
    `<export_dir>/<stem>/<stem> - <label>.<container>` where `<label>` is
    `60fps` or `48fps` style (rounded frame rate, plus ` HEVC` when the
    codec is hevc). Jellyfin treats `Title (Year) - Label.ext` inside a
    `Title (Year)` folder as a named version of that film.
- `<stem>` keeps the source file's stem with the existing character
  sanitising.
- Only `default_export_name()` and the directory choice in `export()`
  change. Explicit `--export <path>` is untouched.
- The library listing and download endpoint must still find exports in
  the new layout (check `ProfileCache.export_dir`, `store.py` around the
  `exported =` line, and `download_export` in `web/app.py`).

**Tests:** names for 23.976→59.94, 24→48, hevc, a title with a year and
punctuation; `flat` unchanged (existing tests pass untouched); an
end-to-end render with `export_layout = "library"` and an `export_dir`
lands at the expected path and is reported by the library API.

## Task B: a `report` command for the measured numbers (`NEXT_STEPS.md` steps 2–4)

The runbook asks the person to copy benchmark lines into tables by hand.

- New CLI: `python -m streamerframes report [--markdown]`.
- Reads `cache/calibration.json` (via `bench.load_calibration`) and
  every cache manifest (via the store) and prints:
  1. a benchmark table: size, model, scale, CUDA graphs, ms/frame,
     interpolations per second, and whether that sustains 2× for 23.976
     and 29.97 fps sources at the 1.1× margin the `realtime` profile
     uses (take the margin from the code, do not restate the number);
  2. a generations table: film, profile, output fps, status, measured
     source fps and real-time ratio where the manifest or progress file
     records them.
- `--markdown` prints tables ready to paste into the "Measured
  performance" section of `README-StreamerFrames.md`.
- Empty calibration or no caches prints a one-line hint naming the
  command that fills it (`bench …`, `render …`).

**Tests:** a fixture calibration file and two fake manifests produce the
expected rows; the empty case; `--markdown` output parses as a Markdown
table.

## Task C: make the docs agree with the code about sleep

`README-StreamerFrames.md` ("Web UI") says watching an already-generated
video from another device does not count as activity. The code says
otherwise: `streamerframes/web/awake.py` (`PlaybackKeepAwake`) blocks
sleep while media is being fetched and for 120 s after the last fetch,
and `tests/test_awake.py` covers it.

- Rewrite that README paragraph to describe what the code does: idle
  server lets the PC sleep; a running generation holds it awake; playback
  from any device holds it awake until about two minutes after the last
  segment is fetched; a paused player stops fetching and so releases it.
- Add one sentence that this is verified by unit tests only and has not
  yet been confirmed on the Windows machine, pointing to `NEXT_STEPS.md`
  step 4.
- In `NEXT_STEPS.md` step 4, shorten the sleep check to "confirm the
  README's description on the real machine".

No code change. If reading the code shows the README was right after
all, leave both files alone and explain in the PR.

---

## PR description must include

- Which tasks are done and what a person should check on the PC.
- The output of `pytest`.
