# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Neural Canvas** — a static site of interactive, in-browser ML visualizations paired with course
assignments (ERA V5). Each `sN-*` / `SN-*` directory is one session's submission: an
`.html` widget (vanilla JS, no build step), often backed by a real offline Python pipeline whose
output is baked into the page as a JS data blob. There is no bundler, package.json, or dev server —
every HTML file is opened directly or served as static files (deployed to Vercel; see
`.agents/skills/`, a Vercel-CLI skill set unrelated to Claude Code).

Not every session has a widget: S5 (`s05-data-mixtures`) and S8 (`s08-model-architectures`) are
markdown/notes-only deliverables and are deliberately absent from the site nav and homepage.

## Site shell

- `index.html` — homepage: a hand-written grid of `topic-card` tiles, one per widget. Each tile
  carries `data-session="S7"` plus a `.card-badge`, a one-line `.card-hook` (always visible) and the
  full `.card-detail` paragraph (a flyout on hover/focus; shown inline on touch, where the hook is
  hidden instead). `initTopicFilter` in `shared.js` builds the session chips from the tiles'
  `data-session` values and searches each tile's own text plus its `NAV_SECTIONS` label — so tiles
  stay the only place the card list is written.
- `shared.css` / `shared.js` — loaded by every session HTML page.
- `shared.js` has a `NAV_SECTIONS` array that is the **single source of truth for the sidebar**,
  separate from `index.html`'s topic-card grid. Adding a new session widget means updating
  **both**: a `topic-card` tile in `index.html` and an entry in `NAV_SECTIONS` (plus an SVG icon
  in the `ICONS` map) in `shared.js` — neither regenerates from the other. The tile's `href` must
  match the `NAV_SECTIONS` `href` exactly, or it gets no icon and no session label in search.
- Pages set `window.PAGE_ID` (to highlight the active nav item) and optionally `window.ROOT_PATH`
  (relative prefix back to repo root) before loading `shared.js`.

## The generate-then-bake pattern (S4, S6, S7, S9, S10, S11)

Sessions with real data pipelines follow the same shape: a Python pipeline writes JSON/JS into an
`out/` (or `submission_artifacts/`) folder, and a small `build_html_data.py` (S4, S6), the
`run_demo.py` itself (S7), or `build_notebook.py` (S9, S10, S11) injects that data as a literal
`const DATASETS = {...}` / `s6data` / `S9DATA` / `S11DATA` blob directly into the session's `.html` file. The HTML has no fetch/XHR — page data is static and
inlined, so **the generated JS blob in the HTML must be regenerated any time the upstream Python
output changes**; editing the JSON in `out/` alone does nothing until the build script re-runs.

Read a session's own `README.md` before changing its pipeline — S6 and S7 in particular document
non-obvious invariants (e.g. S6's `plan_batch` must stay a pure function of `(seed, branch, step)`
for resume/replay/fork to be provably correct; S7's sinusoidal `base` is deliberately recalibrated
from the Transformer's default).

## Commands

There is no repo-wide test runner or linter — each session with a Python pipeline is invoked and
tested independently, from that session's own directory conventions. Commands are written in each
README as `.venv/Scripts/python ...` (Windows-style); on macOS/Linux use `.venv/bin/python` with
whatever venv is active — no `.venv` or lockfile is currently committed, and `pyproject.toml`
(Poetry, requires Python ≥3.14) only declares `pandas`; per-session scripts pull in `numpy`,
`tokenizers`, `huggingface_hub`, `torch`, etc. directly and expect them to already be installed.

```bash
# S4 — data cleaning (after clean.py / export_examples.py produce out/*.json)
python s04-data-cleaning/build_html_data.py

# S6 — training data execution system (full demo + evidence bundle; ~50s, CPU, offline/deterministic)
python s06-dataset-creation/run_demo.py
# 13 invariant tests, run only after run_demo.py (reads submission_artifacts/ + shards/)
python s06-dataset-creation/tests/test_invariants.py     # also runnable via: python -m pytest
# refresh the committed corpus/ snapshot deliberately (normal runs never touch the network)
python s06-dataset-creation/prepare_corpus.py

# S7 — dynamic Kronecker embeddings (8 experiments, 13 gates, ~70s, CPU)
python s07-model-internals/run_demo.py
python s07-model-internals/dynkron.py     # codec self-check only

# S9 — the loss harness (11 experiments + MTP, 16 gates, ~9 min, CPU)
python s09-loss-functions/build_notebook.py   # .py -> executed .ipynb -> baked loss-harness.html
python s09-loss-functions/s9_loss_harness.py  # or the harness alone, without rebuilding the notebook
python s09-loss-functions/check_page.py       # renders the page in headless Chrome; needs Chrome

# S10 — the training loop (6 tasks, 15 gates, ~2 min, CPU)
python s10-training-loop/build_notebook.py    # .py -> executed .ipynb -> baked training-step.html
python s10-training-loop/check_page.py        # renders the page in headless Chrome; needs Chrome

# S11 — optimizers and schedules (6 tasks, 24 gates, ~55 min, CPU)
python s11-optimizers/build_notebook.py       # .py -> executed .ipynb -> baked setting-the-distance.html
python s11-optimizers/build_notebook.py --bake-only   # re-inject out/evidence.json into the page only
python s11-optimizers/check_page.py
```

S10 and S11 follow S9's shape exactly: a `# %%` cell-delimited `.py` is the source of truth, the
committed `.ipynb` carries its outputs, and `out/evidence.json` is baked into the page. S11's
learning-rate sweeps deliberately cap the vocabulary to 8,192 ids (the page and README say why);
its Tasks 1-4 use the full 68,096-token model.

S9's source of truth is `s9_loss_harness.py` (`# %%` cell-delimited). The `.ipynb` is a build
artifact committed **with its outputs**, and both it and the page must be rebuilt when the `.py`
changes. It downloads the Sarvam-1 tokenizer from the HF hub on first run and caches it; the
corpus it reads is S6's committed `corpus/`, so nothing else needs network.

Each `run_demo.py` exits non-zero if any gate/invariant fails — that exit code is the actual
pass/fail signal, not just the printed summary.

## Committed vs. derived data

Several sessions intentionally commit large generated artifacts as the submission itself, while
excluding data that is cheaply re-derivable — check a session's README table before assuming
`out/`, `submission_artifacts/`, or `corpus/` are build output that's safe to delete or gitignore.
For S6 specifically: `corpus/` (input snapshot) and `submission_artifacts/` (evidence bundle) are
committed; `shards/` (tokenized payloads) is not, because `run_demo.py` regenerates it byte-for-byte
from `corpus/` every run.