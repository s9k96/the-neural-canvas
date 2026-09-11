"""Render setting-the-distance.html in a real browser and assert it actually built.

Same reason S8, S9 and S10 have one: a page can parse cleanly, balance its tags and still
render nothing, because parsing a script does not run it. Every panel here is built by
JavaScript from the baked `S11DATA` blob, so if that blob is missing or a render function
throws, the page is a set of empty boxes and no static check would notice.

    python check_page.py

Exits non-zero on any console error, or if the DOM does not contain what a built page
contains. Requires Google Chrome; the page itself needs nothing.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "setting-the-distance.html"
EVIDENCE = HERE / "out" / "evidence.json"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# Everything below is produced by JS at runtime, so a non-zero count proves the corresponding
# render function ran. Matching happens against the DOM with every <script> block REMOVED --
# the render functions build these strings out of template literals that live in the script
# source, so an unstripped dump counts each one twice and the check silently passes on a page
# that rendered nothing.
EXPECT = [
    ("hero tiles",            r'<div class="tile"><div class="k">',           6,  6),
    ("tables built",          r'<thead><tr><th',                             7,  7),
    ("readout blocks",        r'<div class="k">[^<]+</div><div class="v">',  20, 60),
    # Only chart series: the sidebar's nav icons also carry stroke-linejoin, so match the
    # attribute pair that only a series path has.
    ("chart series paths",    r'fill="none" stroke="#[0-9a-f]{6}" stroke-width=', 28, 38),
    ("chart dots",            r'<circle cx="[\d.]+" cy="[\d.]+" r="[\d.]+"', 60, 400),
    ("reference marks",       r'stroke-dasharray="',                         18, 60),
    ("adam gradient inputs",  r'<input type="number"',                        6,  6),
    ("layer selector options", r'<option value="',                            9,  9),
    ("gate rows",             r'<div class="gate (pass|fail)">',             24, 24),
]

SCRIPT = re.compile(r"<script\b.*?</script>", re.S | re.I)


def js_exp(x: float, n: int) -> str:
    """Python's %e prints e-03 where JavaScript's toExponential prints e-3."""
    mant, exp = f"{x:.{n}e}".split("e")
    return f"{mant}e{int(exp):+d}"


def expected_strings() -> list[tuple[str, str]]:
    """Values that must reach the screen, read from evidence.json rather than hardcoded, so
    this check cannot drift from the run that produced the page."""
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    a = ev["summary"]["answers"]
    return [
        ("tokenizer sha prefix", ev["corpus"]["tokenizer_sha256"][:8]),
        ("bias-correction steps", f'{a["2_steps_until_bias_correction_stops_mattering"]:,} steps'),
        ("warmup step", f'step {a["3_warmup_stops_changing_the_ratio_at_step"]}'),
        ("which model to keep", a["4_loss_at_stop"]["keep"].upper()),
        ("eta at width 4096", js_exp(a["5_lr_at_width_4096"]["mup_measured"], 1)),
        ("adam vs torch", js_exp(a["1_worst_disagreement_with_torch"], 1)),
        ("worst layer prediction error",
         f'{ev["task3_warmup"]["worst_prediction_error"] * 100:.2f}%'),
        ("peak bias-correction gap", f'step {ev["task2_bias_correction"]["peak_gap_step"]}'),
        ("eta-lambda timescale", f'{round(ev["extras"]["eta_lambda"]["rows"][0]["predicted"]):,}'),
        ("adamw bytes per weight", "16"),
    ]


def main() -> int:
    if not CHROME.exists():
        print(f"FATAL: Chrome not found at {CHROME}")
        return 2
    for p in (PAGE, EVIDENCE):
        if not p.exists():
            print(f"FATAL: {p} missing")
            return 2

    proc = subprocess.run(
        [str(CHROME), "--headless=new", "--disable-gpu", "--virtual-time-budget=9000",
         "--enable-logging=stderr", "--v=0", "--dump-dom", PAGE.as_uri()],
        capture_output=True, text=True, timeout=180)
    raw, log = proc.stdout, proc.stderr
    dom = SCRIPT.sub("", raw)                  # see EXPECT: script source would double every count

    print("=" * 74)
    print("S11 page runtime check -- does the page actually build in a browser?")
    print("=" * 74)
    print(f"dumped DOM {len(raw):,} bytes; {len(dom):,} after removing <script> blocks")
    fails = []

    console = [l for l in log.splitlines() if ":CONSOLE:" in l]
    real = [l for l in console if re.search(r"error|uncaught|exception", l, re.I)]
    print(f"console messages: {len(console)}   errors: {len(real)}")
    for l in real:
        m = re.search(r'"(.*?)", source', l)
        fails.append("console error: " + (m.group(1) if m else l.strip()))

    print()
    print(f"{'what':<26}{'found':>7}  expected")
    print("-" * 74)
    for name, pat, lo, hi in EXPECT:
        n = len(re.findall(pat, dom))
        ok = lo <= n <= hi
        rng = str(lo) if lo == hi else f"{lo}-{hi}"
        print(f"{name:<26}{n:>7}  {rng:<8} {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{name}: found {n}, expected {rng}")

    print()
    print("values from evidence.json that must reach the screen")
    print("-" * 74)
    for name, val in expected_strings():
        ok = val in dom
        print(f"{name:<30}{val:>16}  {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{name} ({val}) not found in the rendered DOM")

    if "No baked data" in dom:
        fails.append("page rendered its no-data fallback: S11DATA was not baked in")

    print()
    if fails:
        print(f"{len(fails)} PROBLEM(S):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("page builds clean: every panel rendered, no console errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
