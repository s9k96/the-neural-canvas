"""Render training-step.html in a real browser and assert it actually built.

Same reason S8 has one: a page can parse cleanly, balance its tags and still render
nothing, because parsing a script does not run it. Every panel on this page is built by
JavaScript from the baked `S9DATA` blob, so if that blob is missing or a render function
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
PAGE = HERE / "training-step.html"
EVIDENCE = HERE / "out" / "evidence.json"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# Everything below is produced by JS at runtime, so a non-zero count proves the
# corresponding render function ran. Matching happens against the DOM with every <script>
# block REMOVED -- the render functions build these strings out of template literals that
# live in the script source, so an unstripped dump counts each one twice and the check
# silently passes on a page that rendered nothing.
EXPECT = [
    ("hero + norm tiles",      r'<div class="tile"><div class="k">',        9,  9),
    ("nudge readout",          r'<div class="k">central difference</div>',  1,  1),
    ("shape-ledger rows",      r'<td class="mono" style="font-family:var\(--font-mono\)">', 6, 99),
    ("h-sweep series",         r'stroke-width="2" stroke-linejoin="round"', 5,  5),
    ("h-sweep points",         r'<circle cx="[\d.]+" cy="[\d.]+" r="[\d.]+"', 18, 99),
    ("micro-batch sliders",    r'<input type="range" data-i="\d"',          3,  3),
    ("gradient-difference cells", r'<div class="k">relative L2 difference</div>', 1, 1),
    ("grad-norm panels",       r'stroke-width="1.6" stroke-linejoin="round"', 2,  2),
    ("float bit groups",       r'<span class="grp [sem]">',                12, 12),
    ("memory readout",         r'<div class="k">training state alone</div>', 1,  1),
    ("mfu readout + hero tile", r'<div class="k">MFU</div>',                 2,  2),
    ("gate rows",              r'<div class="gate (pass|fail)">',          15, 15),
    ("data tables built",      r'<thead><tr><th',                           9,  9),
]

SCRIPT = re.compile(r"<script\b.*?</script>", re.S | re.I)

# Values that must reach the screen, read from evidence.json rather than hardcoded, so
# this check cannot drift from the run that produced the page.
def expected_strings() -> list[tuple[str, str]]:
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    s = ev["summary"]["six_answers"]
    return [
        ("tokenizer sha prefix", ev["tokenizer"]["sha256"][:8]),
        ("gradient agreement digits", f'{s["2_worst_agreeing_digits"]:.1f} digits'),
        ("the 2024 bug error", f'{s["3_wrong_average_error_pct"]:.1f}%'),
        ("grad-norm lead", f'{s["4_grad_norm_lead_steps"]} steps'),
        ("MFU", f'{s["5_mfu_pct"]:.1f}%'),
        ("MFU if embeddings counted",
         f'{ev["task5_mfu"]["mfu_pct_counting_embeddings"]:.1f}%'),
        ("gradients apart (L2)",
         f'{ev["task3_gradients"]["relative_l2"] * 100:.1f}%'),
        ("clip cosine", f'{ev["extra8_clipping"]["cosine"]:.12f}'),
        ("bytes per weight", str(ev["extra9_memory"]["total_bytes"])),
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
    print("S10 page runtime check -- does the page actually build in a browser?")
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
        print(f"{name:<26}{val:>18}  {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{name} ({val}) not found in the rendered DOM")

    # The no-data fallback must NOT be showing.
    if "No baked data" in dom:
        fails.append("page rendered its no-data fallback: S9DATA was not baked in")

    print()
    print("=" * 74)
    for f in fails:
        print("FAIL  " + f)
    print(f"RESULT: {'FAIL -- %d problem(s)' % len(fails) if fails else 'PASS -- page builds clean'}")
    print("=" * 74)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
