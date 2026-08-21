"""Render attention-timeline.html in a real browser and assert it actually built.

This exists because of a bug that shipped past every other check. `hueOf` was declared
with `const` AFTER the code that used it, so the cards block threw a temporal-dead-zone
ReferenceError at runtime. The page still parsed cleanly, its tags still balanced, and
every date still verified -- but zero of the thirty cards rendered, and because the
sources section runs last it never executed either. A syntax check cannot catch that.
Only running the page can.

    python s8-model-architectures/check_page.py

Exits non-zero on any console error, or if the DOM does not contain what it should.
Requires Google Chrome. Pairs with verify_sources.py: that one checks the facts are
right, this one checks they actually reach the screen.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "attention-timeline.html"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# The page must contain one of each of these once rendered. Counts are "+1" wherever the
# string also appears literally inside the template in the <script> source.
EXPECT = [
    ("mechanism cards",        r"<article class=\"mcard\"",                   31, 31),
    ("bonus/origin rows",      r"<tr><td class=\"n\" style=\"text-align:left\">", 43, 43),
    ("finding call-outs",      r"<div class=\"verdict\"><strong>\d+\.",                 8,  8),
    ("pattern canvases",       r"<canvas",                                    10, 10),
    ("viz slots filled",       r"<div class=\"viz-art\" id=\"v-[\w-]+\"><",   30, 30),
    ("timeline dots",          r"class=\"tl-dot",                             33, 99),
    ("trap call-outs",         r"class=\"verdict\"><strong>",                  4, 99),
]
# Spot-check that specific generated ids exist -- these cannot appear unless the
# template loop actually ran, since the source only contains ${m.id}.
IDS = ["c-rope", "v-rope", "c-qwen36-hybrid", "c-standard-attention", "v-deepseek-v4-csa"]


def main() -> int:
    if not CHROME.exists():
        print(f"FATAL: Chrome not found at {CHROME}")
        return 2
    if not PAGE.exists():
        print(f"FATAL: {PAGE} missing")
        return 2

    proc = subprocess.run(
        [str(CHROME), "--headless=new", "--disable-gpu", "--virtual-time-budget=9000",
         "--enable-logging=stderr", "--v=0", "--dump-dom", PAGE.as_uri()],
        capture_output=True, text=True, timeout=180)
    dom, log = proc.stdout, proc.stderr

    print("=" * 74)
    print("S8 page runtime check -- does the page actually build in a browser?")
    print("=" * 74)

    fails = []

    # 1. console errors. GPU/mailbox chatter from headless is noise, not our problem.
    console = [l for l in log.splitlines() if ":CONSOLE:" in l]
    real = [l for l in console if re.search(r"error|uncaught|exception", l, re.I)]
    print(f"console messages: {len(console)}   errors: {len(real)}")
    for l in real:
        msg = re.search(r'"(.*?)", source', l)
        fails.append("console error: " + (msg.group(1) if msg else l.strip()))

    # 2. the DOM contains what a built page contains
    print()
    print(f"{'what':<26}{'found':>7}  expected")
    print("-" * 74)
    for name, pat, lo, hi in EXPECT:
        n = len(re.findall(pat, dom))
        ok = lo <= n <= hi
        print(f"{name:<26}{n:>7}  {lo if lo == hi else f'{lo}-{hi}':<8} {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{name}: found {n}, expected {lo if lo == hi else f'{lo}-{hi}'}")

    # 3. the chronology table specifically -- counted inside #srcTable, because the
    #    bonus tables in section 04 use the same row markup and would otherwise inflate it.
    tbl = re.search(r'id="srcTable".*?</table>', dom, re.S)
    n = len(re.findall(r"<tr><td", tbl.group(0))) if tbl else 0
    print()
    print(f"chronology rows inside #srcTable: {n}  expected 30 {'ok' if n == 30 else 'FAIL'}")
    if n != 30:
        fails.append(f"chronology table has {n} rows, expected 30")

    # 4. generated ids that prove the template loop ran
    print()
    missing = [i for i in IDS if f'id="{i}"' not in dom]
    print(f"generated ids present: {len(IDS) - len(missing)}/{len(IDS)}")
    for i in missing:
        fails.append(f"generated id missing from DOM: {i} (the render loop did not run)")

    print()
    print("=" * 74)
    for f in fails:
        print("FAIL  " + f)
    if fails:
        print(f"RESULT: FAIL -- {len(fails)} problem(s). The page does not build.")
    else:
        print("RESULT: PASS -- page builds clean, no console errors, all content present.")
    print("=" * 74)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
