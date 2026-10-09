"""Render one-becomes-eight.html in a real browser and assert it actually built.

Same reason S8-S13 have one: a page can parse cleanly and still render nothing, because parsing
a script does not run it. Every panel here is built by JavaScript from the baked `S14DATA` blob,
so if that blob is missing or a render function throws, the page is a set of empty boxes and no
static check would notice.

    python check_page.py

Exits non-zero on any console error, or if the DOM does not contain what a built page contains.
Requires Google Chrome; the page itself needs nothing.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "one-becomes-eight.html"
EVIDENCE = HERE / "out" / "evidence.json"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# Everything below is produced by JS at runtime, so a non-zero count proves the corresponding
# render function ran. Matched against the DOM with every <script> block REMOVED: the template
# literals in the script source would otherwise count each pattern a second time.
EXPECT = [
    ("hero tiles",          r'<div class="tile"><div class="k">',                       5,   5),
    ("tables built",        r'<tbody><tr><th',                                          12,  18),
    ("readout blocks",      r'<div class="k">[^<]+</div><div class="v"',                30,  50),
    # Only chart series: the sidebar's nav icons also carry stroke attributes.
    ("chart series paths",  r'fill="none" stroke="#[0-9a-f]{6}" stroke-width=',        12,  20),
    ("chart dots",          r'<circle cx="[\d.]+" cy="[\d.]+" r="[\d.]+"',             150, 900),
    ("heatmap cells",       r'<rect x="[\d.]+" y="[\d.]+" width="[\d.]+" height="[\d.]+" fill="rgba\(57,135,229', 80, 600),
    ("router bars",         r'<rect x="[\d.]+" y="[\d.]+" width="[\d.]+" height="[\d.]+" rx="3"', 8, 8),
    ("buttons",             r'<button class="btn[^"]*" data-v="',                       28,  34),
    ("capacity slider",     r'<input type="range"',                                      1,   1),
    ("gate rows",           r'<div class="gate (pass|fail)">',                          14,  20),
    ("layout fit marks",    r'<span class="gate (pass|fail)">',                          3,   3),
]

SCRIPT = re.compile(r"<script\b.*?</script>", re.S | re.I)


def expected_strings() -> list[tuple[str, str]]:
    """Values that must reach the screen, read from evidence.json rather than hardcoded, so this
    check cannot drift from the run that produced the page."""
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    b, cv, f = ev["branches"], ev["conversion"], ev["findings"]
    n_gates, passed = len(ev["gates"]), sum(1 for v in ev["gates"].values() if v)
    main = b["moe-bias"]
    return [
        ("gate tally", f"{passed}/{n_gates}"),
        ("dense val at conversion", f'{cv["dense_val"]:.6f}'),
        ("converted val", f'{cv["moe_val"]:.6f}'),
        ("MoE final val", f'{main["val"][-1][1]:.4f}'),
        ("dense-continued final val", f'{b["dense-continued"]["val"][-1][1]:.4f}'),
        ("throughput ratio", f'{f["moe_throughput_ratio"]:.2f}×'),
        ("notes figures", f'{sum(c["ok"] for c in ev["notes_check"])}/{len(ev["notes_check"])}'),
        ("reference total", f'{ev["reference"]["total"] / 1e9:.2f}B'),
        ("clone dead, hard-none", f'{f["clone_dead_end"]["clone-hard-none"]} / {ev["model"]["layers"] * 32}'),
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
    dom = SCRIPT.sub("", raw)

    print("=" * 74)
    print("S14 page runtime check -- does the page actually build in a browser?")
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
        fails.append("page rendered its no-data fallback: S14DATA was not baked in")

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
