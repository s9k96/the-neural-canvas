"""Render the-redundancy-tax.html in a real browser and assert it actually built.

Same reason S8, S9 and S10 have one: a page can parse cleanly, balance its tags and still
render nothing, because parsing a script does not run it. Every panel here is built by
JavaScript from the baked `S12DATA` blob, so if that blob is missing or a render function
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
PAGE = HERE / "the-redundancy-tax.html"
EVIDENCE = HERE / "out" / "evidence.json"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# Everything below is produced by JS at runtime, so a non-zero count proves the corresponding
# render function ran. Matching happens against the DOM with every <script> block REMOVED --
# the render functions build these strings out of template literals that live in the script
# source, so an unstripped dump counts each one twice and the check silently passes on a page
# that rendered nothing.
EXPECT = [
    ("hero tiles",            r'<div class="tile"><div class="k">',           5,  5),
    ("tables built",          r'<tbody><tr><th',                             12, 15),
    ("readout blocks",        r'<div class="k">[^<]+</div>\s*<div class="v"', 28, 70),
    # Only chart series: the sidebar's nav icons also carry stroke attributes, so match the
    # pair that only a series path has.
    ("chart series paths",    r'fill="none" stroke="#[0-9a-f]{6}" stroke-width=', 6, 10),
    ("memory-wall dots",      r'<circle cx="[\d.]+" cy="[\d.]+" r="[\d.]+"', 12, 80),
    ("reference marks",       r'stroke-dasharray="',                          2, 12),
    ("byte-ledger squares",   r'<b class="(w|g|m|gone)"',                     30, 60),
    ("ring cells",            r'<div class="cell[^"]*">',                     16, 80),
    ("stage buttons",         r'<button class="btn[^"]*" data-m="',            4,  4),
    ("gate rows",             r'<div class="gate (pass|fail)">',             45, 55),
    ("hardware buttons",      r'<button class="btn[^"]*" data-v="',            6,  6),
    ("open-question blocks",  r'<div class="qa">',                             4,  4),
    ("global-batch factors",  r'<div class="f[^"]*"><div class="fv">',         4,  4),
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
    st, pr = ev["stages"], ev["precision"]
    n_gates = len(ev["gates"])
    passed = sum(1 for v in ev["gates"].values() if v)
    return [
        ("tokenizer sha prefix", ev["corpus"]["tokenizer_sha256"][:8]),
        ("dp bytes per weight", f'{st["dp"]["ladder_bpp"]:.4f}'),
        ("zero1 bytes per weight", f'{st["zero1"]["ladder_bpp"]:.4f}'),
        ("zero2 bytes per weight", f'{st["zero2"]["ladder_bpp"]:.4f}'),
        ("zero3 bytes per weight", f'{st["zero3"]["ladder_bpp"]:.4f}'),
        ("dp wire cost", f'{st["dp"]["measured_P"]:.3f}P'),
        ("zero3 wire cost", f'{st["zero3"]["measured_P"]:.3f}P'),
        ("bf16 ring sum", f'{ev["collectives"]["bf16_ring_sum"]:.0f}'),
        ("mxfp8 bytes", f'{pr["mxfp8_bytes"]:.4f}'),
        ("gate tally", f'{passed}/{n_gates}'),
        ("replica hash", ev["data_parallel"]["replica_hash"][:16]),
        ("bucketed message count", f'{ev["bucketing"]["dp_flat"]["msgs"]:.0f}'),
        ("H100 comm ratio", f'{ev["hardware"]["cards"]["64 x H100"]["ratio"] * 100:.0f}%'),
        ("NVLink 2P", f'{ev["hardware"]["links"]["NVLink, inside one node"]["t_2p"]:.2f}s'),
        ("PCIe bandwidth", f'{ev["offload"]["pcie_gbs"]:.0f}'),
        ("world size", str(ev["meta"]["main_world"])),
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
    print("S12 page runtime check -- does the page actually build in a browser?")
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
        fails.append("page rendered its no-data fallback: S12DATA was not baked in")

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
