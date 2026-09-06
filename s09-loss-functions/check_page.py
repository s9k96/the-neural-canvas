"""Render loss-harness.html in a real browser and assert it actually built.

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
PAGE = HERE / "loss-harness.html"
EVIDENCE = HERE / "out" / "evidence.json"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# Everything below is produced by JS at runtime, so a non-zero count proves the
# corresponding render function ran. Matching happens against the DOM with every <script>
# block REMOVED -- the render functions build these strings out of template literals that
# live in the script source, so an unstripped dump counts each one twice and the check
# silently passes on a page that rendered nothing.
EXPECT = [
    ("stat tiles (all panels)", r'<div class="tile"><div class="k">',      16, 16),
    ("switch toggles",         r'<div class="sw[^"]*" data-sw=',            3,  3),
    ("switch readout cells",   r'<div class="k">reported loss</div>',       1,  1),
    ("run chart bars",         r'<rect x="150" y="[\d.]+" width=',          4,  4),
    ("four-decimal cells",     r'<td class="n"[^>]*>[+-]?\d+\.\d{4}</td>',  8, 99),
    ("alignment pairs",        r'<div class="apair',                       20, 26),
    ("context readout",        r'<div class="k">logits \[1, T, V\]</div>',  1,  1),
    ("chunk readout",          r'<div class="k">chunked peak logits</div>', 1,  1),
    ("mtp 2 + gap 1 + stab 5", r'stroke-width="2" stroke-linejoin="round"', 8,  8),
    ("mtp crosshair target",   r'id="mtpHit"',                              1,  1),
    ("gap chart hit target",   r'id="gapHit"',                              1,  1),
    ("gate rows",              r'<div class="gate (pass|fail)">',          16, 16),
    ("data tables built",      r'<thead><tr><th',                          12, 12),
    # sections added so the page stands alone without the class notes
    ("spine logit sliders",    r'<input type="range" id="lg\d"',            5,  5),
    ("spine truth chips",      r'<button class="chip[^"]*" data-t="\d"',    5,  5),
    ("stability series labels", r'>(plain|centering|soft-cap c=30)</text>', 3,  3),
    ("loss-map rows",          r'✓ measured here',                          4,  4),
    ("bits/byte language rows", r'<td class="n"[^>]*>\d\.\d{3}</td>',       6, 99),
]

SCRIPT = re.compile(r"<script\b.*?</script>", re.S | re.I)

# Values that must reach the screen, read from evidence.json rather than hardcoded, so
# this check cannot drift from the run that produced the page.
def expected_strings() -> list[tuple[str, str]]:
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    s, two = ev["summary"]["seven_numbers"], ev["summary"]["two_losses"]
    return [
        ("tokenizer sha prefix", ev["tokenizer"]["sha256"][:8]),
        ("untrained perplexity", f'{round(s["5_untrained_perplexity"]):,}'),
        ("head 1 final loss", f'{two["head1_t_plus_1"]:.3f}'),
        ("head 2 final loss", f'{two["head2_t_plus_2"]:.3f}'),
        ("memory ratio", str(s["7_memory_ratio_measured"])),
        ("ln(V) reference", f'{ev["exp5_perplexity"]["ln_V"]:.4f}'),
        ("acceptance rate", f'{ev["part2_acceptance"]["acceptance_rate"] * 100:.1f}%'),
        ("centering final log Z", f'{ev["exp8_stability"]["final"]["centering"]["logZ"]:.4f}'),
        ("SFT completion tokens", f'{ev["exp11_sft_mask"]["tokens_completion_only"]:,}'),
        ("Devanagari bytes/char",
         f'{[r for r in ev["exp10_bits_per_byte"]["languages"] if r["bytes_per_char"] > 1.5][0]["bytes_per_char"]:.2f}'),
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
    print("S9 page runtime check -- does the page actually build in a browser?")
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
