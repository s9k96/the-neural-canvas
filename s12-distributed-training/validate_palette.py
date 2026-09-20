"""Python twin of the dataviz skill's validate_palette.js — same constants, same math.

Exists because this machine has no node, and the palette checks are meant to be computed
rather than eyeballed. Ported faithfully: OKLab/OKLCH conversions, Machado-Oliveira-Fernandes
(2009) CVD transforms at severity 1.0, and the same five thresholds.

    python validate_palette.py "#3987e5,#d95926" --mode dark --surface "#0a0a0c"

Exit code 1 on any hard FAIL (lightness band, chroma floor, normal-vision floor).
WARN bands (CVD 6-8, contrast under 3:1) report but do not fail.
"""
import argparse
import math
import re
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}      # OKLCH L
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0                          # OKLab dE x100, min(protan, deutan)
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}

WS = "[ \t\n\v\f\r   -     　]+"
HEX = re.compile(r"^#?[0-9a-fA-F]{6}$")


def strip_ws(v):
    return re.sub(f"^{WS}|{WS}$", "", v)


def split_colors(raw):
    return [c for c in (strip_ws(x) for x in (raw or "").split(",")) if c]


def hex2srgb(h):
    h = strip_ws(h).lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted((rel_lum(a), rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="dark", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    for c in list(palette) + [surface]:
        if not HEX.match(c):
            raise SystemExit(f"not a hex color: {c!r}")
    lo, hi = BAND[mode]
    rows, failed = [], False

    print(f"\nmode={mode}  surface={surface}  pairs={pairs}\n")
    print(f"{'#':<3}{'hex':<10}{'OKLCH L':>9}{'C':>8}{'contrast':>10}   checks")
    print("-" * 74)
    for i, c in enumerate(palette):
        L, C = oklch(c)
        ct = contrast(c, surface)
        notes = []
        if not (lo <= L <= hi):
            notes.append(f"FAIL band L={L:.3f} not in [{lo},{hi}]")
            failed = True
        if C < CHROMA_FLOOR:
            notes.append(f"FAIL chroma C={C:.3f} < {CHROMA_FLOOR}")
            failed = True
        if ct < CONTRAST_MIN:
            notes.append(f"WARN contrast {ct:.2f}:1 < {CONTRAST_MIN}")
        print(f"{i:<3}{c:<10}{L:>9.3f}{C:>8.3f}{ct:>9.2f}:1   {'; '.join(notes) or 'ok'}")
        rows.append((c, L, C, ct))

    idx = ([(i, i + 1) for i in range(len(palette) - 1)] if pairs == "adjacent"
           else [(i, j) for i in range(len(palette)) for j in range(i + 1, len(palette))])
    print(f"\n{'pair':<16}{'normal':>9}{'protan':>9}{'deutan':>9}{'tritan':>9}   checks")
    print("-" * 74)
    worst_normal = math.inf
    for i, j in idx:
        a, b = palette[i], palette[j]
        n = delta_e(a, b)
        p, d, t = (delta_e(a, b, k) for k in ("protan", "deutan", "tritan"))
        worst_normal = min(worst_normal, n)
        notes = []
        cvd = min(p, d)
        if cvd < CVD_FLOOR:
            notes.append(f"FAIL cvd {cvd:.1f} < {CVD_FLOOR}")
            failed = True
        elif cvd < CVD_TARGET:
            notes.append(f"WARN cvd {cvd:.1f} in [{CVD_FLOOR},{CVD_TARGET}) — needs secondary encoding")
        if n < NORMAL_FLOOR:
            notes.append(f"FAIL normal {n:.1f} < {NORMAL_FLOOR}")
            failed = True
        print(f"{f'{i}-{j}':<16}{n:>9.1f}{p:>9.1f}{d:>9.1f}{t:>9.1f}   {'; '.join(notes) or 'ok'}")

    print(f"\nworst normal-vision dE on the active pairlist: {worst_normal:.1f} (floor {NORMAL_FLOOR})")
    print("RESULT:", "FAIL" if failed else "PASS")
    return not failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("palette")
    ap.add_argument("--mode", default="dark", choices=["light", "dark"])
    ap.add_argument("--surface")
    ap.add_argument("--pairs", default="adjacent", choices=["adjacent", "all"])
    a = ap.parse_args()
    sys.exit(0 if validate(split_colors(a.palette), a.mode, a.surface, a.pairs) else 1)
