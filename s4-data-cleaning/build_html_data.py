# -*- coding: utf-8 -*-
"""Inject the per-tier pipeline outputs into data-cleaning.html as one DATASETS blob.
Run after clean.py + export_examples.py for each tier. Verified is required;
unverified is included if out/unverified/ exists.

Run: .venv/Scripts/python s4-data-cleaning/build_html_data.py
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = HERE / 'data-cleaning.html'

def blob(outdir):
    if not (outdir / 'stats.json').exists():
        return None
    d = json.loads((outdir / 'stats.json').read_text(encoding='utf-8'))
    d['manifest'] = json.loads((outdir / 'manifest.json').read_text(encoding='utf-8'))
    d['samples'] = json.loads((outdir / 'samples.json').read_text(encoding='utf-8'))
    return d

datasets = {}
v = blob(HERE / 'out')
if v: datasets['verified'] = v
u = blob(HERE / 'out' / 'unverified')
if u: datasets['unverified'] = u
assert 'verified' in datasets, 'verified outputs missing — run clean.py first'

line = ('        const DATASETS = ' + json.dumps(datasets, ensure_ascii=False)
        + "; let TIER = 'verified'; let DATA = DATASETS[TIER];")
s = HTML.read_text(encoding='utf-8')
s2 = re.sub(r'^\s*const (?:DATA|DATASETS) = .*$', lambda _: line, s, count=1, flags=re.M)
assert s2 != s, 'no const DATA/DATASETS line found to replace'
HTML.write_text(s2, encoding='utf-8')
print('injected tiers:', list(datasets))
