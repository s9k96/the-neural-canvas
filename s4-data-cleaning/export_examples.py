"""Pick ~12 diverse REAL docs from the raw slice and record clean.py's exact
per-stage output for each, so data-cleaning.html can (a) seed its live-JS
playground with real text and (b) self-check the JS ports against the truth.

Run: .venv/Scripts/python s4-data-cleaning/export_examples.py
Writes: out/examples.json  (list of {id,type,raw, expect:{...}})
"""
import json, re
from pathlib import Path
import clean

HERE = Path(__file__).resolve().parent
CAP = 600                      # trim raw text so the HTML stays small; compute expects on the TRIMMED text

def expect_for(raw):
    norm, garbage = clean.normalize(raw)
    ghosted = clean.GHOST.sub(' ', norm)
    ok_sa, why_sa = clean.quality_ok(ghosted, True)
    ok_en, why_en = clean.quality_ok(ghosted, False)
    masked, pii = clean.scrub_pii(ghosted)
    return {
        'normalize': norm, 'garbage': garbage,
        # full-match string per marker (group 0 == whole match; JS .match() counts these)
        'ghost_hits': [g[0] if isinstance(g, tuple) else g for g in clean.GHOST.findall(norm)],
        'deva_ratio': round(clean.deva_ratio(ghosted), 3),
        'quality_sa': [ok_sa, why_sa], 'quality_en': [ok_en, why_en],
        'pii_masked': masked, 'pii_counts': pii,
    }

def main():
    import pyarrow.parquet as pq
    d = next(pq.ParquetFile(clean.RAW).iter_batches(batch_size=12000)).to_pydict()
    n = len(d['text'])
    types = d.get('type') or ['web'] * n                         # unverified crawl omits 'type'
    ids = d.get('doc_id') or [f'{clean.TIER}-hin-{i}' for i in range(n)]
    docs = [{'id': ids[i], 'type': types[i], 'text': d['text'][i]} for i in range(n)]

    buckets = {'erasure': [], 'pii': [], 'langdrop': [], 'qualdrop': [], 'pdf': []}
    for doc in docs:
        raw = doc['text'][:CAP].strip()
        if len(raw) < 60:
            continue
        e = expect_for(raw)
        item = {'id': doc['id'], 'type': doc['type'], 'raw': raw, 'expect': e}
        keep_sa, keep_en = e['quality_sa'][0], e['quality_en'][0]
        if e['pii_counts'] and len(buckets['pii']) < 2:
            buckets['pii'].append(item)
        elif e['deva_ratio'] < 0.5 and len(buckets['langdrop']) < 2:
            buckets['langdrop'].append(item)
        elif not keep_sa and len(buckets['qualdrop']) < 2:
            buckets['qualdrop'].append(item)
        elif doc['type'] == 'pdf' and keep_sa and len(buckets['pdf']) < 2:
            buckets['pdf'].append(item)
        elif keep_sa and not keep_en and len(buckets['erasure']) < 5:
            buckets['erasure'].append(item)

    # one clearly-labeled synthetic doc so the playground exercises the stages that
    # are rare-to-absent in verified Hindi (ghost markers: 0 in corpus; PII: ~0.3%).
    synth = ("नमस्ते &amp; स्वागत है। <|user|> कृपया संपर्क करें: a@b.com या "
             "9876543210, IP 192.168.0.1, आधार 1234 5678 9012। [INST] ### Human: "
             "यह एक सामान्य हिंदी वाक्य है जिसमें पर्याप्त शब्द मौजूद हैं और यह "
             "स्क्रिप्ट-अवेयर फ़िल्टर को आसानी से पास कर जाता है। यहाँ एक शून्य-चौड़ाई "
             "स्थान​ और एक BOM﻿ भी छिपा है।</s>")
    picked = [x for b in buckets.values() for x in b]
    picked.insert(0, {'id': 'synthetic-messy', 'type': 'synthetic',
                      'raw': synth, 'expect': expect_for(synth)})
    out = clean.OUT / 'examples.json'
    out.write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding='utf-8')
    # same-origin JS so the HTML playground can load real seed docs without fetch/CORS;
    # keyed by tier so verified + unverified example sets merge into one window.EXAMPLES map
    (clean.OUT / 'examples.js').write_text(
        'window.EXAMPLES = Object.assign(window.EXAMPLES || {}, '
        + json.dumps({clean.TIER: picked}, ensure_ascii=False) + ');\n', encoding='utf-8')
    print('wrote', len(picked), 'examples ->', out, '(+ examples.js)')
    for k, v in buckets.items():
        print(f'  {k}: {len(v)}')
    # emit the exact regex source so the JS ports can be built faithfully
    print('\nWS pattern     :', repr(clean.WS.pattern))
    print('MULTINL pattern:', repr(clean.MULTINL.pattern))
    print('CTRL pattern   :', repr(clean.CTRL.pattern))
    print('NOISE_INVIS cps:', [hex(c) for c in clean.NOISE_INVIS])
    print('GHOST pattern  :', clean.GHOST.pattern)
    print('EN_STOP        :', sorted(clean.EN_STOP))

if __name__ == '__main__':
    main()
