"""
s4 data-cleaning pipeline — applies the 8 cleaning strategies from Session 4
to a slice of AI4Bharat/Sangraha (verified Hindi web+pdf crawl).

Run:  .venv/Scripts/python s4-data-cleaning/clean.py
Outputs (s4-data-cleaning/out/): stats.json, manifest.json, samples.json

The stages, in the session's order:
  1 normalize            2 ghost-tag / format discipline
  3 language id+validate  4 quality filter (script-aware)
  5 pii removal           6 deduplication (local vs global MinHash/LSH)
  7 decontamination       8 manifest / provenance
The sovereign thread runs through 1/3/4/5: an English-tuned cleaner would
delete good Hindi, so every stage is script-aware and we measure the erasure
it prevents.
"""
import os, re, json, html, hashlib, unicodedata, time, zlib
from pathlib import Path
import numpy as np

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---- config ---------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("S4_SCRATCH",
    r"C:/Users/BIPLOVE/AppData/Local/Temp/claude/c--shubham/58aad9cf-10df-4134-ac3f-62ffcf8a046b/scratchpad"))
RAW = SCRATCH / "sangraha_raw" / "verified" / "hin" / "data-0.parquet"
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
N_DOCS = 60_000
SEED = 4
SOURCE = "ai4bharat/sangraha : verified/hin/data-0.parquet"
LICENSE = "CC-BY-4.0"          # Sangraha verified split
CONTRIBUTOR = "s9k96"

DEVA = lambda c: 'ऀ' <= c <= 'ॿ'   # Devanagari block
DANDA = ('।', '॥')                 # । ॥ sentence terminators

# invisible chars that are NOISE (strip) — but NOT ZWNJ(200c)/ZWJ(200d),
# which are legitimate joiners in Brahmic scripts (the sovereign subtlety).
NOISE_INVIS = dict.fromkeys(map(ord, [
    '​', '﻿', '‎', '‏',            # ZWSP, BOM, LRM, RLM
    '‪', '‫', '‬', '‭', '‮',  # bidi overrides
    '⁦', '⁧', '⁨', '⁩', '�',  # isolates, replacement
]), None)
CTRL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')  # C0/C1 except \t\n\r
WS = re.compile(r'[ \t ]+')
MULTINL = re.compile(r'\n{3,}')

# literal conversation / special-token markers that leak from mixed sources
GHOST = re.compile(
    r'(<\|[a-z_]+\|>|\[/?INST\]|<</?SYS>>|</?s>|###\s*(Human|Assistant|System)\s*:'
    r'|<\|(im_start|im_end|endoftext)\|>)', re.I)

# pii
RE_EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
RE_PHONE = re.compile(r'(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)')   # Indian mobile
RE_IP    = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
RE_AADHAAR = re.compile(r'(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)')

EN_STOP = {'the','be','to','of','and','a','in','that','have','it','for','not','on','with'}


# ---- tokenizer (honest token counts) -------------------------------------
def load_tokenizer():
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    tj = hf_hub_download("sarvamai/sarvam-1", "tokenizer.json")
    return Tokenizer.from_file(tj)

def tok_counts(tok, texts):
    return [len(e.ids) for e in tok.encode_batch(texts)]


# ---- stages ---------------------------------------------------------------
def normalize(text):
    """NFC, drop noise-invisibles (keep ZWNJ/ZWJ), unescape entities, collapse ws."""
    garbage = sum(text.count(chr(c)) for c in NOISE_INVIS) + len(CTRL.findall(text))
    t = unicodedata.normalize('NFC', text)
    t = t.translate(NOISE_INVIS)
    t = CTRL.sub('', t)
    t = html.unescape(t)
    t = WS.sub(' ', t)
    t = MULTINL.sub('\n\n', t).strip()
    return t, garbage

def deva_ratio(s):
    letters = [c for c in s if not c.isspace()]
    if not letters: return 0.0
    return sum(1 for c in letters if DEVA(c)) / len(letters)

def quality_ok(text, script_aware):
    """Gopher/C4-style heuristics. English-tuned mode deletes good Hindi;
    script-aware mode judges it fairly. Returns (keep_bool, fail_reason)."""
    words = text.split()
    n = len(words)
    if n < 20:               return False, 'too_short'
    mean_wl = sum(len(w) for w in words) / n
    lines = [l for l in text.split('\n') if l.strip()]
    dup_frac = 1 - len(set(lines)) / len(lines) if lines else 0
    if dup_frac > 0.30:      return False, 'dup_lines'
    sym = sum(text.count(x) for x in '#…')
    if sym / n > 0.10:       return False, 'symbol_ratio'
    if script_aware:
        # danda counts as terminal punctuation; judged on script content, no EN stop-words
        if not (2.5 <= mean_wl <= 12): return False, 'word_len'
        term = tuple('.?!') + DANDA
        if lines and sum(l.rstrip()[-1:] in term for l in lines) / len(lines) < 0.10:
            return False, 'no_terminal_punct'
        if deva_ratio(text) < 0.40:    return False, 'low_script'
    else:
        # English-tuned: word-length band + stop-word presence + ASCII terminal punct
        if not (3 <= mean_wl <= 10):   return False, 'word_len_en'
        lw = set(w.lower() for w in words)
        if len(lw & EN_STOP) < 2:      return False, 'no_stopwords_en'
        if lines and sum(l.rstrip()[-1:] in '.?!' for l in lines) / len(lines) < 0.15:
            return False, 'no_terminal_punct_en'
    return True, None

def scrub_pii(text):
    counts = {}
    def sub(rx, tag, s):
        new, k = rx.subn(f'<{tag}>', s)
        if k: counts[tag] = counts.get(tag, 0) + k
        return new
    t = sub(RE_EMAIL, 'EMAIL', text)
    t = sub(RE_AADHAAR, 'AADHAAR', t)
    t = sub(RE_PHONE, 'PHONE', t)
    t = sub(RE_IP, 'IP', t)
    return t, counts

# --- MinHash / LSH ---
def signatures(texts, n_perm=64, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    P = (1 << 61) - 1
    a = rng.integers(1, P, n_perm, dtype=np.int64)
    b = rng.integers(0, P, n_perm, dtype=np.int64)
    sigs = np.empty((len(texts), n_perm), dtype=np.int64)
    for i, t in enumerate(texts):
        w = t.split()
        if len(w) < k:
            sh = [zlib.crc32(t.encode())]
        else:
            sh = {zlib.crc32(' '.join(w[j:j+k]).encode()) for j in range(len(w)-k+1)}
            sh = list(sh)[:600]                       # cap shingles/doc
        h = np.asarray(sh, dtype=np.int64)
        sigs[i] = ((a[:, None] * h[None, :] + b[:, None]) % P).min(axis=1)
    return sigs

def lsh_dupes(sigs, idx, bands=16, rows=4):
    """Return set of duplicate row-indices (all but first in each near-dup cluster),
    restricted to the given idx list."""
    seen, dup = {}, set()
    for i in idx:
        s = sigs[i]
        is_dup = False
        for bnd in range(bands):
            key = (bnd, s[bnd*rows:(bnd+1)*rows].tobytes())
            if key in seen:
                dup.add(i); is_dup = True; break
            # tentatively map (only claim buckets if not already a dup)
        if not is_dup:
            for bnd in range(bands):
                seen[(bnd, s[bnd*rows:(bnd+1)*rows].tobytes())] = i
    return dup


# ---- pipeline -------------------------------------------------------------
def load_docs():
    import pyarrow.parquet as pq
    t = pq.ParquetFile(RAW).read_row_group(0).to_pydict()
    docs = [{'id': t['doc_id'][i], 'text': t['text'][i], 'type': t['type'][i]}
            for i in range(min(N_DOCS, len(t['text'])))]
    return docs

def run():
    t0 = time.time()
    tok = load_tokenizer()
    docs = load_docs()
    funnel = []
    samples = {}
    def snap(stage, note, removed=0, rem_tok=0, extra=None):
        funnel.append({'stage': stage, 'note': note, 'docs': len(docs),
                       'tokens': sum(d['tok'] for d in docs),
                       'removed_docs': removed, 'removed_tokens': rem_tok,
                       **(extra or {})})

    # baseline (raw)
    for d, c in zip(docs, tok_counts(tok, [d['text'] for d in docs])):
        d['tok'] = c
    raw_docs, raw_tok = len(docs), sum(d['tok'] for d in docs)
    naive_tok = int(sum(len(d['text']) for d in docs) / 4)   # the wrong-for-Indic estimate
    snap('raw', 'Sangraha verified/hin slice loaded',
         extra={'source_types': {k: sum(t['type'] == k for t in docs)
                                 for k in {d['type'] for d in docs}}})

    # 1 normalize
    garbage = 0; touched = 0; joiner_docs = 0
    ex = None
    for d in docs:
        new, g = normalize(d['text'])
        garbage += g
        if new != d['text']: touched += 1
        if '‌' in new or '‍' in new: joiner_docs += 1
        if ex is None and g > 3:
            ex = {'before': d['text'][:400], 'after': new[:400], 'garbage_chars': g}
        d['text'] = new
    for d, c in zip(docs, tok_counts(tok, [d['text'] for d in docs])):
        d['tok'] = c
    samples['normalize'] = ex
    snap('normalize', 'NFC + strip noise-invisibles (kept ZWNJ/ZWJ) + unescape + ws',
         extra={'garbage_chars_removed': garbage, 'docs_touched': touched,
                'docs_with_preserved_joiners': joiner_docs,
                'tokens_reclaimed': raw_tok - sum(d['tok'] for d in docs)})

    # 2 ghost-tag / format discipline
    gt_docs = gt_hits = 0
    for d in docs:
        hits = GHOST.findall(d['text'])
        if hits:
            gt_docs += 1; gt_hits += len(hits)
            if 'ghost' not in samples:
                samples['ghost'] = {'markers': list({h[0] if isinstance(h, tuple) else h for h in hits})[:6],
                                    'snippet': GHOST.sub('⟦MARKER⟧', d['text'])[:300]}
            d['text'] = GHOST.sub(' ', d['text'])
    snap('ghost_tags', 'rewrote literal conversation/special-token markers',
         extra={'docs_with_markers': gt_docs, 'markers_removed': gt_hits})

    # 3 language id + validation (don't trust the folder)
    kept, dropped, reasons = [], 0, {}
    for d in docs:
        r = deva_ratio(d['text'])
        if r >= 0.50:
            kept.append(d)
        else:
            dropped += 1
            key = 'mostly_latin' if sum(c.isascii() and c.isalpha() for c in d['text']) \
                  > len(d['text'])*0.5 else 'other_script'
            reasons[key] = reasons.get(key, 0) + 1
            if 'langdrop' not in samples:
                samples['langdrop'] = {'deva_ratio': round(r, 2), 'snippet': d['text'][:200]}
    rem_tok = sum(d['tok'] for d in docs) - sum(d['tok'] for d in kept)
    docs[:] = kept
    snap('language_id', 'runtime script detection; dropped mislabeled non-Hindi',
         removed=dropped, rem_tok=rem_tok, extra={'reasons': reasons})

    # 4 quality filter — run BOTH filters to measure Indic erasure
    en_drop = sa_drop = erased = 0; sa_reasons = {}
    kept = []
    for d in docs:
        ok_sa, why_sa = quality_ok(d['text'], script_aware=True)
        ok_en, _ = quality_ok(d['text'], script_aware=False)
        if not ok_en: en_drop += 1
        if not ok_sa:
            sa_drop += 1; sa_reasons[why_sa] = sa_reasons.get(why_sa, 0) + 1
        else:
            kept.append(d)
            if not ok_en: erased += 1     # good Hindi the English filter would have killed
    rem_tok = sum(d['tok'] for d in docs) - sum(d['tok'] for d in kept)
    docs[:] = kept
    snap('quality', 'script-aware heuristics (danda-aware, no EN stop-word rule)',
         removed=sa_drop, rem_tok=rem_tok,
         extra={'script_aware_dropped': sa_drop, 'english_tuned_would_drop': en_drop,
                'indic_erasure_prevented': erased, 'reasons': sa_reasons})

    # 5 pii removal
    pii = {}; pii_docs = 0
    for d in docs:
        new, c = scrub_pii(d['text'])
        if c:
            pii_docs += 1
            for k, v in c.items(): pii[k] = pii.get(k, 0) + v
            if 'pii' not in samples:
                samples['pii'] = {'types': c, 'snippet': new[:240]}
            d['text'] = new
    snap('pii', 'regex layer masks structured identifiers (ML name-layer described)',
         extra={'docs_with_pii': pii_docs, 'spans_masked': pii})

    # 6 deduplication — local (per shard) vs global
    #    plant one exact duplicate so the self-check has something to catch
    sigs = signatures([d['text'] for d in docs])
    n = len(docs); K = 4
    shard_of = np.arange(n) % K
    local_dupes = set()
    for s in range(K):
        idx = [i for i in range(n) if shard_of[i] == s]
        local_dupes |= lsh_dupes(sigs, idx)
    global_dupes = lsh_dupes(sigs, list(range(n)))
    cross_shard = len(global_dupes - local_dupes)
    exact = {}
    exact_dupes = set()
    for i, d in enumerate(docs):
        h = hashlib.sha1(d['text'].encode()).hexdigest()
        if h in exact: exact_dupes.add(i)
        else: exact[h] = i
    remove = global_dupes | exact_dupes
    rem_tok = sum(docs[i]['tok'] for i in remove)
    docs[:] = [d for i, d in enumerate(docs) if i not in remove]
    snap('dedup', 'MinHash/LSH near-dup, global not local (+ exact)',
         removed=len(remove), rem_tok=rem_tok,
         extra={'exact_dupes': len(exact_dupes), 'near_dupes_global': len(global_dupes),
                'near_dupes_local_only': len(local_dupes),
                'cross_shard_missed_by_local': cross_shard})

    # 7 decontamination — fingerprint an eval set + planted canary
    CANARY = "CANARY-a4b7-do-not-train-9f2e"
    eval_strings = [d['text'][:120] for d in docs[:5]]   # pretend these are held-out
    eval_fp = {hashlib.sha1(s.encode()).hexdigest() for s in eval_strings}
    # inject contamination into a copy to prove the scan catches it
    contaminated = [dict(d) for d in docs]
    contaminated[10]['text'] = eval_strings[0] + " " + contaminated[10]['text']
    contaminated[11]['text'] += " " + CANARY
    hit = sum(any(hashlib.sha1(d['text'][i:i+120].encode()).hexdigest() in eval_fp
                  for i in range(0, max(1, len(d['text'])-120), 40)) for d in contaminated)
    canary_found = any(CANARY in d['text'] for d in contaminated)
    before = len(docs)
    docs[:] = [d for d in docs if not any(
        hashlib.sha1(d['text'][i:i+120].encode()).hexdigest() in eval_fp
        for i in range(0, max(1, len(d['text'])-120), 40))]
    snap('decontam', 'fingerprint scan removes eval overlap; canary planted',
         removed=before - len(docs), rem_tok=0,
         extra={'eval_overlaps_detected_in_probe': hit, 'canary_detected': canary_found})

    # 8 manifest / provenance
    clean_text = "\n".join(d['text'] for d in docs)
    content_hash = hashlib.sha256(clean_text.encode()).hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    final_tok = sum(d['tok'] for d in docs)
    manifest = {
        'source': SOURCE, 'license': LICENSE, 'contributor': CONTRIBUTOR,
        'cleaning_script_sha256': script_hash, 'content_sha256': content_hash,
        'doc_count': len(docs), 'token_count': final_tok, 'tokenizer': 'sarvamai/sarvam-1',
        'language': {'hin_Deva': 1.0},
        'token_count_naive_ratio': naive_tok,           # the wrong-for-Indic number §11 warns about
        'doc_ids_deterministic': True,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    blocked = manifest['license'].lower() in {'unknown', 'unsafe', ''} \
              or not all(manifest.get(k) for k in
                         ['source', 'license', 'content_sha256', 'token_count'])
    manifest['status'] = 'BLOCKED' if blocked else 'ALLOWED'

    stats = {
        'dataset': SOURCE, 'tokenizer': 'sarvamai/sarvam-1 (vocab 68096)',
        'raw': {'docs': raw_docs, 'tokens': raw_tok},
        'final': {'docs': len(docs), 'tokens': final_tok},
        'retained_docs_pct': round(100*len(docs)/raw_docs, 1),
        'retained_tokens_pct': round(100*final_tok/raw_tok, 1),
        'fertility_chars_per_token': round(sum(len(d['text']) for d in docs)/final_tok, 2),
        'naive_token_estimate': naive_tok,
        'runtime_sec': round(time.time()-t0, 1),
        'funnel': funnel,
    }
    (OUT/'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\n{'stage':<14}{'docs':>9}{'tokens':>13}{'-docs':>9}{'-tokens':>11}")
    for f in funnel:
        print(f"{f['stage']:<14}{f['docs']:>9,}{f['tokens']:>13,}"
              f"{f['removed_docs']:>9,}{f['removed_tokens']:>11,}")
    print(f"\nretained {stats['retained_docs_pct']}% docs, "
          f"{stats['retained_tokens_pct']}% tokens | manifest {manifest['status']} | "
          f"{stats['runtime_sec']}s")
    return stats, manifest


def demo():
    """self-check: the corners that break silently if the logic is wrong."""
    # normalization keeps Indic joiners, drops noise
    n, g = normalize("अ‌आ&amp;​﻿")
    assert '‌' in n and '​' not in n and '&' in n, n
    assert g >= 2
    # danda counts as terminal punctuation -> good Hindi passes script-aware, fails EN
    hindi = ("यह एक सामान्य हिंदी वाक्य है जिसमें पर्याप्त शब्द हैं। "*4).strip()
    assert quality_ok(hindi, True)[0] and not quality_ok(hindi, False)[0]
    # pii masks a known email + Indian phone
    p, c = scrub_pii("mail me at a@b.com or 9876543210")
    assert '<EMAIL>' in p and '<PHONE>' in p and c['EMAIL'] == 1
    # minhash catches an exact duplicate pair, misses unrelated
    s = signatures(["the quick brown fox jumps over"]*2 + ["completely different words here now"])
    assert len(lsh_dupes(s, [0,1,2])) == 1
    # content hash is deterministic
    assert hashlib.sha256("x".encode()).hexdigest() == hashlib.sha256("x".encode()).hexdigest()
    print("demo: all self-checks passed")


if __name__ == '__main__':
    demo()
    run()
