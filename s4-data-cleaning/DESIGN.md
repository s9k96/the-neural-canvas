# S4 — Data Cleaning: Design Document

**What this is:** the plan, the reasoning, and the exact steps behind
[clean.py](clean.py) and the [widget](data-cleaning.html). Read this to understand
*what* we do, *how* we do it, and *why* — without reading the code.

---

## 1. The assignment (what we were asked)

From the Session 4 notes:

1. Find **how many cleaning strategies** the session lists, and describe them.
2. Pick a **10–100M dataset**, ideally one seen in Session 3.
3. **Apply those cleanups** to it.
4. Call out **any other strategy or concern** we cleaned.
5. Ship a **widget** covering: the strategies, the dataset, what/why/how we cleaned,
   the extra concern, and **final statistics**. Deploy to Netlify.

## 2. The one-line thesis

> **Raw data is not training data.** A web crawl becomes a corpus only after it is
> normalized, filtered, deduplicated, decontaminated, and stamped with provenance —
> **and every one of those steps must be script-aware, or it silently deletes the
> Indic text it was supposed to keep.**

That second clause is the "other concern" (§7 below). It is the through-line of the
whole session and the reason a generic English cleaner is the wrong tool.

## 3. How many strategies — the count

The session's own recap (§14) names the pipeline stages it builds. We count **8**:

| #   | Strategy                       | Counts as…                                                                  |
| --- | ------------------------------ | --------------------------------------------------------------------------- |
| 1   | Normalization                  | one                                                                         |
| 2   | Format discipline (ghost-tags) | one                                                                         |
| 3   | Quality filtering              | one                                                                         |
| 4   | Deduplication                  | one (taught in two parts: the MinHash/LSH *mechanism*, then *global* scale) |
| 5   | Language ID & validation       | one                                                                         |
| 6   | PII removal                    | one                                                                         |
| 7   | Decontamination                | one                                                                         |
| 8   | Manifest / provenance          | one                                                                         |

**Extraction** (HTML → text) is a 9th stage but the session explicitly inherits it
from Session 3 ("treated as known"), so it is not one of the 8 we build here.

## 4. The dataset and why

**AI4Bharat / Sangraha**, split `verified/hin` (verified-human Hindi web + PDF crawl).

- **Why this one:** Session 3 held up Sangraha as the honest measure of real Indic
  coverage, and the session repeatedly points back to *"our previous Indic crawl had
  no deduplication at any level."* So we clean the exact kind of asset the course said
  was never cleaned. It also keeps the India-first thread running from S2 → S3 → S4.
- **Size:** one shard = 174,763 docs ≈ 107M tokens. We process the first **60,000 docs
  (~37M tokens)** — inside the 10–100M window, small enough for a pure-Python MinHash
  pass to finish.
- **Shape:** columns `doc_id, text, type`; `type ∈ {web, pdf}`; ~95% Devanagari.

### Token counting — the deliberate choice

We count tokens with the **sarvam-1** Indic tokenizer (68k vocab), **not** a naive
`chars/4` ratio. Session 3/4 flag a real V4 bug: token counts *"estimated with a ratio
that is wrong for Indic by several times"* corrupt every downstream budget. Sangraha
Hindi has a fertility of ~**3.5 chars/token** under a real Indic tokenizer — the widget
shows the naive number beside the real one to make the gap concrete.

## 5. The pipeline — what / how / why, stage by stage

Stages run in a fixed order; each depends on the one before. **The invariant:** the
content hash is computed *last*, on the cleaned text, so provenance reflects what we
actually keep.

### 1 · Normalization  *(Indic-sensitive)*
- **What:** put every doc in one canonical form.
- **How:** Unicode NFC → strip invisible control chars → unescape HTML entities
  (`&amp;` → `&`) → collapse whitespace.
- **Why:** a byte-level tokenizer spends vocabulary on zero-width spaces and broken
  byte fragments. V4 shipped **46 garbage tokens** for exactly this reason.
- **Indic fix:** we **keep** the zero-width joiner (ZWJ) and non-joiner (ZWNJ) — they
  are real letters in Brahmic scripts — while stripping true noise (ZWSP, BOM, bidi
  overrides, replacement char). A cleaner that strips *all* invisibles mangles Hindi.

### 2 · Format discipline (ghost-tags)
- **What:** stop fake conversation structure leaking into pretraining.
- **How:** regex-detect literal markers from mixed sources (`<|user|>`, `[INST]`,
  `### Human:`, `<s>`…) and rewrite/remove them.
- **Why:** if pretraining learns `[INST]` as ordinary subwords, it later fights the
  tokenizer's *real* special tokens during fine-tuning. This was a priority-0 V4 bug
  (4 sources, 4 formats).

### 3 · Language ID & validation  *(Indic-sensitive)*
- **What:** confirm each doc is actually Hindi.
- **How:** compute the Devanagari character ratio at runtime; drop docs below 50%.
- **Why:** web crawls are mislabeled often enough that trusting the folder pollutes
  per-language pools and skews the fertility numbers we size budgets with. (This is
  where V4's silent Telugu language-code bug lived.)

### 4 · Quality filtering  *(Indic-sensitive — see §7)*
- **What:** decide if a doc is worth keeping.
- **How:** heuristic rules — min length, mean word length, symbol-to-word ratio,
  duplicate-line fraction, and terminal-punctuation rate. (A trained educational-value
  classifier is the second layer in production; we run the heuristic layer.)
- **Why:** a smaller filtered corpus beats a larger raw one at equal compute.
- **Indic fix:** thresholds are **script-aware** — the danda `।` counts as sentence-
  ending punctuation, and we **drop** the English stop-word rule that would fail good
  Hindi. We run the English-tuned filter too, only to *measure* how much Hindi it
  would wrongly delete.

### 5 · PII removal  *(Indic-sensitive)*
- **What:** remove personal data (for people, and for the corpus's legal usability).
- **How:** a regex layer masks structured identifiers — emails, Indian phone numbers,
  IPs, Aadhaar-shaped numbers.
- **Why / Indic fix:** the ML name-layer (described, not run here) trades precision vs
  recall, and the trade is sharper for Indic names where a common name is also a place.

### 6 · Deduplication  →  local vs global
- **What:** remove repeated learning signal.
- **How:** exact-hash dedup, then near-dup via **shingles → MinHash → LSH banding**.
  We split the slice into 4 shards, dedup each *locally*, then run one *global* pass.
- **Why:** near-duplicates (same article, different header) are most of the real
  duplication and exact-match misses them. And **duplication is global** — two shards
  each look clean yet share documents neither pass ever saw. The widget reports the
  **cross-shard duplicates local passes miss**, which is why a real run needs one
  large-memory machine holding the whole index.

### 7 · Decontamination
- **What:** keep the firewall between training data and benchmarks.
- **How:** fingerprint the eval sets, scan every shard for overlap, remove any doc
  carrying a test item, and plant a **canary string** to detect future leaks.
- **Why:** once a test item trains the model, its score is no longer real. We
  demonstrate this by injecting a contaminated shard and watching the scan catch it.

### 8 · Manifest / provenance
- **What:** make the corpus reproducible and auditable.
- **How:** deterministic IDs from content (not a counter) + a per-shard manifest:
  source, license, contributor, cleaning-script hash, content hash, token count,
  language breakdown. A missing/unsafe license stamps the shard **BLOCKED**.
- **Why:** same input → same output → same hash. The manifest is the datasheet we'd
  publish and the gate a shard must pass to enter the corpus. It catches the V4
  defects — copy-pasted file sizes, IDs that changed every run, wrong Indic token
  counts.

## 6. The "other concern" we cleaned up

The headline extra concern is **Indic erasure**: filters and normalizers tuned on
English quietly delete good low-resource text. We make it measurable — running both
the English-tuned and script-aware quality filters over the *same* Hindi documents and
reporting how many good docs the English filter would have thrown away. The sovereign
fix lives in stages 1, 3, 4, and 5.

## 7. Outputs (produced by `clean.py`, in `out/`)

| File            | Contents                                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| `stats.json`    | the full funnel: docs/tokens surviving each stage, per-stage removals and reasons, final retained %      |
| `manifest.json` | the provenance record for the cleaned shard (with ALLOWED/BLOCKED status)                                |
| `samples.json`  | before/after examples (normalization, a dropped non-Hindi doc, masked PII, ghost markers) for the widget |

## 8. The widget (`data-cleaning.html`)

Neural Canvas style, self-contained, data-bound to the JSON above. It shows: the
**8 strategies** (Indic-sensitive ones flagged), the **dataset** and the naive-vs-real
token gap, an **interactive stage funnel** (what/why/how + before/after per stage), the
**Indic-erasure** comparison, **dedup local-vs-global**, the **provenance manifest**,
and **final statistics**. Numbers are rendered from the pipeline's real output, so the
prose can't drift from the data.

## 9. Reproducibility & honesty

- Run: `.venv/Scripts/python s4-data-cleaning/clean.py`. Deterministic seed; same input
  → same content hash → same numbers.
- **Run for real** on the text: normalization, language ID, quality filtering, PII
  regex, dedup, manifest. **Demonstrated** (mechanism shown, not production-scale):
  the ghost-tag rewrite and decontamination on web text where such cases are rare, and
  the ML name-layer of PII, which is described rather than run. The widget labels which
  is which.

## 10. Performance (why the run takes ~19 min)

By design there are **no heavy dependencies** — MinHash/LSH and the Devanagari-ratio
checks are hand-written Python looping over every character of every document. That is
honest and portable but slow: ~1130s on the 60k slice.

It is left as-is on purpose. The manifest commits to
`cleaning_script_sha256 = sha256(clean.py)`, and the widget embeds that hash, so any
edit to `clean.py` — even a comment — would break the "same script → same hash" claim
until the pipeline is re-run and the numbers re-injected. The results are already in and
self-consistent, so we don't churn them for a one-time speedup.

**If a fast re-run is wanted**, the two hot loops vectorize cleanly: replace the
per-char `deva_ratio` with a compiled regex count (`re.compile(r'[ऀ-ॿ]')`)
memoized once per doc so `language_id` and `quality` share it, and batch the MinHash
signature math in NumPy across docs instead of one doc at a time. Expect roughly a
3–5× speedup. Doing so regenerates the script hash, so re-run `clean.py` and re-inject
`out/*.json` into the widget afterward.
