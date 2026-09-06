"""
Tokenized shards, manifests, the admission gate and the manifest validation pass.
(Session 6 §6, and the Session 4 admission contract it depends on.)

A shard is an immutable training object. Its content hash identifies it; the tokenizer hash
gives its token ids meaning; the cleaning pipeline hash explains how raw text became
admitted data. Modifying a shard produces a NEW shard with a new hash and a parent link --
never an edit in place. The shard id embeds its content hash so that property is visible.

Cleaning is deliberately not uniform: the ghost-tag strip runs on prose lanes only. Applying
it to code or agentic trajectories would damage legitimate `</s>`-like sequences and
tool-call markers -- the Session 4 lesson that an English-prose-tuned cleaner destroys
structured data. Regexes are lifted from s04-data-cleaning/clean.py so the lineage is
continuous; s6 keeps its own copy because it ships as a standalone submission.
"""
import hashlib
import html
import inspect
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

# ---- frozen tokenizer ----------------------------------------------------
TOKENIZER_REPO = "sarvamai/sarvam-1"
TOKENIZER_FILE = "tokenizer.json"
# Frozen at submission time. run_demo verifies the file on disk against this constant;
# a mismatch means the token ids in every shard would silently change meaning.
FROZEN_TOKENIZER_SHA256 = "bb5115a36ddb956a4ee0fd534e9870dd69157835622aec9c53062896f883c072"
TOKENIZER_VERSION = "sarvam-1/tokenizer.json@bb5115a3"

UNK, BOS, EOS, PAD = 0, 1, 2, 3  # <unk> <s> </s> <<reserved_token_0>> (used as PAD)
MODEL_VOCAB = 4096               # frequency-truncated softmax; see build_vocab_remap

PIPELINE_VERSION = "s6-clean-1"
PROSE_LANES = {"general_web", "indic", "long_context"}

# Permissive licenses admitted for training. github-code-clean carries a per-row license,
# so real gpl-2.0 rows are genuinely rejected by the gate -- not a staged rejection.
ALLOWED_LICENSES = {
    "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense",
    "cc-by-4.0", "cc-by-sa-4.0", "cc0-1.0", "odc-by-1.0", "public-domain",
}

DOCS_PER_SHARD = 20

# ---- cleaning (s4 lineage) ----------------------------------------------
NOISE_INVIS = dict.fromkeys(map(ord, [
    '​', '﻿', '‎', '‏',
    '‪', '‫', '‬', '‭', '‮',
    '⁦', '⁧', '⁨', '⁩', '�',
]), None)  # NB: ZWNJ/ZWJ (200c/200d) are legitimate Brahmic joiners -- never stripped
CTRL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
WS = re.compile(r'[ \t ]+')
MULTINL = re.compile(r'\n{3,}')
GHOST = re.compile(r'(<\|[a-z_]+\|>|\[/?INST\]|<</?SYS>>|###\s*(Human|Assistant|System)\s*:)', re.I)

RE_EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
RE_PHONE = re.compile(r'(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)')
RE_IP = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
RE_AADHAAR = re.compile(r'(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)')


def normalize(text, ghost_strip):
    """NFC, drop noise-invisibles (keep ZWNJ/ZWJ), unescape entities, collapse whitespace."""
    t = unicodedata.normalize('NFC', text)
    t = t.translate(NOISE_INVIS)
    t = CTRL.sub('', t)
    t = html.unescape(t)
    if ghost_strip:
        t = GHOST.sub(' ', t)
    t = WS.sub(' ', t)
    return MULTINL.sub('\n\n', t).strip()


def scrub_pii(text):
    """Replace PII with typed placeholders (removal, not deletion of the sentence)."""
    n = 0
    for rx, tag in ((RE_EMAIL, '[EMAIL]'), (RE_AADHAAR, '[ID]'), (RE_PHONE, '[PHONE]'),
                    (RE_IP, '[IP]')):
        text, k = rx.subn(tag, text)
        n += k
    return text, n


def pipeline_hash():
    """Identity of the cleaning code itself -- what `cleaning_pipeline_hash` must mean."""
    src = PIPELINE_VERSION + inspect.getsource(normalize) + inspect.getsource(scrub_pii)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def load_tokenizer():
    """Reuses the s4 loader pattern (s04-data-cleaning/clean.py:67-71)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(TOKENIZER_REPO, TOKENIZER_FILE)
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return Tokenizer.from_file(path), path, sha


def verify_tokenizer(sha):
    return sha == FROZEN_TOKENIZER_SHA256


# ---- shard construction -------------------------------------------------
def clean_doc(doc):
    """Clean every segment; returns (segments, stats). Roles are preserved."""
    ghost = doc["lane"] in PROSE_LANES
    segs, pii = [], 0
    for s in doc["segments"]:
        t = normalize(s["text"], ghost)
        t, k = scrub_pii(t)
        pii += k
        if t:
            segs.append({"role": s["role"], "text": t})
    return segs, {"pii_removed": pii, "ghost_strip": ghost}


def tokenize_doc(tok, doc):
    """Token ids per segment. No implicit specials: EOS is appended explicitly at the
    document boundary so `boundary_or_eos_flag` in the token trace means something."""
    segs, stats = clean_doc(doc)
    if not segs:
        return None
    out = []
    for s in segs:
        ids = tok.encode(s["text"], add_special_tokens=False).ids
        if ids:
            out.append({"role": s["role"], "ids": ids})
    if not out:
        return None
    out[-1]["ids"] = out[-1]["ids"] + [EOS]
    return {"doc": doc, "segments": out, "clean_stats": stats}


class Shard:
    """Immutable tokenized shard + its §6 manifest."""

    def __init__(self, lane, source_id, index, samples, tokenizer_sha, extra=None):
        self.lane, self.source_id, self.index = lane, source_id, index
        flat, segs, docs = [], [], []
        for si, s in enumerate(samples):
            start_doc, target = len(flat), 0
            for seg in s["segments"]:
                a = len(flat)
                flat.extend(seg["ids"])
                is_target = 1 if seg["role"] == "target" else 0
                segs.append((si, a, len(flat), is_target))
                target += (len(flat) - a) * is_target
            d = s["doc"]
            docs.append(dict(doc_id=d["doc_id"], sample_index=si,
                             token_start=start_doc, token_end=len(flat),
                             language_and_script=d["language_and_script"],
                             license=d["license"], provenance_tier=d.get("provenance_tier"),
                             n_target_tokens=target))
        self.tokens = np.asarray(flat, dtype=np.int32)
        self.segs = np.asarray(segs, dtype=np.int32).reshape(-1, 4)
        self.doc_records = docs
        self.content_hash = self._content_hash()
        self.shard_id = f"{lane}.{source_id}.{index:02d}.{self.content_hash[:12]}"
        self.manifest = self._manifest(tokenizer_sha, extra or {})

    def _content_hash(self):
        h = hashlib.sha256()
        h.update(self.tokens.tobytes())
        h.update(self.segs.tobytes())
        h.update(json.dumps([d["doc_id"] for d in self.doc_records], sort_keys=True).encode())
        return h.hexdigest()

    def _manifest(self, tokenizer_sha, extra):
        langs = sorted({d["language_and_script"] for d in self.doc_records})
        lic = sorted({(d["license"] or "unknown").lower() for d in self.doc_records})
        tiers = sorted({d.get("provenance_tier") or "?" for d in self.doc_records})
        target = int(self.segs[self.segs[:, 3] == 1][:, 2].sum() - self.segs[self.segs[:, 3] == 1][:, 1].sum())
        m = {                                                    # §6's 14 fields, in order
            "shard_id": self.shard_id,
            "source_ids": [self.source_id],
            "doc_ids": [d["doc_id"] for d in self.doc_records],
            "tokenizer_hash": tokenizer_sha,
            "token_count": int(self.tokens.size),
            "language_and_script": langs,
            "capability_lane": self.lane,
            "license_and_provenance_tier": {"licenses": lic, "tiers": tiers},
            "cleaning_pipeline_hash": pipeline_hash(),
            "dedup_status": "pending",
            "contamination_status": "pending",
            "eval_overlap_status": "pending",
            "content_hash": self.content_hash,
            "parent_shard_ids": [],
            # operational extras the rest of the system needs
            "tokenizer_version": TOKENIZER_VERSION,
            "loss_bearing_token_count": target,
            "n_samples": int(self.segs[:, 0].max()) + 1 if self.segs.size else 0,
            "docs": self.doc_records,
            "reserved_for_anneal": False,
            "never_train": False,
            "gradient_bearing": True,
            "admission": {"verdict": "pending", "reasons": []},
        }
        m.update(extra)
        return m

    def write(self, shard_dir):
        np.savez(Path(shard_dir) / f"{self.shard_id}.npz", tokens=self.tokens, segs=self.segs)


def build_shards(tok, docs, tokenizer_sha, docs_per_shard=DOCS_PER_SHARD):
    """Group cleaned documents into shards by (lane, source), in deterministic doc order."""
    groups = {}
    for d in docs:
        s = tokenize_doc(tok, d)
        if s:
            groups.setdefault((d["lane"], d["source_id"]), []).append(s)
    shards = []
    for (lane, src), samples in sorted(groups.items()):
        samples.sort(key=lambda s: s["doc"]["doc_id"])
        for i in range(0, len(samples), docs_per_shard):
            shards.append(Shard(lane, src, i // docs_per_shard,
                                samples[i:i + docs_per_shard], tokenizer_sha))
    return shards


def derive_shard(parent, samples, tokenizer_sha, note):
    """A modified shard is a new shard with a new hash and a parent link (§6)."""
    s = Shard(parent.lane, parent.source_id, 90, samples, tokenizer_sha)
    s.manifest["parent_shard_ids"] = [parent.shard_id]
    s.manifest["derivation_note"] = note
    return s


# ---- admission gate -----------------------------------------------------
def admit(manifest):
    """The Session 4 admission contract, executable. Returns (verdict, reasons)."""
    r = []
    if manifest["tokenizer_hash"] != FROZEN_TOKENIZER_SHA256:
        r.append("tokenizer_hash_mismatch")
    if not manifest.get("cleaning_pipeline_hash"):
        r.append("unknown_cleaning_lineage")
    bad = [l for l in manifest["license_and_provenance_tier"]["licenses"]
           if l not in ALLOWED_LICENSES and l != "per-row"]
    if bad:
        r.append("unsafe_license:" + ",".join(bad))
    if manifest["eval_overlap_status"] not in ("clean", "pending"):
        r.append("eval_overlap:" + manifest["eval_overlap_status"])
    if manifest["contamination_status"] not in ("clean", "pending"):
        r.append("contaminated:" + manifest["contamination_status"])
    if manifest.get("never_train"):
        r.append("never_train_flag")
    if manifest["dedup_status"] == "duplicate":
        r.append("duplicate")
    if manifest["token_count"] == 0:
        r.append("empty_shard")
    verdict = "ADMITTED" if not r else "REJECTED"
    manifest["admission"] = {"verdict": verdict, "reasons": r}
    return verdict, r


def mark_dedup(shards):
    """Exact-duplicate detection over shard content hashes and per-doc text hashes.
    # ponytail: exact-dup only. s4 already has the MinHash/LSH near-dup pass; wire it in
    # here if near-duplicates ever matter for this corpus."""
    seen, dupes = {}, 0
    for s in sorted(shards, key=lambda s: s.shard_id):
        if s.content_hash in seen:
            s.manifest["dedup_status"] = "duplicate"
            s.manifest["duplicate_of"] = seen[s.content_hash]
            dupes += 1
        else:
            seen[s.content_hash] = s.shard_id
            s.manifest["dedup_status"] = "unique"
    return dupes


# ---- validation pass ----------------------------------------------------
def validate_manifests(manifest_dir, shard_dir):
    """Re-read every shard from disk and recompute what its manifest claims.
    This is the `manifests validated` event: the manifest is only worth what a
    recomputation from the bytes confirms."""
    rows = []
    for mp in sorted(Path(manifest_dir).glob("*.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        p = Path(shard_dir) / f"{m['shard_id']}.npz"
        checks = {}
        if not p.exists():
            checks["payload_present"] = False
        else:
            z = np.load(p)
            tokens, segs = z["tokens"], z["segs"]
            h = hashlib.sha256()
            h.update(tokens.tobytes())
            h.update(segs.tobytes())
            h.update(json.dumps(m["doc_ids"], sort_keys=True).encode())
            checks["payload_present"] = True
            checks["content_hash_matches"] = h.hexdigest() == m["content_hash"]
            checks["token_count_matches"] = int(tokens.size) == m["token_count"]
            checks["shard_id_embeds_hash"] = m["shard_id"].endswith(m["content_hash"][:12])
            checks["tokenizer_hash_frozen"] = m["tokenizer_hash"] == FROZEN_TOKENIZER_SHA256
            checks["has_all_14_fields"] = all(k in m for k in (
                "shard_id", "source_ids", "doc_ids", "tokenizer_hash", "token_count",
                "language_and_script", "capability_lane", "license_and_provenance_tier",
                "cleaning_pipeline_hash", "dedup_status", "contamination_status",
                "eval_overlap_status", "content_hash", "parent_shard_ids"))
        rows.append({"shard_id": m["shard_id"], "result": "PASS" if all(checks.values()) else "FAIL",
                     "checks": checks})
    return rows


# ---- shard store (read path + perf accounting) --------------------------
class ShardStore:
    """Reads shard payloads with a real in-process cache: cache_hit_rate and
    shard_read_latency in performance.json are measured here, not estimated."""

    def __init__(self, shard_dir):
        self.dir = Path(shard_dir)
        self._cache = {}
        self.hits = self.misses = 0
        self.read_seconds = 0.0

    def get(self, shard_id):
        if shard_id in self._cache:
            self.hits += 1
            return self._cache[shard_id]
        self.misses += 1
        t0 = time.perf_counter()
        z = np.load(self.dir / f"{shard_id}.npz")
        payload = (z["tokens"], z["segs"])
        self.read_seconds += time.perf_counter() - t0
        self._cache[shard_id] = payload
        return payload

    def stats(self):
        n = self.hits + self.misses
        return {"cache_hit_rate": round(self.hits / n, 4) if n else 0.0,
                "shard_reads": self.misses, "cache_hits": self.hits,
                "shard_read_latency_ms_total": round(self.read_seconds * 1000, 3),
                "shard_read_latency_ms_mean": round(self.read_seconds * 1000 / self.misses, 4)
                if self.misses else 0.0}


# ---- vocab remap --------------------------------------------------------
def build_vocab_remap(shards, size=MODEL_VOCAB):
    """Shards keep true sarvam-1 ids (68k) -- that is what manifests and audits reference.
    The numpy model softmaxes over the top-(size-1) corpus-frequent ids plus one UNK
    bucket, because a 68k softmax over a batch is ~1 GB in float64. Losses and
    perplexities are therefore scoped to this remapped vocab; the table's sha256 travels
    in the tokenizer manifest so the scoping is stated, not hidden."""
    counts = np.zeros(70000, dtype=np.int64)
    for s in shards:
        np.add.at(counts, s.tokens, 1)
    keep = [UNK, BOS, EOS, PAD]
    for tid in np.argsort(-counts):
        if len(keep) >= size:
            break
        if int(tid) not in keep and counts[tid] > 0:
            keep.append(int(tid))
    table = np.zeros(70000, dtype=np.int32)  # everything unseen -> model id 0 (UNK bucket)
    for model_id, tid in enumerate(keep):
        table[tid] = model_id
    covered = int(counts[np.asarray(keep)].sum())
    total = int(counts.sum())
    sha = hashlib.sha256(table.tobytes()).hexdigest()
    return table, {"model_vocab": size, "remap_sha256": sha,
                   "true_vocab": 68096, "coverage": round(covered / total, 6) if total else 0.0,
                   "unk_bucket_model_id": 0, "pad_model_id": int(table[PAD]),
                   "eos_model_id": int(table[EOS])}
