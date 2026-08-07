"""
The evaluation and validation firewall (Session 6 §13).

Evaluation data also needs manifests. The difference is permission: training shards are
admitted into the data stream, test shards are registered in the audit system precisely so
they can be BLOCKED from it. Validation shards sit in between -- readable during training
for evaluation, never gradient-bearing.

Two enforcement points, both real:
  1. admission time -- a shard whose text overlaps the registry is rejected before it can
     become a candidate;
  2. serve time    -- `assert_trainable` is called on every shard id that reaches packing,
     so a never-train shard cannot enter a loss-bearing batch even if the gate is bypassed.

Every read of the registry is logged (§13's access_logs): the registry knowing who looked
at it is part of what makes a later benchmark jump auditable.
"""
import hashlib
import json
import re
from datetime import datetime, timezone

NGRAM = 8  # word-level n-gram for contamination fingerprints; MMLU items are short, so a
           # 13-gram would fingerprint almost nothing. Short docs fall back to a whole-text
           # hash (see _fingerprints).
MIN_NGRAM_HITS = 5   # how many distinct fingerprints must collide before a shard is flagged
# A fingerprint shared by more than this many registry documents is boilerplate, not
# benchmark content (Wikipedia reference/category furniture, "Question:" scaffolding). Left
# in, it flags honest shards for overlap they do not have -- the false-positive side of the
# same trade-off Session 4 makes for cleaning: an over-eager filter destroys good data.
MAX_DOC_FREQUENCY = 3
WORD = re.compile(r'\w+', re.UNICODE)


def _norm(text):
    return " ".join(WORD.findall(text.lower()))


def _h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _fingerprints(text, n=NGRAM):
    w = _norm(text).split()
    if len(w) < n:
        return {_h(" ".join(w))} if w else set()
    return {_h(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


class Firewall:
    def __init__(self):
        self.entries = {}        # doc_id -> registry entry (§13's 6 fields)
        self.fp_index = {}       # fingerprint -> [doc_id]
        self.boilerplate = set()  # fingerprints dropped for exceeding MAX_DOC_FREQUENCY
        self.access_logs = []
        self.blocked = []        # every block event, for run.log + evidence
        self.never_train_ids = set()
        self.non_gradient_ids = set()

    # ---- registration ----
    def register(self, docs, source_meta=None):
        """docs come from corpus/eval_registry_docs.jsonl (kind: test | validation).
        source_meta maps source_id -> {benchmark_id, version_tag} out of corpus/sources.json,
        so the registry carries the real benchmark identity rather than a local alias."""
        source_meta = source_meta or {}
        for d in docs:
            text = "\n".join(s["text"] for s in d["segments"])
            fps = _fingerprints(text)
            never_train = d["kind"] == "test"
            sm = source_meta.get(d["source_id"], {})
            self.entries[d["doc_id"]] = {
                "content_hashes": [_h(_norm(text))],
                "benchmark_ids": [sm.get("benchmark_id") or d["source_id"]],
                "version_tags": [sm.get("version_tag") or d["kind"]],
                "contamination_fingerprints": sorted(fps),
                "access_logs": [],
                "never_train_flag": never_train,
                # operational
                "doc_id": d["doc_id"], "kind": d["kind"], "lane": d["lane"],
                "gradient_bearing": False,
                "language_and_script": d["language_and_script"],
                "n_fingerprints": len(fps),
            }
            for f in fps:
                self.fp_index.setdefault(f, []).append(d["doc_id"])
        self.boilerplate = {f for f, docs in self.fp_index.items()
                            if len(set(docs)) > MAX_DOC_FREQUENCY}
        for f in self.boilerplate:
            del self.fp_index[f]
        return len(self.entries)

    def register_shard(self, shard_id, kind):
        """Eval/validation shards get ids in the same namespace as training shards so the
        ledger check `no never-train shard was consumed` is a set membership test."""
        if kind == "test":
            self.never_train_ids.add(shard_id)
        self.non_gradient_ids.add(shard_id)

    # ---- queries (all logged) ----
    def _log(self, actor, purpose, detail=None):
        self.access_logs.append({
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "actor": actor, "purpose": purpose, "detail": detail})

    def scan(self, text, actor="admission_gate"):
        """Returns the registry entries this text collides with, by fingerprint overlap."""
        self._log(actor, "contamination_scan", {"chars": len(text)})
        hits = {}
        for f in _fingerprints(text):
            for doc_id in self.fp_index.get(f, ()):
                hits.setdefault(doc_id, 0)
                hits[doc_id] += 1
        return hits

    def check_shard(self, shard, texts, min_ngrams=MIN_NGRAM_HITS):
        """Sets contamination_status / eval_overlap_status on a shard manifest.
        min_ngrams guards against a single common n-gram tripping the wire."""
        hits = {}
        for t in texts:
            for doc_id, n in self.scan(t, actor=f"shard:{shard.shard_id}").items():
                hits[doc_id] = hits.get(doc_id, 0) + n
        strong = {k: v for k, v in hits.items() if v >= min_ngrams}
        m = shard.manifest
        if strong:
            worst = max(strong, key=strong.get)
            bench = self.entries[worst]["benchmark_ids"][0]
            m["contamination_status"] = f"overlap:{bench}"
            m["eval_overlap_status"] = f"overlap:{bench}"
            m["contamination_detail"] = {"matched_docs": sorted(strong),
                                         "ngram_hits": {k: strong[k] for k in sorted(strong)},
                                         "ngram_size": NGRAM}
            for doc_id in strong:
                self.entries[doc_id]["access_logs"].append(
                    {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "actor": f"shard:{shard.shard_id}", "purpose": "overlap_detected"})
            return False, m["contamination_status"]
        m["contamination_status"] = "clean"
        m["eval_overlap_status"] = "clean"
        return True, "clean"

    def block(self, shard_id, reason, detail=None):
        ev = {"shard_id": shard_id, "reason": reason, "detail": detail,
              "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        self.blocked.append(ev)
        return ev

    def assert_trainable(self, shard_id):
        """Serve-time enforcement. Raised, not returned: a never-train shard reaching the
        packer is a bug that must stop the run, not a warning."""
        if shard_id in self.never_train_ids:
            self.block(shard_id, "never_train_shard_reached_packer")
            raise PermissionError(f"firewall: {shard_id} is never-train and cannot be packed")
        if shard_id in self.non_gradient_ids:
            self.block(shard_id, "validation_shard_reached_loss_bearing_batch")
            raise PermissionError(f"firewall: {shard_id} is validation-only (not gradient-bearing)")

    # ---- artifact ----
    def registry_manifest(self):
        return {
            "ngram_size": NGRAM,
            "min_ngram_hits_to_flag": MIN_NGRAM_HITS,
            "max_document_frequency": MAX_DOC_FREQUENCY,
            "boilerplate_fingerprints_dropped": len(self.boilerplate),
            "counts": {
                "test_docs": sum(1 for e in self.entries.values() if e["kind"] == "test"),
                "validation_docs": sum(1 for e in self.entries.values() if e["kind"] == "validation"),
                "fingerprints": len(self.fp_index),
                "never_train_shards": len(self.never_train_ids),
                "non_gradient_shards": len(self.non_gradient_ids),
            },
            "never_train_shard_ids": sorted(self.never_train_ids),
            "non_gradient_shard_ids": sorted(self.non_gradient_ids),
            "blocked_events": self.blocked,
            "registry_access_logs": self.access_logs[-200:],
            "access_log_total": len(self.access_logs),
            "entries": {k: self.entries[k] for k in sorted(self.entries)},
        }

    def write(self, path):
        path.write_text(json.dumps(self.registry_manifest(), indent=2, ensure_ascii=False),
                        encoding="utf-8")


def contaminate(text, benchmark_text, where=0.5):
    """Splice a real benchmark item into a training document -- the deliberate attack the
    firewall must catch. Returns the poisoned text."""
    cut = int(len(text) * where)
    return text[:cut] + "\n" + benchmark_text + "\n" + text[cut:]
