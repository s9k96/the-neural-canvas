"""
s6 corpus preparation — the ONLY step that touches the network.

Fetches a small slice of each capability lane from real HuggingFace datasets via the
datasets-server rows API (stdlib urllib; `datasets` is not a dependency), and writes an
immutable, committed snapshot to corpus/ together with corpus/sources.json.

Why a snapshot instead of fetching inside run_demo.py: the grader re-runs the demo and
diffs the artifacts. A run that re-fetches live data cannot produce identical hashes, so
the network is quarantined here and run_demo.py is fully offline and deterministic.

Every fetch records dataset / config / split / offset / length / license / sha256 of the
retrieved text -> that is the Session 3 provenance contract, carried into every shard
manifest downstream.

Run:  .venv/Scripts/python s06-dataset-creation/prepare_corpus.py
"""
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
API = "https://datasets-server.huggingface.co/rows"
MAX_LEN = 100  # rows-API hard cap per request


# ---- source inventory -----------------------------------------------------
# Each entry is one provenance-bearing source. `extract` names the row->doc adapter
# below. Every lane maps to the S5 mixture plan's lanes 1:1.
SOURCES = [
    dict(source_id="fineweb-edu", lane="general_web", dataset="HuggingFaceFW/fineweb-edu",
         config="sample-10BT", split="train", offset=0, rows=100, cap=6000,
         license="ODC-By-1.0", tier="B", lang="eng_Latn", extract="text"),
    dict(source_id="github-code-clean", lane="code", dataset="codeparrot/github-code-clean",
         config="all-all", split="train", offset=0, rows=100, cap=6000,
         license="per-row", tier="B", lang="code", extract="code"),
    dict(source_id="wikipedia-hi", lane="indic", dataset="wikimedia/wikipedia",
         config="20231101.hi", split="train", offset=0, rows=60, cap=6000,
         license="CC-BY-SA-4.0", tier="A", lang="hin_Deva", extract="text"),
    dict(source_id="wikipedia-mr", lane="indic", dataset="wikimedia/wikipedia",
         config="20231101.mr", split="train", offset=0, rows=40, cap=6000,
         license="CC-BY-SA-4.0", tier="A", lang="mar_Deva", extract="text"),
    dict(source_id="wikipedia-te", lane="indic", dataset="wikimedia/wikipedia",
         config="20231101.te", split="train", offset=0, rows=40, cap=6000,
         license="CC-BY-SA-4.0", tier="A", lang="tel_Telu", extract="text"),
    dict(source_id="gsm8k-train", lane="stem_math", dataset="openai/gsm8k",
         config="main", split="train", offset=0, rows=100, cap=4000,
         license="MIT", tier="A", lang="eng_Latn", extract="qa"),
    dict(source_id="openthoughts", lane="reasoning", dataset="open-thoughts/OpenThoughts-114k",
         config="default", split="train", offset=0, rows=40, cap=8000,
         license="Apache-2.0", tier="A", lang="eng_Latn", extract="conversations"),
    dict(source_id="hermes-tools", lane="agentic",
         dataset="NousResearch/hermes-function-calling-v1",
         config="func_calling", split="train", offset=0, rows=40, cap=8000,
         license="Apache-2.0", tier="A", lang="eng_Latn", extract="conversations"),
    dict(source_id="pg19-books", lane="long_context", dataset="emozilla/pg19",
         config="default", split="train", offset=0, rows=2, cap=120000,
         license="Apache-2.0", tier="A", lang="eng_Latn", extract="book"),
]

# Never-train benchmark data (S6 §13) and validation data (read, never gradient-bearing).
EVAL_SOURCES = [
    dict(source_id="mmlu-test", kind="test", benchmark_id="cais/mmlu", version_tag="all/test",
         dataset="cais/mmlu", config="all", split="test", offset=0, rows=60, cap=4000,
         license="MIT", lane="stem_math", lang="eng_Latn", extract="mmlu"),
    dict(source_id="gsm8k-test", kind="test", benchmark_id="openai/gsm8k", version_tag="main/test",
         dataset="openai/gsm8k", config="main", split="test", offset=0, rows=40, cap=4000,
         license="MIT", lane="stem_math", lang="eng_Latn", extract="qa"),
    dict(source_id="val-web", kind="validation", benchmark_id="fineweb-edu/heldout",
         version_tag="sample-10BT/train@5000", dataset="HuggingFaceFW/fineweb-edu",
         config="sample-10BT", split="train", offset=5000, rows=20, cap=6000,
         license="ODC-By-1.0", lane="general_web", lang="eng_Latn", extract="text"),
    dict(source_id="val-indic", kind="validation", benchmark_id="wikipedia-hi/heldout",
         version_tag="20231101.hi/train@5000", dataset="wikimedia/wikipedia",
         config="20231101.hi", split="train", offset=5000, rows=20, cap=6000,
         license="CC-BY-SA-4.0", lane="indic", lang="hin_Deva", extract="text"),
]


# ---- row -> document adapters --------------------------------------------
# A document is always a list of segments: role "target" is loss-bearing, role "context"
# is attention-visible but masked out of the loss. Plain pretraining text is one target
# segment; SFT/agentic data keeps its turn structure so packing can preserve it and the
# loss mask can follow the S5 masking rule (loss on the model's turns only).
def _seg(role, text):
    return {"role": role, "text": text}


def x_text(row, cap):
    t = (row.get("text") or "").strip()[:cap]
    return [_seg("target", t)], {"title": row.get("title"), "url": row.get("url")}


def x_code(row, cap):
    t = (row.get("code") or "").strip()[:cap]
    meta = {"repo": row.get("repo_name"), "path": row.get("path"),
            "language": row.get("language"), "row_license": row.get("license")}
    return [_seg("target", t)], meta


def x_qa(row, cap):
    q = (row.get("question") or "").strip()[:cap]
    a = (row.get("answer") or "").strip()[:cap]
    return [_seg("context", "Question: " + q), _seg("target", "Answer: " + a)], {}


def x_mmlu(row, cap):
    ch = row.get("choices") or []
    if isinstance(ch, str):
        ch = [ch]
    q = (row.get("question") or "").strip()
    body = q + "\n" + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(ch))
    return [_seg("context", body[:cap])], {"subject": row.get("subject"), "answer": row.get("answer")}


def x_conversations(row, cap):
    """OpenThoughts / Hermes multi-turn. Model turns are loss-bearing; user turns,
    system prompts and tool observations are context (S5: applying loss to tool
    observations teaches the model to hallucinate tool results instead of calling tools)."""
    segs = []
    if row.get("system"):
        segs.append(_seg("context", str(row["system"])))
    if row.get("tools"):
        segs.append(_seg("context", "TOOLS: " + str(row["tools"])))
    conv = row.get("conversations") or []
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except Exception:
            conv = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        who = str(turn.get("from") or turn.get("role") or "").lower()
        val = str(turn.get("value") or turn.get("content") or "")
        role = "target" if who in ("gpt", "assistant") else "context"
        segs.append(_seg(role, f"<{who}> {val}"))
    # cap the whole document, keeping whole segments
    out, used = [], 0
    for s in segs:
        if used >= cap:
            break
        s = _seg(s["role"], s["text"][: cap - used])
        used += len(s["text"])
        out.append(s)
    return out, {"n_turns": len(conv)}


def x_book(row, cap):
    """One PG19 book, truncated. Split into consecutive parts so the long-context lane has
    several genuinely-long documents; part index is part of the doc id (real provenance)."""
    t = (row.get("text") or "").strip()[:cap]
    part = max(1, cap // 3)
    parts = [t[i:i + part] for i in range(0, len(t), part) if t[i:i + part].strip()]
    return parts, {"title": row.get("short_book_title"), "url": row.get("url")}


ADAPTERS = {"text": x_text, "code": x_code, "qa": x_qa, "mmlu": x_mmlu,
            "conversations": x_conversations, "book": x_book}


# ---- fetch ---------------------------------------------------------------
def fetch_rows(dataset, config, split, offset, rows, tries=3):
    """rows API, paged at MAX_LEN. Returns list of row dicts."""
    out = []
    while len(out) < rows:
        want = min(MAX_LEN, rows - len(out))
        q = {"dataset": dataset, "split": split, "offset": offset + len(out), "length": want}
        if config:
            q["config"] = config
        url = API + "?" + urllib.parse.urlencode(q)
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    got = [x["row"] for x in json.load(r)["rows"]]
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt == tries - 1:
                    raise RuntimeError(f"fetch failed {dataset}/{config}: {e}") from e
                time.sleep(2 * (attempt + 1))
        if not got:
            break
        out.extend(got)
    return out[:rows]


def build(spec, kind="train"):
    """Fetch one source and turn it into documents + a provenance record."""
    print(f"  {spec['source_id']:18s} {spec['dataset']} {spec.get('config')}/{spec['split']}"
          f" @{spec['offset']} x{spec['rows']} ... ", end="", flush=True)
    rows = fetch_rows(spec["dataset"], spec.get("config"), spec["split"],
                      spec["offset"], spec["rows"])
    adapt = ADAPTERS[spec["extract"]]
    docs, blob = [], []
    for i, row in enumerate(rows):
        segs, meta = adapt(row, spec["cap"])
        if spec["extract"] == "book":  # adapter returned parts, not segments
            for p, text in enumerate(segs):
                if len(text.strip()) < 200:
                    continue
                docs.append(_doc(spec, f"{spec['offset'] + i}p{p}", [_seg("target", text)], meta, kind))
                blob.append(text)
            continue
        segs = [s for s in segs if s["text"].strip()]
        if not segs or sum(len(s["text"]) for s in segs) < 64:
            continue
        docs.append(_doc(spec, str(spec["offset"] + i), segs, meta, kind))
        blob.extend(s["text"] for s in segs)
    text_sha = hashlib.sha256("\n".join(blob).encode("utf-8")).hexdigest()
    rec = dict(source_id=spec["source_id"], kind=kind, lane=spec["lane"],
               dataset=spec["dataset"], config=spec.get("config"), split=spec["split"],
               offset=spec["offset"], length_requested=spec["rows"],
               rows_returned=len(rows), docs_kept=len(docs), char_cap=spec["cap"],
               license=spec["license"], provenance_tier=spec.get("tier"),
               language_and_script=spec["lang"], text_sha256=text_sha,
               retrieved_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               api="huggingface datasets-server /rows")
    for k in ("benchmark_id", "version_tag"):
        if k in spec:
            rec[k] = spec[k]
    print(f"{len(docs)} docs")
    return docs, rec


def _doc(spec, row_key, segs, meta, kind):
    return dict(
        doc_id=f"{spec['source_id']}:{row_key}",
        source_id=spec["source_id"],
        lane=spec["lane"],
        kind=kind,
        language_and_script=spec["lang"],
        license=meta.get("row_license") or spec["license"],
        provenance_tier=spec.get("tier"),
        segments=segs,
        meta={k: v for k, v in meta.items() if v is not None},
    )


def write_jsonl(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    CORPUS.mkdir(parents=True, exist_ok=True)
    sources, by_lane, evals = [], {}, []
    print("training lanes:")
    for spec in SOURCES:
        docs, rec = build(spec, "train")
        sources.append(rec)
        by_lane.setdefault(spec["lane"], []).extend(docs)
    print("eval / validation:")
    for spec in EVAL_SOURCES:
        docs, rec = build(spec, spec["kind"])
        sources.append(rec)
        evals.extend(docs)

    for lane, docs in sorted(by_lane.items()):
        write_jsonl(CORPUS / f"{lane}.jsonl", docs)
    write_jsonl(CORPUS / "eval_registry_docs.jsonl", evals)
    (CORPUS / "sources.json").write_text(
        json.dumps({"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "sources": sources}, indent=2, ensure_ascii=False), encoding="utf-8")

    tot = sum(len(v) for v in by_lane.values())
    print(f"\ncorpus/ written: {tot} training docs across {len(by_lane)} lanes, "
          f"{len(evals)} eval/validation docs, {len(sources)} provenance records")
    for lane, docs in sorted(by_lane.items()):
        chars = sum(len(s["text"]) for d in docs for s in d["segments"])
        print(f"  {lane:14s} {len(docs):4d} docs  {chars:>9,d} chars")


if __name__ == "__main__":
    sys.exit(main())
