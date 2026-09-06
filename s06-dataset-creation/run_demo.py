"""
S6 -- one command that runs the complete demonstration.

    .venv/Scripts/python s06-dataset-creation/run_demo.py

Fully offline and deterministic: it reads the committed corpus/ snapshot and regenerates
submission_artifacts/ from scratch. Exit code 0 means every evidence row passed.

The 13 events the brief requires appear in run.log in this order:
  shards created, manifests validated, evaluation data blocked, mixture compiled,
  batches packed, OPUS decisions recorded, checkpoint saved, crash simulated, run resumed,
  historical stream replayed, branch forked, audit completed, performance measured.
"""
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tds import evidence, firewall as fw, mixture, packing, shards, train  # noqa: E402
from tds.ledger import (SCHEMA_VERSION, ConsumptionLedger, LearningLedger,  # noqa: E402
                        TokenTrace, stream_continuity)

CORPUS = HERE / "corpus"
ART = HERE / "submission_artifacts"
SHARD_DIR = HERE / "shards"
RUN_ID = "v5-s6-demo"
MAIN = "main"


class Log:
    """run.log and stdout. `event` marks the brief's mandated sequence; `check` writes the
    [PASS]/[FAIL] lines."""

    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")
        self.events, self.checks = [], []
        self.t0 = time.perf_counter()

    def _w(self, line):
        stamp = f"{time.perf_counter() - self.t0:7.2f}s"
        self.fh.write(f"[{stamp}] {line}\n")
        self.fh.flush()
        print(line, flush=True)

    def info(self, msg):
        self._w(msg)

    def event(self, name, detail=""):
        self.events.append(name)
        self._w(f"[EVENT] {name}" + (f" -- {detail}" if detail else ""))

    def check(self, name, ok, detail=""):
        self.checks.append((name, bool(ok)))
        self._w(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        return ok

    def close(self, result):
        self._w(f"[RESULT] {result}")
        self.fh.close()


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fresh_dirs():
    if ART.exists():
        shutil.rmtree(ART)
    for d in ("manifests/shards", "ledgers", "checkpoints"):
        (ART / d).mkdir(parents=True, exist_ok=True)
    if SHARD_DIR.exists():
        shutil.rmtree(SHARD_DIR)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not (CORPUS / "sources.json").exists():
        print("corpus/ missing -- running prepare_corpus.py first (needs network)")
        import prepare_corpus
        prepare_corpus.main()

    fresh_dirs()
    log = Log(ART / "run.log")
    cfg = train.Config()
    log.info(f"S6 training data execution system -- run_id={RUN_ID}")
    log.info(f"config: seq_len={cfg.seq_len} seq/step={cfg.seq_per_step} "
             f"({cfg.ranks} ranks x {cfg.grad_accum} accum x {cfg.micro} micro) "
             f"steps={cfg.total_steps} ckpt_every={cfg.checkpoint_every} crash_at={cfg.crash_at}")
    log.info("note on event order: 'evaluation data blocked' precedes 'manifests validated' "
             "because the admission verdict is a manifest field, so the firewall and the gate "
             "must settle before a manifest can be revalidated against its final content. "
             "All 13 mandated events are present; the check at the end of this log verifies it.")

    # ---- 1. frozen tokenizer -------------------------------------------
    tok, tok_path, tok_sha = shards.load_tokenizer()
    ok = shards.verify_tokenizer(tok_sha)
    log.check("tokenizer_hash_verified", ok,
              f"{shards.TOKENIZER_REPO}/{shards.TOKENIZER_FILE} sha256={tok_sha[:16]}...")
    if not ok:
        log.close("FAIL")
        return 1

    # ---- 2. shards created ---------------------------------------------
    sources = json.loads((CORPUS / "sources.json").read_text(encoding="utf-8"))
    source_meta = {s["source_id"]: s for s in sources["sources"]}
    train_docs = []
    for p in sorted(CORPUS.glob("*.jsonl")):
        if p.name == "eval_registry_docs.jsonl":
            continue
        train_docs.extend(read_jsonl(p))
    eval_docs = read_jsonl(CORPUS / "eval_registry_docs.jsonl")
    log.info(f"corpus: {len(train_docs)} training docs, {len(eval_docs)} eval/validation docs, "
             f"{len(source_meta)} provenance records")

    all_shards = shards.build_shards(tok, train_docs, tok_sha)
    dupes = shards.mark_dedup(all_shards)
    log.event("shards created", f"{len(all_shards)} immutable shards, "
                                f"{sum(s.manifest['token_count'] for s in all_shards):,} tokens, "
                                f"{dupes} exact duplicates flagged")

    # ---- 3. firewall registry + eval/validation shards -----------------
    wall = fw.Firewall()
    wall.register(eval_docs, source_meta)
    eval_shards = shards.build_shards(tok, eval_docs, tok_sha)
    for s in eval_shards:
        kind = "test" if any(d["doc_id"] in wall.entries and
                             wall.entries[d["doc_id"]]["kind"] == "test"
                             for d in s.manifest["docs"]) else "validation"
        s.manifest["never_train"] = kind == "test"
        s.manifest["gradient_bearing"] = False
        s.manifest["shard_kind"] = kind
        s.manifest["dedup_status"] = "unique"
        s.manifest["contamination_status"] = "registry_member"
        s.manifest["eval_overlap_status"] = "is_eval_data"
        wall.register_shard(s.shard_id, kind)
    log.info(f"firewall: {len(wall.entries)} registry docs, {len(wall.fp_index)} "
             f"{fw.NGRAM}-gram fingerprints, {len(eval_shards)} eval/validation shards")

    # scan every training shard against the registry
    for s in all_shards:
        toks = s.tokens
        texts = [tok.decode(toks[d["token_start"]:d["token_end"]].tolist())
                 for d in s.manifest["docs"]]
        wall.check_shard(s, texts)

    # deliberate attack 1: offer a never-train test shard to the admission gate
    victim = next(s for s in eval_shards if s.manifest["never_train"])
    v_verdict, v_reasons = shards.admit(victim.manifest)
    wall.block(victim.shard_id, "never_train_shard_offered_to_admission_gate",
               {"verdict": v_verdict, "reasons": v_reasons})
    log.event("evaluation data blocked",
              f"{victim.shard_id} -> {v_verdict} ({', '.join(v_reasons)})")
    log.check("eval_shard_blocked", v_verdict == "REJECTED" and "never_train_flag" in v_reasons,
              f"{victim.shard_id} rejected: {v_reasons}")

    # validation shards are admissible-looking but must not be gradient-bearing
    val_shard = next(s for s in eval_shards if not s.manifest["never_train"])
    val_verdict, val_reasons = shards.admit(val_shard.manifest)
    wall.block(val_shard.shard_id, "validation_shard_offered_to_admission_gate",
               {"verdict": val_verdict, "reasons": val_reasons})
    log.check("validation_shard_blocked_from_training", val_verdict == "REJECTED",
              f"{val_shard.shard_id} rejected: {val_reasons}")

    # deliberate attack 2: splice a real MMLU test item into a training document
    bench_doc = next(d for d in eval_docs if d["kind"] == "test")
    bench_text = "\n".join(x["text"] for x in bench_doc["segments"])
    host = next(s for s in all_shards if s.lane == "general_web")
    poisoned_docs = []
    for i, d in enumerate(host.manifest["docs"][:3]):
        raw = tok.decode(host.tokens[d["token_start"]:d["token_end"]].tolist())
        text = fw.contaminate(raw, bench_text) if i == 0 else raw
        poisoned_docs.append({"doc_id": d["doc_id"] + ".poisoned", "lane": host.lane,
                              "source_id": host.source_id, "kind": "train",
                              "language_and_script": d["language_and_script"],
                              "license": d["license"], "provenance_tier": d["provenance_tier"],
                              "segments": [{"role": "target", "text": text}], "meta": {}})
    poisoned = shards.derive_shard(
        host, [shards.tokenize_doc(tok, d) for d in poisoned_docs], tok_sha,
        "contamination attack: one MMLU test item spliced into a training document")
    poisoned.manifest["dedup_status"] = "unique"
    clean, status = wall.check_shard(poisoned, [d["segments"][0]["text"] for d in poisoned_docs])
    p_verdict, p_reasons = shards.admit(poisoned.manifest)
    wall.block(poisoned.shard_id, f"contamination_detected:{status}",
               {"verdict": p_verdict, "reasons": p_reasons,
                "parent_shard_ids": poisoned.manifest["parent_shard_ids"]})
    log.check("contamination_blocked", (not clean) and p_verdict == "REJECTED",
              f"{poisoned.shard_id} -> {status}; reasons={p_reasons}")

    # ---- admission gate over the real training shards ------------------
    for s in all_shards:
        shards.admit(s.manifest)
    admitted = [s for s in all_shards if s.manifest["admission"]["verdict"] == "ADMITTED"]
    rejected = [s for s in all_shards if s.manifest["admission"]["verdict"] == "REJECTED"]
    log.info(f"admission gate: {len(admitted)} admitted, {len(rejected)} rejected")
    for s in rejected:
        log.info(f"  REJECTED {s.shard_id}: {s.manifest['admission']['reasons']}")

    # vocab remap + write payloads and manifests
    remap, remap_info = shards.build_vocab_remap(admitted, cfg.vocab)
    every = all_shards + eval_shards + [poisoned]
    for s in every:
        s.write(SHARD_DIR)
        (ART / "manifests/shards" / f"{s.shard_id}.json").write_text(
            json.dumps(s.manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (ART / "manifests/tokenizer_manifest.json").write_text(json.dumps({
        "repo": shards.TOKENIZER_REPO, "file": shards.TOKENIZER_FILE,
        "frozen_sha256": shards.FROZEN_TOKENIZER_SHA256, "observed_sha256": tok_sha,
        "verified": ok, "resolved_path": str(tok_path),
        "tokenizer_version": shards.TOKENIZER_VERSION,
        "true_vocab_size": tok.get_vocab_size(),
        "special_tokens": {"unk": shards.UNK, "bos": shards.BOS, "eos": shards.EOS,
                           "pad": shards.PAD},
        "cleaning_pipeline_hash": shards.pipeline_hash(),
        "cleaning_pipeline_version": shards.PIPELINE_VERSION,
        "prose_only_ghost_strip_lanes": sorted(shards.PROSE_LANES),
        **remap_info,
    }, indent=2), encoding="utf-8")
    (ART / "manifests/shards_index.json").write_text(json.dumps({
        "counts": {"total": len(every), "admitted": len(admitted), "rejected": len(rejected),
                   "eval_or_validation": len(eval_shards), "derived_attack_shards": 1},
        "allowed_licenses": sorted(shards.ALLOWED_LICENSES),
        "shards": [{"shard_id": s.shard_id, "capability_lane": s.lane,
                    "token_count": s.manifest["token_count"],
                    "loss_bearing_token_count": s.manifest["loss_bearing_token_count"],
                    "content_hash": s.manifest["content_hash"],
                    "licenses": s.manifest["license_and_provenance_tier"]["licenses"],
                    "language_and_script": s.manifest["language_and_script"],
                    "dedup_status": s.manifest["dedup_status"],
                    "contamination_status": s.manifest["contamination_status"],
                    "never_train": s.manifest["never_train"],
                    "reserved_for_anneal": s.manifest["reserved_for_anneal"],
                    "parent_shard_ids": s.manifest["parent_shard_ids"],
                    "admission": s.manifest["admission"]} for s in every],
    }, indent=2), encoding="utf-8")
    (ART / "manifests/corpus_sources.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 4. manifests validated ----------------------------------------
    rows = shards.validate_manifests(ART / "manifests/shards", SHARD_DIR)
    passed = sum(1 for r in rows if r["result"] == "PASS")
    (ART / "manifests/manifest_validation.json").write_text(json.dumps({
        "method": "reloaded every shard payload and recomputed content hash, token count, "
                  "id/hash binding, tokenizer hash and the presence of all 14 fields",
        "passed": passed, "total": len(rows), "rows": rows}, indent=2), encoding="utf-8")
    log.event("manifests validated", f"{passed}/{len(rows)} shard manifests recomputed from bytes")
    log.check("manifests_validated", passed == len(rows), f"{passed}/{len(rows)}")

    # ---- 5. mixture compiled -------------------------------------------
    schedule = mixture.Schedule(cfg.total_steps, cfg.seq_per_step, cfg.seq_len)
    reserve = mixture.reserve_for_anneal(admitted, schedule)
    supply = {}
    for s in admitted:
        supply[s.lane] = supply.get(s.lane, 0) + s.manifest["token_count"]
    report = schedule.compile_report(supply, reserve)
    mixture.write_schedule(ART / "manifests/mixture_schedule.json", report)
    log.event("mixture compiled",
              f"{len(schedule.stages)} stages, floors={mixture.FLOORS}, "
              f"anneal reserve={sum(reserve.values()):,} tokens")
    log.check("mixture_within_s5_tolerance", not report["lanes_outside_tolerance"],
              f"drift vs S5 headline: {report['drift_vs_s5_headline']}")
    for lane, v in report["supply_reconciliation"].items():
        log.info(f"  supply {lane:14s} demand={v['demand_tokens']:>9,d} "
                 f"available={v['available_tokens']:>9,d} -> {v['verdict']}")

    # re-write manifests that reserve_for_anneal mutated
    for s in admitted:
        (ART / "manifests/shards" / f"{s.shard_id}.json").write_text(
            json.dumps(s.manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- training scaffolding ------------------------------------------
    store = shards.ShardStore(SHARD_DIR)
    pool = packing.Pool(admitted, cfg.seq_len)
    log.info("unit pool: " + ", ".join(
        f"{k}={v['units']}u/{v['tokens']:,}t({v['policy']})" for k, v in pool.stats().items()))

    ledger = ConsumptionLedger(ART / "ledgers/consumption.jsonl", RUN_ID,
                               shards.TOKENIZER_VERSION, packing.DATALOADER_VERSION)
    trace = TokenTrace(ART / "ledgers/token_trace.jsonl", tok, cfg.trace_full_interval,
                       cfg.trace_sample_every)
    learning = LearningLedger()
    opus = mixture.Opus()
    model = train.Model(cfg)
    runner = train.Runner(cfg, schedule, pool, store, remap, tok, wall, ledger, trace,
                          learning, log, {s.shard_id: s.manifest for s in every})
    runner.validation, runner.validation_shards = build_validation(
        eval_shards, store, remap, cfg, wall)
    ledger.note("run_start", run_id=RUN_ID, branch_id=MAIN, schema=SCHEMA_VERSION, config={
        "seq_len": cfg.seq_len, "seq_per_step": cfg.seq_per_step, "total_steps": cfg.total_steps,
        "seed": cfg.seed, "tokenizer": shards.TOKENIZER_VERSION,
        "dataloader": packing.DATALOADER_VERSION})

    # ---- 6-8. train until the deliberate crash -------------------------
    ckpt_dir = ART / "checkpoints"
    crashed = False
    try:
        model, opus, ckpt_id, _ = runner.run_range(
            model, opus, start=0, end=cfg.total_steps, record_branch=MAIN, data_branch=MAIN,
            attempt=1, checkpoint_id="ckpt_step0000", crash_at=cfg.crash_at,
            checkpoint_dir=ckpt_dir)
    except train.SimulatedCrash as e:
        crashed = True
        log.info(f"  packed so far: {runner.perf.microbatches} microbatches, "
                 f"{runner.perf.raw_positions:,} token positions; OPUS " +
                 ", ".join(f"{k}={v}" for k, v in sorted(opus.counts.items())))
        log.event("crash simulated", str(e))
        ledger.note("crash", branch_id=MAIN, reason=str(e), at_step=cfg.crash_at - 1,
                    ledger_offset=ledger.offset)
        log.check("crash_simulated", True, f"at step {cfg.crash_at - 1}, "
                                          f"ledger offset {ledger.offset}")
    if not crashed:
        log.check("crash_simulated", False, "the run did not reach the crash step")
        log.close("FAIL")
        return 1

    # ---- 9. resume -----------------------------------------------------
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.npz"))
    latest = ckpts[-1]
    t0 = time.perf_counter()
    rmodel, state = train.load_checkpoint(latest, cfg)
    ropus = mixture.Opus.from_state(state["opus_scores"])
    pre_resume = [e for e in ledger.events if e["type"] == "consumption"
                  and e["branch_id"] == MAIN and e["global_step"] == state["step"]]
    roll = ledger.rollback(MAIN, state["ledger_offset"], state["step"], "resume_after_crash")
    # resume latency is the cost of RECOVERING -- loading the checkpoint, rolling the ledger
    # back and re-deriving the next batch. The training that follows is not recovery cost.
    resume_seconds = time.perf_counter() - t0
    log.info(f"resume: {latest.name} step={state['step']} ledger_offset={state['ledger_offset']} "
             f"superseding {roll['superseded_count']} events in {resume_seconds:.3f}s")
    rmodel, ropus, ckpt_id, _ = runner.run_range(
        rmodel, ropus, start=state["step"], end=cfg.total_steps, record_branch=MAIN,
        data_branch=MAIN, attempt=2, checkpoint_id=state["meta"].get("checkpoint_id",
                                                                    latest.stem),
        checkpoint_dir=ckpt_dir)
    log.event("run resumed", f"from {latest.name} at step {state['step']}, attempt 2, "
                             f"recovery took {resume_seconds:.3f}s")

    post = [e for e in ledger.events if e["type"] == "consumption" and e["branch_id"] == MAIN
            and e["global_step"] == state["step"] and e["attempt"] == 2]
    expected = state["expected_next"]
    redigest = runner._predict_next(state["step"], MAIN)
    first_match = bool(pre_resume) and bool(post) and all(
        a["batch_hash"] == b["batch_hash"] for a, b in
        zip(sorted(pre_resume, key=lambda e: (e["rank"], e["microbatch_id"])),
            sorted(post, key=lambda e: (e["rank"], e["microbatch_id"]))))
    digest_match = expected["plan_digest"] == redigest["plan_digest"]
    eff = ledger.effective(MAIN)
    cont = stream_continuity(eff)
    (ckpt_dir / "resume_proof.json").write_text(json.dumps({
        "crash": {"simulated": True, "at_step": cfg.crash_at - 1,
                  "steps_lost": cfg.crash_at - 1 - state["step"] + 1},
        "resumed_from_checkpoint": latest.name,
        "checkpoint_step": state["step"], "checkpoint_ledger_offset": state["ledger_offset"],
        "expected_next_batch": expected,
        "recomputed_next_batch": redigest,
        "plan_digest_matched": digest_match,
        "next_batch_matched": first_match,
        "batch_ids": {
            "expected": [[e["global_step"], e["rank"], e["microbatch_id"]]
                         for e in sorted(pre_resume, key=lambda e: (e["rank"], e["microbatch_id"]))],
            "resumed": [[e["global_step"], e["rank"], e["microbatch_id"]]
                        for e in sorted(post, key=lambda e: (e["rank"], e["microbatch_id"]))]},
        "batch_hashes": {
            "before_crash_attempt1": [e["batch_hash"] for e in
                                      sorted(pre_resume, key=lambda e: (e["rank"], e["microbatch_id"]))],
            "after_resume_attempt2": [e["batch_hash"] for e in
                                      sorted(post, key=lambda e: (e["rank"], e["microbatch_id"]))]},
        "superseded_count": roll["superseded_count"],
        "effective_stream": cont,
        "recovery_seconds": round(resume_seconds, 4),
        "recovery_seconds_covers": "checkpoint load + ledger rollback + re-derivation of the "
                                   "next batch; not the training that follows",
    }, indent=2), encoding="utf-8")
    log.check("resume_next_batch_matched", first_match and digest_match,
              f"step {state['step']}: {len(post)} microbatches, hashes identical to attempt 1")
    log.check("resume_no_gaps_or_duplicates", cont["contiguous"],
              f"{cont['events']} effective events over steps {cont['step_range']}, "
              f"{len(cont['missing_steps'])} gaps, {len(cont['duplicate_keys'])} duplicates")

    # ---- 10. replay ----------------------------------------------------
    replay_from = ckpt_dir / f"ckpt_step{cfg.checkpoint_every:04d}.npz"
    original = {(e["global_step"], e["rank"], e["microbatch_id"]): e
                for e in ledger.events if e["type"] == "consumption" and e["branch_id"] == MAIN
                and e["attempt"] == 1}
    t0 = time.perf_counter()
    pmodel, pstate = train.load_checkpoint(replay_from, cfg)
    popus = mixture.Opus.from_state(pstate["opus_scores"])
    rbranch = f"replay-{MAIN}@{pstate['step']}"
    ledger.note("replay_start", branch_id=rbranch, source_branch=MAIN,
                from_checkpoint=replay_from.name, from_step=pstate["step"],
                to_step=pstate["step"] + cfg.checkpoint_every)
    _, _, _, comparisons = runner.run_range(
        pmodel, popus, start=pstate["step"], end=pstate["step"] + cfg.checkpoint_every,
        record_branch=rbranch, data_branch=MAIN, attempt=1,
        checkpoint_id=replay_from.stem, mode="replay", compare=original)
    replay_seconds = time.perf_counter() - t0
    matched = sum(1 for c in comparisons if c["hash_match"])
    (ckpt_dir / "replay_proof.json").write_text(json.dumps({
        "from_checkpoint": replay_from.name, "from_step": pstate["step"],
        "to_step": pstate["step"] + cfg.checkpoint_every,
        "record_branch": rbranch, "data_branch": MAIN,
        "method": "restored the older checkpoint (weights, optimizer moments and OPUS score "
                  "buffer), re-ran the pure planner on the same data branch, and compared "
                  "batch hashes, token span ids and loss-mask hashes against the original "
                  "ledger events",
        "compared": len(comparisons), "matched": matched,
        "all_matched": matched == len(comparisons) and bool(comparisons),
        "replay_seconds": round(replay_seconds, 4),
        "comparisons": comparisons}, indent=2), encoding="utf-8")
    log.event("historical stream replayed",
              f"steps {pstate['step']}..{pstate['step'] + cfg.checkpoint_every} from "
              f"{replay_from.name}")
    log.check("replay_hash_matched", matched == len(comparisons) and bool(comparisons),
              f"{matched}/{len(comparisons)} batch hashes, span ids and loss-mask hashes "
              f"identical to the original run")

    # ---- 11. fork ------------------------------------------------------
    fork_from = ckpt_dir / f"ckpt_step{cfg.checkpoint_every * 2:04d}.npz"
    fmodel, fstate = train.load_checkpoint(fork_from, cfg)
    fopus = mixture.Opus.from_state(fstate["opus_scores"])
    fbranch = "fork-1"
    ledger.note("fork", branch_id=fbranch, parent_branch=MAIN,
                from_checkpoint=fork_from.name, divergence_step=fstate["step"],
                divergence_ledger_offset=fstate["ledger_offset"],
                note="new data branch: the seed identity changes, so the stream diverges "
                     "deliberately and every difference is explicit")
    _, _, _, _ = runner.run_range(
        fmodel, fopus, start=fstate["step"], end=fstate["step"] + 5, record_branch=fbranch,
        data_branch=fbranch, attempt=1, checkpoint_id=fork_from.stem, mode="fork")
    fork_events = {(e["global_step"], e["rank"], e["microbatch_id"]): e for e in ledger.events
                   if e["type"] == "consumption" and e["branch_id"] == fbranch}
    # re-run the fork from the same checkpoint: a branch must be reproducible too
    f2model, _ = train.load_checkpoint(fork_from, cfg)
    f2opus = mixture.Opus.from_state(fstate["opus_scores"])
    _, _, _, fcmp = runner.run_range(
        f2model, f2opus, start=fstate["step"], end=fstate["step"] + 5,
        record_branch=fbranch + "-verify", data_branch=fbranch, attempt=1,
        checkpoint_id=fork_from.stem, mode="fork", compare=fork_events)
    same_step = fstate["step"]
    main_at = {(e["rank"], e["microbatch_id"]): e for e in ledger.effective(MAIN)
               if e["global_step"] == same_step}
    fork_at = {(e["rank"], e["microbatch_id"]): e for k, e in fork_events.items()
               if e["global_step"] == same_step}
    diverged = all(main_at[k]["batch_hash"] != fork_at[k]["batch_hash"]
                   for k in main_at if k in fork_at) and bool(fork_at)
    reproducible = bool(fcmp) and all(c["hash_match"] for c in fcmp)
    (ckpt_dir / "fork_proof.json").write_text(json.dumps({
        "branch_id": fbranch, "parent_branch": MAIN, "from_checkpoint": fork_from.name,
        "divergence": {"step": fstate["step"], "ledger_offset": fstate["ledger_offset"],
                       "parent_checkpoint": fork_from.name},
        "diverged": bool(diverged), "reproducible": reproducible,
        "comparison_at_divergence_step": [{
            "rank": k[0], "microbatch_id": k[1],
            "main_batch_hash": main_at[k]["batch_hash"],
            "fork_batch_hash": fork_at[k]["batch_hash"],
            "differs": main_at[k]["batch_hash"] != fork_at[k]["batch_hash"],
            "main_span_ids": main_at[k]["token_span_ids"],
            "fork_span_ids": fork_at[k]["token_span_ids"]} for k in sorted(main_at)
            if k in fork_at],
        "rerun_verification": {"compared": len(fcmp),
                               "matched": sum(1 for c in fcmp if c["hash_match"])},
    }, indent=2), encoding="utf-8")
    log.event("branch forked", f"{fbranch} from {fork_from.name} at step {fstate['step']}")
    log.check("fork_divergence_recorded", diverged and reproducible,
              f"fork diverges from main at step {same_step} and re-runs identically")

    # ---- checkpoints index ---------------------------------------------
    offset_checks = []
    for r in runner.checkpoints:
        expected_eff = r["step"] * (cfg.ranks * cfg.grad_accum)
        offset_checks.append({
            "checkpoint_id": r["checkpoint_id"], "step": r["step"],
            "ledger_offset": r["ledger_offset"],
            "effective_events_at_checkpoint": r["effective_events_at_checkpoint"],
            "expected_effective_events": expected_eff,
            "ledger_offset_matches_effective_count":
                r["effective_events_at_checkpoint"] == expected_eff})
    (ckpt_dir / "checkpoints_index.json").write_text(json.dumps({
        "microbatches_per_step": cfg.ranks * cfg.grad_accum,
        "checkpoint_every": cfg.checkpoint_every,
        "checkpoints": runner.checkpoints, "offset_checks": offset_checks}, indent=2),
        encoding="utf-8")

    # ---- 12. audit -----------------------------------------------------
    eff = ledger.effective(MAIN)
    total_tokens = sum(e["batch_stats"]["positions"] for e in eff)
    aud = train.audit(eff, opus.decisions + ropus.decisions, runner.step_losses,
                      (total_tokens // 4, total_tokens // 2))
    (ART / "ledgers/audit.json").write_text(json.dumps(aud, indent=2), encoding="utf-8")
    log.event("audit completed",
              f"{aud['microbatches_in_range']} microbatches in the audited token range, "
              f"{len(aud['shard_influence'])} shards implicated")

    # OPUS decision log: attempt 1 (up to the crash) and attempt 2 (after the resume).
    # Both are kept -- the decisions the crash discarded are part of the audit trail.
    with open(ART / "ledgers/opus_decisions.jsonl", "w", encoding="utf-8") as f:
        for attempt, inst in ((1, opus), (2, ropus)):
            for d in inst.decisions:
                f.write(json.dumps({**d, "attempt": attempt, "branch_id": MAIN},
                                   sort_keys=True) + "\n")
    log.info(f"OPUS ledger: {len(opus.decisions)} decisions on attempt 1, "
             f"{len(ropus.decisions)} on attempt 2 "
             f"(accepted/rejected/deferred/protected across both: " +
             ", ".join(f"{k}={opus.counts[k] + ropus.counts[k]}"
                       for k in sorted(opus.counts)) + ")")

    # branches + mixture compliance + learning ledger + token stats
    write_branches(ART, ledger, cfg, runner, fstate, pstate, rbranch, fbranch)
    write_compliance(ART, ledger, schedule, report, every, cfg, log)
    tstats = trace.aggregates()
    (ART / "ledgers/token_stats.json").write_text(json.dumps(tstats, indent=2), encoding="utf-8")
    spike = float(np.percentile([abs(v) for v in runner.step_losses.values()], 95)) * 3
    rows = learning.rows(tstats, spike)
    with open(ART / "ledgers/learning_ledger.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    log.info(f"learning ledger: {sum(1 for r in rows if r['key_type'] == 'shard')} shard report "
             f"cards, {sum(1 for r in rows if r['key_type'] == 'capability_lane')} lane roll-ups; "
             f"token trace {trace.rows_written} detailed rows")

    # ---- 13. performance ----------------------------------------------
    write_packed_report(ART, runner, pool, cfg)
    perf = write_performance(ART, runner, store, opus, ropus, resume_seconds, replay_seconds,
                            cfg, ledger)
    log.event("performance measured",
              f"{perf['metrics']['useful_loss_bearing_tokens_per_sec']:.0f} useful tok/s, "
              f"packing utilization {perf['metrics']['packing_utilization']:.4f}")

    wall.write(ART / "manifests/eval_registry.json")
    ledger.note("run_end", branch_id=MAIN, effective_events=len(eff),
                total_positions=total_tokens)
    trace.close()
    ledger.close()

    # ---- evidence ------------------------------------------------------
    bundle = evidence.write(ART)
    for r in bundle["requirements"]:
        log.check(f"evidence:{r['requirement'].lower().replace(' ', '_')}",
                  r["result"] == "PASS", r["evidence"]["recomputed"][:120])
    log.info(f"evidence: {bundle['passed']}/{bundle['total']} requirements PASS")

    # ---- the write-up page's data ---------------------------------------
    # dataset-creation.html reads out/s6data.js, so one command really does produce
    # everything. This is presentation, not evidence: a failure here is logged and
    # ignored rather than allowed to change the exit code.
    try:
        import build_html_data
        build_html_data.main()
    except Exception as e:                                        # noqa: BLE001
        log.info(f"  note: out/s6data.js not rebuilt ({type(e).__name__}: {e}); "
                 f"run build_html_data.py separately for the write-up page")
    missing = [e for e in REQUIRED_EVENTS if e not in log.events]
    log.check("all_required_events_logged", not missing, f"missing={missing}" if missing else
              f"{len(REQUIRED_EVENTS)} mandated events present")
    result = "PASS" if bundle["result"] == "PASS" and not missing else "FAIL"
    log.close(result)
    print(f"\n{'=' * 70}\n{result}: submission_artifacts/ regenerated "
          f"({bundle['passed']}/{bundle['total']} evidence rows)\n{'=' * 70}")
    return 0 if result == "PASS" else 1


REQUIRED_EVENTS = ["shards created", "manifests validated", "evaluation data blocked",
                   "mixture compiled", "batches packed", "OPUS decisions recorded",
                   "checkpoint saved", "crash simulated", "run resumed",
                   "historical stream replayed", "branch forked", "audit completed",
                   "performance measured"]


def build_validation(eval_shards, store, remap, cfg, wall):
    """Validation sequences: read during training for evaluation, never gradient-bearing.
    They bypass the packer's firewall assertion on purpose -- that assertion exists to keep
    them OUT of loss-bearing batches, and this path produces no gradients."""
    val = [s for s in eval_shards if not s.manifest["never_train"]][:2]
    batches, ids = [], []
    for s in val:
        units = packing.Pool._enumerate(s, cfg.seq_len)
        if not units:
            continue
        plan = packing.SeqPlan("validation", "pad_only", cfg.seq_len)
        plan.place(units[0], 0)
        batches.append(packing.materialize([plan], store, remap, cfg.seq_len, firewall=None))
        ids.append(s.shard_id)
    return batches, ids


def write_branches(ART, ledger, cfg, runner, fstate, pstate, rbranch, fbranch):
    branches = {}
    for e in ledger.events:
        if e["type"] != "consumption":
            continue
        b = branches.setdefault(e["branch_id"], {
            "branch_id": e["branch_id"], "events": 0, "steps": set(), "attempts": set(),
            "modes": set()})
        b["events"] += 1
        b["steps"].add(e["global_step"])
        b["attempts"].add(e["attempt"])
        b["modes"].add(e.get("mode"))
    out = []
    for b in branches.values():
        out.append({**b, "steps": [min(b["steps"]), max(b["steps"])],
                    "attempts": sorted(b["attempts"]), "modes": sorted(b["modes"])})
    (ART / "ledgers/branches.json").write_text(json.dumps({
        "run_id": RUN_ID, "main_branch": MAIN,
        "branches": out,
        "divergences": [
            {"branch_id": rbranch, "kind": "replay", "parent": MAIN,
             "at_step": pstate["step"], "at_ledger_offset": pstate["ledger_offset"],
             "data_branch": MAIN,
             "note": "same data stream as the parent, recorded separately for comparison"},
            {"branch_id": fbranch, "kind": "fork", "parent": MAIN,
             "at_step": fstate["step"], "at_ledger_offset": fstate["ledger_offset"],
             "data_branch": fbranch,
             "note": "new data branch: every difference from the parent is explicit"}],
        "rollbacks": ledger.rollbacks,
    }, indent=2), encoding="utf-8")


def write_compliance(ART, ledger, schedule, report, every, cfg, log):
    """Planned vs actual lane shares, per-window floor checks, anneal-reserve quarantine."""
    eff = ledger.effective(MAIN)
    actual = {}
    for e in eff:
        for lane, n in e["lane_sequence_counts"].items():
            actual[lane] = actual.get(lane, 0) + n
    tot = sum(actual.values()) or 1
    planned = report["planned_allocator_shares"]
    shares = {l: {"planned": round(planned.get(l, 0), 4),
                  "actual": round(actual.get(l, 0) / tot, 4),
                  "actual_sequences": actual.get(l, 0),
                  "drift": round(actual.get(l, 0) / tot - planned.get(l, 0), 4)}
              for l in sorted(set(planned) | set(actual))}

    windows = []
    by_step = {}
    for e in eff:
        for lane, n in e["lane_sequence_counts"].items():
            by_step.setdefault(e["global_step"], {})
            by_step[e["global_step"]][lane] = by_step[e["global_step"]].get(lane, 0) + n
    steps = sorted(by_step)
    W = mixture.FLOOR_WINDOW
    for i in range(0, len(steps), W):
        chunk = steps[i:i + W]
        counts, n = {}, 0
        for s in chunk:
            for lane, k in by_step[s].items():
                counts[lane] = counts.get(lane, 0) + k
                n += k
        lanes = {lane: {"share": round(counts.get(lane, 0) / n, 4), "floor": f,
                        "holds": counts.get(lane, 0) / n >= f - 1e-9}
                 for lane, f in mixture.FLOORS.items()}
        windows.append({"steps": [chunk[0], chunk[-1]], "sequences": n,
                        "protected_lanes": lanes,
                        "all_floors_hold": all(v["holds"] for v in lanes.values())})

    anneal_start = [s for s in schedule.stages if s["stage"].startswith("4")][0]["step_start"]
    reserved = {s.shard_id for s in every if s.manifest.get("reserved_for_anneal")}
    early = [e for e in eff if e["global_step"] < anneal_start
             and reserved & set(e["shard_ids"])]
    (ART / "ledgers/mixture_compliance.json").write_text(json.dumps({
        "method": "recounted lane sequences from every effective consumption event and "
                  "compared them against the compiled per-step quotas",
        "total_sequences": tot, "shares": shares,
        "floor_window_steps": W, "floor_windows": windows,
        "floors_violated_in_windows": sum(1 for w in windows if not w["all_floors_hold"]),
        "anneal_reserve": {
            "anneal_starts_at_step": anneal_start,
            "reserved_shards": len(reserved),
            "reserved_shards_consumed_before_anneal": len(early),
            "offending_events": [[e["global_step"], e["shard_ids"]] for e in early[:10]]},
    }, indent=2), encoding="utf-8")
    bad = sum(1 for w in windows if not w["all_floors_hold"])
    log.check("protected_floors_held", bad == 0,
              f"{len(windows)} windows of {W} steps, {bad} floor violations")
    log.check("anneal_reserve_quarantined", not early,
              f"{len(reserved)} reserved shards, {len(early)} consumed before step {anneal_start}")


def write_packed_report(ART, runner, pool, cfg):
    batches = runner.packed_batches
    train_b = [b for b in batches if b["mode"] == "train"]
    used = sum(b["used_positions"] for b in train_b)
    tot = sum(b["positions"] for b in train_b)
    (ART / "manifests/packed_batch_report.json").write_text(json.dumps({
        "sequence_length": cfg.seq_len,
        "lane_policies": packing.LANE_POLICY,
        "totals": {"batches": len(train_b), "positions": tot, "used_positions": used,
                   "pad_positions": tot - used,
                   "loss_bearing_positions": sum(b["loss_bearing_positions"] for b in train_b),
                   "context_positions": sum(b["context_positions"] for b in train_b),
                   "utilization": round(used / tot, 6) if tot else 0,
                   "truncations": sum(s["truncations"] for b in train_b for s in b["sequences"])},
        "mask_invariants": {**runner.mask_audit,
                            "definition": {
                                "loss_on_padding_positions": "loss_mask set where segment_id < 0",
                                "loss_on_context_positions":
                                    "loss_mask set where token t+1 is not a target token",
                                "cross_segment_visible_pairs":
                                    "attention mask true between different segments",
                                "non_causal_visible_pairs": "attention mask true for s > t",
                                "position_id_violations":
                                    "a segment whose position ids are not 0..n-1"}},
        "pool": pool.stats(),
        "policy_comparison": packing.policy_comparison(pool, cfg.seq_len),
        "batches": batches,
    }, indent=2), encoding="utf-8")


def write_performance(ART, runner, store, opus, ropus, resume_seconds, replay_seconds, cfg,
                      ledger):
    p = runner.perf
    wall = p.wall_seconds
    # Throughput is reported over the EFFECTIVE stream -- what the run actually retained.
    # The 15 steps the crash discarded cost real time but taught the model nothing, so
    # counting them would inflate every rate. The discarded work is reported separately.
    eff = ledger.effective(MAIN)
    eff_raw = sum(e["batch_stats"]["positions"] for e in eff)
    eff_useful = sum(e["batch_stats"]["loss_bearing_positions"] for e in eff)
    decisions = opus.decisions + ropus.decisions
    by_lane = {}
    for d in decisions:
        e = by_lane.setdefault(d["capability_lane"], {"total": 0, "rejected": 0, "deferred": 0,
                                                      "protected": 0, "accepted": 0})
        e["total"] += 1
        e[d["status"]] += 1
    rej = {k: round((v["rejected"] + v["deferred"]) / v["total"], 4) for k, v in by_lane.items()}
    metrics = {
        # raw: every position the loader materialised, including OPUS-rejected candidates
        # and the work the crash discarded. accepted: positions that survived selection and
        # the crash. useful: of those, the ones that actually bore loss.
        "raw_tokens_per_sec": round(p.candidate_positions / wall, 3) if wall else 0,
        "accepted_tokens_per_sec_after_opus": round(eff_raw / wall, 3) if wall else 0,
        "useful_loss_bearing_tokens_per_sec": round(eff_useful / wall, 3) if wall else 0,
        # CPU, not GPU: there is no separate accelerator to starve here, so this is wall time
        # minus all measured model work (forward/backward plus OPUS scoring, which is also
        # model compute) -- i.e. the time the model spent waiting on the loader and bookkeeping.
        "compute_idle_time_seconds": round(
            max(0.0, wall - p.compute_seconds - p.opus_seconds), 4),
        "loader_wait_time_seconds": round(p.loader_seconds, 4),
        "packing_utilization": round(p.packing_used / p.packing_total, 6) if p.packing_total else 0,
        "rejection_rate_by_lane": rej,
        "replay_and_resume_latency_seconds": {"resume": round(resume_seconds, 4),
                                             "replay": round(replay_seconds, 4)},
        **store.stats(),
    }
    out = {
        "run_id": RUN_ID,
        "measured_with": "time.perf_counter around the planner, OPUS scoring, materialisation "
                         "and the forward/backward pass; no figure is reported that cannot be "
                         "recomputed from the ledger",
        "wall_seconds": round(wall, 4),
        "breakdown_seconds": {"loader": round(p.loader_seconds, 4),
                              "opus_scoring": round(p.opus_seconds, 4),
                              "compute": round(p.compute_seconds, 4),
                              "other": round(max(0.0, wall - p.loader_seconds - p.opus_seconds
                                                 - p.compute_seconds), 4)},
        "counts": {"steps": p.steps, "microbatches": p.microbatches,
                   "candidate_positions_scored": p.candidate_positions,
                   "materialised_positions_including_discarded": p.raw_positions,
                   "effective_positions": eff_raw,
                   "effective_loss_bearing_positions": eff_useful,
                   "positions_discarded_by_crash": p.raw_positions - eff_raw,
                   "effective_microbatches": len(eff),
                   "opus_decisions": len(decisions)},
        "metrics": metrics,
        "opus_decisions_by_lane": by_lane,
        "note": "raw > accepted > useful, and the gaps are the point: raw includes the "
                "candidates OPUS rejected and the batches the crash discarded, accepted is "
                "what reached the optimizer, useful is what actually bore loss. Padding and "
                "context-only tokens are carried through the batch either way, which is why "
                "useful loss-bearing tokens/sec is the only rate worth optimising.",
        "run_meta": {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "python": sys.version.split()[0], "numpy": np.__version__},
    }
    (ART / "performance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.exit(main())
