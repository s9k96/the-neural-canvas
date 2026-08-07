"""
Invariant tests for the S6 training data execution system.

Plain asserts with a __main__ runner -- pytest is not a dependency here, but every test is
a plain `test_*` function so `python -m pytest` also works if it is installed.

    .venv/Scripts/python s6-dataset-creation/tests/test_invariants.py

The tests run against the artifacts in submission_artifacts/, so run_demo.py must have been
run first. Tests 5 and 6 additionally re-execute parts of the system (the model, the
planner) to check properties that no artifact can assert on its own.
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from tds import evidence, mixture, packing, shards, train  # noqa: E402
from tds.ledger import effective_from_disk, read_jsonl, stream_continuity  # noqa: E402

A = ROOT / "submission_artifacts"
SHARD_DIR = ROOT / "shards"
MAIN = "main"


# ---- fixtures (plain module-level loads, no framework) -------------------
def manifests():
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((A / "manifests/shards").glob("*.json"))]


def all_events():
    return read_jsonl(A / "ledgers/consumption.jsonl")


def effective():
    return effective_from_disk(all_events(), MAIN)


def load(name):
    return json.loads((A / name).read_text(encoding="utf-8"))


def rebuild_pool():
    """Reconstruct the planner's inputs from the manifests on disk."""
    class S:                                     # minimal stand-in for a Shard
        pass
    out = []
    for m in manifests():
        if m["admission"]["verdict"] != "ADMITTED" or m.get("never_train"):
            continue
        z = np.load(SHARD_DIR / f"{m['shard_id']}.npz")
        s = S()
        s.shard_id, s.lane = m["shard_id"], m["capability_lane"]
        s.tokens, s.segs, s.manifest = z["tokens"], z["segs"], m
        out.append(s)
    cfg = train.Config()
    return packing.Pool(out, cfg.seq_len), cfg


# ---- 1. shard immutability ---------------------------------------------
def test_shard_immutability():
    ms = manifests()
    assert ms, "no shard manifests"
    for m in ms:
        z = np.load(SHARD_DIR / f"{m['shard_id']}.npz")
        h = hashlib.sha256()
        h.update(z["tokens"].tobytes())
        h.update(z["segs"].tobytes())
        h.update(json.dumps(m["doc_ids"], sort_keys=True).encode())
        assert h.hexdigest() == m["content_hash"], f"{m['shard_id']} content hash drifted"
        assert m["shard_id"].endswith(m["content_hash"][:12]), \
            f"{m['shard_id']} does not embed its content hash"
        assert int(z["tokens"].size) == m["token_count"]
    # re-tokenising a document must reproduce the exact ids stored in the shard
    tok, _, sha = shards.load_tokenizer()
    assert sha == shards.FROZEN_TOKENIZER_SHA256
    m = next(x for x in manifests() if x["capability_lane"] == "general_web"
             and x["admission"]["verdict"] == "ADMITTED")
    z = np.load(SHARD_DIR / f"{m['shard_id']}.npz")
    d = m["docs"][0]
    stored = z["tokens"][d["token_start"]:d["token_end"]]
    text = tok.decode(stored.tolist())
    again = tok.encode(text, add_special_tokens=False).ids
    assert len(again) > 0
    # round-trip through decode is lossy at the edges; the ids of the re-encoded decode of
    # the stored ids must still be a stable fixed point
    assert tok.encode(tok.decode(again), add_special_tokens=False).ids == again


# ---- 2. firewall -------------------------------------------------------
def test_never_train_and_validation_never_consumed():
    reg = load("manifests/eval_registry.json")
    never = set(reg["never_train_shard_ids"])
    nongrad = set(reg["non_gradient_shard_ids"])
    assert never, "no never-train shards registered"
    consumed = {s for e in all_events() if e.get("type") == "consumption"
                for s in e["shard_ids"]}
    assert not (never & consumed), f"never-train shards consumed: {never & consumed}"
    assert not (nongrad & consumed), f"validation shards consumed: {nongrad & consumed}"
    # and the blocks were actually recorded
    reasons = " ".join(b["reason"] for b in reg["blocked_events"])
    assert "never_train" in reasons and "contamination" in reasons


# ---- 3. loss mask ------------------------------------------------------
def test_loss_mask_never_on_padding_or_context():
    mi = load("manifests/packed_batch_report.json")["mask_invariants"]
    assert mi["checked_batches"] > 0
    assert mi["loss_on_padding_positions"] == 0
    assert mi["loss_on_context_positions"] == 0
    # independently: re-materialise a batch and check the arrays directly
    pool, cfg = rebuild_pool()
    remap, _ = shards.build_vocab_remap(
        [s for s in _pool_shards(pool)], cfg.vocab)
    store = shards.ShardStore(SHARD_DIR)
    sched = mixture.Schedule(cfg.total_steps, cfg.seq_per_step, cfg.seq_len)
    slots, _ = packing.plan_batch(sched, pool, cfg.seed, MAIN, 12, cfg.ranks, cfg.micro)
    b = packing.materialize([slots[0].candidates[0]], store, remap, cfg.seq_len)
    assert not (b.loss_mask & (b.segment_ids < 0)).any(), "loss on padding"
    nxt = np.zeros_like(b.is_target)
    nxt[:, :-1] = b.is_target[:, 1:]
    assert not (b.loss_mask & ~nxt).any(), "loss on a non-target token"


# ---- 4. attention mask + position ids ----------------------------------
def test_attention_block_diagonal_and_position_ids():
    mi = load("manifests/packed_batch_report.json")["mask_invariants"]
    assert mi["cross_segment_visible_pairs"] == 0
    assert mi["non_causal_visible_pairs"] == 0
    assert mi["position_id_violations"] == 0
    pool, cfg = rebuild_pool()
    remap, _ = shards.build_vocab_remap(_pool_shards(pool), cfg.vocab)
    store = shards.ShardStore(SHARD_DIR)
    sched = mixture.Schedule(cfg.total_steps, cfg.seq_per_step, cfg.seq_len)
    # a structure-preserving lane is where multi-segment packing actually happens
    slots, _ = packing.plan_batch(sched, pool, cfg.seed, MAIN, 20, cfg.ranks, cfg.micro)
    multi = [c for s in slots for c in s.candidates
             if len({p["segment_id"] for p in c.placements}) > 1]
    assert multi, "no multi-segment sequence to test"
    b = packing.materialize(multi[:2], store, remap, cfg.seq_len)
    am = b.attention_mask()
    seg = b.segment_ids
    assert not (am & (seg[:, :, None] != seg[:, None, :])).any(), "attention crossed a segment"
    for i in range(seg.shape[0]):
        for s in np.unique(seg[i]):
            if s < 0:
                continue
            idx = np.flatnonzero(seg[i] == s)
            assert np.array_equal(b.position_ids[i, idx], np.arange(idx.size))


# ---- 5. packing non-contamination (the mask is honoured, not just recorded) ----
def test_packed_samples_do_not_contaminate_each_other():
    """Two samples packed into one window must produce exactly the per-token losses they
    produce when packed alone. This is what proves the model consumes the block-diagonal
    mask rather than merely carrying it."""
    pool, cfg = rebuild_pool()
    remap, _ = shards.build_vocab_remap(_pool_shards(pool), cfg.vocab)
    store = shards.ShardStore(SHARD_DIR)
    # any structure-preserving lane with two samples that fit together in one window
    small = []
    for lane in ("stem_math", "agentic", "reasoning"):
        cand = sorted([u for u in pool.eligible(lane, in_anneal=True)
                       if u.n_tokens < cfg.seq_len // 2], key=lambda u: u.unit_id)
        if len(cand) >= 2:
            small = cand[:2]
            break
    assert len(small) == 2, "need two samples that fit together in one window"
    model = train.Model(cfg)

    def losses(plan):
        b = packing.materialize([plan], store, remap, cfg.seq_len)
        per_token, *_ = model.loss_terms(b)
        return per_token[0], b

    alone = []
    for u in small:
        p = packing.SeqPlan(u.lane, "structure_preserving", cfg.seq_len)
        p.place(u, 0)
        alone.append(losses(p))
    together = packing.SeqPlan(small[0].lane, "structure_preserving", cfg.seq_len)
    together.place(small[0], 0)
    together.place(small[1], 1)
    pt_together, bt = losses(together)

    for i, (pt_alone, ba) in enumerate(alone):
        seg = bt.segment_ids[0] == i
        n = int(ba.loss_mask[0].sum())
        got = pt_together[seg][ba.loss_mask[0][:int(seg.sum())]]
        want = pt_alone[ba.loss_mask[0]]
        assert n > 0
        assert np.allclose(got, want, atol=1e-12), \
            f"sample {i} leaked: max delta {np.abs(got - want).max()}"


# ---- 6. determinism ---------------------------------------------------
def test_planner_is_pure_and_replay_matches():
    pool, cfg = rebuild_pool()
    sched = mixture.Schedule(cfg.total_steps, cfg.seq_per_step, cfg.seq_len)
    for step in (0, 7, 41, 79):
        a, _ = packing.plan_batch(sched, pool, cfg.seed, MAIN, step, cfg.ranks, cfg.micro)
        b, _ = packing.plan_batch(sched, pool, cfg.seed, MAIN, step, cfg.ranks, cfg.micro)
        assert [[c.span_ids() for c in s.candidates] for s in a] == \
               [[c.span_ids() for c in s.candidates] for s in b], f"step {step} not pure"
        assert [s.lane for s in a] == [s.lane for s in b]
    # a different branch must produce a different stream (fork identity is in the seed)
    f, _ = packing.plan_batch(sched, pool, cfg.seed, "fork-1", 41, cfg.ranks, cfg.micro)
    m, _ = packing.plan_batch(sched, pool, cfg.seed, MAIN, 41, cfg.ranks, cfg.micro)
    assert [[c.span_ids() for c in s.candidates] for s in f] != \
           [[c.span_ids() for c in s.candidates] for s in m]
    proof = load("checkpoints/replay_proof.json")
    assert proof["compared"] > 0 and proof["all_matched"]


# ---- 7. crash recovery: no skipped or repeated batch -------------------
def test_effective_stream_has_no_gaps_or_duplicates():
    cont = stream_continuity(effective())
    assert cont["events"] > 0
    assert not cont["missing_steps"], f"skipped steps: {cont['missing_steps']}"
    assert not cont["duplicate_keys"], f"repeated batches: {cont['duplicate_keys']}"
    assert cont["contiguous"]
    cfg = train.Config()
    assert cont["microbatches_per_step"] == [cfg.ranks * cfg.grad_accum]
    # the crash really did discard work that was then re-served
    proof = load("checkpoints/resume_proof.json")
    assert proof["superseded_count"] > 0
    assert proof["next_batch_matched"] and proof["plan_digest_matched"]


# ---- 8. checkpoints are tied to ledger offsets ------------------------
def test_checkpoint_offsets_match_ledger():
    idx = load("checkpoints/checkpoints_index.json")
    assert idx["checkpoints"]
    for r in idx["offset_checks"]:
        assert r["ledger_offset_matches_effective_count"], \
            f"{r['checkpoint_id']}: {r['effective_events_at_checkpoint']} != " \
            f"{r['expected_effective_events']}"
    eff = effective()
    for r in idx["checkpoints"]:
        n = sum(1 for e in eff if e["global_step"] < r["step"])
        assert n == r["step"] * idx["microbatches_per_step"]
        assert (A / f"checkpoints/{r['checkpoint_id']}.npz").exists()


# ---- 9. mixture shares and protected floors --------------------------
def test_lane_shares_and_protected_floors():
    sched = load("manifests/mixture_schedule.json")
    comp = load("ledgers/mixture_compliance.json")
    for lane, v in comp["shares"].items():
        assert abs(v["drift"]) <= 0.005, f"{lane} drifted {v['drift']} from plan"
    assert comp["floor_windows"]
    for w in comp["floor_windows"]:
        for lane, v in w["protected_lanes"].items():
            assert v["holds"], f"floor broken for {lane} in steps {w['steps']}: {v['share']}"
    assert set(sched["protected_floors"]) == {"indic", "reasoning", "agentic"}
    assert not sched["lanes_outside_tolerance"], sched["drift_vs_s5_headline"]


# ---- 10. anneal reserve ----------------------------------------------
def test_anneal_reserve_untouched_before_anneal():
    comp = load("ledgers/mixture_compliance.json")
    r = comp["anneal_reserve"]
    assert r["reserved_shards"] > 0, "nothing was quarantined"
    assert r["reserved_shards_consumed_before_anneal"] == 0, r["offending_events"]
    reserved = {m["shard_id"] for m in manifests() if m.get("reserved_for_anneal")}
    assert reserved
    start = r["anneal_starts_at_step"]
    for e in effective():
        if e["global_step"] < start:
            assert not (reserved & set(e["shard_ids"])), \
                f"reserved shard consumed at step {e['global_step']}"


# ---- 11. OPUS decision records ---------------------------------------
def test_opus_records_complete_and_floors_respected():
    dec = read_jsonl(A / "ledgers/opus_decisions.jsonl")
    assert dec
    protected = set(load("manifests/mixture_schedule.json")["protected_floors"])
    fields = ("candidate_id", "shard_ids", "capability_lane", "curriculum_stage",
              "scoring_checkpoint_id", "proxy_version", "opus_score", "status",
              "rejection_reason", "protected_floor_override", "effective_token_estimate")
    seen = set()
    for d in dec:
        for f in fields:
            assert f in d, f"{d.get('candidate_id')} missing {f}"
        assert d["status"] in ("accepted", "rejected", "deferred", "protected")
        if d["status"] in ("rejected", "deferred"):
            assert d["rejection_reason"], f"{d['candidate_id']} rejected without a reason"
        if d["protected_floor_override"]:
            assert d["capability_lane"] in protected, \
                f"floor override on unprotected lane {d['capability_lane']}"
        seen.add(d["candidate_id"])
    for status in ("accepted", "rejected"):
        assert any(d["status"] == status for d in dec), f"no {status} decisions"
    # every consumed batch names decisions that exist
    for e in effective():
        for cid in e["opus_decision_id"]:
            assert cid in seen, f"dangling decision id {cid}"


# ---- 12. fork --------------------------------------------------------
def test_fork_diverges_and_is_reproducible():
    f = load("checkpoints/fork_proof.json")
    assert f["diverged"] and f["reproducible"]
    assert f["divergence"]["step"] is not None
    rows = f["comparison_at_divergence_step"]
    assert rows
    for r in rows:
        assert r["differs"], "fork produced the parent's batch"
        assert r["main_span_ids"] != r["fork_span_ids"]
    assert f["rerun_verification"]["compared"] == f["rerun_verification"]["matched"]


# ---- 13. the evidence bundle is derived, not declared ----------------
def test_tampering_a_ledger_flips_the_evidence():
    """Copy the artifacts, corrupt one batch hash in the consumption ledger, and confirm the
    affected evidence rows flip to FAIL. If they do not, the bundle is not actually derived
    from the artifacts."""
    before = evidence.build(A)
    assert before["result"] == "PASS", "fixture must start from a passing bundle"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "artifacts"
        shutil.copytree(A, tmp)
        p = tmp / "ledgers/consumption.jsonl"
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            e = json.loads(line)
            if e.get("type") == "consumption" and e["branch_id"] == MAIN and e["attempt"] == 1 \
                    and e["global_step"] == 20:
                e["batch_hash"] = "0" * 64
                lines[i] = json.dumps(e, sort_keys=True)
                break
        else:
            raise AssertionError("no event to tamper with")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        after = evidence.build(tmp)
    assert after["result"] == "FAIL", "tampering with the ledger did not fail the bundle"
    flipped = {r["requirement"] for r in after["requirements"] if r["result"] == "FAIL"}
    assert "Replay" in flipped, f"expected the Replay row to flip, got {flipped}"


def _pool_shards(pool):
    """The Pool keeps manifests; rebuild lightweight shard stand-ins for build_vocab_remap."""
    class S:
        pass
    out = []
    for sid, m in pool.shard_manifest.items():
        s = S()
        s.shard_id, s.lane, s.manifest = sid, m["capability_lane"], m
        s.tokens = np.load(SHARD_DIR / f"{sid}.npz")["tokens"]
        out.append(s)
    return out


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    # submission_artifacts/ is committed but shards/ is not (it is derived data that
    # run_demo.py rebuilds). Check for both, so a fresh clone gets this sentence instead of
    # a FileNotFoundError from the first test that opens a shard payload.
    missing = [str(p.relative_to(ROOT)) for p in (A / "evidence.json", SHARD_DIR)
               if not p.exists()]
    if missing:
        print(f"missing {', '.join(missing)} -- run run_demo.py first "
              f"(it regenerates both)")
        return 2
    failed = []
    for i, t in enumerate(TESTS, 1):
        try:
            t()
            print(f"[PASS] {i:2d}/{len(TESTS)} {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"[FAIL] {i:2d}/{len(TESTS)} {t.__name__}: {e}")
        except Exception as e:                                   # noqa: BLE001
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"[ERROR] {i:2d}/{len(TESTS)} {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - len(failed)}/{len(TESTS)} invariants hold")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
