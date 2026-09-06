"""
The training consumption ledger (§8), the token-level trace (§11) and the two-way learning
ledger (§12).

The consumption ledger is append-only and is the run's memory. It is never rewritten: a
resume does not delete the batches served between the last checkpoint and the crash, it
appends a `rollback` record that supersedes them and re-serves the same steps under a new
attempt number. The EFFECTIVE stream is then "highest non-superseded attempt per
(branch, step, rank, microbatch)" -- which is what "no skipped or repeated batch" is
actually checked against.
"""
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

SCHEMA_VERSION = "s6-ledger-1"  # recorded in the run_start note


class ConsumptionLedger:
    def __init__(self, path, run_id, tokenizer_version, dataloader_version):
        self.path = path
        self.run_id = run_id
        self.tokenizer_version = tokenizer_version
        self.dataloader_version = dataloader_version
        self.offset = 0
        self.events = []
        self.rollbacks = []
        self._pass_count = defaultdict(int)     # unit_id -> times consumed (repeated passes)
        self._fh = open(path, "w", encoding="utf-8")

    def close(self):
        self._fh.close()

    def _append(self, rec):
        rec["ledger_offset"] = self.offset
        self._fh.write(json.dumps(rec, sort_keys=True) + "\n")
        self._fh.flush()
        self.offset += 1
        return rec

    # ---- consumption events (§8) ----
    def record_batch(self, *, branch, step, attempt, checkpoint_id, rank, microbatch,
                     batch, plans, lane, lane_counts, stage, opus_decision_ids, mode="train"):
        for p in plans:
            for u in p.unit_ids():
                self._pass_count[u] += 1
        rec = {                                          # §8's fields, in the notes' order
            "run_id": self.run_id,
            "branch_id": branch,
            "global_step": step,
            "checkpoint_id": checkpoint_id,
            "rank": rank,
            "microbatch_id": microbatch,
            "packed_sample_ids": [u for p in plans for u in p.unit_ids()],
            "shard_ids": sorted({s for p in plans for s in p.shard_ids()}),
            "token_span_ids": [s for p in plans for s in p.span_ids()],
            "loss_mask_hash": batch.loss_mask_hash(),
            "attention_and_position_policy": "causal_block_diagonal/segment_reset_positions",
            "mixture_lane": lane,
            "lane_sequence_counts": lane_counts,   # per-sequence, so shares recompute exactly
            "curriculum_stage": stage,
            "tokenizer_version": self.tokenizer_version,
            "dataloader_version": self.dataloader_version,
            "opus_decision_id": opus_decision_ids,
            # what resume and replay actually assert on
            "batch_hash": batch.batch_hash(),
            "attempt": attempt,
            "superseded": False,
            "mode": mode,
            "packing_policies": sorted({p.policy for p in plans}),
            "batch_stats": batch.stats(),
            "repeated_pass_numbers": {u: self._pass_count[u]
                                      for p in plans for u in p.unit_ids()},
            "type": "consumption",
        }
        self.events.append(rec)
        return self._append(rec)

    def rollback(self, branch, to_offset, to_step, reason):
        """Supersede everything appended after a checkpoint's ledger offset. Append-only:
        the superseded events stay on disk as history."""
        killed = []
        for e in self.events:
            if (e["type"] == "consumption" and e["branch_id"] == branch
                    and not e["superseded"] and e["ledger_offset"] >= to_offset):
                e["superseded"] = True
                killed.append([e["global_step"], e["rank"], e["microbatch_id"]])
        rec = {"type": "rollback", "branch_id": branch, "reason": reason,
               "to_ledger_offset": to_offset, "resumed_from_step": to_step,
               "superseded_events": killed, "superseded_count": len(killed),
               "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        self.rollbacks.append(rec)
        self.events.append(rec)
        return self._append(rec)

    def note(self, kind, **kw):
        rec = {"type": kind, "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        rec.update(kw)
        self.events.append(rec)
        return self._append(rec)

    # ---- queries ----
    def effective(self, branch, mode="train"):
        """One event per (step, rank, microbatch): the surviving attempt."""
        best = {}
        for e in self.events:
            if e["type"] != "consumption" or e["branch_id"] != branch or e.get("mode") != mode:
                continue
            if e["superseded"]:
                continue
            k = (e["global_step"], e["rank"], e["microbatch_id"])
            if k not in best or e["attempt"] > best[k]["attempt"]:
                best[k] = e
        return [best[k] for k in sorted(best)]

    def effective_count(self, branch, upto_step=None):
        ev = self.effective(branch)
        if upto_step is None:
            return len(ev)
        return sum(1 for e in ev if e["global_step"] <= upto_step)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def effective_from_disk(events, branch, mode="train"):
    """The same reduction, computed from the file alone -- what evidence.py and the tests
    use, so the invariant is checked against the artifact rather than the process.

    A rollback supersedes exactly the events appended between the checkpoint's offset and
    the rollback record itself. The re-served attempts come after the rollback record and
    are therefore untouched -- which is what makes the append-only log recoverable without
    rewriting history (the on-disk `superseded` flag is whatever it was at write time; the
    rollback record is the authority)."""
    windows = [(e["to_ledger_offset"], e["ledger_offset"]) for e in events
               if e.get("type") == "rollback" and e["branch_id"] == branch]
    best = {}
    for e in events:
        if e.get("type") != "consumption" or e["branch_id"] != branch or e.get("mode") != mode:
            continue
        if any(lo <= e["ledger_offset"] < hi for lo, hi in windows):
            continue
        k = (e["global_step"], e["rank"], e["microbatch_id"])
        if k not in best or e["attempt"] > best[k]["attempt"]:
            best[k] = e
    return [best[k] for k in sorted(best)]


def stream_continuity(events):
    """Gap / duplicate analysis over an effective stream."""
    keys = [(e["global_step"], e["rank"], e["microbatch_id"]) for e in events]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    steps = sorted({e["global_step"] for e in events})
    gaps = [s for s in range(steps[0], steps[-1] + 1) if s not in set(steps)] if steps else []
    per_step = defaultdict(int)
    for e in events:
        per_step[e["global_step"]] += 1
    sizes = sorted(set(per_step.values()))
    return {"events": len(events), "steps": len(steps),
            "step_range": [steps[0], steps[-1]] if steps else [],
            "duplicate_keys": [list(d) for d in dupes], "missing_steps": gaps,
            "microbatches_per_step": sizes,
            "contiguous": not gaps and not dupes and len(sizes) <= 1}


# ---- token-level trace (§11) -------------------------------------------
class TokenTrace:
    """Three storage tiers, as §11 prescribes: full rows for a flagged interval, quantized
    rows for sampled intervals, aggregates for the whole run."""

    def __init__(self, path, tok, full_interval, sample_every=7):
        self.path = path
        self.tok = tok
        self.full_interval = full_interval
        self.sample_every = sample_every
        self.rows_written = 0
        self.agg = defaultdict(lambda: {"n": 0, "loss": 0.0, "high_ppl": 0})
        self._fh = open(path, "w", encoding="utf-8")

    def close(self):
        self._fh.close()

    def record(self, *, step, stage, batch, per_token_loss, plans, lane, ckpt_before,
               ckpt_after, model_age, opus_ids, pass_numbers, shard_manifest):
        a, b = self.full_interval
        full = a <= step < b
        quant = (not full) and step % self.sample_every == 0
        B, T = batch.tokens.shape
        for i in range(B):
            plan = plans[i]
            seg_of = {p["segment_id"]: p for p in plan.placements}
            for t in np.flatnonzero(batch.loss_mask[i]):
                t = int(t)
                loss = float(per_token_loss[i, t])
                tid = int(batch.tokens[i, t + 1])          # the token the loss is ON
                seg = int(batch.segment_ids[i, t + 1])
                pl = seg_of.get(seg, plan.placements[0])
                m = shard_manifest.get(pl["shard_id"], {})
                langs = m.get("language_and_script") or ["unknown"]
                key = (pl["shard_id"], lane, langs[0], t // 64)
                agg = self.agg[key]
                agg["n"] += 1
                agg["loss"] += loss
                agg["high_ppl"] += 1 if loss > 6.0 else 0
                if not (full or quant):
                    continue
                if quant:
                    # §11's middle tier: "compressed or quantized token losses" for sampled
                    # intervals. A compact row with rounded loss, not the full 17 fields --
                    # storing everything for every interval is exactly the cost the tiering
                    # exists to avoid.
                    self._fh.write(json.dumps({
                        "tier": "quantized", "global_step": step, "shard_id": pl["shard_id"],
                        "capability_lane": lane, "position_in_packed_sequence": t + 1,
                        "token_id": tid, "cross_entropy_loss": round(loss, 2),
                        "token_perplexity": round(float(np.exp(min(loss, 20))), 1),
                        "loss_mask_flag": True}, sort_keys=True) + "\n")
                    self.rows_written += 1
                    continue
                row = {                                     # §11's fields
                    "token_id": tid,
                    "decoded_preview": self.tok.decode([tid]),
                    "position_in_packed_sequence": t + 1,
                    "doc_id": pl.get("doc_id"),
                    "shard_id": pl["shard_id"],
                    "language_and_script": langs[0],
                    "capability_lane": lane,
                    "special_token_flag": tid < 4,
                    "boundary_or_eos_flag": tid == 2,
                    "loss_mask_flag": True,
                    "cross_entropy_loss": round(loss, 6),
                    "token_perplexity": round(float(np.exp(min(loss, 20))), 4),
                    "model_age_tokens": model_age,
                    "checkpoint_before": ckpt_before,
                    "checkpoint_after": ckpt_after,
                    "curriculum_stage": stage,
                    "opus_score_or_decision_id": opus_ids[0] if opus_ids else None,
                    "repeated_pass_number": pass_numbers.get(pl["unit_id"], 1),
                    "global_step": step,
                    "tier": "full" if full else "quantized",
                }
                self._fh.write(json.dumps(row, sort_keys=True) + "\n")
                self.rows_written += 1

    def aggregates(self):
        by_shard, by_lane, by_lang, by_pos = {}, {}, {}, {}
        for (shard, lane, lang, bucket), v in self.agg.items():
            for d, k in ((by_shard, shard), (by_lane, lane), (by_lang, lang),
                         (by_pos, f"{bucket * 64}-{bucket * 64 + 63}")):
                e = d.setdefault(k, {"tokens": 0, "loss_sum": 0.0, "high_ppl_tokens": 0})
                e["tokens"] += v["n"]
                e["loss_sum"] += v["loss"]
                e["high_ppl_tokens"] += v["high_ppl"]
        for d in (by_shard, by_lane, by_lang, by_pos):
            for k, v in d.items():
                v["mean_loss"] = round(v["loss_sum"] / v["tokens"], 6) if v["tokens"] else None
                v["mean_perplexity"] = round(float(np.exp(min(v["mean_loss"], 20))), 4) \
                    if v["tokens"] else None
                v["loss_sum"] = round(v["loss_sum"], 4)
        return {"tier": "aggregate", "rows_written_detailed": self.rows_written,
                "full_interval_steps": list(self.full_interval),
                "quantized_every_n_steps": self.sample_every,
                "by_shard": by_shard, "by_capability_lane": by_lane,
                "by_language_and_script": by_lang, "by_position_bucket": by_pos}


# ---- learning ledger (§12) ---------------------------------------------
class LearningLedger:
    """Attaches the outcome back to the data: what was consumed, how surprising it was, how
    the model responded, and when in training that response happened."""

    def __init__(self):
        self.per_shard = defaultdict(lambda: {
            "exposures": [], "losses": [], "grad_norms": [], "grad_alignments": [],
            "opus_scores": [], "tokens": 0, "phases": set(), "passes": [], "lane": None,
            "first_step": None, "last_step": None})

    def observe(self, *, shard_ids, lane, step, mean_loss, grad_norm, grad_align,
                opus_score, loss_tokens, phase, pass_numbers, tokens_consumed):
        for sid in shard_ids:
            r = self.per_shard[sid]
            r["lane"] = lane
            r["exposures"].append(step)
            r["losses"].append(float(mean_loss))
            r["grad_norms"].append(float(grad_norm))
            r["grad_alignments"].append(float(grad_align))
            r["opus_scores"].append(float(opus_score))
            r["tokens"] += int(loss_tokens)
            r["phases"].add(phase)
            r["passes"].extend(pass_numbers)
            r["first_step"] = step if r["first_step"] is None else min(r["first_step"], step)
            r["last_step"] = step if r["last_step"] is None else max(r["last_step"], step)
            r["tokens_at_first_sight"] = r.get("tokens_at_first_sight", tokens_consumed)

    def rows(self, token_aggregates, spike_threshold):
        """§12's fields per shard, plus a per-lane roll-up."""
        out = []
        by_shard = token_aggregates.get("by_shard", {})
        for sid in sorted(self.per_shard):
            r = self.per_shard[sid]
            losses = r["losses"]
            delta = round(losses[0] - losses[-1], 6) if len(losses) > 1 else None
            gmax = max(r["grad_norms"]) if r["grad_norms"] else 0.0
            spike = gmax > spike_threshold
            passes = max(r["passes"]) if r["passes"] else 1
            repeat_effect = None
            if passes > 1 and len(losses) > 1:
                repeat_effect = round(losses[-1] - losses[len(losses) // 2], 6)
            if spike:
                cls = "harmful"
            elif delta is None:
                cls = "neutral"
            elif delta > 0.01:
                cls = "useful"
            elif delta < -0.01:
                cls = "harmful"
            else:
                cls = "neutral"
            tok = by_shard.get(sid, {})
            out.append({                                     # §12's fields
                "key": sid, "key_type": "shard", "capability_lane": r["lane"],
                "avg_token_loss": round(float(np.mean(losses)), 6) if losses else None,
                "high_perplexity_clusters": tok.get("high_ppl_tokens", 0),
                "loss_delta_before_after": delta,
                "gradient_norm": round(float(np.mean(r["grad_norms"])), 6) if r["grad_norms"] else None,
                "gradient_norm_max": round(gmax, 6),
                "gradient_alignment": round(float(np.mean(r["grad_alignments"])), 6)
                if r["grad_alignments"] else None,
                "opus_score": round(float(np.mean(r["opus_scores"])), 6) if r["opus_scores"] else None,
                "repeated_pass_effect": repeat_effect,
                "max_repeated_pass": passes,
                "model_phase": sorted(r["phases"]),
                "tokens_consumed_when_seen": r.get("tokens_at_first_sight"),
                "loss_bearing_tokens_consumed": r["tokens"],
                "usefulness_classification": cls,
                "exposures": len(r["exposures"]),
                "first_step": r["first_step"], "last_step": r["last_step"],
                "gradient_spike": spike,
            })
        lanes = defaultdict(list)
        for row in out:
            lanes[row["capability_lane"]].append(row)
        for lane, rows in sorted(lanes.items()):
            ok = [r for r in rows if r["avg_token_loss"] is not None]
            out.append({
                "key": lane, "key_type": "capability_lane", "capability_lane": lane,
                "avg_token_loss": round(float(np.mean([r["avg_token_loss"] for r in ok])), 6)
                if ok else None,
                "high_perplexity_clusters": sum(r["high_perplexity_clusters"] for r in rows),
                "loss_delta_before_after": round(float(np.mean(
                    [r["loss_delta_before_after"] for r in rows
                     if r["loss_delta_before_after"] is not None] or [0.0])), 6),
                "gradient_norm": round(float(np.mean([r["gradient_norm"] for r in rows
                                                      if r["gradient_norm"] is not None] or [0.0])), 6),
                "gradient_alignment": round(float(np.mean([r["gradient_alignment"] for r in rows
                                                           if r["gradient_alignment"] is not None] or [0.0])), 6),
                "opus_score": round(float(np.mean([r["opus_score"] for r in rows
                                                   if r["opus_score"] is not None] or [0.0])), 6),
                "repeated_pass_effect": round(float(np.mean(
                    [r["repeated_pass_effect"] for r in rows
                     if r["repeated_pass_effect"] is not None] or [0.0])), 6),
                "max_repeated_pass": max((r["max_repeated_pass"] for r in rows), default=1),
                "model_phase": sorted({p for r in rows for p in r["model_phase"]}),
                "tokens_consumed_when_seen": min(
                    (r["tokens_consumed_when_seen"] for r in rows
                     if r["tokens_consumed_when_seen"] is not None), default=None),
                "shards": len(rows),
                "loss_bearing_tokens_consumed": sum(r["loss_bearing_tokens_consumed"] for r in rows),
                "usefulness_classification": max(
                    ("useful", "neutral", "harmful"),
                    key=lambda c: sum(1 for r in rows if r["usefulness_classification"] == c)),
            })
        return out
