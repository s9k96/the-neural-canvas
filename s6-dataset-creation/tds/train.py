"""
The training loop and the four recovery modes: resume, replay, fork, audit (§14).

A checkpoint without a data position is incomplete, so every checkpoint here carries the
ledger offset, the branch, the OPUS score buffer and the model age in tokens alongside the
weights and optimizer moments. Restoring one restores the data position too.

The model is a small numpy causal LM. It exists for three reasons only: to produce real
per-token losses, real gradient norms and real optimizer state -- and to genuinely CONSUME
the block-diagonal attention mask, which is what makes the packing non-contamination test a
proof rather than an assertion.

# ponytail: masked-mean attention, no learned QK projection. Swap in softmax(QK^T/sqrt(d))
# if the mask ever needs to be shown as *learned* rather than merely honoured.
"""
import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np

from . import packing
from .mixture import model_phase


class SimulatedCrash(RuntimeError):
    """Raised at a configured step to prove recovery works. Caught by run_demo."""


@dataclass
class Config:
    seq_len: int = 256
    # d_model and vocab are sized so a checkpoint (params + both Adam moments, float64) stays
    # a few MB: the checkpoints are committed artifacts, and float32 storage is not an option
    # because replay restores the model and then rescores OPUS candidates -- a rounded
    # checkpoint would change decisions and break the bit-exact replay proof.
    d_model: int = 32
    vocab: int = 2048
    ranks: int = 2
    micro: int = 2
    grad_accum: int = 2
    total_steps: int = 80
    checkpoint_every: int = 20
    crash_at: int = 55
    lr: float = 0.05
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    seed: str = "s6-v5"
    validate_every: int = 20
    # §11's storage tiers: full rows for one flagged interval, quantized rows for sampled
    # intervals, aggregates for everything. Full traces are expensive -- that is the lesson,
    # so the demo stores them for one step rather than for the whole run.
    trace_full_interval: tuple = (30, 31)
    trace_sample_every: int = 40
    candidates_per_slot: int = 2

    @property
    def seq_per_step(self):
        return self.ranks * self.micro * self.grad_accum


class Model:
    """emb -> masked causal mean -> linear readout. float64 for bit-reproducibility on CPU."""

    def __init__(self, cfg, rng=None):
        rng = rng or np.random.default_rng(0)
        d, v, t = cfg.d_model, cfg.vocab, cfg.seq_len
        self.cfg = cfg
        self.p = {
            "E": rng.normal(0, 0.02, (v, d)),
            "P": rng.normal(0, 0.02, (t, d)),
            "W": rng.normal(0, 0.02, (d, v)),
            "b": np.zeros(v),
        }
        self.m = {k: np.zeros_like(v_) for k, v_ in self.p.items()}
        self.v = {k: np.zeros_like(v_) for k, v_ in self.p.items()}
        self.t = 0
        self.age_tokens = 0
        self.grad_ema = None

    # ---- forward ----
    def _hidden(self, batch):
        """Masked causal mean over the attention mask -- the mask is load-bearing here."""
        emb = self.p["E"][batch.model_tokens] + self.p["P"][batch.position_ids]
        mask = batch.attention_mask().astype(np.float64)
        denom = mask.sum(-1, keepdims=True)
        np.maximum(denom, 1.0, out=denom)
        A = mask / denom
        return A, np.einsum("bts,bsd->btd", A, emb)

    def loss_terms(self, batch):
        """Returns (per_token_loss (B,T), idx, targets, h_sel, A, h). Loss at position t
        predicts token t+1; only masked positions are computed."""
        A, h = self._hidden(batch)
        bi, ti = np.nonzero(batch.loss_mask)
        per_token = np.zeros(batch.loss_mask.shape)
        if bi.size == 0:
            return per_token, (bi, ti), None, None, A, h, None
        targets = batch.model_tokens[bi, ti + 1]
        h_sel = h[bi, ti]
        logits = h_sel @ self.p["W"] + self.p["b"]
        logits -= logits.max(-1, keepdims=True)
        exp = np.exp(logits)
        Z = exp.sum(-1, keepdims=True)
        logp = logits - np.log(Z)
        losses = -logp[np.arange(targets.size), targets]
        per_token[bi, ti] = losses
        return per_token, (bi, ti), targets, h_sel, A, h, exp / Z

    def mean_loss(self, batch):
        per_token, (bi, ti), targets, *_ = self.loss_terms(batch)
        if targets is None:
            return 0.0, per_token
        return float(per_token[bi, ti].mean()), per_token

    def score_sequences(self, plans, store, remap, seq_len, chunk=4, target_cache=None):
        """OPUS proxy: mean loss per candidate sequence under the current parameters."""
        out = []
        for i in range(0, len(plans), chunk):
            group = plans[i:i + chunk]
            b = packing.materialize(group, store, remap, seq_len, firewall=None,
                                    target_cache=target_cache)
            per_token, *_ = self.loss_terms(b)
            for j in range(len(group)):
                m = b.loss_mask[j]
                out.append(float(per_token[j][m].mean()) if m.any() else 0.0)
        return out

    # ---- backward + Adam ----
    def backward(self, batch, cached):
        per_token, (bi, ti), targets, h_sel, A, h, probs = cached
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        if targets is None:
            return g, 0.0
        n = targets.size
        dlogits = probs
        dlogits[np.arange(n), targets] -= 1.0
        dlogits /= n
        g["W"] = h_sel.T @ dlogits
        g["b"] = dlogits.sum(0)
        dh_sel = dlogits @ self.p["W"].T
        dh = np.zeros_like(h)
        np.add.at(dh, (bi, ti), dh_sel)
        demb = np.einsum("bts,btd->bsd", A, dh)     # A^T dh, per sequence
        np.add.at(g["E"], batch.model_tokens.ravel(), demb.reshape(-1, demb.shape[-1]))
        np.add.at(g["P"], batch.position_ids.ravel(), demb.reshape(-1, demb.shape[-1]))
        norm = float(np.sqrt(sum(float((v ** 2).sum()) for v in g.values())))
        return g, norm

    def step_adam(self, grads):
        self.t += 1
        b1, b2 = self.cfg.betas
        for k, gk in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.cfg.lr * mhat / (np.sqrt(vhat) + self.cfg.eps)

    def alignment(self, grads):
        """Cosine between this batch's gradient and the EMA direction (§12's
        gradient_alignment, 'where available' -- here it is)."""
        g = np.concatenate([grads["W"].ravel(), grads["b"].ravel()])
        if self.grad_ema is None:
            self.grad_ema = g.copy()
            return 1.0
        na, nb = np.linalg.norm(g), np.linalg.norm(self.grad_ema)
        cos = float(g @ self.grad_ema / (na * nb)) if na and nb else 0.0
        self.grad_ema = 0.9 * self.grad_ema + 0.1 * g
        return cos


# ---- checkpoints --------------------------------------------------------
def save_checkpoint(path, model, opus, *, step, ledger_offset, branch, data_branch,
                    expected_next, meta):
    np.savez(path,
             **{f"p_{k}": v for k, v in model.p.items()},
             **{f"m_{k}": v for k, v in model.m.items()},
             **{f"v_{k}": v for k, v in model.v.items()},
             adam_t=np.int64(model.t), age_tokens=np.int64(model.age_tokens),
             grad_ema=model.grad_ema if model.grad_ema is not None else np.zeros(1),
             opus_scores=opus.state(),
             step=np.int64(step), ledger_offset=np.int64(ledger_offset),
             branch=np.array(branch), data_branch=np.array(data_branch),
             expected_next=np.array(json.dumps(expected_next)),
             meta=np.array(json.dumps(meta)))


def load_checkpoint(path, cfg):
    z = np.load(path, allow_pickle=False)
    model = Model(cfg)
    for k in model.p:
        model.p[k] = z[f"p_{k}"].copy()
        model.m[k] = z[f"m_{k}"].copy()
        model.v[k] = z[f"v_{k}"].copy()
    model.t = int(z["adam_t"])
    model.age_tokens = int(z["age_tokens"])
    ge = z["grad_ema"]
    model.grad_ema = None if ge.size == 1 else ge.copy()
    state = {"step": int(z["step"]), "ledger_offset": int(z["ledger_offset"]),
             "branch": str(z["branch"]), "data_branch": str(z["data_branch"]),
             "expected_next": json.loads(str(z["expected_next"])),
             "opus_scores": z["opus_scores"].copy(),
             "meta": json.loads(str(z["meta"]))}
    return model, state


# ---- the runner ---------------------------------------------------------
@dataclass
class Perf:
    loader_seconds: float = 0.0
    opus_seconds: float = 0.0
    compute_seconds: float = 0.0
    wall_seconds: float = 0.0
    raw_positions: int = 0
    useful_positions: int = 0
    candidate_positions: int = 0
    accepted_positions: int = 0
    steps: int = 0
    microbatches: int = 0
    packing_used: int = 0
    packing_total: int = 0


class Runner:
    def __init__(self, cfg, schedule, pool, store, remap, tok, firewall, ledger, trace,
                 learning, log, manifests):
        self.cfg, self.schedule, self.pool, self.store = cfg, schedule, pool, store
        self.remap, self.tok, self.firewall = remap, tok, firewall
        self.ledger, self.trace, self.learning, self.log = ledger, trace, learning, log
        self.manifests = manifests
        self.perf = Perf()
        self.step_losses = {}
        self.packed_batches = []
        self.checkpoints = []
        self.target_cache = {}
        self.mask_audit = {}
        self.validation = None
        self.validation_shards = []
        self._announced = False

    # -- one step --
    def run_range(self, model, opus, *, start, end, record_branch, data_branch, attempt,
                  checkpoint_id, crash_at=None, mode="train", checkpoint_dir=None,
                  ledger_offset_base=0, compare=None):
        """Runs [start, end). Returns (model, opus, last_ckpt, comparisons).

        `record_branch` is the ledger identity; `data_branch` is the SEED identity fed to
        plan_batch. Replay keeps data_branch = the original branch (identical stream,
        recorded separately); fork changes it (deliberate divergence).
        """
        comparisons = []
        # replay and fork must not pollute the main run's throughput accounting; their cost
        # is reported separately as replay_and_resume_latency
        perf = self.perf if mode == "train" else Perf()
        for step in range(start, end):
            t_step = time.perf_counter()
            _, stage = self.schedule.stage_at(step)
            phase = model_phase(step / max(1, self.cfg.total_steps))

            t0 = time.perf_counter()
            slots, meta = packing.plan_batch(
                self.schedule, self.pool, self.cfg.seed, data_branch, step,
                self.cfg.ranks, self.cfg.micro, self.cfg.candidates_per_slot)
            perf.loader_seconds += time.perf_counter() - t0

            # ---- OPUS: score every candidate, select one per slot ----
            t0 = time.perf_counter()
            chosen = []
            for slot in slots:
                scores = model.score_sequences(slot.candidates, self.store, self.remap,
                                               self.cfg.seq_len, target_cache=self.target_cache)
                perf.candidate_positions += len(slot.candidates) * self.cfg.seq_len
                rec, plan = opus.select(slot, slot.candidates, scores, step, stage["stage"],
                                        self._next_stage(step), checkpoint_id)
                chosen.append((slot, rec, plan))
            perf.opus_seconds += time.perf_counter() - t0

            # ---- group the accepted sequences into microbatches and train ----
            groups = {}
            for slot, rec, plan in chosen:
                groups.setdefault((slot.rank, slot.microbatch), []).append((slot, rec, plan))
            grads_acc, n_acc, step_loss = None, 0, []
            for (rank, mb) in sorted(groups):
                items = groups[(rank, mb)]
                plans = [p for _, _, p in items]
                t0 = time.perf_counter()
                batch = packing.materialize(plans, self.store, self.remap, self.cfg.seq_len,
                                            firewall=self.firewall,
                                            target_cache=self.target_cache)
                perf.loader_seconds += time.perf_counter() - t0

                t0 = time.perf_counter()
                cached = model.loss_terms(batch)
                per_token = cached[0]
                grads, gnorm = model.backward(batch, cached)
                perf.compute_seconds += time.perf_counter() - t0
                if grads_acc is None:
                    grads_acc = {k: v.copy() for k, v in grads.items()}
                else:
                    for k in grads_acc:
                        grads_acc[k] += grads[k]
                n_acc += 1

                mask = batch.loss_mask
                mloss = float(per_token[mask].mean()) if mask.any() else 0.0
                step_loss.append(mloss)
                lane = ",".join(sorted({s.lane for s, _, _ in items}))
                lane_counts = {}
                for s, _, _ in items:
                    lane_counts[s.lane] = lane_counts.get(s.lane, 0) + 1
                dec_ids = [r["candidate_id"] for _, r, _ in items]
                ev = self.ledger.record_batch(
                    branch=record_branch, step=step, attempt=attempt,
                    checkpoint_id=checkpoint_id, rank=rank, microbatch=mb, batch=batch,
                    plans=plans, lane=lane, lane_counts=lane_counts, stage=stage["stage"],
                    opus_decision_ids=dec_ids, mode=mode)
                ev["data_branch_id"] = data_branch
                stats = batch.stats()
                perf.raw_positions += stats["positions"]
                perf.useful_positions += stats["loss_bearing_positions"]
                perf.accepted_positions += stats["positions"]
                perf.packing_used += stats["used_positions"]
                perf.packing_total += stats["positions"]
                perf.microbatches += 1
                audit_counts = packing.mask_audit(batch)
                for k, v in audit_counts.items():
                    self.mask_audit[k] = self.mask_audit.get(k, 0) + v
                self.packed_batches.append({
                    "global_step": step, "branch_id": record_branch, "mode": mode,
                    "rank": rank, "microbatch_id": mb, "batch_hash": ev["batch_hash"],
                    "loss_mask_hash": ev["loss_mask_hash"],
                    "attention_mask_hash": _mask_hash(batch),
                    "lanes": sorted({s.lane for s, _, _ in items}),
                    "policies": sorted({p.policy for p in plans}),
                    "sequences": [p.to_json() for p in plans],
                    **stats,
                    "mean_loss": round(mloss, 6),
                })
                if compare is not None:
                    comparisons.append(_compare(compare, ev, step, rank, mb))

                # token-level trace + learning ledger
                self.trace.record(step=step, stage=stage["stage"], batch=batch,
                                  per_token_loss=per_token, plans=plans, lane=lane,
                                  ckpt_before=checkpoint_id,
                                  ckpt_after=self._next_ckpt_id(step),
                                  model_age=model.age_tokens, opus_ids=dec_ids,
                                  pass_numbers=ev["repeated_pass_numbers"],
                                  shard_manifest=self.manifests)
                align = model.alignment(grads)
                for s, r, p in items:
                    self.learning.observe(
                        shard_ids=p.shard_ids(), lane=s.lane, step=step, mean_loss=mloss,
                        grad_norm=gnorm, grad_align=align, opus_score=r["opus_score"],
                        loss_tokens=int(mask.sum()), phase=phase,
                        pass_numbers=[ev["repeated_pass_numbers"].get(u, 1)
                                      for u in p.unit_ids()],
                        tokens_consumed=model.age_tokens)
                model.age_tokens += stats["positions"]

            t0 = time.perf_counter()
            for k in grads_acc:
                grads_acc[k] /= max(1, n_acc)
            model.step_adam(grads_acc)
            perf.compute_seconds += time.perf_counter() - t0
            perf.steps += 1
            if mode == "train":
                self.step_losses[step] = round(float(np.mean(step_loss)), 6)
            perf.wall_seconds += time.perf_counter() - t_step

            # the two mandated events belong at the point they first happen, not in the
            # crash handler; totals for the whole run are logged again at the end
            if mode == "train" and not self._announced:
                self._announced = True
                self.log.event("batches packed",
                               f"first optimizer step: {len(groups)} microbatches x "
                               f"{self.cfg.micro} sequences of {self.cfg.seq_len} tokens, "
                               f"policies {sorted({p.policy for _, _, p in chosen})}")
                self.log.event("OPUS decisions recorded",
                               f"{len(opus.decisions)} candidates scored at step {step}: " +
                               ", ".join(f"{k}={v}" for k, v in sorted(opus.counts.items())))

            if mode == "train" and (step + 1) % self.cfg.validate_every == 0:
                self._validate(model, step, record_branch)

            # checkpoint AFTER the step is fully recorded, so ledger_offset is exact
            if mode == "train" and checkpoint_dir is not None and \
                    (step + 1) % self.cfg.checkpoint_every == 0:
                checkpoint_id = self._checkpoint(model, opus, step, record_branch,
                                                 data_branch, checkpoint_dir)

            if crash_at is not None and step + 1 == crash_at:
                raise SimulatedCrash(f"deliberate crash after step {step}")
        return model, opus, checkpoint_id, comparisons

    # -- helpers --
    def _next_stage(self, step):
        i, _ = self.schedule.stage_at(step)
        return self.schedule.stages[min(i + 1, len(self.schedule.stages) - 1)]["stage"]

    def _next_ckpt_id(self, step):
        nxt = ((step // self.cfg.checkpoint_every) + 1) * self.cfg.checkpoint_every
        return f"ckpt_step{nxt:04d}"

    def _checkpoint(self, model, opus, step, branch, data_branch, ckpt_dir):
        cid = f"ckpt_step{step + 1:04d}"
        expected = self._predict_next(step + 1, data_branch)
        offset = self.ledger.offset
        save_checkpoint(ckpt_dir / f"{cid}.npz", model, opus, step=step + 1,
                        ledger_offset=offset, branch=branch, data_branch=data_branch,
                        expected_next=expected,
                        meta={"stage": self.schedule.stage_at(step)[1]["stage"],
                              "age_tokens": model.age_tokens,
                              "mean_loss": self.step_losses.get(step)})
        rec = {"checkpoint_id": cid, "step": step + 1, "ledger_offset": offset,
               "branch_id": branch, "data_branch_id": data_branch,
               "effective_events_at_checkpoint": self.ledger.effective_count(branch, step),
               "expected_next_batch": expected,
               "parent_checkpoint": self.checkpoints[-1]["checkpoint_id"] if self.checkpoints else None,
               "mean_loss": self.step_losses.get(step),
               "age_tokens": model.age_tokens}
        self.checkpoints.append(rec)
        self.ledger.note("checkpoint", **rec)
        self.log.event("checkpoint saved", f"{cid} at ledger offset {offset}")
        self.log.check("checkpoint_saved", True, f"{cid} offset={offset}")
        return cid

    def _predict_next(self, step, data_branch):
        """What the next batch MUST be. Written into the checkpoint before any crash, so the
        resume proof compares against a prediction made in the past, not a post-hoc replay."""
        slots, _ = packing.plan_batch(self.schedule, self.pool, self.cfg.seed, data_branch,
                                      step, self.cfg.ranks, self.cfg.micro,
                                      self.cfg.candidates_per_slot)
        return {"global_step": step,
                "slot_ids": [s.slot_id for s in slots],
                "lanes": [s.lane for s in slots],
                "candidate_span_ids": [[c.span_ids() for c in s.candidates] for s in slots],
                "plan_digest": hashlib.sha256(json.dumps(
                    [[c.span_ids() for c in s.candidates] for s in slots],
                    sort_keys=True).encode()).hexdigest()}

    def _validate(self, model, step, branch):
        """Validation shards are read for evaluation and never become gradient-bearing."""
        if not getattr(self, "validation", None):
            return
        losses = []
        for b in self.validation:
            ml, _ = model.mean_loss(b)
            losses.append(ml)
        self.ledger.note("validation_eval", global_step=step, branch_id=branch,
                         gradient_bearing=False, sequences=len(self.validation),
                         mean_loss=round(float(np.mean(losses)), 6),
                         shard_ids=sorted(self.validation_shards))
        self.log.info(f"  validation @step{step}: mean_loss={np.mean(losses):.4f} "
                      f"(gradient_bearing=false)")


def _mask_hash(batch):
    return hashlib.sha256(np.ascontiguousarray(
        batch.attention_mask()).tobytes()).hexdigest()[:32]


def _compare(original_index, ev, step, rank, mb):
    """Replay comparison: original vs replayed batch identity for one microbatch."""
    key = (step, rank, mb)
    o = original_index.get(key)
    return {
        "global_step": step, "rank": rank, "microbatch_id": mb,
        "original_batch_hash": o["batch_hash"] if o else None,
        "replay_batch_hash": ev["batch_hash"],
        "hash_match": bool(o and o["batch_hash"] == ev["batch_hash"]),
        "original_span_ids": o["token_span_ids"] if o else None,
        "replay_span_ids": ev["token_span_ids"],
        "span_match": bool(o and o["token_span_ids"] == ev["token_span_ids"]),
        "loss_mask_hash_match": bool(o and o["loss_mask_hash"] == ev["loss_mask_hash"]),
        "original_batch_id": [o["global_step"], o["rank"], o["microbatch_id"]] if o else None,
    }


# ---- audit (§14) --------------------------------------------------------
def audit(events, opus_decisions, step_losses, token_range, spike_lookback=3):
    """Reconstructs which data trained an interval, and what OPUS did before a loss spike."""
    by_step = {}
    cum = 0
    for e in sorted(events, key=lambda e: (e["global_step"], e["rank"], e["microbatch_id"])):
        n = e["batch_stats"]["positions"]
        by_step.setdefault(e["global_step"], {"token_start": cum, "events": []})
        by_step[e["global_step"]]["events"].append(e)
        cum += n
        by_step[e["global_step"]]["token_end"] = cum
    a, b = token_range
    in_range = [e for s, v in by_step.items() if v["token_end"] > a and v["token_start"] < b
                for e in v["events"]]
    shards, lanes = {}, {}
    for e in in_range:
        for sid in e["shard_ids"]:
            shards[sid] = shards.get(sid, 0) + 1
        for lane in e["mixture_lane"].split(","):
            lanes[lane] = lanes.get(lane, 0) + e["batch_stats"]["loss_bearing_positions"]

    losses = sorted(step_losses.items(), key=lambda kv: int(kv[0]))
    spike_step, spike_delta = None, 0.0
    for (s0, l0), (s1, l1) in zip(losses, losses[1:]):
        if l1 - l0 > spike_delta:
            spike_step, spike_delta = int(s1), l1 - l0
    pre = [d for d in opus_decisions
           if spike_step is not None and spike_step - spike_lookback <= d["global_step"] <= spike_step]
    return {
        "question_1": f"which shards influenced the model between tokens {a} and {b}?",
        "token_range": [a, b],
        "microbatches_in_range": len(in_range),
        "steps_in_range": sorted({e["global_step"] for e in in_range}),
        "shard_influence": dict(sorted(shards.items(), key=lambda kv: -kv[1])),
        "loss_bearing_tokens_by_lane": dict(sorted(lanes.items(), key=lambda kv: -kv[1])),
        "question_2": "which OPUS-selected batches appeared before the largest loss spike?",
        "largest_loss_spike": {"step": spike_step, "delta": round(spike_delta, 6),
                               "lookback_steps": spike_lookback},
        "opus_decisions_before_spike": {
            "total": len(pre),
            "by_status": {s: sum(1 for d in pre if d["status"] == s)
                          for s in ("accepted", "rejected", "deferred", "protected")},
            "accepted_lanes": sorted({d["capability_lane"] for d in pre
                                      if d["status"] in ("accepted", "protected")}),
            "candidate_ids": [d["candidate_id"] for d in pre][:40],
        },
    }
