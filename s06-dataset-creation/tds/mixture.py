"""
Compiling the mixture timeline (§7) and OPUS selection (§10).

Session 5 described the mixture in human terms. This module converts it into per-step
quotas: which stage the run is in, which lanes are active, what share each must receive,
which floors are protected, what is reserved for the anneal, and how transitions warm up.

Two properties make the whole recovery story work:
  * quota_for_step(step) is a PURE function of step -- it reconstructs cumulative
    allocation from step 0 every time, so no scheduler state has to survive a crash.
  * the protected floor is enforced on a rolling window, not per batch. With 8 sequences
    per step a 4 % floor would round up to 1 whole sequence (12.5 %), which would silently
    inflate the scarce lanes it is meant to merely protect.

The stage vectors below are derived from S5 §7's described emphasis; the compiler reports
the token-weighted integral of the stages against the S5 headline mixture and flags any
lane that drifts more than TOLERANCE, so the deviation is stated rather than hidden.
"""
import json
from functools import lru_cache

import numpy as np

LANES = ["general_web", "code", "indic", "stem_math", "reasoning", "long_context", "agentic"]

# S5 §2 headline mixture (share of total pretraining tokens) -- the compliance target.
S5_HEADLINE = {"general_web": .34, "code": .24, "indic": .16, "stem_math": .12,
               "reasoning": .06, "long_context": .06, "agentic": .02}
# S5 §5 protected always-on floors. The selector may not cross these.
FLOORS = {"indic": .12, "reasoning": .04, "agentic": .02}
# S5 §6 anneal reserve: 3 % of the budget, quarantined before the selector can spend it.
ANNEAL_FRACTION = .03
RESERVE_MARGIN = 1.15          # S5 §6 keeps ~15 % margin so the anneal still has choice
TOLERANCE = .03                # allowed drift of integrated stages vs the S5 headline
FLOOR_WINDOW = 10              # steps over which a protected floor must hold

# S5 §7 curriculum. `frac` is the share of total tokens; `warmup` is the fraction of the
# stage over which the mixture blends in from the previous stage (never a hard step --
# V4 saw a ~150x gradient-norm spike from an abrupt Hindi-share change).
STAGES = [
    dict(stage="0-seed", frac=.05, warmup=.0, mixture={
        "general_web": .68, "code": .08, "indic": .16, "stem_math": .02,
        "reasoning": .04, "long_context": .00, "agentic": .02}),
    dict(stage="1-general-foundation", frac=.45, warmup=.15, mixture={
        "general_web": .44, "code": .22, "indic": .17, "stem_math": .08,
        "reasoning": .04, "long_context": .03, "agentic": .02}),
    dict(stage="2-capability-ramp", frac=.32, warmup=.15, mixture={
        "general_web": .18, "code": .32, "indic": .15, "stem_math": .19,
        "reasoning": .08, "long_context": .04, "agentic": .04}),
    dict(stage="3-long-context", frac=.15, warmup=.20, mixture={
        "general_web": .14, "code": .16, "indic": .14, "stem_math": .10,
        "reasoning": .08, "long_context": .34, "agentic": .04}),
    dict(stage="4-anneal", frac=.03, warmup=.10, mixture={   # S5 §6 anneal preset
        "general_web": .08, "code": .20, "indic": .28, "stem_math": .10,
        "reasoning": .18, "long_context": .08, "agentic": .08}),
]

PHASES = [(.25, "early"), (.60, "mid"), (.97, "late"), (1.01, "anneal")]


def model_phase(progress):
    for edge, name in PHASES:
        if progress < edge:
            return name
    return "anneal"


class Schedule:
    """The compiled timeline. Everything here is a pure function of the step index."""

    def __init__(self, total_steps, seq_per_step, seq_len):
        self.total_steps = total_steps
        self.seq_per_step = seq_per_step
        self.seq_len = seq_len
        self.tokens_per_step = seq_per_step * seq_len
        self.total_tokens = total_steps * self.tokens_per_step
        bounds, t = [], 0
        for s in STAGES:
            n = max(1, round(s["frac"] * total_steps))
            bounds.append(dict(s, step_start=t, step_end=t + n,
                               token_start=t * self.tokens_per_step,
                               token_end=(t + n) * self.tokens_per_step,
                               warmup_steps=max(0, round(s["warmup"] * n))))
            t += n
        bounds[-1]["step_end"] = max(bounds[-1]["step_end"], total_steps)
        bounds[-1]["token_end"] = bounds[-1]["step_end"] * self.tokens_per_step
        self.stages = bounds

    # ---- stage lookup / warmup blending ----
    def stage_at(self, step):
        for i, s in enumerate(self.stages):
            if s["step_start"] <= step < s["step_end"]:
                return i, s
        return len(self.stages) - 1, self.stages[-1]

    def mixture_at(self, step):
        """Effective lane shares at `step`, linearly blended across the stage boundary."""
        i, s = self.stage_at(step)
        mix = dict(s["mixture"])
        if i > 0 and s["warmup_steps"]:
            into = step - s["step_start"]
            if into < s["warmup_steps"]:
                a = (into + 1) / (s["warmup_steps"] + 1)
                prev = self.stages[i - 1]["mixture"]
                mix = {k: (1 - a) * prev[k] + a * mix[k] for k in LANES}
        tot = sum(mix.values())
        return {k: v / tot for k, v in mix.items()}

    # ---- per-step quotas ----
    @lru_cache(maxsize=None)
    def _alloc_through(self, step):
        """Deterministic water-filling allocation for steps 0..step inclusive.
        Returns (per-step allocation tuple for `step`, cumulative allocation tuple)."""
        if step < 0:
            return (), tuple([0] * len(LANES))
        _, cum = self._alloc_through(step - 1)
        cum = list(cum)
        target = [0.0] * len(LANES)
        for s in range(step + 1):
            m = self.mixture_at(s)
            for j, lane in enumerate(LANES):
                target[j] += m[lane] * self.seq_per_step
        # window floors: how much each protected lane still owes over the trailing window
        alloc = [0] * len(LANES)
        floor_debt = self._floor_debt(step, cum)
        for _ in range(self.seq_per_step):
            best, best_key = None, None
            for j, lane in enumerate(LANES):
                deficit = target[j] - (cum[j] + alloc[j])
                if self.mixture_at(step)[lane] <= 0 and floor_debt[j] <= 0:
                    continue
                # a protected lane still owing its floor outranks pure share deficit
                key = (1 if alloc[j] < floor_debt[j] else 0, round(deficit, 9), -j)
                if best_key is None or key > best_key:
                    best, best_key = j, key
            if best is None:
                best = int(np.argmax([target[j] - (cum[j] + alloc[j]) for j in range(len(LANES))]))
            alloc[best] += 1
        return tuple(alloc), tuple(c + a for c, a in zip(cum, alloc))

    def _floor_debt(self, step, cum):
        """Sequences each protected lane must still get this step for its rolling-window
        floor to hold. Window = FLOOR_WINDOW steps ending at `step`."""
        w0 = max(0, step - FLOOR_WINDOW + 1)
        prior = self._alloc_through(w0 - 1)[1] if w0 else tuple([0] * len(LANES))
        span = step - w0 + 1
        debt = [0] * len(LANES)
        for j, lane in enumerate(LANES):
            f = FLOORS.get(lane)
            if not f:
                continue
            need = int(np.ceil(f * span * self.seq_per_step))
            have = cum[j] - prior[j]
            debt[j] = max(0, min(self.seq_per_step, need - have))
        return debt

    def quota_for_step(self, step):
        """{lane: n_sequences} for this step. Pure: same step -> same quota, always."""
        alloc, _ = self._alloc_through(step)
        return {lane: alloc[j] for j, lane in enumerate(LANES) if alloc[j]}

    def floor_slots(self, step):
        """{lane: n} sequences this step that exist to satisfy a protected floor. OPUS is not
        allowed to reject these. Pure function of `step`, like the quota itself."""
        _, cum_prev = self._alloc_through(step - 1) if step else ((), tuple([0] * len(LANES)))
        debt = self._floor_debt(step, cum_prev)
        alloc, _ = self._alloc_through(step)
        return {lane: min(debt[j], alloc[j]) for j, lane in enumerate(LANES)
                if debt[j] and alloc[j]}

    # ---- reporting ----
    def integrated_mixture(self):
        tot = [0.0] * len(LANES)
        for s in range(self.total_steps):
            m = self.mixture_at(s)
            for j, lane in enumerate(LANES):
                tot[j] += m[lane]
        return {lane: tot[j] / self.total_steps for j, lane in enumerate(LANES)}

    def planned_shares(self):
        """Shares actually planned by the quota allocator (what the run will really serve)."""
        _, cum = self._alloc_through(self.total_steps - 1)
        n = sum(cum)
        return {lane: cum[j] / n for j, lane in enumerate(LANES)}

    def compile_report(self, supply_tokens, reserve_tokens):
        integ = self.integrated_mixture()
        planned = self.planned_shares()
        drift = {l: round(integ[l] - S5_HEADLINE[l], 4) for l in LANES}
        supply = {}
        for lane in LANES:
            demand = planned[lane] * self.total_tokens
            avail = supply_tokens.get(lane, 0)
            ratio = demand / avail if avail else float("inf")
            supply[lane] = {
                "demand_tokens": int(demand), "available_tokens": int(avail),
                "reserved_for_anneal_tokens": int(reserve_tokens.get(lane, 0)),
                "demand_over_supply": round(ratio, 4) if avail else None,
                # S5 §3 verdict rule
                "verdict": ("covered" if ratio <= 1 else
                            "needs_repetition" if ratio <= 4 else "must_synthesize"),
                "resolution": ("consume_unique" if ratio <= 1 else
                               "repeat_within_epoch_cap" if ratio <= 4 else
                               "synthesize_or_reduce_lane"),
            }
        anneal = [s for s in self.stages if s["stage"].startswith("4")][0]
        return {
            "run_shape": {"total_steps": self.total_steps, "seq_per_step": self.seq_per_step,
                          "sequence_length": self.seq_len,
                          "tokens_per_step": self.tokens_per_step,
                          "total_token_budget": self.total_tokens},
            "stages": self.stages,
            "protected_floors": FLOORS,
            "floor_window_steps": FLOOR_WINDOW,
            "anneal": {"fraction": ANNEAL_FRACTION, "stage": anneal["stage"],
                       "step_start": anneal["step_start"], "step_end": anneal["step_end"],
                       "reserve_margin": RESERVE_MARGIN,
                       "reserved_tokens_by_lane": {k: int(v) for k, v in reserve_tokens.items()}},
            "s5_headline_mixture": S5_HEADLINE,
            "integrated_stage_mixture": {k: round(v, 4) for k, v in integ.items()},
            "planned_allocator_shares": {k: round(v, 4) for k, v in planned.items()},
            "drift_vs_s5_headline": drift,
            "drift_tolerance": TOLERANCE,
            "lanes_outside_tolerance": [l for l in LANES if abs(drift[l]) > TOLERANCE],
            "supply_reconciliation": supply,
            "per_step_quotas": {str(s): self.quota_for_step(s) for s in range(self.total_steps)},
        }


# ---- anneal reserve ------------------------------------------------------
def reserve_for_anneal(shards, schedule):
    """Quarantine the best data per lane before the selector can spend it (S5 §6).
    Ranking is deterministic: provenance tier A first, then larger shards, then shard id.
    Reserve sizing = anneal demand for the lane x RESERVE_MARGIN."""
    anneal_mix = [s for s in STAGES if s["stage"].startswith("4")][0]["mixture"]
    anneal_tokens = ANNEAL_FRACTION * schedule.total_tokens
    reserved = {}
    for lane in LANES:
        want = anneal_mix[lane] * anneal_tokens * RESERVE_MARGIN
        pool = sorted([s for s in shards if s.lane == lane and
                       s.manifest["admission"]["verdict"] == "ADMITTED"],
                      key=lambda s: ("A" not in (s.manifest["license_and_provenance_tier"]["tiers"] or []),
                                     -s.manifest["token_count"], s.shard_id))
        got, cap = 0, max(0, len(pool) - 1)   # never quarantine a lane's last shard
        for s in pool[:cap]:
            if got >= want:
                break
            s.manifest["reserved_for_anneal"] = True
            got += s.manifest["token_count"]
        reserved[lane] = got
    return reserved


# ---- OPUS ---------------------------------------------------------------
class Opus:
    """OPUS sits inside the data path; its decisions are training events (§10).

    The proxy is the model's own mean loss on the candidate's loss-bearing tokens: high
    loss = high remaining learning value. It is scored under the CURRENT parameters, which
    is why replay must restore the scoring checkpoint -- and why replay then reproduces the
    same decisions rather than merely a similar stream.
    """
    PROXY_VERSION = "opus-proxy-1/mean-token-loss"
    KEEP_PERCENTILE = 60          # keep the top ~40 % (S5 §5: OPUS retains ~40 %)
    DEFER_BAND = 0.85             # within 15 % below threshold -> defer, not reject
    BUFFER = 64
    CANDIDATES_PER_SLOT = 2

    def __init__(self, scores=None):
        self.scores = list(scores or [])
        self.decisions = []
        self.counts = {"accepted": 0, "rejected": 0, "deferred": 0, "protected": 0}

    # -- state travels with the checkpoint so replay is exact --
    def state(self):
        return np.asarray(self.scores[-self.BUFFER:], dtype=np.float64)

    @classmethod
    def from_state(cls, arr):
        return cls(list(np.asarray(arr).ravel()))

    def threshold(self):
        if len(self.scores) < 8:
            return None                       # warmup: nothing to compare against yet
        return float(np.percentile(self.scores[-self.BUFFER:], self.KEEP_PERCENTILE))

    def observe(self, score):
        self.scores.append(float(score))

    def decide(self, cand, score, thr, lane_needs_floor, stage, next_stage, ckpt_id):
        """One candidate -> one §10 record. `cand` is the descriptor built by `select` from a
        planned sequence. Status is one of accepted/rejected/deferred/protected."""
        if thr is None:
            status, reason = "accepted", None
        elif score >= thr:
            status, reason = "accepted", None
        elif lane_needs_floor:
            status, reason = "protected", "protected_floor_override"
        elif score >= thr * self.DEFER_BAND:
            status, reason = "deferred", "below_threshold_scarce_lane"
        else:
            status, reason = "rejected", "low_proxy_utility"
        rec = {                                            # §10's 11 fields
            "candidate_id": cand["candidate_id"],
            "shard_ids": cand["shard_ids"],
            "capability_lane": cand["lane"],
            "curriculum_stage": stage,
            "scoring_checkpoint_id": ckpt_id,
            "proxy_version": self.PROXY_VERSION,
            "opus_score": round(float(score), 6),
            "status": status,
            "rejection_reason": reason,
            "protected_floor_override": status == "protected",
            "effective_token_estimate": int(cand["loss_tokens"] *
                                            (min(2.0, score / thr) if thr else 1.0)),
            # operational extras
            "threshold": round(thr, 6) if thr is not None else None,
            "global_step": cand["step"],
            "loss_bearing_tokens": cand["loss_tokens"],
            "backfill": False,
            "deferred_until_stage": next_stage if status == "deferred" else None,
        }
        return rec

    def select(self, slot, plans, scores, step, stage, next_stage, ckpt_id):
        """Score every candidate for one slot, record all of them, return the winner.

        A slot must be filled: if nothing clears the threshold and the lane is not
        floor-protected, the best candidate is promoted with backfill=true. That keeps the
        rejection rate honest instead of quietly dropping the slot.
        """
        thr = self.threshold()
        cands = [{"candidate_id": f"{slot.slot_id}-c{i}", "shard_ids": p.shard_ids(),
                  "lane": slot.lane, "loss_tokens": p.loss_tokens, "step": step}
                 for i, p in enumerate(plans)]
        recs = [self.decide(c, s, thr, slot.floor_protected, stage, next_stage, ckpt_id)
                for c, s in zip(cands, scores)]
        winner = None
        for r, p in zip(recs, plans):
            if r["status"] in ("accepted", "protected"):
                winner = (r, p)
                break
        if winner is None:
            best = int(np.argmax(scores))
            recs[best]["status"] = "accepted"
            recs[best]["backfill"] = True
            recs[best]["rejection_reason"] = "none:backfill_after_all_rejected"
            winner = (recs[best], plans[best])
        for s in scores:
            self.observe(s)
        for r in recs:
            self.counts[r["status"]] += 1
            self.decisions.append(r)
        return winner


def write_schedule(path, report):
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
