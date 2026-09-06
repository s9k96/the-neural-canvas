"""
Packing policies, batch assembly, and the pure planner (§3, §4, §5).

The whole recovery story rests on one property of this module:

    plan_batch(schedule, pool, seed, branch, step)  is a PURE FUNCTION of its arguments.

No cursor, no iterator state, no RNG carried between steps -- every draw is seeded from
(seed, branch, step, lane, slot). Resume, replay and fork are therefore the same call with
the same arguments, and "no skipped or repeated batch" is a property of the design rather
than a property of careful state restoration.

Planning is metadata-only (fast, touches no payload). `materialize` is what reads shard
bytes through the ShardStore -- which is where the cache-hit and read-latency numbers in
performance.json come from.

Loss-mask convention: position t predicts token t+1, so
    loss_mask[t] = is_target(t+1) AND segment(t+1) == segment(t)
The last position of every segment therefore bears no loss, and loss can never cross a
segment boundary or land on padding.
"""
import hashlib
import json

import numpy as np

from .shards import EOS, PAD

POLICIES = ["pad_only", "concat_chop", "greedy", "best_fit", "structure_preserving", "long_context"]

# The correct policy depends on the data type (§5).
LANE_POLICY = {
    "general_web": "concat_chop",       # plain prose tolerates concatenation
    "indic": "concat_chop",
    "code": "best_fit",                 # keep whole files, fill tightly
    "stem_math": "structure_preserving",
    "reasoning": "structure_preserving",  # a trace needs room to finish its argument
    "agentic": "structure_preserving",  # tool observations must not leak across samples
    "long_context": "long_context",     # every unused position wastes a high-value slot
}
DATALOADER_VERSION = "s6-dataloader-1"
MAX_PLACEMENTS = 8


def _rng(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(h[:8], "big")))


# ---- unit pool ----------------------------------------------------------
class Unit:
    """One drawable piece of a shard: either a whole sample (structured / file-level) or a
    contiguous token window (plain pretraining). `segments` are (start, end, is_target) in
    shard-flat coordinates."""
    __slots__ = ("unit_id", "shard_id", "lane", "start", "end", "segments", "sample_index",
                 "language", "doc_id", "reserved")

    def __init__(self, unit_id, shard_id, lane, start, end, segments, sample_index,
                 language, doc_id, reserved):
        self.unit_id, self.shard_id, self.lane = unit_id, shard_id, lane
        self.start, self.end, self.segments = start, end, segments
        self.sample_index, self.language, self.doc_id = sample_index, language, doc_id
        self.reserved = reserved

    @property
    def n_tokens(self):
        return self.end - self.start

    @property
    def n_target(self):
        return sum(e - a for a, e, t in self.segments if t)


class Pool:
    """Deterministic, ordered enumeration of every drawable unit per lane."""

    def __init__(self, shards, seq_len):
        self.seq_len = seq_len
        self.by_lane = {}
        self.units = {}
        self.shard_manifest = {}
        for s in sorted(shards, key=lambda s: s.shard_id):
            m = s.manifest
            if m["admission"]["verdict"] != "ADMITTED" or m.get("never_train"):
                continue
            self.shard_manifest[s.shard_id] = m
            for u in self._enumerate(s, seq_len):
                self.by_lane.setdefault(u.lane, []).append(u)
                self.units[u.unit_id] = u
        for lane in self.by_lane:
            self.by_lane[lane].sort(key=lambda u: u.unit_id)

    @staticmethod
    def _enumerate(shard, seq_len):
        m, segs = shard.manifest, shard.segs
        lane, policy = shard.lane, LANE_POLICY[shard.lane]
        lang = (m["language_and_script"] or ["unknown"])[0]
        out = []
        if policy in ("concat_chop", "long_context"):
            # windows over the shard's flat token stream; EOS tokens inside mark doc
            # boundaries, which is exactly what §4 describes for plain pretraining
            n = int(m["token_count"])
            for i, a in enumerate(range(0, n, seq_len)):
                b = min(n, a + seq_len)
                if b - a < 32:     # a tail shorter than this is not worth a window
                    continue
                out.append(Unit(f"{shard.shard_id}#w{i:04d}", shard.shard_id, lane, a, b,
                                [(a, b, 1)], None, lang, None, m["reserved_for_anneal"]))
        else:
            # Sample-level units for SFT / agentic / reasoning data: one unit per MODEL turn,
            # preceded by as much of its immediately-preceding context as the window allows
            # (left truncation, the standard SFT treatment). This is the §5 masking rule made
            # concrete -- the user request and tool observations are carried as context so the
            # response is conditioned on them, and only the model's turn bears loss. A turn is
            # never split away from its own prompt, and the unit is always a contiguous token
            # range, so nothing is stitched together that was not adjacent in the document.
            # ponytail: with seq_len 256 only the nearest context survives for a long
            # trajectory. Raise seq_len (cost is O(T^2) in the attention mask) to keep more.
            for si in sorted({int(r[0]) for r in segs}):
                rows = sorted([r for r in segs if int(r[0]) == si], key=lambda r: int(r[1]))
                doc = next((d for d in m["docs"] if d["sample_index"] == si), None)
                sample_start = int(rows[0][1])
                for ti, r in enumerate(rows):
                    a, b, t = int(r[1]), int(r[2]), int(r[3])
                    if not t:
                        continue                       # context turns are not units of their own
                    if b - a > seq_len:                # a model turn longer than the window
                        b = a + seq_len
                    room = seq_len - (b - a)
                    start = max(sample_start, a - room)
                    if b - start < 8:
                        continue
                    parts = ([(start, a, 0)] if a > start else []) + [(a, b, 1)]
                    out.append(Unit(f"{shard.shard_id}#s{si:04d}t{ti:02d}", shard.shard_id,
                                    lane, start, b, parts, si, lang,
                                    doc["doc_id"] if doc else None, m["reserved_for_anneal"]))
        return out

    def eligible(self, lane, in_anneal):
        """The anneal reserve is invisible to the selector until the anneal begins (S5 §6)."""
        return [u for u in self.by_lane.get(lane, ()) if in_anneal or not u.reserved]

    def stats(self):
        out = {}
        for lane, units in sorted(self.by_lane.items()):
            out[lane] = {"units": len(units),
                         "tokens": sum(u.n_tokens for u in units),
                         "loss_bearing_tokens": sum(u.n_target for u in units),
                         "reserved_units": sum(1 for u in units if u.reserved),
                         "reserved_tokens": sum(u.n_tokens for u in units if u.reserved),
                         "policy": LANE_POLICY[lane]}
        return out


# ---- packing policies ---------------------------------------------------
class SeqPlan:
    """A planned sequence: where each placement lands in the fixed-length window."""

    def __init__(self, lane, policy, seq_len):
        self.lane, self.policy, self.seq_len = lane, policy, seq_len
        self.placements = []          # dicts: unit_id, shard_id, src_a, src_b, dst_a, dst_b, segment_id
        self.truncations = 0
        self.used = 0
        self.loss_tokens = 0          # planned loss-bearing estimate, for the OPUS record

    def place(self, unit, segment_id, limit=None):
        room = self.seq_len - self.used if limit is None else min(limit, self.seq_len - self.used)
        if room <= 0:
            return False
        take = min(unit.n_tokens, room)
        if take < unit.n_tokens:
            self.truncations += 1
        self.placements.append({
            "unit_id": unit.unit_id, "shard_id": unit.shard_id, "doc_id": unit.doc_id,
            "src_a": unit.start, "src_b": unit.start + take,
            "dst_a": self.used, "dst_b": self.used + take,
            "segment_id": segment_id,
            "span_id": f"{unit.shard_id}#{unit.start}-{unit.start + take}",
        })
        self.used += take
        self.loss_tokens += (unit.n_target if take == unit.n_tokens
                             else int(unit.n_target * take / max(1, unit.n_tokens)))
        return True

    @property
    def utilization(self):
        return self.used / self.seq_len

    def span_ids(self):
        return [p["span_id"] for p in self.placements]

    def unit_ids(self):
        return [p["unit_id"] for p in self.placements]

    def shard_ids(self):
        return sorted({p["shard_id"] for p in self.placements})

    def to_json(self):
        return {"lane": self.lane, "policy": self.policy, "used_positions": self.used,
                "pad_positions": self.seq_len - self.used,
                "utilization": round(self.utilization, 4), "truncations": self.truncations,
                "segments": len({p["segment_id"] for p in self.placements}),
                "placements": self.placements}


def pack(pool, lane, primary, rng, seq_len, policy=None, in_anneal=False):
    """Fill one fixed-length sequence starting from `primary`, per the lane's policy."""
    policy = policy or LANE_POLICY[lane]
    plan = SeqPlan(lane, policy, seq_len)
    units = pool.eligible(lane, in_anneal)

    if policy == "pad_only":
        plan.place(primary, 0)
        return plan
    if policy in ("concat_chop", "long_context"):
        # the unit is already a window over a concatenated token stream: one attention
        # segment, boundaries carried by the EOS tokens inside it
        plan.place(primary, 0)
        return plan
    if policy == "structure_preserving":
        # never split a sample across sequences, never let two samples see each other:
        # each sample is its own attention segment (§5)
        plan.place(primary, 0)
        seg = 1
        start = int(rng.integers(0, max(1, len(units))))
        for k in range(len(units)):
            if seg >= MAX_PLACEMENTS or plan.used >= seq_len:
                break
            u = units[(start + k) % len(units)]
            if u.unit_id == primary.unit_id or u.n_tokens > seq_len - plan.used:
                continue        # only whole samples -- a partial trajectory teaches nothing
            plan.place(u, seg)
            seg += 1
        return plan
    if policy in ("greedy", "best_fit"):
        plan.place(primary, 0)
        seg = 1
        order = (sorted(units, key=lambda u: (-u.n_tokens, u.unit_id)) if policy == "best_fit"
                 else units)
        start = int(rng.integers(0, max(1, len(order))))
        for k in range(len(order)):
            if seg >= MAX_PLACEMENTS or plan.used >= seq_len:
                break
            u = order[(start + k) % len(order)]
            if u.unit_id == primary.unit_id:
                continue
            if u.n_tokens <= seq_len - plan.used:      # first-fit vs tightest-fit ordering
                plan.place(u, seg)
                seg += 1
        return plan
    raise ValueError(f"unknown policy {policy}")


# ---- the planner --------------------------------------------------------
class Slot:
    __slots__ = ("slot_id", "lane", "index", "rank", "microbatch", "candidates", "floor_protected")

    def __init__(self, slot_id, lane, index, rank, microbatch, candidates, floor_protected):
        self.slot_id, self.lane, self.index = slot_id, lane, index
        self.rank, self.microbatch = rank, microbatch
        self.candidates = candidates
        self.floor_protected = floor_protected


def plan_batch(schedule, pool, seed, branch, step, ranks, micro, candidates_per_slot=2):
    """Pure. Returns (slots, meta). Every slot carries its OPUS candidate slate, each
    candidate being a fully planned sequence -- so OPUS scores exactly what would be
    consumed, not a proxy for it.

    A slot is `floor_protected` when the allocator granted it to satisfy a protected lane's
    rolling-window floor. That flag is a pure function of the step, so OPUS's floor override
    reproduces identically on replay -- it must not depend on how much of the lane the
    process happens to have served so far.
    """
    _, stage = schedule.stage_at(step)
    in_anneal = stage["stage"].startswith("4")
    quota = schedule.quota_for_step(step)
    floor_slots = schedule.floor_slots(step)
    slots, i = [], 0
    for lane in sorted(quota):
        units = pool.eligible(lane, in_anneal)
        if not units:
            continue
        for k in range(quota[lane]):
            cands = []
            for c in range(candidates_per_slot):
                crng = _rng(seed, branch, step, lane, k, c)
                primary = units[int(crng.integers(0, len(units)))]
                cands.append(pack(pool, lane, primary, crng, schedule.seq_len,
                                  in_anneal=in_anneal))
            group = i // micro                      # one microbatch per group
            slots.append(Slot(f"s{step:05d}-{lane}-{k}", lane, k,
                              group % ranks, group // ranks, cands,
                              k < floor_slots.get(lane, 0)))
            i += 1
    return slots, {"stage": stage["stage"], "in_anneal": in_anneal, "quota": quota,
                   "floor_slots": floor_slots, "mixture": schedule.mixture_at(step)}


# ---- materialisation ----------------------------------------------------
class Batch:
    """One microbatch: the arrays the training step consumes, plus the provenance the
    ledger records."""

    def __init__(self, seq_plans, tokens, loss_mask, segment_ids, position_ids, remap,
                 is_target):
        self.plans = seq_plans
        self.tokens = tokens                # true tokenizer ids (audited)
        self.model_tokens = remap[tokens]   # frequency-truncated ids (what the model sees)
        self.loss_mask = loss_mask
        self.segment_ids = segment_ids
        self.position_ids = position_ids
        self.is_target = is_target          # per-token target flag, for the mask audit

    @property
    def shape(self):
        return self.tokens.shape

    def attention_mask(self):
        """Causal AND same-segment. Padding (segment -1) sees nothing and is seen by nothing."""
        B, T = self.tokens.shape
        causal = np.tril(np.ones((T, T), dtype=bool))
        same = self.segment_ids[:, :, None] == self.segment_ids[:, None, :]
        valid = self.segment_ids >= 0
        return causal[None] & same & valid[:, None, :] & valid[:, :, None]

    def batch_hash(self):
        h = hashlib.sha256()
        for a in (self.tokens, self.loss_mask.astype(np.int8),
                  self.segment_ids, self.position_ids):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(json.dumps([p.span_ids() for p in self.plans], sort_keys=True).encode())
        return h.hexdigest()

    def loss_mask_hash(self):
        return hashlib.sha256(np.ascontiguousarray(
            self.loss_mask.astype(np.int8)).tobytes()).hexdigest()[:32]

    def stats(self):
        pad = int((self.segment_ids < 0).sum())
        return {"positions": int(self.tokens.size), "pad_positions": pad,
                "used_positions": int(self.tokens.size) - pad,
                "loss_bearing_positions": int(self.loss_mask.sum()),
                "context_positions": int(self.tokens.size) - pad - int(self.loss_mask.sum()),
                "utilization": round(1 - pad / self.tokens.size, 4),
                "loss_bearing_share": round(float(self.loss_mask.sum()) / self.tokens.size, 4),
                "segments": int(self.segment_ids.max()) + 1 if (self.segment_ids >= 0).any() else 0,
                "eos_inside": int((self.tokens == EOS).sum())}


def _target_bitmap(store, shard_id, n_tokens, cache):
    """Per-token target flag for a shard, from its segment table."""
    if shard_id in cache:
        return cache[shard_id]
    _, segs = store.get(shard_id)
    bm = np.zeros(n_tokens, dtype=bool)
    for _, a, b, t in segs:
        if t:
            bm[a:b] = True
    cache[shard_id] = bm
    return bm


def materialize(seq_plans, store, remap, seq_len, firewall=None, target_cache=None):
    """Read the planned spans out of the shards and build the arrays. This is the only
    place shard payloads are touched."""
    target_cache = target_cache if target_cache is not None else {}
    B = len(seq_plans)
    tokens = np.full((B, seq_len), PAD, dtype=np.int32)
    is_target = np.zeros((B, seq_len), dtype=bool)
    segment_ids = np.full((B, seq_len), -1, dtype=np.int32)
    position_ids = np.zeros((B, seq_len), dtype=np.int32)
    for b, plan in enumerate(seq_plans):
        for p in plan.placements:
            if firewall is not None:
                firewall.assert_trainable(p["shard_id"])   # serve-time enforcement
            toks, _ = store.get(p["shard_id"])
            a, z, da, dz = p["src_a"], p["src_b"], p["dst_a"], p["dst_b"]
            tokens[b, da:dz] = toks[a:z]
            bm = _target_bitmap(store, p["shard_id"], toks.size, target_cache)
            is_target[b, da:dz] = bm[a:z]
            segment_ids[b, da:dz] = p["segment_id"]
        # position ids restart at 0 in every attention segment
        for seg in sorted({p["segment_id"] for p in plan.placements}):
            idx = np.flatnonzero(segment_ids[b] == seg)
            position_ids[b, idx] = np.arange(idx.size, dtype=np.int32)
    # loss at t predicts token t+1: same segment, and t+1 must be a target token
    nxt_target = np.zeros_like(is_target)
    nxt_target[:, :-1] = is_target[:, 1:]
    same_seg = np.zeros_like(is_target)
    same_seg[:, :-1] = (segment_ids[:, 1:] == segment_ids[:, :-1]) & (segment_ids[:, :-1] >= 0)
    loss_mask = nxt_target & same_seg
    return Batch(seq_plans, tokens, loss_mask, segment_ids, position_ids, remap, is_target)


def mask_audit(batch):
    """Array-level verification of the three mask invariants, run on every real training
    batch. These counts are what the packing evidence row asserts are zero."""
    seg, lm = batch.segment_ids, batch.loss_mask
    pad = seg < 0
    on_pad = int((lm & pad).sum())
    # loss at t must land on a target token at t+1
    nxt_target = np.zeros_like(batch.is_target)
    nxt_target[:, :-1] = batch.is_target[:, 1:]
    on_context = int((lm & ~nxt_target).sum())
    am = batch.attention_mask()
    cross = int((am & (seg[:, :, None] != seg[:, None, :])).sum())
    non_causal = int((am & ~np.tril(np.ones(am.shape[1:], dtype=bool))[None]).sum())
    bad_pos = 0
    for b in range(seg.shape[0]):
        for s in np.unique(seg[b]):
            if s < 0:
                continue
            idx = np.flatnonzero(seg[b] == s)
            if not np.array_equal(batch.position_ids[b, idx], np.arange(idx.size)):
                bad_pos += 1
    return {"loss_on_padding_positions": on_pad, "loss_on_context_positions": on_context,
            "cross_segment_visible_pairs": cross, "non_causal_visible_pairs": non_causal,
            "position_id_violations": bad_pos, "checked_batches": 1}


def policy_comparison(pool, seq_len, seed="policy-lab", per_policy=24):
    """Run every policy over the same data so §4/§5's utilisation claims are
    reconstructible rather than asserted. Metadata-only: no payload reads."""
    rows = []
    for lane in sorted(pool.by_lane):
        units = pool.eligible(lane, in_anneal=True)
        if not units:
            continue
        for policy in POLICIES:
            if policy in ("concat_chop", "long_context") and \
                    LANE_POLICY[lane] not in ("concat_chop", "long_context"):
                continue        # a window policy needs window units; skip incompatible pairs
            if policy not in ("concat_chop", "long_context") and \
                    LANE_POLICY[lane] in ("concat_chop", "long_context"):
                continue
            plans = []
            for i in range(min(per_policy, len(units))):
                rng = _rng(seed, lane, policy, i)
                plans.append(pack(pool, lane, units[int(rng.integers(0, len(units)))],
                                  rng, seq_len, policy=policy, in_anneal=True))
            used = sum(p.used for p in plans)
            rows.append({
                "lane": lane, "policy": policy, "sequences": len(plans),
                "utilization": round(used / (len(plans) * seq_len), 4),
                "pad_positions": len(plans) * seq_len - used,
                "truncations": sum(p.truncations for p in plans),
                "mean_segments_per_sequence": round(
                    sum(len({q["segment_id"] for q in p.placements}) for p in plans) / len(plans), 2),
                "is_lane_default": policy == LANE_POLICY[lane],
                "structure_safe": policy in ("pad_only", "structure_preserving", "long_context"),
            })
    return rows
