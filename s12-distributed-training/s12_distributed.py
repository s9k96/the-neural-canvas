# %% [markdown]
# # Session 12 — Distributed Training I: Data Parallel and ZeRO
#
# **ERA V5 · 32 virtual GPUs, one model, and four different answers to the question of what
# each card has to keep.**
#
# The assignment: *create 32 virtual GPUs, write a demo model that runs on top of them,
# simulate ZeRO-1, ZeRO-2 and ZeRO-3, and show how the memory and computation change.*
#
# One word in that sentence is doing a lot of work, and this harness takes a position on it.
# **Nothing here is simulated.** The 32 GPUs are 32 real operating-system processes in a
# `gloo` process group. The collectives are real sockets. Every byte reported as crossing the
# wire was handed to `dist.isend` by code in this repository, and counted there.
#
# That distinction matters because the whole session is a set of claims about *quantities* —
# 16 bytes per weight, 5.50, 3.75, 2.00; 2P, 2P, 2P, 3P. A simulation that computes those
# numbers from the same formula that predicted them has proved nothing. So the rule for this
# file is: **every number in the notes' tables is measured here, and then compared with the
# formula afterwards.**
#
# ---
#
# ### What I had to build myself, and why it turned out to be the interesting part
#
# `gloo` — the CPU backend, the only one available without GPUs — **does not implement
# `reduce_scatter`**. It raises `ProcessGroupGloo does not support reduce_scatter`. That is
# the one operation ZeRO-2 and ZeRO-3 are built on.
#
# So the ring from §4 is written out by hand here on `isend`/`irecv`, and that accident is why
# this submission can measure communication instead of quoting it. Section 2 below checks the
# hand-written ring against `gloo`'s own `all_reduce`, and checks §4's central claim —
# *a reduce-scatter followed by an all-gather is an all-reduce* — as an exact equality.

# %%
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch

import s12_ranks as R

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if HERE.name != "s12-distributed-training" and (HERE / "s12-distributed-training").is_dir():
    HERE = HERE / "s12-distributed-training"          # notebook launched from the repo root
ROOT = HERE.parent
OUT = HERE / "out"
RUNS = OUT / "runs"
OUT.mkdir(exist_ok=True)
RUNS.mkdir(exist_ok=True)

T_START = time.time()
EVIDENCE = {"meta": {
    "python": platform.python_version(), "torch": torch.__version__,
    "platform": platform.platform(), "cpu_count": __import__("os").cpu_count(),
}}
print(f"python {platform.python_version()} · torch {torch.__version__} · "
      f"{EVIDENCE['meta']['cpu_count']} cores · gloo available "
      f"{torch.distributed.is_gloo_available()}")


# %% [markdown]
# ## 1 · The arithmetic that starts the session
#
# §1 of the notes settles the question before any code runs. Each weight carries four things,
# and they add to sixteen bytes:
#
# | what is stored for one weight | bytes | dtype here |
# |---|---|---|
# | the weight, in the 16-bit format used for arithmetic | 2 | `bfloat16` |
# | its gradient | 2 | `bfloat16` |
# | a 32-bit copy of the weight, kept for accuracy | 4 | `float32` |
# | two running averages the optimizer keeps (Adam's m and v) | 8 | `float32` × 2 |
# | **total** | **16** | |
#
# Those dtypes are not a description of the model below — they *are* the model below. The
# engine in `s12_ranks.py` holds `bfloat16` parameters and gradients and a `float32` master
# copy with `float32` Adam moments, so the 16 bytes are the ones actually allocated.
#
# The consequence for V5, at 30 billion parameters:

# %%
PARAMS_30B = 30e9
GIB = 1024 ** 3
BYTES_PER_WEIGHT = {"weight_bf16": 2, "grad_bf16": 2, "master_fp32": 4, "adam_m_v": 8}
FULL_STATE = sum(BYTES_PER_WEIGHT.values())
CARD_GIB = 80e9 / GIB                                  # an 80 GB card, in GiB

state_30b_gib = PARAMS_30B * FULL_STATE / GIB
print(f"{FULL_STATE} bytes/weight × 30e9 weights = {PARAMS_30B * FULL_STATE / 1e9:.0f} GB "
      f"= {state_30b_gib:.1f} GiB")
print(f"an 80 GB card holds {CARD_GIB:.1f} GiB → {state_30b_gib / CARD_GIB:.2f} cards "
      f"just to store the state, before a single calculation")

# P, the unit every communication cost in this session is measured in: one complete copy of
# the parameters in 16-bit format.
P_30B_GB = PARAMS_30B * 2 / 1e9
print(f"P = {P_30B_GB:.0f} GB, so data parallelism's 2P moves {2 * P_30B_GB:.0f} GB "
      f"per GPU per step")

EVIDENCE["ledger"] = {
    "bytes_per_weight": BYTES_PER_WEIGHT, "total": FULL_STATE,
    "params_30b": PARAMS_30B, "state_30b_gib": round(state_30b_gib, 1),
    "card_gib": round(CARD_GIB, 1), "cards_needed": round(state_30b_gib / CARD_GIB, 2),
    "P_30b_gb": P_30B_GB,
}


# %% [markdown]
# ## 2 · The seven words, with numbers
#
# Seven words carry the whole session, and each one has a number attached. They are set out
# here rather than assumed, because every later section is denominated in them.
#
# * A **GPU** is one card. Here every GPU has 80 GB, which is 74.5 GiB — and GB against GiB is
#   not pedantry, it is the 6.6% that decides whether 68.1 GiB fits.
# * A **node** is one physical machine holding several GPUs, almost always eight.
# * **World size** is the total number of GPUs in the run. Four nodes of eight is a world size
#   of 32 — which is exactly the world size this assignment asks for, and not a coincidence.
# * A **process rank** is a GPU's index in that set, 0 to 31. (Unrelated to the rank of a
#   matrix.)
# * An **interconnect** is the wiring between GPUs. Inside a node it is NVLink at roughly
#   450 GB/s; between nodes it is a network cable, usually InfiniBand, at roughly 50 GB/s.
#   **Nine times slower**, and §12 below is entirely about what that ratio does.
# * A **collective** is an operation every GPU performs together, on data each holds a piece of.
# * **P** is one complete copy of the parameters in the compute format. For a 30B model in
#   16-bit, P is 60 GB. Every communication cost in this session is a multiple of P.

# %%
NODE_GPUS = 8
NVLINK_GBS, INFINIBAND_GBS, PCIE_GBS = 450.0, 50.0, 60.0

print(f"{'GPU':<14s} 80 GB = {CARD_GIB:.1f} GiB")
print(f"{'node':<14s} {NODE_GPUS} GPUs")
print(f"{'world size':<14s} {MAIN_WORLD if 'MAIN_WORLD' in dir() else 32} "
      f"= {32 // NODE_GPUS} nodes of {NODE_GPUS}")
print(f"{'ranks':<14s} 0 .. 31")
print(f"{'NVLink':<14s} {NVLINK_GBS:.0f} GB/s   (inside one node)")
print(f"{'InfiniBand':<14s} {INFINIBAND_GBS:.0f} GB/s   (between nodes) "
      f"\u2014 {NVLINK_GBS / INFINIBAND_GBS:.0f}x slower")
print(f"{'PCIe':<14s} {PCIE_GBS:.0f} GB/s   (GPU to system memory, \u00a714)")
print(f"{'P':<14s} {P_30B_GB:.0f} GB   (30e9 params x 2 bytes)")

EVIDENCE["terms"] = {
    "gpu_gb": 80, "gpu_gib": round(CARD_GIB, 2), "node_gpus": NODE_GPUS,
    "world_size": 32, "nodes": 32 // NODE_GPUS,
    "nvlink_gbs": NVLINK_GBS, "infiniband_gbs": INFINIBAND_GBS, "pcie_gbs": PCIE_GBS,
    "interconnect_ratio": NVLINK_GBS / INFINIBAND_GBS, "P_gb": P_30B_GB,
}


# %% [markdown]
# ## 3 · The demo model and the tokens it trains on
#
# **Continuity.** Same frozen Sarvam-1 tokenizer as Sessions 9, 10 and 11, and the same S6
# corpus. The model is the four-layer decoder those sessions used, copied rather than
# imported — each session ships standalone — with the muP switch dropped, since nothing here
# changes width.
#
# **Two scope limits, stated up front.** Thirty-two processes have to fit on one laptop, and
# every byte of state is held thirty-two times over under data parallelism. So this model is
# `d_model` 192 and the vocabulary is capped to the **top 2,048 Sarvam-1 ids** by frequency in
# this corpus, with the rest mapped to `UNK`. That is about 2.6M parameters.
#
# Neither limit touches what is being measured. Bytes *per parameter* and wire traffic *in
# multiples of P* are both ratios, and §7's projection to 30B is done from the measured ratio,
# not from this model's absolute size. What the small vocabulary does affect is one honest
# caveat recorded in §4 below: the largest single tensor is a larger share of a small model
# than of a real one, and that shows up in the transient buffers.

# %%
TOKENIZER_REPO = "sarvamai/sarvam-1"
FROZEN_TOKENIZER_SHA256 = "bb5115a36ddb956a4ee0fd534e9870dd69157835622aec9c53062896f883c072"
UNK, BOS, EOS, PAD = 0, 1, 2, 3
VOCAB_CAP = 2048


def load_tokenizer():
    import hashlib
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
    return Tokenizer.from_file(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()


tok, tok_sha = load_tokenizer()
V_FULL = tok.get_vocab_size()
assert tok_sha == FROZEN_TOKENIZER_SHA256, f"tokenizer drifted: {tok_sha}"
print(f"tokenizer verified · sha256 {tok_sha[:8]}… · vocab {V_FULL:,}")


def load_stream(limit_per_lane=200):
    corpus = ROOT / "s06-dataset-creation" / "corpus"
    ids, lanes = [], Counter()
    for lane_file in sorted(corpus.glob("*.jsonl")):
        if lane_file.stem == "eval_registry_docs":     # S6's eval firewall — never train text
            continue
        with lane_file.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= limit_per_lane:
                    break
                d = json.loads(line)
                text = "\n".join(s["text"] for s in d["segments"] if s.get("text"))
                if len(text) <= 200:
                    continue
                ids.extend([BOS] + tok.encode(text).ids + [EOS])
                lanes[d["lane"]] += 1
    return torch.tensor(ids, dtype=torch.long), lanes


STREAM, LANES = load_stream()
counts = Counter(STREAM.tolist())
keep = [i for i, _ in counts.most_common(VOCAB_CAP)]
for special in (UNK, BOS, EOS, PAD):
    if special not in keep:
        keep[-1] = special
remap = torch.zeros(V_FULL, dtype=torch.long)
for new_id, old_id in enumerate(sorted(keep)):
    remap[old_id] = new_id
STREAM_CAPPED = remap[STREAM]
coverage = sum(counts[i] for i in keep) / len(STREAM)

STREAM_PATH = RUNS / "stream.pt"
torch.save(STREAM_CAPPED, STREAM_PATH)
print(f"{len(STREAM):,} tokens · {sum(LANES.values())} documents · {len(LANES)} lanes")
print(f"capped vocab {VOCAB_CAP:,} covers {coverage:.2%} of token occurrences")

CFG = {"vocab_size": VOCAB_CAP, "d_model": 192, "n_layer": 4, "block_size": 128}
_probe = R.Model(R.Config(**CFG))
N_PARAMS = sum(p.numel() for p in _probe.parameters())
BIGGEST = max((p.numel(), n) for n, p in _probe.named_parameters())
del _probe
print(f"model: {N_PARAMS:,} parameters · largest tensor {BIGGEST[1]} "
      f"({BIGGEST[0]:,} = {BIGGEST[0] / N_PARAMS:.1%} of the model)")

EVIDENCE["corpus"] = {"tokens": len(STREAM), "docs": sum(LANES.values()),
                      "lanes": dict(sorted(LANES.items())), "vocab_full": V_FULL,
                      "vocab_cap": VOCAB_CAP, "cap_coverage": round(coverage, 5),
                      "tokenizer_sha256": tok_sha}
EVIDENCE["model"] = {"params": N_PARAMS, "largest_tensor": BIGGEST[1],
                     "largest_tensor_params": BIGGEST[0],
                     "largest_tensor_frac": round(BIGGEST[0] / N_PARAMS, 4), **CFG}


# %% [markdown]
# ## 4 · Thirty-two virtual GPUs
#
# The world size is 32 and each rank is a separate `python` process with `gloo`, one thread
# each. They talk over loopback TCP. In the vocabulary of §2 of the notes: world size 32,
# ranks 0 to 31, and an interconnect whose bandwidth is whatever the kernel's loopback gives.
#
# **Why the ranks are launched through a subprocess rather than `mp.spawn` here.** `spawn`
# re-imports the parent's `__main__` in every child. From a `# %%` script that has no
# `if __name__ == "__main__"` guard the children re-run the whole harness; from a notebook it
# is worse. Putting the spawn behind `s12_ranks.py`'s own entry point means this file — and
# the notebook built from it — only ever launches a subprocess, and behaves identically in
# both. That is also why the rank worker lives in a module of its own: `spawn` pickles it by
# module path, and a function defined in a notebook cell has no module path to pickle.

# %%
MAIN_WORLD = 32
MAIN_STEPS = 12
MICRO = 1                                  # sequences per GPU; global batch = 32 × 128 tokens
LR = 3e-4
STAGES = ["dp", "zero1", "zero2", "zero3"]
LABEL = {"dp": "data parallelism", "zero1": "ZeRO-1", "zero2": "ZeRO-2", "zero3": "ZeRO-3"}
_port = [29700]


def launch(mode, world, steps=MAIN_STEPS, micro=MICRO, accum=1, selfcheck=False, tag=""):
    """Run one configuration to completion on `world` real processes. Returns every rank."""
    _port[0] += 1
    opts = {"port": _port[0], "stream_path": str(STREAM_PATH), "cfg": CFG, "lr": LR,
            "micro": micro, "accum": accum, "steps": steps, "selfcheck": selfcheck, "tag": tag}
    spec = RUNS / f"spec_{mode}{tag}_w{world}.json"
    spec.write_text(json.dumps({"world": world, "mode": mode, "opts": opts,
                                "result_dir": str(RUNS)}))
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(HERE / "s12_ranks.py"), str(spec)],
                          capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"{mode} w{world} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    ranks = [json.loads((RUNS / f"rank_{mode}{tag}_w{world}_{r}.json").read_text())
             for r in range(world)]
    for r in ranks:
        r["launch_wall"] = time.time() - t0
    return ranks


def mean_losses(ranks):
    """The run's loss, which is not any single rank's loss.

    Each rank sees different sequences, so rank 0's number is the loss on one thirty-second
    of the batch. §3's claim is about the average, and averaging the per-rank losses is the
    same operation on the loss that the all-reduce performs on the gradient.
    """
    return [sum(c) / len(c) for c in zip(*(r["losses"] for r in ranks))]


def bpp(ranks, key):
    """Bytes per parameter, as the worst rank sees it — the one that sets the requirement."""
    return max(r[key] for r in ranks) / ranks[0]["params"]


def predicted_bpp(mode, world):
    """§7's ledger, as a formula: which of the 16 bytes this stage still replicates."""
    w, g, opt = 2, 2, 12
    if mode == "dp":
        return w + g + opt
    if mode == "zero1":
        return w + g + opt / world
    if mode == "zero2":
        return w + (g + opt) / world
    return (w + g + opt) / world                     # zero3


# %% [markdown]
# ## 5 · Do the hand-written collectives actually work?
#
# Everything downstream rests on the ring in `s12_ranks.Ring`, so it is checked before it is
# used, against two independent references.
#
# 1. **Against `gloo`.** The hand-written `all_reduce` is compared with `dist.all_reduce` on
#    the same data. They will not be bit-identical — floating-point addition is not
#    associative and the two use different orders — so what matters is that the disagreement
#    is at the level of `float32` rounding and not at the level of a wrong algorithm.
# 2. **Against §4's own claim.** *A reduce-scatter followed by an all-gather is an
#    all-reduce.* Both sides of that sentence run through the same ring in the same order, so
#    here the check is for **exact bitwise equality**, and anything less is a bug.
#
# The same cell also sums the integers 1…32 across the ranks twice, once in `bfloat16` and
# once in `float32`, which is the shortest demonstration I know of why §1's 4-byte master copy
# exists at all.

# %%
_selfcheck_ranks = launch("dp", MAIN_WORLD, steps=1, selfcheck=True)
CHECK = _selfcheck_ranks[0]["selfcheck"]
print(f"hand-written ring vs gloo all_reduce : max abs {CHECK['vs_gloo_max_abs']:.3e} "
      f"(relative {CHECK['vs_gloo_rel']:.2e})")
print(f"reduce_scatter + all_gather == all_reduce, bitwise : {CHECK['two_phase_identical']}")
print(f"\nsumming 1…{MAIN_WORLD} around the ring:")
print(f"  exact          {CHECK['exact_sum']:.0f}")
print(f"  in float32     {CHECK['fp32_ring_sum']:.0f}")
print(f"  in bfloat16    {CHECK['bf16_ring_sum']:.0f}   "
      f"← off by {abs(CHECK['bf16_ring_sum'] - CHECK['exact_sum']):.0f} "
      f"({abs(CHECK['bf16_ring_sum'] - CHECK['exact_sum']) / CHECK['exact_sum']:.2%})")
print("\nThat last line is §1's 32-bit master copy, justified in one measurement: accumulating")
print("32 contributions in 16-bit loses the low bits, and a weight update is exactly such an")
print("accumulation repeated for every step of training.")

EVIDENCE["collectives"] = dict(CHECK, world=MAIN_WORLD)


# %% [markdown]
# ## 6 · Data parallelism, and the two things it promises
#
# The simplest way to use 32 GPUs is to give each of them a complete copy of the model and a
# different portion of the data. Each reads its own examples, runs them through its own copy,
# and produces its own gradients — and those gradients *differ*, because each GPU saw different
# text. The copies are then brought back into agreement: every GPU sends its gradients to every
# other, all of them compute the average, and each applies that same average to its own copy.
#
# That arrangement makes two promises, and both are checkable rather than assumable.
#
# **One: the copies stay identical.** They started identical and applied an identical update,
# so they must remain identical — forever, with no drift. This is the property the whole
# arrangement rests on, because the moment two ranks disagree about the weights they are no
# longer training one model. Every rank hashes the entire model it can see, and the check is
# that 32 hex strings are the same string.
#
# **Two: the global batch is a product, and only the product matters.**
#
# > global batch = sequences per GPU × GPUs × accumulation steps
#
# Accumulation steps are repeats of the forward and backward pass before the weights move, used
# when the batch you want is larger than what fits at once. If that formula is real, then any
# factorization of the same global batch is the same training run. So this runs the same 32
# sequences two ways: **32 GPUs × 1 sequence × 1 step**, and **8 GPUs × 2 sequences × 2
# accumulation steps** — the second being exactly the configuration §16 records the previous
# run using.

# %%
RUNS_MAIN = {"dp": launch("dp", MAIN_WORLD)}

replica_hashes = {r["weight_sha256"] for r in RUNS_MAIN["dp"]}
print(f"promise 1 — after {MAIN_STEPS} steps, the {MAIN_WORLD} replicas hash to "
      f"{len(replica_hashes)} distinct value(s):")
print(f"  {sorted(replica_hashes)[0][:32]}…  on all {MAIN_WORLD} ranks"
      if len(replica_hashes) == 1 else f"  DIVERGED: {replica_hashes}")

alt = launch("dp", NODE_GPUS, steps=MAIN_STEPS, micro=2, accum=2, tag="_accum")
w_32 = torch.load(RUNS / f"weights_dp_w{MAIN_WORLD}.pt")
w_alt = torch.load(RUNS / f"weights_dp_accum_w{NODE_GPUS}.pt")
loss_32 = mean_losses(RUNS_MAIN["dp"])
loss_alt = mean_losses(alt)
factor_rel = abs(loss_alt[-1] - loss_32[-1]) / loss_32[-1]

print(f"\npromise 2 — the same global batch of {alt[0]['global_batch']} sequences, factored twice:")
print(f"  {'32 GPUs x 1 seq x 1 step':<34s} final loss {loss_32[-1]:.6f}")
print(f"  {'8 GPUs x 2 seq x 2 accum steps':<34s} final loss {loss_alt[-1]:.6f}")
print(f"  relative difference {factor_rel:.2e} · max weight difference "
      f"{float((w_alt - w_32).abs().max()):.3e}")
print("\nNot bitwise, and it should not be: 32 ranks summing one sequence each around a ring")
print("and 8 ranks summing four apiece are different orders of the same addition. The point is")
print("that the arrangement of the hardware does not change what the model learns — only the")
print("product does, which is what makes a recipe portable from 8 GPUs to 32.")

EVIDENCE["data_parallel"] = {
    "replica_hashes_distinct": len(replica_hashes),
    "replica_hash": sorted(replica_hashes)[0],
    "global_batch": alt[0]["global_batch"],
    "factorizations": [
        {"gpus": MAIN_WORLD, "seq_per_gpu": MICRO, "accum": 1,
         "loss_curve": loss_32, "final_loss": loss_32[-1]},
        {"gpus": NODE_GPUS, "seq_per_gpu": 2, "accum": 2,
         "loss_curve": loss_alt, "final_loss": loss_alt[-1]},
    ],
    "loss_rel": factor_rel,
    "weight_max_abs": float((w_alt - w_32).abs().max()),
}

# %% [markdown]
# ## 7 · The four stages, measured
#
# Now the substance. The same model, the same seed, the same tokens in the same order, run
# four times on 32 processes — changing only what each rank is allowed to keep.
#
# * **data parallelism** — every rank holds all 16 bytes for every weight.
# * **ZeRO-1** — the 12 optimizer bytes are split 32 ways. Each rank updates its own slice
#   and the updated slices are shared back out.
# * **ZeRO-2** — the gradient is split too. A rank only ever needs the gradients for the slice
#   it updates, so each gradient is reduced to its owner *the moment backward finishes
#   producing it* and discarded everywhere else.
# * **ZeRO-3** — the weights themselves are split. A layer's weights are collected from all
#   ranks when the forward pass reaches it, used, and discarded again; then collected a second
#   time on the way back.
#
# ### Two numbers per stage, not one
#
# The notes' table is about the state a rank *holds*. A running implementation also needs a
# buffer for whatever collective is in flight, which the table does not mention and no real
# implementation escapes. So both are reported:
#
# * **ladder** — everything resident at the instant `backward()` returns: weights, whichever
#   gradients the stage kept, optimizer state. No collective in flight. This is the number
#   §6's table is about.
# * **peak** — the same, plus the largest buffer any single collective needs.
#
# On this model the gap is larger than it would be on a real one, for a reason worth naming:
# the buffer is sized by the *largest single tensor*, and in a 2.6M-parameter model with a
# 2,048-token vocabulary the embedding matrix is a far bigger share of the whole than it would
# be at 30B. The gap is an artifact of the toy, the ladder is not.

# %%
for mode in STAGES:                       # dp already ran in §6
    if mode not in RUNS_MAIN:
        RUNS_MAIN[mode] = launch(mode, MAIN_WORLD)
    print(f"{LABEL[mode]:<20s} done in {RUNS_MAIN[mode][0]['launch_wall']:5.1f}s")

print(f"\n{'stage':<20s} {'measured':>9s} {'predicted':>10s} {'peak':>8s} "
      f"{'30B/GPU':>10s}  (bytes per weight, {MAIN_WORLD} GPUs)")
STAGE_TABLE = {}
for mode in STAGES:
    ranks = RUNS_MAIN[mode]
    ladder, peak = bpp(ranks, "ladder_bytes"), bpp(ranks, "peak_bytes")
    pred = predicted_bpp(mode, MAIN_WORLD)
    gib30 = PARAMS_30B * ladder / GIB
    STAGE_TABLE[mode] = {"ladder_bpp": ladder, "peak_bpp": peak, "predicted_bpp": pred,
                         "gib_30b": gib30}
    print(f"{LABEL[mode]:<20s} {ladder:9.4f} {pred:10.4f} {peak:8.2f} {gib30:9.1f}G")

spread = {mode: (max(r["ladder_bytes"] for r in RUNS_MAIN[mode])
                 / min(r["ladder_bytes"] for r in RUNS_MAIN[mode])) for mode in STAGES}
print(f"\nload balance (heaviest rank / lightest rank): "
      f"{', '.join(f'{m} {spread[m]:.4f}' for m in STAGES)}")
print("Every rank receives a different slice and none of them sits idle — §6's sentence,")
print("checked as a number rather than assumed.")


# %% [markdown]
# ## 8 · What it costs on the wire
#
# §6's claim is precise and slightly surprising: **stages 1 and 2 cost nothing extra**, and
# only stage 3 raises the bill, from 2P to 3P. The reason is §4's equivalence. Data
# parallelism's all-reduce already *is* a reduce-scatter followed by an all-gather; ZeRO-1 and
# ZeRO-2 run those same two phases and simply keep the intermediate slice instead of throwing
# it away. Stage 3 adds a gather of the weights in the forward pass and another in the
# backward pass.
#
# ### A correction the measurement forces, which the notes round away
#
# A ring collective does not move P. It moves **P·(N−1)/N** — each rank sends every chunk but
# its own, N−1 of the N chunks. At N=32 that is 0.969P, so data parallelism's real cost is
# 1.94P and ZeRO-3's is 2.91P. The notes' 2P and 3P are the large-N limit of this, and at 32
# GPUs the limit is 3% away. I would rather report 1.94P and explain it than report 2P and be
# quietly wrong.

# %%
print(f"{'stage':<20s} {'measured':>10s} {'ring':>9s} {'notes':>7s}   {'30B, per GPU per step':>22s}")
COMM_TABLE = {}
for mode in STAGES:
    ranks = RUNS_MAIN[mode]
    P = ranks[0]["P_bytes"]
    per_step = [r["per_step"][-1]["bytes"] / P for r in ranks]      # last step: steady state
    measured = sum(per_step) / len(per_step)
    phases = 3 if mode == "zero3" else 2
    ring_pred = phases * (MAIN_WORLD - 1) / MAIN_WORLD
    COMM_TABLE[mode] = {"measured_P": measured, "ring_predicted_P": ring_pred,
                        "notes_P": phases, "gb_30b": measured * P_30B_GB,
                        "spread": max(per_step) - min(per_step)}
    print(f"{LABEL[mode]:<20s} {measured:9.3f}P {ring_pred:8.3f}P {phases:6d}P   "
          f"{measured * P_30B_GB:18.0f} GB")

print(f"\nZeRO-1 and ZeRO-2 move the same bytes as plain data parallelism "
      f"({COMM_TABLE['zero1']['measured_P']:.3f}P vs {COMM_TABLE['dp']['measured_P']:.3f}P) "
      f"while holding\n{STAGE_TABLE['dp']['ladder_bpp'] / STAGE_TABLE['zero2']['ladder_bpp']:.1f}× "
      f"less state. ZeRO-3 pays "
      f"{COMM_TABLE['zero3']['measured_P'] / COMM_TABLE['dp']['measured_P']:.2f}× the traffic "
      f"for {STAGE_TABLE['dp']['ladder_bpp'] / STAGE_TABLE['zero3']['ladder_bpp']:.0f}× less "
      f"state.")

EVIDENCE["stages"] = {mode: dict(STAGE_TABLE[mode], **COMM_TABLE[mode],
                                 label=LABEL[mode],
                                 losses=mean_losses(RUNS_MAIN[mode]),
                                 losses_rank0=RUNS_MAIN[mode][0]["losses"],
                                 load_balance=round(spread[mode], 5))
                      for mode in STAGES}


# %% [markdown]
# ## 9 · Is it still the same training run?
#
# A memory saving that changed the answer would be worthless, so this is the gate that matters
# most. There are two separate claims and they deserve different standards of proof.
#
# **Across the stages, the standard is bitwise.** All four run the identical ring in the
# identical order on the identical data; the only difference is which rank keeps which slice
# of the result. Every arithmetic operation is therefore performed on the same inputs in the
# same order, and the final weights must agree to the last bit. A tolerance here would be
# hiding something.
#
# **Against a single GPU, the standard is a tolerance, and the size of it is the finding.**
# §3 says a distributed run is *mathematically* identical to a single-GPU run on a batch 32
# times larger. Mathematically, yes. In floating point, no: one rank computing the gradient of
# 32 sequences in one matrix multiply, and 32 ranks computing 1 sequence each and summing
# around a `bfloat16` ring, are different summation orders of the same quantity. So the
# comparison run below is that single GPU, with the *same 32 sequences* in one batch.

# %%
oracle = launch("dp", 1, steps=MAIN_STEPS, micro=MAIN_WORLD * MICRO)
w_ref = torch.load(RUNS / f"weights_dp_w{MAIN_WORLD}.pt")

EQUIV = {}
for mode in ["zero1", "zero2", "zero3"]:
    w = torch.load(RUNS / f"weights_{mode}_w{MAIN_WORLD}.pt")
    same_loss = RUNS_MAIN[mode][0]["losses"] == RUNS_MAIN["dp"][0]["losses"]
    EQUIV[mode] = {"bitwise": bool(torch.equal(w, w_ref)),
                   "max_abs": float((w - w_ref).abs().max()),
                   "losses_identical": bool(same_loss)}
    print(f"{LABEL[mode]:<20s} vs data parallelism : bitwise {EQUIV[mode]['bitwise']}, "
          f"loss curve identical {same_loss}")

w_single = torch.load(RUNS / "weights_dp_w1.pt")
denom = w_ref.abs().max().item()
dp_mean = mean_losses(RUNS_MAIN["dp"])
rank_spread = (max(r["losses"][-1] for r in RUNS_MAIN["dp"])
               - min(r["losses"][-1] for r in RUNS_MAIN["dp"]))
EQUIV["single_gpu"] = {
    "max_abs": float((w_single - w_ref).abs().max()),
    "rel": float((w_single - w_ref).abs().max() / denom),
    "loss_curve_1gpu": oracle[0]["losses"],
    "loss_curve_32gpu": dp_mean,
    "loss_last_1gpu": oracle[0]["losses"][-1], "loss_last_32gpu": dp_mean[-1],
    "loss_rel": abs(dp_mean[-1] - oracle[0]["losses"][-1]) / oracle[0]["losses"][-1],
    "rank_loss_spread": rank_spread,
}
print(f"\n32 GPUs vs 1 GPU on the same 32 sequences, after {MAIN_STEPS} steps:")
print(f"  final loss   {dp_mean[-1]:.6f}  (32 ranks, averaged)   "
      f"{oracle[0]['losses'][-1]:.6f}  (1 rank)   "
      f"relative difference {EQUIV['single_gpu']['loss_rel']:.2e}")
print(f"  max weight difference {EQUIV['single_gpu']['max_abs']:.3e} "
      f"({EQUIV['single_gpu']['rel']:.2e} of the largest weight)")
print(f"\n  (Averaged is the operative word. At the last step the 32 ranks report losses "
      f"{rank_spread:.2f}\n  apart from each other, because each one sees a different "
      f"sequence. Rank 0's loss is not\n  the run's loss; the mean of them is, and that is "
      f"the same averaging the all-reduce does\n  to the gradient.)")
print("\nThat residue is the bfloat16 ring of §4 meeting the non-associativity of floating-point")
print("addition. It is the price of the arrangement, and it is small — but it is not zero, and")
print("§3's 'mathematically identical' is a statement about the maths, not about the bits.")

EVIDENCE["equivalence"] = EQUIV


# %% [markdown]
# ## 10 · Bucketing: the cost is paid per message, not per byte
#
# The four stages above reduce **one tensor at a time**: each of the model's 25 parameter
# tensors gets its own ring, and a ring across 32 ranks is 31 hops in each direction. That is
# about 1,550 separate sends per step, each one a few kilobytes.
#
# The session says gradients are collected into *buckets* and a bucket is sent when it fills,
# and that the bucket size sets a balance: smaller buckets start transferring earlier, larger
# ones pay the fixed cost of starting a transfer less often. Rather than quote that, the run
# below takes it to its limit — `dp_flat` packs **every gradient in the model into one buffer**
# and reduces it once — and measures both ends of the balance.

# %%
RUNS_MAIN["dp_flat"] = launch("dp_flat", MAIN_WORLD)


def per_step_mean(ranks, key):
    xs = [s[key] for r in ranks for s in r["per_step"][1:]]
    return sum(xs) / len(xs)


BUCKETS = {}
for m in ("dp", "dp_flat"):
    ranks = RUNS_MAIN[m]
    msgs = per_step_mean(ranks, "msgs")
    byts = per_step_mean(ranks, "bytes")
    comm = per_step_mean(ranks, "comm")
    BUCKETS[m] = {"msgs": msgs, "bytes": byts, "comm_s": comm,
                  "bytes_per_msg": byts / msgs, "ms_per_msg": comm / msgs * 1000,
                  "effective_gbs": byts / comm / 1e9,
                  "peak_bpp": bpp(ranks, "peak_bytes"),
                  "ladder_bpp": bpp(ranks, "ladder_bytes"),
                  "step_s": per_step_mean(ranks, "wall")}

print(f"{'':<26s}{'messages':>10s}{'bytes/msg':>12s}{'comm':>9s}{'step':>9s}{'peak B/param':>14s}")
for m, lab in (("dp", "one ring per tensor"), ("dp_flat", "one ring for everything")):
    b = BUCKETS[m]
    print(f"{lab:<26s}{b['msgs']:10.0f}{b['bytes_per_msg']:11.0f}B{b['comm_s']:8.3f}s"
          f"{b['step_s']:8.3f}s{b['peak_bpp']:14.2f}")

print(f"\nSame bytes on the wire ({BUCKETS['dp']['bytes'] / 1e6:.1f} MB vs "
      f"{BUCKETS['dp_flat']['bytes'] / 1e6:.1f} MB per rank per step), "
      f"{BUCKETS['dp']['msgs'] / BUCKETS['dp_flat']['msgs']:.0f}x fewer messages, "
      f"{BUCKETS['dp']['comm_s'] / BUCKETS['dp_flat']['comm_s']:.1f}x less time waiting.")
print(f"Effective throughput rises from {BUCKETS['dp']['effective_gbs'] * 1000:.1f} MB/s to "
      f"{BUCKETS['dp_flat']['effective_gbs'] * 1000:.1f} MB/s on the same link, which says the "
      f"small-message\ncase was never moving bytes — it was paying "
      f"{BUCKETS['dp']['ms_per_msg']:.2f} ms of fixed cost per send.")
print(f"\nAnd the other side of the balance, which is the reason production uses a few hundred")
print(f"megabytes rather than 'everything': one buffer holding every gradient is a second copy")
print(f"of them, so peak memory goes from {BUCKETS['dp']['peak_bpp']:.2f} to "
      f"{BUCKETS['dp_flat']['peak_bpp']:.2f} bytes per weight — "
      f"+{BUCKETS['dp_flat']['peak_bpp'] - BUCKETS['dp']['peak_bpp']:.2f}, while the state it "
      f"holds\nis unchanged at {BUCKETS['dp_flat']['ladder_bpp']:.2f}. A bucket buys latency "
      f"with memory.")

w_flat = torch.load(RUNS / f"weights_dp_flat_w{MAIN_WORLD}.pt")
BUCKETS["weight_max_abs"] = float((w_flat - torch.load(RUNS / f"weights_dp_w{MAIN_WORLD}.pt")).abs().max())
print(f"\nBucketing changes the summation order, so it is not bitwise identical to the "
      f"per-tensor run:\nlargest weight difference {BUCKETS['weight_max_abs']:.3e}, which is "
      f"bfloat16's own resolution. Grouping\ntensors differently is a numerical choice as well "
      f"as a performance one.")

EVIDENCE["bucketing"] = BUCKETS

# %% [markdown]
# ## 11 · What it costs in time
#
# §5 measures communication as a *fraction of step time*, because that is the test for whether
# a run is limited by its arithmetic or by its wiring. The same fraction is measured here, with
# one caveat stated plainly: 32 processes are sharing far fewer physical cores, so the absolute
# seconds describe this laptop and nothing else. The **shape** — which stage waits more, and by
# how much — is the transferable part.

# %%
print(f"{'stage':<20s} {'step':>8s} {'compute':>9s} {'comm':>8s} {'comm/compute':>13s}")
TIME_TABLE = {}
for mode in STAGES:
    ranks = RUNS_MAIN[mode]
    steps = [s for r in ranks for s in r["per_step"][1:]]          # drop step 0, it is warm-up
    wall = sum(s["wall"] for s in steps) / len(steps)
    comm = sum(s["comm"] for s in steps) / len(steps)
    TIME_TABLE[mode] = {"step_s": wall, "comm_s": comm, "compute_s": wall - comm,
                        "comm_frac": comm / wall, "comm_over_compute": comm / (wall - comm)}
    print(f"{LABEL[mode]:<20s} {wall:7.3f}s {wall - comm:8.3f}s {comm:7.3f}s "
          f"{comm / (wall - comm):12.0%}")

print(f"\nZeRO-3 moves {COMM_TABLE['zero3']['measured_P'] / COMM_TABLE['dp']['measured_P']:.2f}× "
      f"the bytes of data parallelism and spends "
      f"{TIME_TABLE['zero3']['comm_s'] / TIME_TABLE['dp']['comm_s']:.2f}× the time doing it.")
print("§10's answer to this — overlapping the transfers with the backward pass — is out of")
print("scope for this submission and is the first thing I would add.")

EVIDENCE["timing"] = TIME_TABLE


# %% [markdown]
# ## 12 · What this costs on hardware that is not a laptop
#
# §11's seconds describe 32 processes contending for a handful of cores. The session's own
# question is different and more useful: given a real interconnect and a real card, is the
# transfer small compared with the work?
#
# This section is arithmetic, not measurement — I do not have an H100 — and it is marked as
# such wherever it appears. But it is the arithmetic that decides the shape of V5, so it is
# worth doing precisely rather than reading past.

# %%
LINKS = {"NVLink, inside one node": NVLINK_GBS, "InfiniBand, between nodes": INFINIBAND_GBS}
CARDS = {"64 x H100": 7.10, "64 x B200": 3.12}        # seconds of compute for a 1M-token step

print(f"{'path':<28s}{'2P = 120 GB':>14s}{'3P = 180 GB':>14s}")
LINK_TABLE = {}
for name, gbs in LINKS.items():
    t2, t3 = 2 * P_30B_GB / gbs, 3 * P_30B_GB / gbs
    LINK_TABLE[name] = {"gbs": gbs, "t_2p": t2, "t_3p": t3}
    print(f"{name:<28s}{t2:13.2f}s{t3:13.2f}s")

print(f"\n{'':<12s}{'compute/step':>14s}{'2P on InfiniBand':>18s}{'comm/compute':>14s}{'if not hidden':>15s}")
CARD_TABLE = {}
comm_ib = 2 * P_30B_GB / INFINIBAND_GBS
for name, compute in CARDS.items():
    CARD_TABLE[name] = {"compute_s": compute, "comm_s": comm_ib,
                        "ratio": comm_ib / compute, "serial_s": compute + comm_ib}
    print(f"{name:<12s}{compute:13.2f}s{comm_ib:17.2f}s{comm_ib / compute:13.0%}"
          f"{compute + comm_ib:14.2f}s")

print(f"\nThe volume does not move. {2 * P_30B_GB:.0f} GB crosses the wire either way. What moves")
print(f"is the compute it has to hide behind, and it got "
      f"{CARDS['64 x H100'] / CARDS['64 x B200']:.1f}x shorter.")
print(f"**Faster GPUs raise the ratio of communication to compute** — from "
      f"{CARD_TABLE['64 x H100']['ratio']:.0%} to {CARD_TABLE['64 x B200']['ratio']:.0%} — "
      f"which is the least\nintuitive consequence of buying newer hardware and the reason §10's "
      f"overlap stops being an\noptimization and becomes a requirement. Both ratios are still "
      f"below 1, so a transfer that runs\nentirely during the computation is still fully hidden. "
      f"That is the whole margin.")

EVIDENCE["hardware"] = {"links": LINK_TABLE, "cards": CARD_TABLE,
                        "gb_2p": 2 * P_30B_GB, "gb_3p": 3 * P_30B_GB,
                        "interconnect_ratio": NVLINK_GBS / INFINIBAND_GBS}

# %% [markdown]
# ## 13 · The memory wall
#
# §7 applies the ledger at four world sizes and finds that the choice for a 30B model narrows
# to two arrangements. That table is reproduced here from **measured** bytes per parameter —
# the four stages are run again at 4, 8 and 16 GPUs, and the 30B column is the measured ratio
# multiplied out.
#
# The projection is a ratio argument and it is worth being explicit about what it assumes:
# bytes-per-parameter is independent of parameter count for all four arrangements, which is
# exactly what makes §7's table a function of world size alone. The sweep below is the check
# of that assumption on this model.

# %%
SWEEP_WORLDS = [4, 8, 16, MAIN_WORLD]
SWEEP = {}
for world in SWEEP_WORLDS:
    SWEEP[world] = {}
    for mode in STAGES:
        ranks = RUNS_MAIN[mode] if world == MAIN_WORLD else launch(mode, world, steps=2)
        SWEEP[world][mode] = {
            "measured_bpp": bpp(ranks, "ladder_bytes"),
            "predicted_bpp": predicted_bpp(mode, world),
            "gib_30b": PARAMS_30B * bpp(ranks, "ladder_bytes") / GIB,
        }
    print(f"world {world:2d}: " + "  ".join(
        f"{LABEL[m]} {SWEEP[world][m]['measured_bpp']:.4f}" for m in STAGES))

print(f"\n{'':<20s}" + "".join(f"{w:>12d}" for w in SWEEP_WORLDS)
      + "    (GiB per GPU, 30B model)")
for mode in STAGES:
    row = "".join(f"{SWEEP[w][mode]['gib_30b']:11.1f}G" for w in SWEEP_WORLDS)
    fits = [w for w in SWEEP_WORLDS if SWEEP[w][mode]["gib_30b"] <= CARD_GIB]
    note = f"fits from {min(fits)} GPUs" if fits else "never fits"
    print(f"{LABEL[mode]:<20s}{row}    {note}")
print(f"\na card holds {CARD_GIB:.1f} GiB")

# §7's floor: the four bytes per weight that data parallelism and ZeRO-1 both replicate, no
# matter how many GPUs there are, fill a card exactly at this many parameters.
floor_params = CARD_GIB * GIB / 4
print(f"\nData parallelism and ZeRO-1 both leave weight+gradient — 4 bytes — on every card at "
      f"every\nworld size. 4 bytes fills a {CARD_GIB:.1f} GiB card at "
      f"{floor_params / 1e9:.1f}B parameters, so a 30B model is past that line and no GPU count "
      f"rescues it.")

EVIDENCE["sweep"] = {str(w): SWEEP[w] for w in SWEEP_WORLDS}
EVIDENCE["wall"] = {"card_gib": CARD_GIB, "floor_params": floor_params,
                    "fits_from": {mode: (min([w for w in SWEEP_WORLDS
                                              if SWEEP[w][mode]["gib_30b"] <= CARD_GIB],
                                             default=None)) for mode in STAGES}}


# %% [markdown]
# ## 14 · Offload, and the trade it actually makes
#
# A GPU is not the only memory in the machine. The optimizer state is the natural thing to move
# out of it: twelve of the sixteen bytes, and touched exactly once per step. There are two
# versions — park the state in system memory and copy it to the GPU for the update, or perform
# the update on the CPU as well, so the state never moves and the GPU is freed of that work.
#
# The cost is PCIe, at roughly 60 GB/s: far below NVLink and comparable to a network cable.
# **Offload converts a memory problem into a bandwidth problem**, which is only a good trade
# when memory is the binding constraint. Not implemented here — the arithmetic is.

# %%
OFF = {}
for mode in STAGES:
    on_gpu = predicted_bpp(mode, MAIN_WORLD)
    opt_part = (12 if mode == "dp" else 12 / MAIN_WORLD)
    OFF[mode] = {"gib_before": PARAMS_30B * on_gpu / GIB,
                 "gib_after": PARAMS_30B * (on_gpu - opt_part) / GIB,
                 "bpp_after": on_gpu - opt_part}

print(f"{'arrangement':<20s}{'GiB/GPU':>10s}{'with optimizer offloaded':>26s}")
for mode in STAGES:
    print(f"{LABEL[mode]:<20s}{OFF[mode]['gib_before']:9.1f}G{OFF[mode]['gib_after']:25.1f}G")

# What has to cross PCIe each step if the update runs on the CPU: this rank's shard of the
# gradient goes out, and its shard of the updated weights comes back.
shard_grad_gb = PARAMS_30B * 2 / MAIN_WORLD / 1e9
shard_state_gb = PARAMS_30B * 12 / MAIN_WORLD / 1e9
print(f"\nper step, per GPU, at 30B on {MAIN_WORLD} GPUs:")
print(f"  update on the CPU  : {2 * shard_grad_gb:.2f} GB over PCIe "
      f"= {2 * shard_grad_gb / PCIE_GBS:.3f}s   (gradient shard out, weight shard back)")
print(f"  update on the GPU  : {2 * shard_state_gb:.2f} GB over PCIe "
      f"= {2 * shard_state_gb / PCIE_GBS:.3f}s   (the whole state shard, both ways)")
print(f"\nThat second line is why the CPU-side update is the version that earns its place: it")
print(f"moves {shard_state_gb / shard_grad_gb:.0f}x less, because the twelve bytes never leave "
      f"system memory at all.")

EVIDENCE["offload"] = {"per_stage": OFF, "pcie_gbs": PCIE_GBS,
                       "cpu_update_gb": 2 * shard_grad_gb,
                       "gpu_update_gb": 2 * shard_state_gb,
                       "cpu_update_s": 2 * shard_grad_gb / PCIE_GBS,
                       "gpu_update_s": 2 * shard_state_gb / PCIE_GBS}

# %% [markdown]
# ## 15 · The 8-bit ledger
#
# §11's arithmetic, which is smaller than it first looks and is worth doing precisely because
# of that. Moving the weights and gradients to 8-bit removes two of the sixteen bytes, and
# MXFP8 adds back a shared 8-bit scale for every block of 32 values — across two tensors that
# is 2/32 = 0.0625 bytes per parameter. The twelve bytes of master copy and optimizer moments
# are untouched, because the update arithmetic still needs the accuracy.
#
# This one is computed rather than measured: `bfloat16` is as low as this CPU goes, and
# claiming to have measured MXFP8 on hardware that cannot multiply it would be exactly the
# kind of thing this harness exists to avoid.

# %%
MX_BLOCK = 32
fp8_state = 1 + 1 + 4 + 8 + 2 / MX_BLOCK           # weight, grad, master, m+v, two scale bytes
print(f"{'':<34s}{'bytes/weight':>13s}{'30B model':>12s}")
print(f"{'16-bit weights and gradients':<34s}{FULL_STATE:13.2f}{PARAMS_30B * FULL_STATE / GIB:11.1f}G")
print(f"{'8-bit weights and gradients':<34s}{fp8_state:13.2f}"
      f"{PARAMS_30B * fp8_state / GIB:11.1f}G")
print(f"\nreduction in stored state: {1 - fp8_state / FULL_STATE:.1%}")
print(f"reduction in what crosses the wire: {1 - 8 / 16:.0%}, because P is the parameters in the")
print("compute format and halving that format halves every multiple of P in this session.")

EVIDENCE["precision"] = {
    "bf16_bytes": FULL_STATE, "mxfp8_bytes": fp8_state, "mx_block": MX_BLOCK,
    "scale_bytes_per_param": 2 / MX_BLOCK,
    "state_reduction": 1 - fp8_state / FULL_STATE,
    "gib_30b_bf16": PARAMS_30B * FULL_STATE / GIB,
    "gib_30b_fp8": PARAMS_30B * fp8_state / GIB,
    "wire_reduction": 0.5,
}


# %% [markdown]
# ## 16 · What the previous run did, and what this one says about the next
#
# The previous run, LightningLM v0.1, used DeepSpeed at ZeRO stage 2 on 8 GPUs, with
# `overlap_comm` enabled, `round_robin_gradients` enabled as an out-of-memory fix, bucket sizes
# of 2×10⁸ bytes, and weight decay deliberately at zero. Its global batch of 32 came from
# 2 sequences per GPU × 8 GPUs × 2 accumulation steps — **the second factorization measured in
# §6**, and the reason that comparison was worth running rather than asserting.
#
# It never needed stage 3. At 30B, §13 says that is no longer available: data parallelism and
# ZeRO-1 do not fit at any world size, so the starting point is ZeRO-2 from 32 GPUs or ZeRO-3
# from 8. Four questions stay open, and what this harness can and cannot say about each is
# recorded honestly below.

# %%
V5_QUESTIONS = [
    {"q": "ZeRO-2 on 32 GPUs, or ZeRO-3 on 8?",
     "settled_by": "A measured step time for both on the real architecture, with activation "
                   "memory included.",
     "this_harness": f"Gives the memory side exactly — {EVIDENCE['stages']['zero2']['gib_30b']:.1f} "
                     f"GiB against {EVIDENCE['stages']['zero3']['gib_30b']:.1f} GiB per GPU — and "
                     f"the communication ratio, 1.50x. It cannot give step time on hardware it "
                     f"does not have, and it excludes activations."},
    {"q": "How many GPUs per node, and how many nodes?",
     "settled_by": "How much traffic can be kept inside a node.",
     "this_harness": f"Not addressed: one loopback link, no node boundary. §12's "
                     f"{NVLINK_GBS / INFINIBAND_GBS:.0f}x is the whole of the argument."},
    {"q": "Is 8-bit arithmetic committed from the start?",
     "settled_by": "A short run in bf16 and in MXFP8 on the same architecture, comparing loss "
                   "and step time. Commits to Blackwell hardware.",
     "this_harness": f"Gives the storage arithmetic ({EVIDENCE['precision']['state_reduction']:.1%}) "
                     f"and nothing else. bfloat16 is as low as this CPU goes."},
    {"q": "Does any state go to system memory?",
     "settled_by": "Whether the run is memory-bound or communication-bound once the stage is "
                   "chosen.",
     "this_harness": "§14's arithmetic only. Offload is not implemented."},
]
for i, q in enumerate(V5_QUESTIONS, 1):
    print(f"{i}. {q['q']}\n   settled by : {q['settled_by']}\n   here       : {q['this_harness']}\n")

EVIDENCE["v4"] = {
    "stage": 2, "format": "bf16", "seq_per_gpu": 2, "accum": 2, "global_batch": 32, "gpus": 8,
    "peak_lr": 3e-4, "warmup": 500, "weight_decay": 0.0, "grad_clip": 1.0,
    "bucket_bytes": 2e8, "overlap_comm": True, "round_robin_gradients": True,
}
EVIDENCE["v5_questions"] = V5_QUESTIONS

# %% [markdown]
# ## 17 · Gates
#
# Every claim this file makes, as a boolean. `build_notebook.py` exits non-zero if any of them
# is false, so the exit code is the pass/fail signal and not the printed summary.

# %%
def close(a, b, tol=0.01):
    return abs(a - b) <= tol * max(abs(b), 1e-12)


GATES = {
    "tokenizer_frozen": tok_sha == FROZEN_TOKENIZER_SHA256,
    "world_is_32": all(len(RUNS_MAIN[m]) == 32 for m in STAGES),
    "ranks_are_real_processes": all(r["world"] == MAIN_WORLD for r in RUNS_MAIN["dp"]),

    # §4 — the collectives
    "ring_matches_gloo": CHECK["vs_gloo_rel"] < 1e-5,
    "reduce_scatter_plus_all_gather_is_all_reduce": CHECK["two_phase_identical"],
    "bf16_ring_loses_bits": CHECK["bf16_ring_sum"] != CHECK["exact_sum"],
    "fp32_ring_is_exact": CHECK["fp32_ring_sum"] == CHECK["exact_sum"],

    # §6 — the memory ladder, measured against the formula
    **{f"memory_{m}_matches_ledger": close(STAGE_TABLE[m]["ladder_bpp"],
                                           predicted_bpp(m, MAIN_WORLD), 0.001)
       for m in STAGES},
    "memory_ladder_is_monotonic": (STAGE_TABLE["dp"]["ladder_bpp"]
                                   > STAGE_TABLE["zero1"]["ladder_bpp"]
                                   > STAGE_TABLE["zero2"]["ladder_bpp"]
                                   > STAGE_TABLE["zero3"]["ladder_bpp"]),
    "every_rank_holds_an_equal_share": all(spread[m] <= 1.02 for m in STAGES),

    # §6 — communication
    **{f"comm_{m}_matches_ring": close(COMM_TABLE[m]["measured_P"],
                                       COMM_TABLE[m]["ring_predicted_P"], 0.001)
       for m in STAGES},
    "zero1_and_zero2_are_free": (close(COMM_TABLE["zero1"]["measured_P"],
                                       COMM_TABLE["dp"]["measured_P"], 0.001)
                                 and close(COMM_TABLE["zero2"]["measured_P"],
                                           COMM_TABLE["dp"]["measured_P"], 0.001)),
    "zero3_costs_half_as_much_again": close(
        COMM_TABLE["zero3"]["measured_P"] / COMM_TABLE["dp"]["measured_P"], 1.5, 0.001),

    # §3 — correctness
    **{f"{m}_bitwise_identical_to_dp": EQUIV[m]["bitwise"] for m in ["zero1", "zero2", "zero3"]},
    **{f"{m}_loss_curve_identical": EQUIV[m]["losses_identical"]
       for m in ["zero1", "zero2", "zero3"]},
    "matches_single_gpu_within_bf16_error": EQUIV["single_gpu"]["rel"] < 1e-2,
    "loss_matches_single_gpu": EQUIV["single_gpu"]["loss_rel"] < 1e-3,
    "model_actually_trained": (mean_losses(RUNS_MAIN["dp"])[-1]
                               < mean_losses(RUNS_MAIN["dp"])[0] - 0.5),

    # §7 — the wall
    "zero3_scales_as_one_over_world": all(
        close(SWEEP[w]["zero3"]["measured_bpp"], FULL_STATE / w, 0.001) for w in SWEEP_WORLDS),
    "dp_never_fits_a_card": EVIDENCE["wall"]["fits_from"]["dp"] is None,
    "zero1_never_fits_a_card": EVIDENCE["wall"]["fits_from"]["zero1"] is None,
    "zero2_fits_from_32_gpus": EVIDENCE["wall"]["fits_from"]["zero2"] == 32,
    "zero3_fits_from_8_gpus": EVIDENCE["wall"]["fits_from"]["zero3"] == 8,

    # §6 — data parallelism's two promises
    "replicas_stay_identical": EVIDENCE["data_parallel"]["replica_hashes_distinct"] == 1,
    "global_batch_is_a_product": EVIDENCE["data_parallel"]["loss_rel"] < 1e-2,

    # §10 — bucketing
    "bucketing_cuts_messages": BUCKETS["dp"]["msgs"] / BUCKETS["dp_flat"]["msgs"] > 10,
    "bucketing_moves_the_same_bytes": close(BUCKETS["dp_flat"]["bytes"],
                                            BUCKETS["dp"]["bytes"], 0.02),
    "bucketing_is_faster": BUCKETS["dp_flat"]["comm_s"] < BUCKETS["dp"]["comm_s"],
    "bucketing_costs_memory": BUCKETS["dp_flat"]["peak_bpp"] > BUCKETS["dp"]["peak_bpp"],
    "bucketing_does_not_change_state_held": close(BUCKETS["dp_flat"]["ladder_bpp"],
                                                  BUCKETS["dp"]["ladder_bpp"], 1e-9),

    # §12 — the hardware arithmetic
    "faster_card_raises_comm_ratio": (CARD_TABLE["64 x B200"]["ratio"]
                                      > CARD_TABLE["64 x H100"]["ratio"]),
    "both_ratios_still_below_one": all(c["ratio"] < 1 for c in CARD_TABLE.values()),
    "interconnect_gap_is_9x": close(NVLINK_GBS / INFINIBAND_GBS, 9.0, 1e-9),
    "nvlink_hides_2p_in_under_a_third_of_a_second":
        close(LINK_TABLE["NVLink, inside one node"]["t_2p"], 0.267, 0.02),

    # §14 — offload
    "cpu_side_update_moves_less": (EVIDENCE["offload"]["cpu_update_gb"]
                                   < EVIDENCE["offload"]["gpu_update_gb"]),

    # §11 — precision
    "mxfp8_ledger": close(fp8_state, 14.0625, 1e-6),
    "mxfp8_saves_12_percent": close(1 - fp8_state / FULL_STATE, 0.121, 0.01),
}

EVIDENCE["gates"] = {k: bool(v) for k, v in GATES.items()}
EVIDENCE["meta"]["wall_seconds"] = round(time.time() - T_START, 1)
EVIDENCE["meta"]["main_world"] = MAIN_WORLD
EVIDENCE["meta"]["main_steps"] = MAIN_STEPS

for name, ok in GATES.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
passed = sum(1 for v in GATES.values() if v)
print(f"\n{passed}/{len(GATES)} gates pass · {EVIDENCE['meta']['wall_seconds'] / 60:.1f} minutes")

(OUT / "evidence.json").write_text(json.dumps(EVIDENCE, indent=2), encoding="utf-8")
print(f"wrote {(OUT / 'evidence.json').relative_to(ROOT)}")
assert passed == len(GATES), f"{len(GATES) - passed} gate(s) failed"
