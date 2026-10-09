# %% [markdown]
# # Session 14 — Mixture-of-Experts
#
# **ERA V5 · one feed-forward block becomes eight, and the model keeps learning.**
#
# The assignment: *train a Linear model and convert that into an MoE. Your call on model size
# and data trained on, but must show they continue to train and reduce loss.*
#
# "Linear model" is read here as the **dense** model: a transformer whose feed-forward block
# is one SwiGLU network that every token passes through. It is converted into a mixture of
# experts by **sparse upcycling** (§15 of the notes): every expert starts as a copy of that
# network, a new router picks 2 of 8 per token, and the router's weights for the chosen pair
# are rescaled to sum to one.
#
# ---
#
# ### The one claim that has to be checked before any curve means anything
#
# "It continues to train" is only meaningful if training continues *from where the dense model
# was*. If the conversion itself moves the model, a falling loss afterwards might just be the
# MoE recovering from damage the conversion did. So §5 checks that the converted model computes
# the **same function** as the dense one — same logits, same validation loss — before a single
# MoE step is taken. It does, by construction: with identical experts,
# `Σ gᵢ·E(x) = E(x)·Σ gᵢ = E(x)`.
#
# That same identity has a consequence that shapes everything after it: at the moment of
# conversion, **the router receives no gradient from the language loss at all**. Its choice
# cannot change the output, because every choice gives the same output. The experts have to
# drift apart first, on the different tokens the random router happens to send them, before
# the router has anything to learn. §6 measures that.

# %%
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if HERE.name != "s14-moe" and (HERE / "s14-moe").is_dir():
    HERE = HERE / "s14-moe"
OUT = HERE / "out"
STAGES = OUT / "stages"
STAGES.mkdir(parents=True, exist_ok=True)

# CPU, deliberately. Measured on this machine (torch 2.2.2): MPS ran the dense model no faster
# than CPU and the MoE at half CPU speed (routing's gather/scatter backward is slow there), and
# CPU runs are deterministic, as S9-S12's were.
DEV = torch.device("cpu")
# Smoke mode: a toy that proves the code runs end to end in a minute or two. Its numbers are
# not findings.
QUICK = os.environ.get("S14_QUICK") == "1"
if QUICK:
    STAGES = OUT / "stages_quick"
    STAGES.mkdir(parents=True, exist_ok=True)


def stage(name, fn, version="1"):
    """Run `fn` once, cache its JSON result, reload it on later runs.

    `version` invalidates a cache when the code that produced it changes; bump it rather than
    deleting files by hand.
    """
    path = STAGES / f"{name}.json"
    if path.exists():
        blob = json.loads(path.read_text())
        if blob.get("__stage__") == version:
            print(f"[resume] {name}")
            return blob["value"]
        print(f"[stale]  {name} · cached v{blob.get('__stage__')}, need v{version}")
    t0 = time.time()
    value = fn()
    path.write_text(json.dumps({"__stage__": version, "value": value}, default=float))
    print(f"[done]   {name} · {time.time() - t0:.0f}s")
    return value


T_START = time.time()
EVIDENCE = {"meta": {"python": platform.python_version(), "torch": torch.__version__,
                     "platform": platform.platform(), "device": str(DEV), "quick": QUICK,
                     "chip": platform.processor() or platform.machine()}}


def snapshot(label):
    EVIDENCE["meta"]["wall_seconds"] = round(time.time() - T_START, 1)
    EVIDENCE["meta"]["last_section"] = label
    (OUT / ("evidence_smoke.json" if QUICK else "evidence.json")).write_text(
        json.dumps(EVIDENCE, indent=1, default=float))


print(f"python {platform.python_version()} · torch {torch.__version__} · device {DEV}"
      + ("  [QUICK smoke mode]" if QUICK else ""))


# %% [markdown]
# ## 1 · The reference model, by arithmetic
#
# The notes carry one model throughout: the published shape of Qwen3-30B-A3B. Nothing here can
# train it, but every figure the notes quote about it is arithmetic, and arithmetic can be
# checked. This cell recomputes them from the shape alone. On the page these are marked
# `computed`; the model trained below is marked `measured`.

# %%
GIB = 1024 ** 3
REF = dict(layers=48, d=2048, q_heads=32, kv_heads=4, head_dim=128, experts=128, k=8,
           expert_width=768, vocab=151_936)


def reference_arithmetic(r=REF):
    d, L = r["d"], r["layers"]
    attn = d * r["q_heads"] * r["head_dim"] * 2 + d * r["kv_heads"] * r["head_dim"] * 2
    expert = 3 * d * r["expert_width"]
    router = d * r["experts"]
    emb = 2 * r["vocab"] * d
    total = L * (attn + r["experts"] * expert + router) + emb
    active = L * (attn + r["k"] * expert + router) + emb
    dense_parts = L * (attn + router) + emb
    kv_per_token = 2 * r["kv_heads"] * r["head_dim"] * 2 * L
    # §16: dispatch + combine, 7/8 leaves the GPU, x2 for backward, one 8,192-token sequence
    a2a_bytes = 2 * (r["k"] * d * 2) * 8192 * 7 / 8 * L * 2
    flops_seq = 6 * active * 8192
    # §17: per GPU at EP=8
    experts_per_gpu = L * (r["experts"] // 8) * expert
    act = 8192 * d * 34 * L
    return {
        "attention_per_layer": attn, "expert": expert, "experts_per_layer": r["experts"] * expert,
        "experts_all_layers": L * r["experts"] * expert, "router_per_layer": router,
        "embeddings": emb, "total": total, "active": active, "active_per_layer": attn + r["k"] * expert + router,
        "ratio": total / active, "dense_parts": dense_parts,
        "train_state_gib": total * 16 / GIB,
        "gflop_per_token_moe": 6 * active / 1e9, "gflop_per_token_dense30b": 6 * 30.2e9 / 1e9,
        "kv_cache_kib": kv_per_token / 1024,
        "kv_cache_mha_kib": kv_per_token / 1024 * r["q_heads"] / r["kv_heads"],
        "dense_ffn_width": r["k"] * r["expert_width"],
        "even_load": r["k"] / r["experts"],
        "capacity_avg": 8192 * r["k"] // r["experts"], "capacity_125": int(8192 * r["k"] / r["experts"] * 1.25),
        "combos_mixtral": math.comb(8, 2), "combos_ref": math.comb(128, 8),
        "combos_16c2": math.comb(16, 2), "combos_64c8": math.comb(64, 8),
        "a2a_gb": a2a_bytes / 1e9,
        "a2a_seconds": {"NVLink 5 (B200)": a2a_bytes / 900e9, "NVLink 4 (H100)": a2a_bytes / 450e9,
                        "network card": a2a_bytes / 50e9},
        "compute_tflop_seq": flops_seq / 1e12, "compute_seconds_b200": flops_seq / 2.25e15,
        "tokens_per_expert_ep8": 8 * 8192 * r["k"] // r["experts"],
        "ep8_experts_gib": experts_per_gpu * 16 / GIB,
        "ep8_dense_gib": dense_parts * (4 + 12 / 8) / GIB,
        "ep8_activations_gib": act / GIB,
        "ep8_total_gib": (experts_per_gpu * 16 + dense_parts * (4 + 12 / 8) + act) / GIB,
        "ep16_state_gib": (experts_per_gpu * 16 / 2 + dense_parts * (4 + 12 / 16)) / GIB,
        "ep16_total_gib": (experts_per_gpu * 16 / 2 + dense_parts * (4 + 12 / 16) + act) / GIB,
    }


def router_example():
    """§7's 8-expert, top-2 worked example, and §13's bias example on top of it."""
    z = torch.tensor([2.0, 0.5, 1.2, -0.3, 0.9, 1.8, -1.0, 0.1], dtype=torch.float64)
    out = {"logits": z.tolist()}
    for name, s in (("softmax", z.softmax(-1)), ("sigmoid", z.sigmoid())):
        top = s.topk(2)
        w = top.values / top.values.sum()
        out[name] = {"scores": s.tolist(), "chosen": top.indices.tolist(), "weights": w.tolist()}
    sig = z.sigmoid()
    bias = torch.zeros(8, dtype=torch.float64)
    bias[5] = -0.5
    pick = (sig + bias).topk(2).indices
    out["bias_example"] = {"bias_on": 5, "bias": -0.5, "expert5_for_selection": float(sig[5] + bias[5]),
                           "chosen": pick.tolist(),
                           "weights_from_original": (sig[pick] / sig[pick].sum()).tolist()}
    # §12: Switch aux loss at alpha 0.01, N 128
    N, a = 128, 0.01
    out["aux_even"] = a * N * N * (1 / N) * (1 / N)
    out["aux_collapsed"] = a * N * 1.0
    return out


REF_ARITH = reference_arithmetic()
ROUTER_EX = router_example()
for k in ("total", "active", "experts_all_layers", "dense_parts"):
    print(f"{k:<22}{REF_ARITH[k] / 1e9:>8.2f} B")
print(f"{'train state @16B':<22}{REF_ARITH['train_state_gib']:>8.0f} GiB   "
      f"work/token {REF_ARITH['gflop_per_token_moe']:.1f} GFLOP")
print(f"{'all-to-all, 1 seq':<22}{REF_ARITH['a2a_gb']:>8.1f} GB    "
      f"EP=8 on B200: {REF_ARITH['ep8_total_gib']:.1f} GiB per card")
print(f"router example, softmax top-2 weights {[round(w, 3) for w in ROUTER_EX['softmax']['weights']]}"
      f" · sigmoid {[round(w, 3) for w in ROUTER_EX['sigmoid']['weights']]}")

# Each value the notes print, against the recomputation, at the precision the notes print it.
NOTES_FIGURES = [
    ("§1 attention per layer (M)", REF_ARITH["attention_per_layer"] / 1e6, 18.87, 0.005),
    ("§1 one expert (M)", REF_ARITH["expert"] / 1e6, 4.72, 0.005),
    ("§1 experts, 48 layers (B)", REF_ARITH["experts_all_layers"] / 1e9, 28.99, 0.005),
    ("§1 total (B)", REF_ARITH["total"] / 1e9, 30.53, 0.005),
    ("§1 active (B)", REF_ARITH["active"] / 1e9, 3.35, 0.005),
    ("§1 training state (GiB)", REF_ARITH["train_state_gib"], 455, 0.5),
    ("§1 work per token (GFLOP)", REF_ARITH["gflop_per_token_moe"], 20.1, 0.05),
    ("§1 KV cache per token (KiB)", REF_ARITH["kv_cache_kib"], 96, 0.5),
    ("§7 softmax weight, expert 0", ROUTER_EX["softmax"]["weights"][0], 0.55, 0.005),
    ("§7 sigmoid weight, expert 0", ROUTER_EX["sigmoid"]["weights"][0], 0.507, 0.0005),
    ("§13 expert 5 score with bias", ROUTER_EX["bias_example"]["expert5_for_selection"], 0.358, 0.0005),
    ("§8 groups of 8 from 128", REF_ARITH["combos_ref"], 1_429_702_652_400, 0),
    ("§8 groups of 8 from 64", REF_ARITH["combos_64c8"], 4_426_165_368, 0),
    ("§11 capacity at 1.25", REF_ARITH["capacity_125"], 640, 0),
    ("§12 aux loss, even", ROUTER_EX["aux_even"], 0.01, 1e-9),
    ("§12 aux loss, collapsed", ROUTER_EX["aux_collapsed"], 1.28, 1e-9),
    ("§16 all-to-all bytes (GB)", REF_ARITH["a2a_gb"], 45.1, 0.05),
    ("§16 compute per sequence (TFLOP)", REF_ARITH["compute_tflop_seq"], 164.8, 0.05),
    ("§17 experts per GPU (GiB)", REF_ARITH["ep8_experts_gib"], 54.0, 0.05),
    ("§17 dense parts, ZeRO-1 (GiB)", REF_ARITH["ep8_dense_gib"], 7.9, 0.05),
    ("§17 activations (GiB)", REF_ARITH["ep8_activations_gib"], 25.5, 0.05),
    ("§17 total per GPU (GiB)", REF_ARITH["ep8_total_gib"], 87.4, 0.05),
    ("§17 EP=16 total (GiB)", REF_ARITH["ep16_total_gib"], 59.3, 0.05),
]
NOTES_CHECK = [{"figure": n, "recomputed": got, "notes": want, "ok": abs(got - want) <= tol}
               for n, got, want, tol in NOTES_FIGURES]
bad = [c for c in NOTES_CHECK if not c["ok"]]
print(f"\n{len(NOTES_CHECK) - len(bad)}/{len(NOTES_CHECK)} of the notes' figures reproduce")
for c in bad:
    print(f"  MISMATCH {c['figure']}: recomputed {c['recomputed']}, notes {c['notes']}")
EVIDENCE.update(reference=REF_ARITH, router_example=ROUTER_EX, notes_check=NOTES_CHECK)
snapshot("reference")


# %% [markdown]
# ## 2 · Tokens
#
# About 20 million of them, read once from a public dataset and cached. The S6 corpus this repo
# used through Session 12 is 656,920 tokens; the runs below need 11M, so taking them from S6
# would mean ~17 passes, and §5 of the notes records that mixtures of experts overfit faster
# than dense models. A falling training loss over repeated data would then be memorising, not
# learning. No token is used twice in this file. The tokenizer is still the frozen Sarvam-1
# one from Sessions 9–13.
#
# The vocabulary is capped to the 8,191 most frequent ids plus one `UNK` (id 0), as in S11's
# sweeps and S13: an uncapped 68,096-row embedding would be three quarters of this dense model.

# %%
TOKENIZER_REPO = "sarvamai/sarvam-1"
FROZEN_TOKENIZER_SHA256 = "bb5115a36ddb956a4ee0fd534e9870dd69157835622aec9c53062896f883c072"
TARGET_TOKENS = 20_000_000                 # smoke mode reads the same cache, less of it
VOCAB_CAP = 8192
CACHE = OUT / f"tokens_{TARGET_TOKENS}_{VOCAB_CAP}.npy"
DATASETS = [("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
            ("wikitext", "wikitext-103-raw-v1", "text")]


def load_tokenizer():
    import hashlib
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
    return Tokenizer.from_file(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_stream():
    """Stream text to TARGET_TOKENS, tokenize, cap the vocabulary, cache with a sidecar.

    The sidecar records the source and the sha256 of the token array itself, so a cache that
    was rebuilt from a different source cannot pass itself off as the one the numbers came from.
    """
    import hashlib
    meta_path = CACHE.with_suffix(".meta.json")
    if CACHE.exists() and meta_path.exists():
        arr = np.load(CACHE)
        meta = json.loads(meta_path.read_text())
        if hashlib.sha256(arr.tobytes()).hexdigest() != meta["token_sha256"]:
            raise RuntimeError("token cache does not match its sidecar")
        print(f"cached: {len(arr):,} tokens · source {meta['source']}")
        return arr, meta
    from collections import Counter
    from datasets import load_dataset
    tok, sha = load_tokenizer()
    assert sha == FROZEN_TOKENIZER_SHA256, f"tokenizer drifted: {sha}"
    ids, source, docs = [], None, 0
    for name, cfg, field in DATASETS:
        try:
            ds = load_dataset(name, cfg, split="train", streaming=True)
            t0, buf = time.time(), []
            for rec in ds:
                text = rec.get(field) or ""
                if len(text) < 200:
                    continue
                buf.append(text)
                if len(buf) == 256:
                    for e in tok.encode_batch(buf):
                        ids.extend(e.ids)
                    docs += len(buf)
                    buf = []
                    if len(ids) >= TARGET_TOKENS:
                        break
            if len(ids) >= TARGET_TOKENS:
                source = name
                print(f"streamed {len(ids):,} tokens from {docs:,} {name} documents "
                      f"in {time.time() - t0:.0f}s")
                break
            ids, docs = [], 0
        except Exception as exc:                               # noqa: BLE001 -- try the next source
            print(f"{name} unavailable ({str(exc)[:80]}); trying the next source")
            ids, docs = [], 0
    if source is None:
        raise RuntimeError("no dataset reachable; this harness needs the network once")
    ids = ids[:TARGET_TOKENS]
    counts = Counter(ids)
    keep = sorted(i for i, _ in counts.most_common(VOCAB_CAP - 1))
    remap = {old: new + 1 for new, old in enumerate(keep)}         # 0 is UNK
    coverage = sum(counts[i] for i in keep) / len(ids)
    arr = np.fromiter((remap.get(i, 0) for i in ids), dtype=np.uint16, count=len(ids))
    np.save(CACHE, arr)
    meta = {"source": source, "dataset_config": dict((n, c) for n, c, _ in DATASETS)[source],
            "documents": docs, "coverage": coverage, "distinct_ids": len(counts),
            "tokenizer_sha256": sha, "token_sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"capped vocabulary to {VOCAB_CAP:,} ids, covering {coverage:.2%} of occurrences")
    return arr, meta


STREAM_NP, CORPUS_META = build_stream()
STREAM = torch.from_numpy(STREAM_NP.astype(np.int64))
EVIDENCE["corpus"] = {"tokens": len(STREAM), "vocab_cap": VOCAB_CAP, **CORPUS_META}
snapshot("corpus")


# %% [markdown]
# ## 3 · The model, dense and mixed
#
# One class covers both. A feed-forward block holds its experts as stacked tensors
# `[E, d, f]`; with `E = 1` and no router it is the dense model, and upcycling is a
# `repeat` along the first axis. Every expert is a SwiGLU network, as in §3 of the notes:
# `Eᵢ(x) = W_down(SiLU(W_gate x) ⊙ W_up x)`.
#
# The router follows the notes' practical advice (§7): it is computed in float32 and it starts
# at a tenth of the usual scale. It scores with softmax, keeps the top `k`, and rescales the
# kept scores to sum to one. Two selection rules are supported:
#
# * **hard** — the top `k` of `score + bias`. The bias is §13's aux-loss-free balancer: it
#   steers *which* experts are chosen and never enters the weights, so it never touches the
#   language-loss gradient.
# * **probabilistic** — `k` experts drawn without replacement in proportion to their scores
#   (Gumbel-top-k). §15's fix for clone-family collapse, used only for an early window.
#
# Expert outputs are computed dropless (§11): tokens are sorted by expert and each expert runs
# on exactly the tokens it was sent, however many.

# %%
D_MODEL, N_LAYER, N_HEAD, FFN = (128, 2, 4, 256) if QUICK else (256, 4, 4, 768)
SEQ, BATCH = (64, 16) if QUICK else (256, 32)
INIT_STD = 0.02


class FFNBlock(nn.Module):
    """E SwiGLU experts plus, when E > 1, a router. E == 1 is the dense feed-forward network."""

    def __init__(self, d, f, E=1, k=1, router_std=INIT_STD / 10):
        super().__init__()
        self.E, self.k = E, k
        self.w_gate = nn.Parameter(torch.randn(E, d, f) * INIT_STD)
        self.w_up = nn.Parameter(torch.randn(E, d, f) * INIT_STD)
        self.w_down = nn.Parameter(torch.randn(E, f, d) * INIT_STD / math.sqrt(2 * N_LAYER))
        if E > 1:
            self.router = nn.Parameter(torch.randn(d, E) * router_std)
            self.register_buffer("bias", torch.zeros(E))
        self.select = "hard"
        self.stats = None

    def expert(self, e, h):
        return (F.silu(h @ self.w_gate[e]) * (h @ self.w_up[e])) @ self.w_down[e]

    def forward(self, x):
        if self.E == 1:
            return self.expert(0, x)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        rdt = torch.float64 if x.dtype == torch.float64 else torch.float32
        logits = x.to(rdt) @ self.router.to(rdt)                     # float32 router (§7)
        probs = logits.softmax(-1)
        with torch.no_grad():
            choose = probs + self.bias                               # bias steers choice only
            if self.select == "probabilistic" and self.training:
                u = torch.rand_like(probs).clamp_(1e-9, 1 - 1e-9)
                choose = probs.log() - (-u.log()).log()              # Gumbel-top-k sampling
            idx = choose.topk(self.k, -1).indices
        w = probs.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)                              # weights from raw scores
        flat = idx.reshape(-1)
        order = flat.argsort()
        tok = order // self.k
        counts = torch.bincount(flat, minlength=self.E)
        xs = x.index_select(0, tok)
        ws = w.reshape(-1)[order].unsqueeze(-1).to(x.dtype)
        # unbind once: indexing a stacked parameter per expert makes autograd build a full
        # [E, d, f] gradient for every slice, which costs E^2 and dominated the step at E = 32.
        G, U, Dn = self.w_gate.unbind(0), self.w_up.unbind(0), self.w_down.unbind(0)
        outs, a = [], 0
        for e, c in enumerate(counts.tolist()):
            if c:
                h = xs[a:a + c]
                outs.append((F.silu(h @ G[e]) * (h @ U[e])) @ Dn[e])
            a += c
        y = torch.zeros_like(x).index_add_(0, tok, torch.cat(outs) * ws)
        self.stats = {"counts": counts, "probs_mean": probs.mean(0), "weight_sum": w.sum(-1),
                      "n": x.shape[0]}
        return y.reshape(shape)


class Block(nn.Module):
    def __init__(self, d, H, ffn):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.H, self.ffn = H, ffn

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.H, d // self.H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ffn(self.n2(x))


class LM(nn.Module):
    def __init__(self, E=1, k=1, router_std=INIT_STD / 10):
        super().__init__()
        self.E, self.k = E, k
        self.emb = nn.Embedding(VOCAB_CAP, D_MODEL)
        self.pos = nn.Embedding(SEQ, D_MODEL)
        nn.init.normal_(self.emb.weight, std=INIT_STD)
        nn.init.normal_(self.pos.weight, std=INIT_STD)
        self.blocks = nn.ModuleList(Block(D_MODEL, N_HEAD, FFNBlock(D_MODEL, FFN, E, k, router_std))
                                    for _ in range(N_LAYER))
        for b in self.blocks:
            nn.init.normal_(b.qkv.weight, std=INIT_STD)
            nn.init.normal_(b.o.weight, std=INIT_STD / math.sqrt(2 * N_LAYER))
        self.nf = nn.LayerNorm(D_MODEL)

    def ffns(self):
        return [b.ffn for b in self.blocks]

    def forward(self, x):
        h = self.emb(x) + self.pos.weight[: x.shape[1]]
        for b in self.blocks:
            h = b(h)
        return self.nf(h) @ self.emb.weight.T                       # tied head


def count_params(E, k):
    """Total and active parameters from the shape alone, as §1 counts them."""
    attn = 4 * D_MODEL * D_MODEL
    norms = 4 * D_MODEL
    expert = 3 * D_MODEL * FFN
    router = D_MODEL * E if E > 1 else 0
    emb = VOCAB_CAP * D_MODEL + SEQ * D_MODEL + 2 * D_MODEL
    per_layer_total = attn + norms + E * expert + router
    per_layer_active = attn + norms + k * expert + router
    return {"total": N_LAYER * per_layer_total + emb, "active": N_LAYER * per_layer_active + emb,
            "expert": expert, "experts_total": N_LAYER * E * expert}


def grow(src, E, k, router_noise=0.0, router_std=INIT_STD / 10, seed=0):
    """Upcycle `src` into a model with E experts, each a copy of one of src's.

    Dense -> MoE: every expert is the dense network and the router is new.
    MoE -> bigger MoE: each expert is cloned E/src.E times, the router's columns are tiled
    the same way, and `router_noise` (relative to the router's own scale) is added so the
    clones are not exact ties.
    """
    torch.manual_seed(seed)
    dst = LM(E, k, router_std).to(next(src.parameters()).device)
    sd = src.state_dict()
    rep = E // src.E
    new = {}
    for name, t in sd.items():
        if name.endswith(("w_gate", "w_up", "w_down")):
            new[name] = t.repeat_interleave(rep, 0).clone()
        elif name.endswith("router"):
            r = t.repeat_interleave(rep, 1)
            new[name] = r + torch.randn_like(r) * router_noise * t.std()
        elif name.endswith("ffn.bias"):
            new[name] = t.repeat_interleave(rep, 0).clone()
        else:
            new[name] = t.clone()
    for name, t in dst.state_dict().items():
        if name not in new:                                 # dense source: fresh router and bias
            new[name] = t
    dst.load_state_dict(new)
    return dst


for E, k in ((1, 1), (8, 2), (32, 4)):
    m, c = LM(E, k), count_params(E, k)
    n = sum(p.numel() for p in m.parameters())
    assert n == c["total"], (E, n, c["total"])
    print(f"E={E:<3} k={k}: total {c['total'] / 1e6:6.2f}M  active {c['active'] / 1e6:6.2f}M  "
          f"(counted {n:,} = formula)")
MODEL_SIZES = {f"{E}x{k}": count_params(E, k) for E, k in ((1, 1), (8, 2), (32, 4))}
EVIDENCE["model"] = {"d_model": D_MODEL, "layers": N_LAYER, "heads": N_HEAD, "ffn": FFN,
                     "seq": SEQ, "batch": BATCH, "vocab": VOCAB_CAP, "sizes": MODEL_SIZES}
snapshot("model")


# %% [markdown]
# ### Does the router do what §7 and §13 say?
#
# Three properties, checked on a tiny layer where they can be checked exactly:
#
# 1. the chosen weights sum to one for every token;
# 2. a bias that changes *which* experts are chosen leaves the *weights* of the chosen experts
#    equal to their raw scores rescaled — the bias never enters the output;
# 3. the bias receives no gradient: it is a buffer, so the language loss cannot see it.

# %%
def router_unit_checks():
    torch.manual_seed(0)
    f = FFNBlock(16, 32, E=8, k=2, router_std=0.5).double()
    x = torch.randn(64, 16, dtype=torch.float64)
    f(x)
    sums_ok = bool(torch.allclose(f.stats["weight_sum"].double(), torch.ones(64, dtype=torch.float64)))
    probs = (x @ f.router).softmax(-1)
    base = probs.topk(2, -1).indices
    f.bias[base[0, 0]] = -1.0                                        # evict token 0's favourite
    y = f(x)
    idx = (probs + f.bias).topk(2, -1).indices
    changed = bool((idx[0] != base[0]).any())
    w = probs.gather(-1, idx)
    w = w / w.sum(-1, keepdim=True)
    ref = sum(w[:, j:j + 1] * torch.stack([f.expert(int(e), x[t:t + 1])[0]
                                           for t, e in enumerate(idx[:, j])])
              for j in range(2))
    weights_ok = bool(torch.allclose(y, ref, atol=1e-12))
    y.sum().backward()
    no_grad = f.bias.grad is None and "bias" not in dict(f.named_parameters())
    return {"weights_sum_to_one": sums_ok, "bias_changed_selection": changed,
            "weights_from_raw_scores": weights_ok, "bias_has_no_gradient": no_grad}


ROUTER_CHECK = router_unit_checks()
for k_, v in ROUTER_CHECK.items():
    print(f"  {'ok  ' if v else 'FAIL'} {k_}")
EVIDENCE["router_check"] = ROUTER_CHECK
snapshot("router_check")


# %% [markdown]
# ## 4 · Training, one loop for every run
#
# The token stream is cut into disjoint ranges before anything trains: the dense phase, the
# continuation phase that every branch shares, the clone phase, and a held-out validation slice
# at the very end. Branches that start from the same checkpoint read **the same tokens in the
# same order**, so the only difference between two branches is the model.
#
# Every run uses AdamW (β = 0.9/0.95, weight decay 0.1 on matrices, none on the router), clip 1.0,
# a short warmup and then a constant learning rate. There is deliberately no decay: a cosine
# decay to the end of the dense phase would make the "continue training" phase start by
# re-warming, and the bump that causes would be the schedule's, not the conversion's.
#
# Balancing, per §12 and §13:
#
# * **none** — the router trains on the language loss alone;
# * **aux** — Switch's `α·N·Σ fᵢ·Pᵢ` with α = 0.01, summed over layers;
# * **bias** — after every step, `bᵢ ← bᵢ + γ·sign(mean load − loadᵢ)` with γ = 0.001, the load
#   counted over the whole step's batch (§14's whole-batch scope).

# %%
LR, WARMUP = 1e-3, 100
TOK_PER_STEP = SEQ * BATCH
if QUICK:
    DENSE_TOKENS, CONT_TOKENS, CLONE_TOKENS, VAL_TOKENS = 300_000, 300_000, 150_000, 40_960
else:
    DENSE_TOKENS, CONT_TOKENS, CLONE_TOKENS, VAL_TOKENS = 5_000_000, 4_000_000, 2_000_000, 262_144
RANGES = {"dense": (0, DENSE_TOKENS),
          "continue": (DENSE_TOKENS, DENSE_TOKENS + CONT_TOKENS),
          "clone": (DENSE_TOKENS + CONT_TOKENS, DENSE_TOKENS + CONT_TOKENS + CLONE_TOKENS),
          "val": (len(STREAM) - VAL_TOKENS, len(STREAM))}
assert RANGES["clone"][1] <= RANGES["val"][0], "training ranges overlap validation"
LOG_EVERY = 10 if QUICK else 25


def windows(lo, hi):
    """Consecutive SEQ+1 windows of [lo, hi), batched, in order: every token read once."""
    n = (hi - lo - 1) // SEQ
    starts = lo + torch.arange(n) * SEQ
    for i in range(0, n - BATCH + 1, BATCH):
        b = torch.stack([STREAM[s:s + SEQ + 1] for s in starts[i:i + BATCH].tolist()])
        yield b[:, :-1].to(DEV), b[:, 1:].to(DEV)


@torch.no_grad()
def val_loss(model):
    model.eval()
    tot, n = 0.0, 0
    for x, y in windows(*RANGES["val"]):
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum").item()
        n += y.numel()
    model.train()
    return tot / n


def optimizer(model):
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and not n.endswith("router")]
    rest = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and not n.endswith("router"))]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                              {"params": rest, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.95))


def load_stats(counts):
    """MaxVio (§10), dead and starved experts from per-layer token counts over a window."""
    out = []
    for c in counts:
        c = c.float()
        mean = c.mean()
        out.append({"maxvio": float((c.max() - mean) / mean), "dead": int((c == 0).sum()),
                    "starved": int((c < 0.1 * mean).sum())})
    return out


def train(model, rng, balance="none", gamma=1e-3, alpha=0.01, prob_window=0, name="run",
          val_every=None):
    """Train over the token range `rng` once. Returns curves and per-window load records."""
    torch.manual_seed(1234)
    opt = optimizer(model)
    steps = (rng[1] - rng[0] - 1) // SEQ // BATCH
    val_every = val_every or max(steps // 12, 1)
    moe = model.E > 1
    rec = {"name": name, "balance": balance, "steps": steps, "tokens": steps * TOK_PER_STEP,
           "train": [], "val": [[0, val_loss(model)]], "load": [], "counts": [], "router_grad": []}
    win_loss, win_n = 0.0, 0
    win_counts = [torch.zeros(model.E, device=DEV) for _ in range(N_LAYER)] if moe else None
    t0 = time.time()
    for step, (x, y) in enumerate(windows(*rng), start=1):
        for f in model.ffns():
            f.select = "probabilistic" if step <= prob_window else "hard"
        lr = LR * min(1.0, step / WARMUP)
        for g in opt.param_groups:
            g["lr"] = lr
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        total = loss
        if moe and balance == "aux":
            aux = sum(model.E * (f.stats["counts"].float() / (f.stats["n"] * model.k)
                                 * f.stats["probs_mean"]).sum() for f in model.ffns())
            total = loss + alpha * aux
        opt.zero_grad(set_to_none=True)
        total.backward()
        if moe and (step % LOG_EVERY == 0 or step <= 5):
            # language-loss gradient only: under "aux" the aux term also reaches the router
            rg = math.sqrt(sum(float(f.router.grad.norm()) ** 2 for f in model.ffns()))
            rec["router_grad"].append([step, rg])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if moe:
            for i, f in enumerate(model.ffns()):
                c = f.stats["counts"].float()
                win_counts[i] += c
                if balance == "bias":
                    with torch.no_grad():
                        f.bias += gamma * torch.sign(c.mean() - c)
        win_loss += loss.item()
        win_n += 1
        if step % LOG_EVERY == 0 or step == steps:
            rec["train"].append([step, win_loss / win_n])
            if moe:
                cnt = [c.long().tolist() for c in win_counts]
                rec["counts"].append([step, cnt])
                rec["load"].append([step, load_stats(win_counts)])
                win_counts = [torch.zeros(model.E, device=DEV) for _ in range(N_LAYER)]
            win_loss, win_n = 0.0, 0
        if step % val_every == 0 or step == steps:
            rec["val"].append([step, val_loss(model)])
            el = time.time() - t0
            tr = rec["train"][-1][1] if rec["train"] else loss.item()
            print(f"  {name:<18} step {step:>5}/{steps}  train {tr:.3f}  "
                  f"val {rec['val'][-1][1]:.3f}  {step * TOK_PER_STEP / el:,.0f} tok/s")
    rec["seconds"] = time.time() - t0
    rec["tok_per_s"] = rec["tokens"] / rec["seconds"]
    return rec


def ckpt(name):
    return STAGES / f"{name}.pt"


def trained(name, make, rng, version="1", **kw):
    """A training run as a stage: the record is cached as JSON, the weights as a .pt beside it."""
    holder = {}

    def run():
        model = make()
        rec = train(model, rng, name=name, **kw)
        torch.save(model.state_dict(), ckpt(name))
        holder["model"] = model
        return rec

    rec = stage(name, run, version)
    model = holder.get("model")
    if model is None:
        model = make()
        model.load_state_dict(torch.load(ckpt(name), map_location=DEV))
    return rec, model


# %% [markdown]
# ## 5 · The dense model, and the conversion
#
# The dense model trains on the first 5M tokens. It is then converted into eight experts with
# top-2 routing, and the converted model is compared against the dense one **before any MoE
# step**: on the full validation slice, and logit by logit on one batch.

# %%
def dense_model():
    torch.manual_seed(0)
    return LM(1, 1).to(DEV)


DENSE, DENSE_M = trained("dense", dense_model, RANGES["dense"])
print(f"dense: val {DENSE['val'][0][1]:.3f} -> {DENSE['val'][-1][1]:.3f} over "
      f"{DENSE['tokens']:,} tokens at {DENSE['tok_per_s']:,.0f} tok/s")


def conversion_check():
    moe = grow(DENSE_M, 8, 2)
    x, _ = next(windows(*RANGES["val"]))
    DENSE_M.eval(), moe.eval()
    with torch.no_grad():
        a, b = DENSE_M(x), moe(x)
    w_sum = torch.cat([f.stats["weight_sum"] for f in moe.ffns()])
    # The router's language-loss gradient at the moment of conversion. With identical experts
    # the output is E(x) whatever the router picks, so this should be zero up to rounding;
    # the attention weights' gradient on the same batch is the scale to read it against.
    moe.train()
    xb, yb = next(windows(*RANGES["continue"]))
    F.cross_entropy(moe(xb).flatten(0, 1), yb.flatten()).backward()
    router_g = math.sqrt(sum(float(f.router.grad.norm()) ** 2 for f in moe.ffns()))
    attn_g = math.sqrt(sum(float(b.qkv.weight.grad.norm()) ** 2 for b in moe.blocks))
    return {"router_grad_at_conversion": router_g, "attn_grad_at_conversion": attn_g,"dense_val": val_loss(DENSE_M), "moe_val": val_loss(moe),
            "max_logit_diff": float((a - b).abs().max()), "logit_scale": float(a.abs().max()),
            "weight_sum_err": float((w_sum - 1).abs().max())}


CONV = stage("conversion", conversion_check)
print(f"dense val {CONV['dense_val']:.6f}   converted val {CONV['moe_val']:.6f}   "
      f"|Δ| {abs(CONV['dense_val'] - CONV['moe_val']):.2e}")
print(f"router gradient at conversion {CONV['router_grad_at_conversion']:.2e} "
      f"(attention's on the same batch: {CONV['attn_grad_at_conversion']:.2e})")
print(f"largest logit difference {CONV['max_logit_diff']:.2e} on logits up to "
      f"{CONV['logit_scale']:.1f}; router weights sum to 1 within {CONV['weight_sum_err']:.1e}")
EVIDENCE.update(dense=DENSE, conversion=CONV)
snapshot("conversion")


# %% [markdown]
# ## 6 · Continue training: dense vs mixture-of-experts
#
# Four branches from the same dense checkpoint, over the same next 4M tokens:
#
# | branch | model | balancing |
# |---|---|---|
# | `dense-continued` | the dense model, unchanged | — |
# | `moe-bias` | 8 experts, top-2 | aux-loss-free bias (§13) — **the main run** |
# | `moe-aux` | 8 experts, top-2 | Switch auxiliary loss (§12) |
# | `moe-none` | 8 experts, top-2 | none |
#
# `dense-continued` is the control the assignment does not ask for and the claim needs: a dense
# model also keeps reducing its loss on new tokens, so "the MoE's loss falls" alone shows only
# that training was not broken. Whether the MoE *learns more* is the comparison between the two.
# The comparison is at equal tokens; per token the MoE does about 1.6× the dense model's
# feed-forward work (two experts instead of one), which the page reports alongside.

# %%
def from_dense(E=8, k=2):
    return lambda: grow(DENSE_M, E, k)


def dense_copy():
    m = dense_model()
    m.load_state_dict(DENSE_M.state_dict())
    return m


BRANCHES = {}
BRANCHES["dense-continued"], _ = trained("dense-continued", dense_copy, RANGES["continue"])
BRANCHES["moe-bias"], MOE_M = trained("moe-bias", from_dense(), RANGES["continue"], balance="bias")
BRANCHES["moe-aux"], _ = trained("moe-aux", from_dense(), RANGES["continue"], balance="aux")
BRANCHES["moe-none"], _ = trained("moe-none", from_dense(), RANGES["continue"], balance="none")
EVIDENCE["branches"] = BRANCHES
snapshot("branches")

print(f"\n{'branch':<18}{'val start':>10}{'val end':>9}{'Δ':>8}{'tok/s':>9}{'MaxVio':>8}{'dead':>6}")
for n_, r in BRANCHES.items():
    lv = r["load"][-1][1] if r["load"] else None
    mv = f"{np.mean([l['maxvio'] for l in lv]):.2f}" if lv else "—"
    dd = f"{sum(l['dead'] for l in lv)}" if lv else "—"
    print(f"{n_:<18}{r['val'][0][1]:>10.3f}{r['val'][-1][1]:>9.3f}"
          f"{r['val'][-1][1] - r['val'][0][1]:>8.3f}{r['tok_per_s']:>9,.0f}{mv:>8}{dd:>6}")


# %% [markdown]
# ### Did the experts actually become different?
#
# Upcycled experts start identical, and the router has no gradient until they differ (see the
# top of this notebook). If they never separated, the "MoE" would be the dense model computed
# eight times. This measures, per layer, how far each expert has moved from the dense network
# it was copied from, and how far apart the experts are from one another, both relative to the
# size of the dense network's weights.

# %%
def expert_spread(model, family=None):
    rows = []
    for i, (f, fd) in enumerate(zip(model.ffns(), DENSE_M.ffns())):
        W = torch.cat([f.w_gate.flatten(1), f.w_up.flatten(1), f.w_down.flatten(1)], 1).float()
        W0 = torch.cat([fd.w_gate.flatten(1), fd.w_up.flatten(1), fd.w_down.flatten(1)], 1).float()
        scale = W0.norm()
        from_dense = ((W - W0).norm(dim=1) / scale).tolist()
        D = torch.cdist(W, W) / scale
        off = ~torch.eye(len(W), dtype=torch.bool, device=W.device)
        row = {"layer": i, "from_dense": from_dense, "pairwise_mean": float(D[off].mean())}
        if family:
            fam = torch.arange(len(W), device=W.device) // family
            same = (fam[:, None] == fam[None, :]) & off
            row["within_family"] = float(D[same].mean())
            row["across_family"] = float(D[~same & off].mean())
        rows.append(row)
    return rows


SPREAD = expert_spread(MOE_M)
for r in SPREAD:
    print(f"layer {r['layer']}: moved from dense {np.mean(r['from_dense']):.3f} · "
          f"apart from each other {r['pairwise_mean']:.3f}")
EVIDENCE["spread"] = SPREAD
snapshot("spread")


# %% [markdown]
# ## 7 · Growing again: clone families
#
# §15 records a failure the published growth studies do not: when the Lightning LM team cloned
# experts without redrawing any neurons and kept hard top-k routing, whole families of clones
# collapsed onto a few members, because the router could not tell near-identical clones apart.
# The fix was probabilistic selection during an early window.
#
# The trained `moe-bias` model is grown here from 8 experts to 32 — each expert cloned 4 times,
# the router's columns tiled with small noise — and top-k rises from 2 to 4 (the same direction
# as the notes' 20 → 460 experts with top-k 2 → 12). Three arms over the same next 2M tokens:
#
# | arm | selection | balancing |
# |---|---|---|
# | `clone-hard-none` | hard top-4 | none |
# | `clone-hard-bias` | hard top-4 | bias |
# | `clone-prob-bias` | probabilistic for the first 150 steps, then hard | bias |
#
# This growth is **not** function-preserving, and that is measured rather than hidden: tiled
# logits rank all four clones of a family together, so a token's top-4 is one family's four
# clones where it used to be two different experts.

# %%
PROB_WINDOW = 20 if QUICK else 150
CLONE = {}
for arm, bal, pw in (("clone-hard-none", "none", 0), ("clone-hard-bias", "bias", 0),
                     ("clone-prob-bias", "bias", PROB_WINDOW)):
    CLONE[arm], m_ = trained(arm, lambda: grow(MOE_M, 32, 4, router_noise=0.01, seed=7),
                             RANGES["clone"], balance=bal, prob_window=pw)
    CLONE[arm]["spread"] = expert_spread(m_, family=4)
CLONE_START_VAL = BRANCHES["moe-bias"]["val"][-1][1]
EVIDENCE["clone"] = {"arms": CLONE, "from_val": CLONE_START_VAL, "prob_window": PROB_WINDOW}
snapshot("clone")

print(f"\ngrown from moe-bias at val {CLONE_START_VAL:.3f}")
print(f"{'arm':<18}{'val@0':>8}{'val end':>9}{'dead (max over run)':>21}{'dead end':>10}"
      f"{'within/across':>15}")
for n_, r in CLONE.items():
    dead_series = [sum(l["dead"] for l in ls) for _, ls in r["load"]]
    sp = r["spread"]
    ratio = np.mean([s["within_family"] for s in sp]) / np.mean([s["across_family"] for s in sp])
    print(f"{n_:<18}{r['val'][0][1]:>8.3f}{r['val'][-1][1]:>9.3f}{max(dead_series):>21}"
          f"{dead_series[-1]:>10}{ratio:>15.2f}")


# %% [markdown]
# ## 8 · Gates
#
# Each gate is a claim the page makes. The script exits non-zero if any fails.

# %%
def mean_maxvio(rec):
    return float(np.mean([l["maxvio"] for l in rec["load"][-1][1]]))


main = BRANCHES["moe-bias"]
dc = BRANCHES["dense-continued"]
GATES = {
    "the notes' reference figures reproduce": all(c["ok"] for c in NOTES_CHECK),
    "parameter counts equal the formula": True,                     # asserted in §3
    "router weights sum to one and the bias stays out of the gradient": all(ROUTER_CHECK.values()),
    "dense pretraining reduces validation loss": DENSE["val"][-1][1] < DENSE["val"][0][1] - 0.5,
    "conversion preserves the function (val loss within 1e-4)":
        abs(CONV["dense_val"] - CONV["moe_val"]) < 1e-4,
    "conversion preserves the function (logits within 1e-3)": CONV["max_logit_diff"] < 1e-3,
    "MoE starts exactly where the dense model stopped":
        abs(main["val"][0][1] - DENSE["val"][-1][1]) < 1e-4,
    "MoE keeps reducing validation loss after conversion": main["val"][-1][1] < main["val"][0][1] - 0.05,
    "no MoE checkpoint is worse than where the dense model stopped":
        all(v < main["val"][0][1] for _, v in main["val"][1:]),
    "the router gets no language-loss gradient at conversion":
        CONV["router_grad_at_conversion"] < 1e-4 * CONV["attn_grad_at_conversion"],
    "experts separated from one another": min(r["pairwise_mean"] for r in SPREAD) > 1e-3,
    "bias balancing beats no balancing on MaxVio":
        mean_maxvio(BRANCHES["moe-bias"]) < mean_maxvio(BRANCHES["moe-none"]),
    "aux loss beats no balancing on MaxVio":
        mean_maxvio(BRANCHES["moe-aux"]) < mean_maxvio(BRANCHES["moe-none"]),
    "no token range is read twice and none overlaps validation":
        RANGES["dense"][1] <= RANGES["continue"][0] and RANGES["continue"][1] <= RANGES["clone"][0]
        and RANGES["clone"][1] <= RANGES["val"][0],
}
# Findings are reported as measured, whichever way they come out; they are not gates.
FINDINGS = {
    "moe_minus_dense_val": main["val"][-1][1] - dc["val"][-1][1],
    "moe_beats_dense_at_equal_tokens": main["val"][-1][1] < dc["val"][-1][1],
    "moe_throughput_ratio": main["tok_per_s"] / dc["tok_per_s"],
    "maxvio": {n_: mean_maxvio(r) for n_, r in BRANCHES.items() if r["load"]},
    "clone_dead_max": {n_: max(sum(l["dead"] for l in ls) for _, ls in r["load"])
                       for n_, r in CLONE.items()},
    "clone_dead_end": {n_: sum(l["dead"] for l in r["load"][-1][1]) for n_, r in CLONE.items()},
}
EVIDENCE.update(gates=GATES, findings=FINDINGS, ranges=RANGES)
snapshot("gates")
for g, ok in GATES.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {g}")
print(f"\n{sum(GATES.values())}/{len(GATES)} gates pass · wall {time.time() - T_START:.0f}s")
print(f"MoE vs dense-continued at equal tokens: {FINDINGS['moe_minus_dense_val']:+.3f} val loss, "
      f"{FINDINGS['moe_throughput_ratio']:.2f}x the throughput")
if __name__ == "__main__" and not all(GATES.values()):
    raise SystemExit(1)
