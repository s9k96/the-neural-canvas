# %% [markdown]
# # Session 13 — Reversibility
#
# **ERA V5 · a reversible stack keeps the two states at the ends and rebuilds everything in
# between on the way back.**
#
# The assignment: *train a 20M LLM for 50M tokens, fix a batch size you can run; train again
# with reversibility, reporting which variant worked; train again with reversibility pushed to
# the maximum batch size; report final loss, speed, memory peak and other findings.*
#
# Sections 1–15 of this session are about dividing a model across many GPUs. This assignment
# draws on §§16–17 only, and runs on one. What the earlier sections supply is the reason to
# care: ZeRO divides the *training state* across GPUs but never divides the activations, and
# one 8,192-token sequence through a 30B model stores **127.5 GiB** of them. Reversibility
# does not divide activations. It removes them.
#
# ---
#
# ### The one claim that has to be checked before any measurement means anything
#
# A reversible stack that reconstructs slightly wrong still trains. The loss still falls. So
# the loss curve is not evidence that the implementation is correct, and §4 below compares
# reversible gradients against ordinary autograd instead — in float64, where an exact
# implementation must agree to rounding.
#
# Three bugs were found that way while building this, every one of which left the loss curve
# looking perfectly healthy.

# %%
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import s13_reversible as R

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if HERE.name != "s13-distributed-training-2" and (HERE / "s13-distributed-training-2").is_dir():
    HERE = HERE / "s13-distributed-training-2"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUICK = os.environ.get("S13_QUICK") == "1"       # CPU smoke mode: tiny, seconds, not a result

# --- resumable stages -------------------------------------------------------------------
# A hosted runtime can disappear mid-run, and this harness books about an hour of GPU. Every
# expensive section therefore writes its result to disk the moment it finishes, and re-running
# the notebook reloads what is already there instead of recomputing it.
#
# If Google Drive is mounted, stages go there: a dropped *connection* keeps the VM's local
# disk, but a dropped *VM* does not, and Drive is the only thing that survives both. Mount it
# first if you want that safety:
#     from google.colab import drive; drive.mount('/content/drive')
DRIVE = Path("/content/drive/MyDrive")
STAGES = (DRIVE / "s13_stages") if DRIVE.is_dir() else (OUT / "stages")
STAGES.mkdir(parents=True, exist_ok=True)
print(f"stage cache: {STAGES}"
      + ("  (Google Drive — survives a lost VM)" if DRIVE.is_dir()
         else "  (local disk — mount Drive to survive a lost VM)"))


def stage(name, fn, version="1", force=False):
    """Run `fn` once, cache its result, and reload it on any later run.

    `version` is how a cached stage gets invalidated when the code that produced it changes.
    Without it, fixing a stage's logic leaves every existing cache silently wrong, and the
    only remedy is asking whoever ran it to go and delete files by hand — which is a fine way
    to end up publishing numbers from the previous version of an experiment. Bump the version
    and the stage recomputes itself everywhere.
    """
    path = STAGES / f"{name}.json"
    if path.exists() and not force:
        try:
            blob = json.loads(path.read_text())
        except json.JSONDecodeError:
            blob = None
        if isinstance(blob, dict) and "__stage__" in blob:
            cached_version, value = blob["__stage__"], blob["value"]
        else:
            cached_version, value = "1", blob          # written before versioning existed
        if value is not None and cached_version == version:
            print(f"[resume] {name}")
            return value
        print(f"[stale]  {name} · cached v{cached_version}, need v{version} · recomputing")
    t0 = time.time()
    result = fn()
    path.write_text(json.dumps({"__stage__": version, "value": result}, default=float))
    print(f"[done]   {name} · {time.time() - t0:.0f}s · cached")
    return result


def release():
    """Drop whatever the last stage was holding before the next one allocates."""
    import gc
    gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

T_START = time.time()
EVIDENCE = {"meta": {
    "python": platform.python_version(), "torch": torch.__version__,
    "platform": platform.platform(), "device": str(DEV), "quick": QUICK,
}}
if DEV.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    # is_bf16_supported() counts emulation by default, so a Turing card answers yes without
    # any native support. Ask for the honest answer as well.
    try:
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        native_bf16 = torch.cuda.is_bf16_supported()
    EVIDENCE["meta"].update({
        "gpu": props.name, "vram_gb": round(props.total_memory / 1e9, 1),
        "capability": f"{props.major}.{props.minor}",
        "bf16_native": bool(native_bf16),
        "bf16_including_emulation": bool(torch.cuda.is_bf16_supported()),
    })
    print(f"{props.name} · {props.total_memory/1e9:.1f} GB · capability "
          f"{props.major}.{props.minor} · bf16 native {native_bf16}")
print(f"python {platform.python_version()} · torch {torch.__version__} · device {DEV}"
      + ("  [QUICK smoke mode]" if QUICK else ""))


# %% [markdown]
# ## 1 · What reversibility removes
#
# Every layer of an ordinary stack stores the intermediate results its backward pass will need.
# Across 96 layers of the session's 30B model that is 127.5 GiB for a single 8,192-token
# sequence. A reversible stack stores the state entering the stack, the state leaving it, and
# the working memory of the one layer it is currently rebuilding — and **the layer count
# disappears from that expression**.

# %%
GIB = 1024 ** 3
D_REF, LAYERS_REF, BYTES_PER_TOKEN_UNIT = 5120, 96, 34     # the session's reference model

ACT_TABLE = []
for seq in (8192, 32768, 131072):
    stored = seq * D_REF * BYTES_PER_TOKEN_UNIT * LAYERS_REF / GIB
    boundary = 2 * seq * D_REF * 2 / GIB                    # two states, 16-bit
    one_layer = seq * D_REF * BYTES_PER_TOKEN_UNIT / GIB
    ACT_TABLE.append({"seq": seq, "stored_gib": stored,
                      "reversible_gib": boundary + one_layer,
                      "ratio": stored / (boundary + one_layer)})
    print(f"{seq:>7} tokens: stored {stored:8.1f} GiB   reversible "
          f"{boundary + one_layer:6.1f} GiB   {stored / (boundary + one_layer):5.1f}x")
print("\nThe ratio is the same at every length because the 96 cancels: what remains grows with")
print("the number of tokens, not with the depth of the model. That is the whole claim, and §6")
print("below measures it on a model small enough to fit here.")
EVIDENCE["activation_arithmetic"] = ACT_TABLE


# %% [markdown]
# ## 2 · Tokens
#
# 50 million of them, streamed rather than repeated: the S6 corpus this repo has used since
# Session 6 is 656,920 tokens, and reaching 50M from it would mean 76 epochs, which trains a
# memoriser and makes "final loss" mean something other than what the assignment intends. The
# **tokenizer is still the frozen Sarvam-1** one from Sessions 9–12, so that thread holds.

# %%
TOKENIZER_REPO = "sarvamai/sarvam-1"
FROZEN_TOKENIZER_SHA256 = "bb5115a36ddb956a4ee0fd534e9870dd69157835622aec9c53062896f883c072"
TARGET_TOKENS = 2_000_000 if QUICK else 50_000_000
# In smoke mode the vocabulary has to shrink too: at a 2M-parameter target the
# embedding and head alone would be 2 x 8192 x 128 = 2.1M, leaving the width search
# no room to match parameter counts across rules.
VOCAB_CAP = 1024 if QUICK else 8192
CACHE = OUT / f"tokens_{TARGET_TOKENS}_{VOCAB_CAP}.npy"

DATASETS = [("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
            ("wikitext", "wikitext-103-raw-v1", "text"),
            ("roneneldan/TinyStories", None, "text")]


def load_tokenizer():
    import hashlib
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
    return Tokenizer.from_file(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_corpus(tok, n_tokens):
    """Offline fallback: S6's committed corpus, repeated to length.

    Only for the CPU smoke path, where `datasets` is absent and the point is to exercise the
    code rather than to train anything. A real run streams, because reaching 50M tokens from
    657k would mean 76 epochs.
    """
    corpus = HERE.parent / "s06-dataset-creation" / "corpus"
    base = []
    for f in sorted(corpus.glob("*.jsonl")):
        if f.stem == "eval_registry_docs":
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                text = "\n".join(sg["text"] for sg in d["segments"] if sg.get("text"))
                if len(text) > 200:
                    base.extend(tok.encode(text).ids)
    if not base:
        raise RuntimeError("no local corpus either")
    out = []
    while len(out) < n_tokens:
        out.extend(base)
    return out[:n_tokens]


def build_stream():
    """Stream text until TARGET_TOKENS, tokenize, cap the vocabulary, cache to disk.

    The cache carries a sidecar recording where the tokens came from. Without it, a resumed
    run reports its corpus as "cache" and the evidence no longer says which dataset produced
    the numbers, which tokenizer hashed them, or what the vocabulary cap covered -- provenance
    that cannot be reconstructed afterwards from the token ids alone.
    """
    import numpy as np
    meta_path = CACHE.with_suffix(".meta.json")
    if CACHE.exists():
        arr = np.load(CACHE)
        if meta_path.exists():
            build_stream.meta = json.loads(meta_path.read_text())
            src = build_stream.meta.get("source", "unknown")
            import hashlib
            actual = hashlib.sha256(arr.tobytes()).hexdigest()
            recorded = build_stream.meta.get("token_sha256")
            if recorded and recorded != actual:
                raise RuntimeError(
                    f"token cache does not match its sidecar: {actual[:12]} vs "
                    f"{recorded[:12]}. The corpus changed under a resumed run.")
            build_stream.meta["token_sha256_verified"] = bool(recorded)
        else:
            build_stream.meta = {"source": "unknown (cache predates provenance recording)"}
            src = build_stream.meta["source"]
        print(f"cached: {len(arr):,} tokens from {CACHE.name} · source {src}")
        return arr, src
    from collections import Counter
    tok, sha = load_tokenizer()
    assert sha == FROZEN_TOKENIZER_SHA256, f"tokenizer drifted: {sha}"

    ids, source = [], None
    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None
        print("`datasets` not installed — falling back to the local S6 corpus")
    for name, cfg, field in (DATASETS if load_dataset else []):
        try:
            ds = load_dataset(name, cfg, split="train", streaming=True)
            t0 = time.time()
            for rec in ds:
                text = rec.get(field) or ""
                if len(text) < 200:
                    continue
                ids.extend(tok.encode(text).ids)
                if len(ids) >= TARGET_TOKENS:
                    break
            if len(ids) >= TARGET_TOKENS * 0.9:
                source = name
                print(f"streamed {len(ids):,} tokens from {name} in {time.time()-t0:.0f}s")
                break
            print(f"{name} gave only {len(ids):,} tokens; trying the next source")
            ids = []
        except Exception as exc:
            print(f"{name} unavailable ({str(exc)[:70]}); trying the next source")
            ids = []
    if source is None:
        ids, source = local_corpus(tok, TARGET_TOKENS), "s06-corpus (local fallback)"
        print(f"using {len(ids):,} tokens from {source}")

    ids = ids[:TARGET_TOKENS]
    counts = Counter(ids)
    keep = [i for i, _ in counts.most_common(VOCAB_CAP)]
    remap = {old: new for new, old in enumerate(sorted(keep))}
    coverage = sum(counts[i] for i in keep) / len(ids)
    arr = np.fromiter((remap.get(i, 0) for i in ids), dtype=np.uint16, count=len(ids))
    np.save(CACHE, arr)
    print(f"capped vocabulary to {VOCAB_CAP:,} ids, covering {coverage:.1%} of occurrences")
    # Hash the tokens themselves. The sidecar says where they came from; this says whether a
    # later rebuild actually reproduced them. Without it, re-streaming to recover a lost
    # source could relabel a run with a corpus it never saw -- a worse outcome than an
    # unknown source, because it would look authoritative.
    import hashlib
    token_sha = hashlib.sha256(arr.tobytes()).hexdigest()
    build_stream.meta = {"source": source, "coverage": coverage, "tokenizer_sha256": sha,
                         "dataset_config": next((c for n, c, _ in DATASETS if n == source), None),
                         "token_sha256": token_sha,
                         "built_utc": __import__("datetime").datetime.utcnow().isoformat(
                             timespec="seconds") + "Z"}
    meta_path.write_text(json.dumps(build_stream.meta, indent=2))
    return arr, source


STREAM, SOURCE = build_stream()
STREAM_T = torch.from_numpy(STREAM.astype("int64"))
print(f"{len(STREAM_T):,} tokens ready · source {SOURCE}")
EVIDENCE["corpus"] = {"tokens": int(len(STREAM_T)), "source": str(SOURCE),
                      "vocab_cap": VOCAB_CAP, "target": TARGET_TOKENS,
                      **getattr(build_stream, "meta", {})}


# %% [markdown]
# ## 3 · One size for every rule
#
# A revnet block runs on half the channels, so at equal width it has about a quarter of a
# standard block's parameters and a quarter of its FLOPs. Benchmarking the two at the same
# `d_model` compares a model against a smaller model — which is exactly the mistake the sizing
# probe made, and it made revnet look 1.8× faster than it is. Every rule below is fitted to
# the same parameter count instead.

# %%
SEQ = 128 if QUICK else 512
BASE_LAYERS = 4 if QUICK else 10
TARGET_PARAMS = 2_000_000 if QUICK else 20_000_000
RULE_SET = ("standard", "midpoint", "blended", "revnet")
H_STEP, GAMMA = 0.25, 0.5                     # the Lightning LM values §16 records


def cfg_for(rule, n_layer=None, target=None):
    """Same depth, same residual-stream width, same parameter count — differ only in the rule.

    Letting each rule pick its own d_model reaches 20M too, but by a different route: a
    revnet at d_model 224 and a standard model at 288 have equal parameters and unequal
    stream widths, which changes activation sizes and FLOPs and quietly makes the throughput
    comparison meaningless. So the width is fitted once, from the standard rule, and every
    other rule matches the parameter count through d_ff alone.
    """
    n_layer = n_layer or BASE_LAYERS
    target = target or TARGET_PARAMS
    d, _, _ = R.fit_config(target, "standard", n_layer, VOCAB_CAP, SEQ)
    d_ff = R.match_params(target, rule, n_layer, d, VOCAB_CAP, SEQ)
    cfg = R.Config(rule=rule, n_layer=n_layer, d_model=d, d_ff=d_ff, vocab_size=VOCAB_CAP,
                   block_size=SEQ, h=1.0 if rule == "standard" else H_STEP,
                   gamma=GAMMA if rule == "blended" else 0.0)
    return cfg, R.count_params(cfg)


CFG = {}
print(f"{'rule':<10}{'layers':>7}{'d_model':>9}{'d_ff':>7}{'params':>12}{'err':>8}")
for rule in RULE_SET:
    CFG[rule], n = cfg_for(rule)
    print(f"{rule:<10}{CFG[rule].n_layer:>7}{CFG[rule].d_model:>9}{CFG[rule].d_ff:>7}"
          f"{n:>12,}{abs(n - TARGET_PARAMS) / TARGET_PARAMS:>7.1%}")
EVIDENCE["configs"] = {r: {"d_model": c.d_model, "d_ff": c.d_ff, "n_layer": c.n_layer,
                           "params": R.count_params(c), "h": c.h, "gamma": c.gamma}
                       for r, c in CFG.items()}


# %% [markdown]
# ## 4 · Is the reversible stack correct?
#
# Two separate questions, and conflating them is how a correct implementation gets mistaken
# for a broken one.
#
# * **Is the arithmetic exact?** Ask in float64. An exact reversible stack differs from
#   ordinary autograd only by rounding, so the error must collapse when precision rises.
# * **How much does it drift in practice?** Ask in float32. That error is real, it grows with
#   depth because each reconstruction feeds the next, and §7 measures it properly.
#
# Euler is included precisely because it *fails*: `p + h·f(p)` has the same flaw as the
# ordinary residual, since inverting it needs `f` at the state being recovered.

# %%
def _correctness():
    out = {}
    for rule in R.REVERSIBLE:
        small = R.Config(rule=rule, n_layer=6, d_model=128, d_ff=256, vocab_size=512,
                         block_size=64, h=1.0 if rule == "revnet" else H_STEP,
                         gamma=GAMMA if rule == "blended" else 0.0)
        e64 = R.gradient_check(small, batch=2, seq=16, device="cpu",
                               dtype=torch.float64)["worst_rel_grad_error"]
        e32 = R.gradient_check(small, batch=2, seq=16, device=str(DEV),
                               dtype=torch.float32)["worst_rel_grad_error"]
        out[rule] = {"exact_fp64": e64, "drift_fp32": e32, "exact": e64 < 1e-9}
    try:
        R.invert("euler", torch.zeros(1), torch.zeros(1), torch.zeros(1), H_STEP, 0.0)
        out["euler_reversible"] = True
    except ValueError:
        out["euler_reversible"] = False
    return out


CORRECT = stage("correctness", _correctness)
print(f"{'rule':<10}{'float64 (exactness)':>22}{'float32 (drift)':>18}{'':>8}")
for rule in R.REVERSIBLE:
    c = CORRECT[rule]
    print(f"{rule:<10}{c['exact_fp64']:>22.2e}{c['drift_fp32']:>18.2e}"
          f"{'  ok' if c['exact'] else '  FAIL':>8}")
print(f"\neuler reversible: {CORRECT['euler_reversible']} — it is in the comparison as the"
      " negative result")

EVIDENCE["correctness"] = CORRECT
release()


# %% [markdown]
# ## 5 · The three runs the assignment asks for
#
# Same tokens, same seed, same schedule. Run 1 against run 2 isolates what reversibility
# *costs*, because only the memory path changes. Run 2 against run 3 isolates what it *buys*,
# because the saved memory is spent on batch size.

# %%
LR, WARMUP = 3e-4, 100
AMP = DEV.type == "cuda"                     # fp16 matmuls, fp32 residual stream and master weights


def batches(stream, step, batch, seq, seed=1234):
    g = torch.Generator().manual_seed(seed + step)
    ix = torch.randint(len(stream) - seq - 1, (batch,), generator=g)
    x = torch.stack([stream[i:i + seq] for i in ix])
    y = torch.stack([stream[i + 1:i + 1 + seq] for i in ix])
    return x.to(DEV), y.to(DEV)


def train(rule, reversible, batch, tokens, seq=None, n_layer=None, log_every=50,
          chunks=8, seed=1337):
    """One training run. Returns loss curve, throughput and peak memory."""
    seq = seq or SEQ
    cfg = CFG[rule] if n_layer is None else cfg_for(rule, n_layer)[0]
    torch.manual_seed(seed)
    model = R.Model(cfg, reversible=reversible).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=AMP)      # torch >= 2.4
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    steps = max(1, tokens // (batch * seq))
    R.peak_bytes_reset(DEV)
    losses, t0, seen = [], None, 0
    for s in range(steps):
        for grp in opt.param_groups:
            grp["lr"] = LR * min(1.0, (s + 1) / WARMUP)
        x, y = batches(STREAM_T, s, batch, seq)
        with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=AMP):
            hidden = model.hidden(x)
        loss = R.chunked_ce(hidden.float(), model.head_for_loss(), y, chunks=chunks)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        seen += batch * seq
        if s == 1:                                   # start the clock after warm-up
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            t0, seen = time.time(), 0
        if s % log_every == 0 or s == steps - 1:
            losses.append({"step": s, "tokens": s * batch * seq, "loss": loss.item()})
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    wall = max(time.time() - (t0 or time.time()), 1e-9)
    peak = R.peak_bytes_read(DEV)
    out = {"rule": rule, "reversible": reversible, "batch": batch, "seq": seq,
           "n_layer": cfg.n_layer, "params": R.count_params(cfg), "steps": steps,
           "tokens": steps * batch * seq, "final_loss": losses[-1]["loss"],
           "losses": losses, "tok_s": seen / wall, "wall_s": wall, "peak_bytes": peak}
    del model, opt
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return out


def largest_batch(rule, reversible, lo=1, hi=1024, seq=None):
    """Binary search, not doubling: the probe's powers of two only resolve to a factor of 2."""
    seq = seq or SEQ
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            train(rule, reversible, mid, tokens=mid * seq * 3, seq=seq, log_every=10 ** 9)
            best, lo = mid, mid + 1
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
            hi = mid - 1
    return best


# %%
# The batch the assignment asks you to "fix": the largest the ordinary path can run.
def _find_batches():
    return {"fixed": 2 if QUICK else largest_batch("standard", False),
            "reversible_max": 4 if QUICK else largest_batch("midpoint", True)}


_b = stage("batches", _find_batches)
# Run at the very top of the card and a long run will eventually lose to fragmentation: the
# first attempt peaked at 14,293 MB of 15,360 and the session died during run 2. Back off one
# notch. "A batch size that you can run" is the assignment's own wording, and this is it.
B_FIX = max(1, int(_b["fixed"] * 0.85))
B_REV = max(1, int(_b["reversible_max"] * 0.85))
print(f"largest batch, stored activations : {_b['fixed']}  -> using {B_FIX}")
print(f"largest batch, reversible         : {_b['reversible_max']}  -> using {B_REV}"
      f"   ({_b['reversible_max'] / max(_b['fixed'],1):.1f}x headroom)")
release()

RUN_TOKENS = 200_000 if QUICK else TARGET_TOKENS
est = RUN_TOKENS / 20000 / 60
print(f"\nthree runs of {RUN_TOKENS:,} tokens — roughly {3 * est:.0f} minutes if this card "
      f"sustains 20k tok/s\n")

RUNS = {}
RUNS["standard"] = stage("run_standard",
                         lambda: train("standard", False, B_FIX, RUN_TOKENS))
release()
print(f"1. standard,   batch {B_FIX:<4} loss {RUNS['standard']['final_loss']:.4f}  "
      f"{RUNS['standard']['tok_s']:,.0f} tok/s  peak "
      f"{(RUNS['standard']['peak_bytes'] or 0)/1e6:,.0f} MB")

RUNS["reversible_same"] = stage("run_reversible_same",
                                lambda: train("midpoint", True, B_FIX, RUN_TOKENS))
release()
print(f"2. reversible, batch {B_FIX:<4} loss {RUNS['reversible_same']['final_loss']:.4f}  "
      f"{RUNS['reversible_same']['tok_s']:,.0f} tok/s  peak "
      f"{(RUNS['reversible_same']['peak_bytes'] or 0)/1e6:,.0f} MB")

RUNS["reversible_max"] = stage("run_reversible_max",
                               lambda: train("midpoint", True, B_REV, RUN_TOKENS))
release()
print(f"3. reversible, batch {B_REV:<4} loss {RUNS['reversible_max']['final_loss']:.4f}  "
      f"{RUNS['reversible_max']['tok_s']:,.0f} tok/s  peak "
      f"{(RUNS['reversible_max']['peak_bytes'] or 0)/1e6:,.0f} MB")

a, b, c = RUNS["standard"], RUNS["reversible_same"], RUNS["reversible_max"]
print(f"\nwhat it costs  (1 -> 2, batch held): {b['tok_s']/a['tok_s']:.2f}x throughput, "
      f"{(b['peak_bytes'] or 1)/(a['peak_bytes'] or 1):.2f}x memory")
print(f"what it buys   (2 -> 3, batch freed): {c['tok_s']/b['tok_s']:.2f}x throughput at "
      f"{B_REV/max(B_FIX,1):.1f}x the batch")
EVIDENCE["runs"] = RUNS
EVIDENCE["batches"] = {"fixed": B_FIX, "reversible_max": B_REV}


# %% [markdown]
# ## 6 · Does activation memory really stop depending on depth?
#
# This is the session's central claim and the notes assert it without showing it. Parameter
# count is held at 20M while the depth changes, so the only thing moving is the number of
# layers whose activations would have to be stored.

# %%
DEPTHS = (2, 4) if QUICK else (4, 8, 16, 32)
def _depth_sweep():
    """Vary depth at FIXED WIDTH, letting the parameter count move.

    The first version of this held parameters fixed instead, which sounds like the tighter
    control and is not. Parameters go as n_layer x d_model^2, so pinning them forces
    d_model ~ 1/sqrt(depth); activations go as n_layer x d_model, so they then grow as
    sqrt(depth) rather than linearly, and the reversible column *shrinks* because the stream
    narrowed. Width and depth both moved, so neither column measured depth.

    Section 17's claim is about depth at a given width: a 96-layer and a 20-layer model store
    the same two boundary states. So width is what has to be held.
    """
    rows = []
    base_d = CFG["midpoint"].d_model
    for n_layer in DEPTHS:
        # Width and d_ff both held at the base config's; only n_layer moves, so the
        # parameter count is free to vary and is reported alongside.
        cfg = R.Config(rule="midpoint", n_layer=n_layer, d_model=base_d, d_ff=CFG["midpoint"].d_ff,
                       vocab_size=VOCAB_CAP, block_size=SEQ, h=H_STEP)
        n = R.count_params(cfg)
        x = torch.randint(VOCAB_CAP, (2, SEQ), device=DEV)
        y = torch.randint(VOCAB_CAP, (2, SEQ), device=DEV)
        row = {"n_layer": n_layer, "d_model": cfg.d_model, "params": n}
        for key, rev in (("stored", False), ("reversible", True)):
            torch.manual_seed(0)
            m = R.Model(cfg, reversible=rev).to(DEV)
            with R.activation_bytes() as acts:
                loss = R.chunked_ce(m.hidden(x).float(), m.head_for_loss(), y)
            loss.backward()
            row[key] = acts[0]
            del m
            release()
        row["ratio"] = row["stored"] / max(row["reversible"], 1)
        rows.append(row)
    return rows


DEPTH_SWEEP = stage("depth_sweep", _depth_sweep, version="2")  # v2: width held fixed
print(f"{'layers':>7}{'d_model':>9}{'params':>12}{'stored act MB':>15}"
      f"{'reversible act MB':>19}{'ratio':>8}")
for row in DEPTH_SWEEP:
    print(f"{row['n_layer']:>7}{row['d_model']:>9}{row['params']:>12,}"
          f"{row['stored']/1e6:>15.1f}{row['reversible']/1e6:>19.1f}{row['ratio']:>7.1f}x")

first, last = DEPTH_SWEEP[0], DEPTH_SWEEP[-1]
growth_stored = last["stored"] / first["stored"]
growth_rev = last["reversible"] / first["reversible"]
depth_growth = last["n_layer"] / first["n_layer"]
print(f"\n{depth_growth:.0f}x the depth at fixed width: stored activations grew "
      f"{growth_stored:.2f}x, reversible grew {growth_rev:.2f}x.")
print("The first number should track the depth. The second should not move at all, because")
print("the only thing a reversible stack keeps is the pair of states at the ends.")
EVIDENCE["depth_sweep"] = {"rows": DEPTH_SWEEP, "depth_growth": depth_growth,
                           "stored_growth": growth_stored, "reversible_growth": growth_rev}


# %% [markdown]
# ## 7 · Reconstruction drift
#
# Not in the session notes, and it is the thing that limits how deep a reversible stack can
# go. Reversal *subtracts*, so each rebuilt state carries the error of the one above it, and
# the error compounds down the stack. In low precision it compounds faster — the same failure
# mode Session 12 measured when a bfloat16 ring summed 1…32 to 524 instead of 528.

# %%
DRIFT_DTYPES = [("float32", torch.float32)]
if DEV.type == "cuda":
    DRIFT_DTYPES.append(("float16", torch.float16))
def _drift():
    rows = []
    for rule in ("midpoint", "revnet"):
        for name, dt in DRIFT_DTYPES:
            row = {"rule": rule, "dtype": name, "by_depth": {}}
            for n_layer in (2, 4, 8, 16, 32):
                small = R.Config(rule=rule, n_layer=n_layer, d_model=128, d_ff=256,
                                 vocab_size=512, block_size=64,
                                 h=1.0 if rule == "revnet" else H_STEP)
                try:
                    e = R.gradient_check(small, batch=2, seq=16, device=str(DEV),
                                         dtype=dt)["worst_rel_grad_error"]
                except Exception:
                    e = float("nan")
                row["by_depth"][str(n_layer)] = e
            rows.append(row)
    return rows


DRIFT = stage("drift", _drift)
print(f"{'rule':<10}{'dtype':>10}" + "".join(f"{d:>11}" for d in (2, 4, 8, 16, 32)))
for row in DRIFT:
    print(f"{row['rule']:<10}{row['dtype']:>10}"
          + "".join(f"{row['by_depth'][str(d)]:>11.2e}" for d in (2, 4, 8, 16, 32)))
EVIDENCE["drift"] = DRIFT


# %% [markdown]
# ## 8 · The blend coefficient, and an arithmetic constraint on what it can mean
#
# §16 records Lightning LM running a step size of 0.25 and a **blend coefficient of 0.5**, and
# says these must be set explicitly because library defaults differ. It never gives the
# formula. Under the natural reading — `p⁺ = (1−γ)p⁻ + γp + 2h·f(p)` — the inverse divides by
# `(1−γ)`, so it multiplies any reconstruction error by `1/(1−γ)` **once per layer**.
#
# At γ = 0.5 across 20 layers that is 2²⁰ ≈ 10⁶. So either the formula is different from this
# reading, or the coefficient cannot be 0.5 at depth. The sweep below measures the law.

# %%
def _gamma_sweep():
    rows, base = [], None
    for g in (0.0, 0.125, 0.25, 0.5):
        small = R.Config(rule="blended" if g else "midpoint", n_layer=8, d_model=128,
                         d_ff=256, vocab_size=512, block_size=64, h=H_STEP, gamma=g)
        e = R.gradient_check(small, batch=2, seq=16, device=str(DEV),
                             dtype=torch.float32)["worst_rel_grad_error"]
        base = base or e
        rows.append({"gamma": g, "predicted_amplification": (1 / (1 - g)) ** 8,
                     "drift": e, "measured_amplification": e / base})
    return rows


GAMMA_SWEEP = stage("gamma_sweep", _gamma_sweep)
print(f"{'gamma':>7}{'predicted 1/(1-g)^L':>22}{'measured drift':>17}{'vs gamma=0':>13}")
for r in GAMMA_SWEEP:
    print(f"{r['gamma']:>7}{r['predicted_amplification']:>22.1f}{r['drift']:>17.2e}"
          f"{r['measured_amplification']:>13.1f}x")
EVIDENCE["gamma_sweep"] = GAMMA_SWEEP


# %% [markdown]
# ## 9 · The frontier
#
# The three required runs are three points. This is the curve they sit on: throughput against
# peak memory as the batch grows, for both paths, until the card refuses.

# %%
def _frontier():
    rows = []
    cands = (1, 2) if QUICK else (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    for label, rule, rev, cap in (("stored", "standard", False, B_FIX),
                                  ("reversible", "midpoint", True, B_REV)):
        for b in cands:
            if b > cap:
                break
            try:
                r = train(rule, rev, b, tokens=b * SEQ * 6, log_every=10 ** 9)
                rows.append({"path": label, "batch": b, "tok_s": r["tok_s"],
                             "peak_bytes": r["peak_bytes"]})
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                release()
                break
            release()
    return rows


FRONTIER = stage("frontier", _frontier)
print(f"{'path':<14}{'batch':>7}{'tok/s':>11}{'peak MB':>10}")
for r in FRONTIER:
    print(f"{r['path']:<14}{r['batch']:>7}{r['tok_s']:>11,.0f}"
          f"{(r['peak_bytes'] or 0)/1e6:>10.0f}")
EVIDENCE["frontier"] = FRONTIER


# %% [markdown]
# ## 10 · Gates

# %%
def close(a, b, tol=0.05):
    return abs(a - b) <= tol * max(abs(b), 1e-12)


GATES = {
    # §4 — the claim everything else rests on
    **{f"{r}_is_exact_in_fp64": CORRECT[r]["exact"] for r in R.REVERSIBLE},
    "euler_is_not_reversible": not CORRECT["euler_reversible"],
    "no_dropout_anywhere": all("drop" not in n.lower()
                               for n, _ in R.Model(CFG["midpoint"]).named_modules()),

    # §3 — the comparison is between equals
    "all_rules_same_param_count": (
        max(EVIDENCE["configs"][r]["params"] for r in RULE_SET)
        / min(EVIDENCE["configs"][r]["params"] for r in RULE_SET) < 1.05),

    # §5 — the three runs
    "reversible_uses_less_memory": (
        (RUNS["reversible_same"]["peak_bytes"] or 0) < (RUNS["standard"]["peak_bytes"] or 1)
        if DEV.type == "cuda" else True),
    "reversible_costs_throughput": (
        RUNS["reversible_same"]["tok_s"] < RUNS["standard"]["tok_s"]),
    "reversibility_buys_batch": B_REV > B_FIX,
    "all_runs_saw_the_same_tokens": (
        RUNS["standard"]["tokens"] > 0
        and abs(RUNS["standard"]["tokens"] - RUNS["reversible_same"]["tokens"])
        <= RUNS["standard"]["tokens"] * 0.01),
    "every_run_trained": all(r["losses"][-1]["loss"] < r["losses"][0]["loss"] - 0.1
                             for r in RUNS.values()),

    # §6 — the session's central claim
    # At fixed width the reversible column must not move at all, and the stored column must
    # track the depth. The earlier version of this gate asked for linear growth from a sweep
    # that held parameters fixed instead, which forces sub-linear growth by construction --
    # the gate was wrong, not the measurement.
    "reversible_memory_is_depth_independent":
        0.9 < EVIDENCE["depth_sweep"]["reversible_growth"] < 1.1,
    "stored_memory_grows_with_depth":
        EVIDENCE["depth_sweep"]["stored_growth"] > 0.7 * depth_growth,

    # §8 — the blend coefficient
    "blending_amplifies_drift": GAMMA_SWEEP[-1]["drift"] > GAMMA_SWEEP[0]["drift"] * 10,
}
EVIDENCE["gates"] = {k: bool(v) for k, v in GATES.items()}
EVIDENCE["meta"]["wall_seconds"] = round(time.time() - T_START, 1)

for name, ok in GATES.items():
    print(f"  {'PASS' if ok else 'FAIL':<5} {name}")
passed = sum(1 for v in GATES.values() if v)
print(f"\n{passed}/{len(GATES)} gates pass · {EVIDENCE['meta']['wall_seconds']/60:.1f} minutes")

# Smoke output must never land on the real evidence file. It did once, and it silently
# replaced a 74-minute GPU run's numbers with a 40-second CPU run's.
_ev_path = OUT / ("evidence_smoke.json" if QUICK else "evidence.json")
_ev_path.write_text(json.dumps(EVIDENCE, indent=2, default=float))
print(f"wrote {_ev_path}")
if not QUICK:
    assert passed == len(GATES), f"{len(GATES) - passed} gate(s) failed"
