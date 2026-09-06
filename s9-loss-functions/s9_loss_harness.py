# %% [markdown]
# # Session 9 — Loss Functions & Output Heads
#
# **ERA V5 · the loss harness.** Four lines stand between the model's output and the scalar
# the optimiser pushes down:
#
# ```python
# hidden = model(tokens)
# logits = output_head(hidden)
# loss = cross_entropy(logits[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1))
# ```
#
# Every bug that lives in those four lines is silent. None of them raise. Three of them make
# the loss look *better*. This notebook makes each one produce a number instead.
#
# **Configuration.** The real frozen tokenizer this course's data pipeline already uses —
# **Sarvam-1, 68,096 tokens**, sha256 `bb5115a3…`, the same hash `s6-dataset-creation/tds/shards.py`
# pins and `s7-model-internals` measured against. The text is S6's committed `corpus/`, so the
# documents in the shift audit and the packing experiment are real Hindi, Telugu and code —
# not toy strings.
#
# The V5 target configuration from the class notes (`V = 131,072`, `D = 4,096`) appears
# wherever the arithmetic is what matters rather than the run: the parameter counts in
# Experiment 6 and the logits-tensor bill in Experiment 1.
#
# **One structural note.** Experiments 3, 4 and the switch table need a model that has
# actually learned something — on an untrained model every token costs `ln(V)` nats and a
# cross-document boundary is no more surprising than anything else, which makes those tests
# vacuous. So Experiment 2 trains a base model and the later experiments reuse it.
# Experiment 5 is the exception: it *must* run on a fresh model, because being untrained is
# the entire point of it.

# %%
import hashlib
import json
import math
import platform
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if HERE.name != "s9-loss-functions" and (HERE / "s9-loss-functions").is_dir():
    HERE = HERE / "s9-loss-functions"                     # notebook launched from the repo root
ROOT = HERE.parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVIDENCE = {"meta": {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "device": str(DEVICE),
    "platform": platform.platform(),
}}

print(f"python {platform.python_version()} · torch {torch.__version__} · device {DEVICE}")


# %% [markdown]
# ## 0. The tokenizer, the corpus, and the model
#
# Nothing below is mocked. The tokenizer is hash-verified against the constant S6 froze; a
# mismatch would mean every token id in this notebook means something different from every
# token id in the shards, so it is a hard gate rather than a warning.

# %%
TOKENIZER_REPO = "sarvamai/sarvam-1"
FROZEN_TOKENIZER_SHA256 = "bb5115a36ddb956a4ee0fd534e9870dd69157835622aec9c53062896f883c072"
UNK, BOS, EOS, PAD = 0, 1, 2, 3        # <unk> <s> </s> <<reserved_token_0>>, exactly as S6 assigns them


def load_tokenizer():
    """Same loader S4 and S6 use. Downloads once, then hits the HF cache."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return Tokenizer.from_file(path), sha


tok, tok_sha = load_tokenizer()
V = tok.get_vocab_size()
assert tok_sha == FROZEN_TOKENIZER_SHA256, f"tokenizer drifted: {tok_sha}"
print(f"tokenizer verified · sha256 {tok_sha[:8]}… · vocab V = {V:,}")
EVIDENCE["tokenizer"] = {"repo": TOKENIZER_REPO, "sha256": tok_sha, "vocab_size": V}


# %%
def load_corpus_docs(limit_per_lane=40):
    """Real documents from the S6 corpus snapshot committed in this repo."""
    corpus = ROOT / "s6-dataset-creation" / "corpus"
    docs = []
    for lane_file in sorted(corpus.glob("*.jsonl")):
        if lane_file.stem == "eval_registry_docs":         # S6's eval firewall — never train text
            continue
        with lane_file.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= limit_per_lane:
                    break
                d = json.loads(line)
                text = "\n".join(s["text"] for s in d["segments"] if s.get("text"))
                if len(text) > 200:
                    docs.append({"doc_id": d["doc_id"], "lane": d["lane"],
                                 "lang": d.get("language_and_script", "?"), "text": text})
    return docs


DOCS = load_corpus_docs()
print(f"{len(DOCS)} documents from {len({d['lane'] for d in DOCS})} lanes")
for lane in sorted({d["lane"] for d in DOCS}):
    n = sum(1 for d in DOCS if d["lane"] == lane)
    langs = sorted({d["lang"] for d in DOCS if d["lane"] == lane})
    print(f"  {lane:<14} {n:>3} docs   {','.join(langs)}")


def pick(pred, why):
    """First document matching pred, with a loud fallback rather than a StopIteration."""
    for d in DOCS:
        if pred(d):
            return d
    print(f"  [note] no document matched {why}; falling back to the first document")
    return DOCS[0]


# %% [markdown]
# A small pre-norm decoder, built the way Session 9 §2 describes the block: RMSNorm on the
# branch rather than on the stream, SwiGLU in the feed-forward, and a residual path that
# nothing overwrites. It is deliberately tiny — the point of this notebook is the last layer,
# not the trunk — but the output head is the **real** 68,096-wide one, because the head is
# what every experiment here is about.

# %%
class Config:
    def __init__(self, **kw):
        self.vocab_size, self.d_model, self.n_layer, self.n_head = V, 256, 4, 4
        self.d_ff, self.block_size, self.n_extra_heads = 704, 128, 0
        self.__dict__.update(kw)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class SwiGLU(nn.Module):
    """down(silu(gate(h)) ⊙ up(h)) — three matrices, §2 of the notes."""

    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.d_head = cfg.n_head, cfg.d_model // cfg.n_head
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ffn = SwiGLU(cfg)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))      # residual: nothing overwrites
        return x + self.ffn(self.n2(x))


class Model(nn.Module):
    """Trunk + one output head per predicted offset. head[0] predicts t+1, head[1] t+2, …"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.heads = nn.ModuleList(
            nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            for _ in range(1 + cfg.n_extra_heads))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        # Small init on every projection. This is what puts an untrained model near ln(V):
        # near-zero logits are a near-uniform softmax. Experiment 5 measures how near.
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def trunk(self, idx):
        B, T = idx.shape
        x = self.embed(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.norm_f(x)                                       # hidden state h, [B, T, D]

    def forward(self, idx):
        h = self.trunk(idx)
        return h, [head(h) for head in self.heads]                  # logits, each [B, T, V]


# %% [markdown]
# Two batch builders. The first is the ordinary one — one document per row. The second packs
# **two** documents per row with an `<eos>` join and gives each row a different length, so a
# single batch contains both of the structural hazards §6 describes: padding, and a
# cross-document boundary. Experiments 3, 4 and the switch table all read from it, which is
# what makes their numbers comparable to each other.

# %%
def encode_batch(docs, block_size, batch_size, offset=0):
    """One document per row, truncated to block_size, PAD-filled."""
    ids = torch.full((batch_size, block_size), PAD, dtype=torch.long)
    for r in range(batch_size):
        d = docs[(offset + r) % len(docs)]
        e = [BOS] + tok.encode(d["text"]).ids[: block_size - 1]
        ids[r, : len(e)] = torch.tensor(e, dtype=torch.long)
    return ids


def build_packed_batch(docs, block_size, batch_size, offset=0, seed=11):
    """Two documents per row joined by <eos>, rows of deliberately different lengths.

    Returns ids [B, T], the boundary index per row (position of doc B's first token),
    and the true length per row. Rows are shorter than block_size on purpose: a batch
    with no padding cannot demonstrate anything about padding.
    """
    g = torch.Generator().manual_seed(seed)
    ids = torch.full((batch_size, block_size), PAD, dtype=torch.long)
    boundaries, lengths = [], []
    for r in range(batch_size):
        want = int(torch.randint(int(block_size * 0.45), int(block_size * 0.95), (1,), generator=g))
        a = docs[(offset + 2 * r) % len(docs)]
        b = docs[(offset + 2 * r + 1) % len(docs)]
        a_ids = [BOS] + tok.encode(a["text"]).ids[: want // 2 - 2] + [EOS]
        b_ids = tok.encode(b["text"]).ids[: want - len(a_ids)]
        seq = a_ids + b_ids
        ids[r, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        boundaries.append(len(a_ids))                  # index of doc B's first token
        lengths.append(len(seq))
    return ids, boundaries, lengths


def shifted_masks(ids, boundaries):
    """Masks over the FLAT shifted targets, so every experiment indexes them identically.

    Flat index j corresponds to row j//(T-1), original target position j%(T-1) + 1.
    """
    B, T = ids.shape
    targets = ids[:, 1:].reshape(-1)
    is_pad = targets == PAD
    is_boundary = torch.zeros_like(is_pad)
    for r, bd in enumerate(boundaries):
        if 1 <= bd <= T - 1:                           # the prediction that crosses the join
            is_boundary[r * (T - 1) + (bd - 1)] = True
    return targets, is_pad, is_boundary


cfg = Config()
model = Model(cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
n_head_params = sum(p.numel() for p in model.heads.parameters())
print(f"model {n_params/1e6:.1f}M params · output head {n_head_params/1e6:.1f}M "
      f"({100*n_head_params/n_params:.0f}% of the model)")


# %% [markdown]
# ---
# # Part 1 — the harness
#
# ## Experiment 1 — every tensor shape, and what each dimension is
#
# The bullet asks for one line per dimension. The reason it is worth the space is the last
# rows: the logits tensor is the only object in the forward pass whose size is set by the
# **vocabulary**, and it is created in the final layer, after all the attention work is done.

# %%
B, T = 4, cfg.block_size
tokens = encode_batch(DOCS, T, B).to(DEVICE)

with torch.no_grad():
    h, logits_list = model(tokens)
logits = logits_list[0]

shift_logits = logits[:, :-1, :]
shift_targets = tokens[:, 1:]
flat_logits = shift_logits.reshape(-1, V)
flat_targets = shift_targets.reshape(-1)

rows = [
    ("tokens",         tuple(tokens.shape),        "B=batch rows · T=positions in the sequence"),
    ("embed(tokens)",  (B, T, cfg.d_model),        "B · T · D=width of one token's vector"),
    ("h (hidden)",     tuple(h.shape),             "B · T · D — each position has seen its whole past"),
    ("logits",         tuple(logits.shape),        "B · T · V=one raw score per vocabulary token"),
    ("logits[:, :-1]", tuple(shift_logits.shape),  "B · T-1 — drop the last position: nothing follows it"),
    ("tokens[:, 1:]",  tuple(shift_targets.shape), "B · T-1 — drop the first token: nothing predicts it"),
    ("flat logits",    tuple(flat_logits.shape),   "(B·(T-1)) predictions · V scores each"),
    ("flat targets",   tuple(flat_targets.shape),  "(B·(T-1)) correct answers, one id each"),
    ("loss",           (),                         "one scalar the optimiser can push down"),
]
print(f"{'tensor':<16}{'shape':<22}{'what each dimension is'}")
print("-" * 96)
for name, shape, meaning in rows:
    print(f"{name:<16}{str(shape):<22}{meaning}")

hidden_elems, logit_elems = h.numel(), logits.numel()
ratio_here = logit_elems / hidden_elems
V5_V, V5_D = 131_072, 4_096
print(f"\nlogits / hidden elements, this run   : {logit_elems:,} / {hidden_elems:,} = {ratio_here:.0f}x  (V/D = {V:,}/{cfg.d_model})")
print(f"logits / hidden elements, V5 target  : V/D = {V5_V:,}/{V5_D:,} = {V5_V//V5_D}x")
print(f"V5 at B=1, T=262,144, bf16           : {1*262144*V5_V*2/2**30:.0f} GiB for one logits tensor"
      f"  ({2*1*262144*V5_V*2/2**30:.0f} GiB with its gradient)")

EVIDENCE["exp1_shapes"] = {
    "B": B, "T": T, "D": cfg.d_model, "V": V,
    "hidden_elements": hidden_elems, "logit_elements": logit_elems,
    "logits_to_hidden_ratio": round(ratio_here, 2),
    "v5_ratio": V5_V // V5_D,
    "v5_logits_gib_at_256k_ctx": round(1 * 262144 * V5_V * 2 / 2**30, 1),
}


# %% [markdown]
# ## Experiment 2 — verify the shift by reading the strings
#
# > *"You will not catch an off-by-one in a wall of integers."*
#
# So here are the strings. Inputs on top, targets below, offset by one. The correct table
# reads as running text: each target is the word that actually follows its input.

# %%
def show_alignment(ids_row, name, inp, tgt, n=12, start=1):
    """inp/tgt are index tensors into one row; print the decoded strings side by side."""
    print(f"\n{name}")
    print(f"  {'pos':>4}  {'input':<24}{'target':<24}")
    for k in range(start, start + n):
        i_s = tok.decode([int(ids_row[inp[k]])], skip_special_tokens=False)
        t_s = tok.decode([int(ids_row[tgt[k]])], skip_special_tokens=False)
        print(f"  {k:>4}  {repr(i_s):<24}{repr(t_s):<24}")


hindi = pick(lambda d: d["lang"].startswith("hin"), "language=hin*")
row = torch.tensor([BOS] + tok.encode(hindi["text"]).ids[:40])
idx = torch.arange(len(row))

print(f"document {hindi['doc_id']} · lane {hindi['lane']} · {hindi['lang']}")
print(f"text: {hindi['text'][:90]}…")

show_alignment(row, "CORRECT   logits[:, :-1] vs tokens[:, 1:]  — target is the NEXT token",
               idx[:-1], idx[1:])
show_alignment(row, "NO SHIFT  logits vs tokens                 — target IS the input. The answer is given.",
               idx, idx)
show_alignment(row, "REVERSED  logits[:, 1:] vs tokens[:, :-1]  — target is the PREVIOUS token",
               idx[1:], idx[:-1])

# The correct alignment must reproduce the document when targets are read in order.
decoded_targets = tok.decode([int(t) for t in row[1:16]])
targets_are_substring = decoded_targets.strip()[:40] in hindi["text"]
print(f"\ntargets 1..15 read in order: {decoded_targets!r}")
print(f"appears in the source text : {targets_are_substring}")


# %% [markdown]
# That is the verification. What follows is the *consequence* — the reason the assignment
# warns about it twice.
#
# An untrained model scores all three alignments at about `ln(V)`, so a single loss value at
# step 0 tells you nothing. The bug only shows itself once you train: a model handed its own
# input as the target learns to copy, and copying is easy.

# %%
TRAIN_STEPS = 200


def train_run(mode, steps=TRAIN_STEPS, bs=4, lr=1e-3, log_every=50, quiet=False):
    """Train a fresh model under one harness. Returns (model, loss curve).

    The curve is *the loss that run would have shown you* — which is the whole point:
    a broken harness is discovered by reading that number, or not at all.
    """
    torch.manual_seed(1337)
    m = Model(Config()).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    curve = []
    for s in range(steps):
        ids = encode_batch(DOCS, cfg.block_size, bs, offset=s * bs).to(DEVICE)
        lg = m(ids)[1][0]
        ignore = PAD
        if mode == "correct":
            lo, ta = lg[:, :-1, :], ids[:, 1:]
        elif mode == "no_shift":
            lo, ta = lg, ids
        elif mode == "reversed":
            lo, ta = lg[:, 1:, :], ids[:, :-1]
        elif mode == "pad_counted":                      # correct shift, padding NOT excluded
            lo, ta, ignore = lg[:, :-1, :], ids[:, 1:], -100
        loss = F.cross_entropy(lo.reshape(-1, V), ta.reshape(-1), ignore_index=ignore)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append(loss.item())
        if not quiet and (s % log_every == 0 or s == steps - 1):
            print(f"  {mode:<12} step {s:>3}  loss {loss.item():7.4f}  ppl {math.exp(min(loss.item(), 20)):>12,.1f}")
    return m, curve


print(f"Alignments, identical seed, identical data. ln(V) = {math.log(V):.3f}\n")
t0 = time.time()
trained, curves = {}, {}
for mode in ("correct", "no_shift", "reversed", "pad_counted"):
    trained[mode], curves[mode] = train_run(mode)
    print()
print(f"4 x {TRAIN_STEPS} steps in {time.time()-t0:.0f}s on {DEVICE}\n")

VERDICT = {"correct": "the honest number",
           "no_shift": "the model learned to copy its input",
           "reversed": "predicting backwards, also learnable",
           "pad_counted": "a third of the credit is <pad> -> <pad>"}
print(f"{'harness':<14}{'final loss':>12}{'final ppl':>14}   what it means")
print("-" * 88)
for mode, c in curves.items():
    tail = sum(c[-20:]) / 20
    print(f"{mode:<14}{tail:>12.4f}{math.exp(min(tail, 20)):>14,.1f}   {VERDICT[mode]}")

base = trained["correct"]                                 # reused by experiments 3, 4 and the switches
base.eval()

EVIDENCE["exp2_shift"] = {
    "doc_id": hindi["doc_id"], "language": hindi["lang"],
    "targets_are_substring_of_source": bool(targets_are_substring),
    "steps": 200,
    "final_loss": {k: round(sum(c[-20:]) / 20, 4) for k, c in curves.items()},
    "curves": {k: [round(x, 4) for x in c] for k, c in curves.items()},
}


# %% [markdown]
# ## Experiment 3 — mask the padding, and watch the count change
#
# Padding is not a prediction. It is also trivially predictable — `<pad>` is always followed
# by `<pad>` — which is why counting it makes the loss look better than it is.
#
# **This only works as a comparison between two training runs, not two ways of scoring one
# model.** A model trained with `ignore_index=PAD` never learns padding at all, so scoring it
# with padding counted makes the loss *worse*, not better — the opposite of the effect §6
# warns about. The warning is about a run that was trained the wrong way and reported the
# number it saw. So both runs above are used here: `correct` and `pad_counted`, same seed,
# same data, differing in one argument.

# %%
pk_ids, pk_bounds, pk_lens = build_packed_batch(DOCS, cfg.block_size, 32, offset=17)
pk_ids = pk_ids.to(DEVICE)
pk_targets, pk_is_pad, pk_is_boundary = shifted_masks(pk_ids, pk_bounds)
n_all = pk_targets.numel()
n_real = int((~pk_is_pad).sum())


def per_token_loss(m, ids):
    with torch.no_grad():
        lg = m(ids)[1][0]
    return F.cross_entropy(lg[:, :-1, :].reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none")


pk_per = per_token_loss(base, pk_ids)                      # the correctly-trained model
pad_per = per_token_loss(trained["pad_counted"], pk_ids)   # the one trained with padding counted

print(f"row lengths (of {cfg.block_size}): {pk_lens[:8]} … ({len(pk_lens)} rows)")
print(f"{n_all - n_real:,} of {n_all:,} target positions ({100*(n_all-n_real)/n_all:.1f}%) are padding\n")

print(f"{'run':<22}{'scored over':<26}{'contributing':>13}{'loss':>10}{'perplexity':>13}")
print("-" * 84)
rows3 = [
    ("trained correctly", "real tokens only", n_real, pk_per[~pk_is_pad].mean().item()),
    ("trained pad-counted", "everything (as reported)", n_all, pad_per.mean().item()),
    ("trained pad-counted", "real tokens only (truth)", n_real, pad_per[~pk_is_pad].mean().item()),
]
for run, scope, n, l in rows3:
    print(f"{run:<22}{scope:<26}{n:>13,}{l:>10.4f}{math.exp(min(l, 20)):>13,.1f}")

flattery = rows3[2][3] - rows3[1][3]
print(f"\nThe pad-counted run reports {rows3[1][3]:.4f} but is really at {rows3[2][3]:.4f} on the")
print(f"tokens that matter: the reported number is {flattery:.4f} nats too kind.")
print(f"mean loss on padding positions, pad-counted run: {pad_per[pk_is_pad].mean().item():.4f} nats"
      f"  ({100*(n_all-n_real)/n_all:.0f}% of the mean is this)")
print(f"mean loss on padding positions, correct run    : {pk_per[pk_is_pad].mean().item():.4f} nats"
      f"  (never learned it — it was masked)")

EVIDENCE["exp3_padding"] = {
    "row_lengths": pk_lens, "block_size": cfg.block_size, "rows": len(pk_lens),
    "tokens_counted": n_all, "tokens_masked": n_real,
    "padding_fraction": round((n_all - n_real) / n_all, 4),
    "correct_run_real_tokens": round(rows3[0][3], 4),
    "pad_counted_run_as_reported": round(rows3[1][3], 4),
    "pad_counted_run_real_tokens": round(rows3[2][3], 4),
    "flattery_nats": round(flattery, 4),
    "loss_on_padding_pad_counted_run": round(pad_per[pk_is_pad].mean().item(), 4),
    "loss_on_padding_correct_run": round(pk_per[pk_is_pad].mean().item(), 4),
    "delta": round(flattery, 4),
}


# %% [markdown]
# ## Experiment 4 — two documents packed into one sequence
#
# S6 packs many documents into fixed-length sequences to stop wasting compute on padding.
# That packing creates the trap §6 names: the last token of one document has no relationship
# to the first token of the next, and training that pair teaches the model that unrelated
# things follow each other.
#
# Each row of this batch joins two documents drawn from different lanes, so the boundary pair
# is as unrelated as the corpus can make it. The baseline here already excludes padding —
# otherwise this experiment would be measuring Experiment 3 again.

# %%
print(f"{'row':>4}{'boundary at':>13}{'doc A lane':>16}{'doc B lane':>16}")
print("-" * 51)
for r, bd in enumerate(pk_bounds[:6]):
    a = DOCS[(17 + 2 * r) % len(DOCS)]
    b = DOCS[(17 + 2 * r + 1) % len(DOCS)]
    print(f"{r:>4}{bd:>13}{a['lane']:>16}{b['lane']:>16}")
print(f"  … {len(pk_bounds)} rows, {len(pk_bounds)} joins")

r0, bd0 = 0, pk_bounds[0]
ctx = tok.decode([int(t) for t in pk_ids[r0, bd0 - 4:bd0]], skip_special_tokens=False)
nxt = tok.decode([int(t) for t in pk_ids[r0, bd0:bd0 + 4]], skip_special_tokens=False)
print(f"\nrow 0, the join:  …{ctx!r}  ||  {nxt!r}…")
print(f"the pair being trained across it: "
      f"{tok.decode([int(pk_ids[r0, bd0-1])], skip_special_tokens=False)!r}"
      f" -> {tok.decode([int(pk_ids[r0, bd0])], skip_special_tokens=False)!r}")

real = ~pk_is_pad                                          # padding is masked in BOTH rows below
with_boundary = pk_per[real].mean()
without_boundary = pk_per[real & ~pk_is_boundary].mean()
boundary_only = pk_per[pk_is_boundary].mean()
n_bd = int(pk_is_boundary.sum())
ratio = boundary_only.item() / without_boundary.item()

print(f"\n{'':<40}{'contributing':>14}{'loss':>10}")
print("-" * 66)
print(f"{'boundary counted':<40}{int(real.sum()):>14,}{with_boundary.item():>10.4f}")
print(f"{'boundary masked':<40}{int((real & ~pk_is_boundary).sum()):>14,}{without_boundary.item():>10.4f}")
print(f"\nper-token: the {n_bd} boundary predictions averaged {boundary_only.item():.4f} nats against "
      f"{without_boundary.item():.4f} in-document — {ratio:.2f}x")
print(f"on the mean: masking {n_bd} of {int(real.sum()):,} targets moved the loss by "
      f"{without_boundary.item()-with_boundary.item():+.4f}")

# The per-token ratio is the finding; the effect on the mean is diluted by how rare boundaries
# are HERE. At S6's packing density the dilution is very different, so state it explicitly.
frac_here = n_bd / int(real.sum())
S6_SEQ, S6_DOC = 8192, 512            # a realistic packed sequence, and a realistic document length
frac_s6 = (S6_SEQ / S6_DOC) / S6_SEQ
excess = boundary_only.item() - without_boundary.item()
print(f"\nboundary targets are {100*frac_here:.2f}% of this batch -> the mean moves {excess*frac_here:+.4f}")
print(f"at {S6_SEQ:,}-token sequences packing ~{S6_SEQ//S6_DOC} docs, they would be {100*frac_s6:.2f}% "
      f"-> {excess*frac_s6:+.4f}")
print("Small either way on the mean. The reason to mask is not the mean — it is that every")
print("one of those targets teaches a transition that does not exist in the language.")

EVIDENCE["exp4_boundary"] = {
    "boundaries": pk_bounds, "n_boundaries": n_bd, "rows": len(pk_bounds),
    "loss_boundary_counted": round(with_boundary.item(), 4),
    "loss_boundary_masked": round(without_boundary.item(), 4),
    "boundary_token_loss": round(boundary_only.item(), 4),
    "in_document_loss": round(without_boundary.item(), 4),
    "boundary_ratio": round(ratio, 3),
    "boundary_excess_nats": round(excess, 4),
    "boundary_fraction_here": round(frac_here, 5),
    "boundary_fraction_s6_scale": round(frac_s6, 5),
    "delta": round(without_boundary.item() - with_boundary.item(), 4),
    "contributing_before": int(real.sum()), "contributing_after": int((real & ~pk_is_boundary).sum()),
}


# %% [markdown]
# ## Experiment 5 — perplexity, and the cheapest sanity check there is
#
# > *"An untrained model on our vocabulary starts at a perplexity of V and a loss of ln(V).
# > If your run does not start near there, inspect your target alignment before trusting the run."*
#
# `ln(68,096) = 11.129`. This is the gate: if it fails, nothing else in the notebook is worth
# reading.
#
# It does not land exactly on `ln(V)`, and the size of the miss is itself predictable. For
# logits drawn as `z ~ N(0, σ²)`, the expected loss is `ln(V) + σ²/2`: the log-sum-exp picks up
# `ln E[e^z] = σ²/2` while the true-token term averages to zero. So the excess is not slack in
# the check — it is the logit spread, and measuring σ predicts it.

# %%
torch.manual_seed(1337)
fresh = Model(Config()).to(DEVICE)
losses, sigmas = [], []
with torch.no_grad():
    for s in range(8):
        ids = encode_batch(DOCS, cfg.block_size, 4, offset=s * 4).to(DEVICE)
        lg = fresh(ids)[1][0]
        losses.append(F.cross_entropy(lg[:, :-1, :].reshape(-1, V),
                                      ids[:, 1:].reshape(-1), ignore_index=PAD).item())
        sigmas.append(lg.std().item())

untrained = sum(losses) / len(losses)
sigma = sum(sigmas) / len(sigmas)
expected = math.log(V)
predicted = expected + sigma ** 2 / 2
ppl = math.exp(untrained)

print(f"{'':<34}{'loss':>10}{'perplexity':>14}")
print("-" * 58)
print(f"{'untrained model, measured':<34}{untrained:>10.4f}{ppl:>14,.1f}")
print(f"{'uniform over V, ln(V)':<34}{expected:>10.4f}{V:>14,.1f}")
print(f"{'ln(V) + sigma^2/2, predicted':<34}{predicted:>10.4f}{math.exp(predicted):>14,.1f}")
print(f"\nmeasured logit sigma       : {sigma:.4f}")
print(f"excess over ln(V), measured: {untrained - expected:+.4f} nats")
print(f"excess over ln(V), predicted by sigma^2/2: {sigma**2/2:+.4f} nats")
print(f"perplexity is {100*ppl/V:.1f}% of the vocabulary size")

GATE_PERPLEXITY = abs(untrained - expected) < 0.25 and abs(untrained - predicted) < 0.05
print(f"\nGATE untrained_loss_at_ln_V: {'PASS' if GATE_PERPLEXITY else 'FAIL'}")
print("  (within 0.25 nats of ln(V), AND the miss explained to 0.05 nats by the logit spread)")

EVIDENCE["exp5_perplexity"] = {
    "untrained_loss": round(untrained, 4), "untrained_perplexity": round(ppl, 1),
    "ln_V": round(expected, 4), "V": V,
    "logit_sigma": round(sigma, 4),
    "predicted_loss": round(predicted, 4),
    "excess_measured": round(untrained - expected, 4),
    "excess_predicted": round(sigma ** 2 / 2, 4),
    "ppl_as_pct_of_V": round(100 * ppl / V, 1),
    "gate_pass": GATE_PERPLEXITY,
}


# %% [markdown]
# ## Experiment 6 — tied against untied head parameters
#
# The input embedding table holds one row per token, and so does the output head, in the same
# width. Tying uses one matrix for both. The saving is exactly `V × D`.
#
# Then §8's finding, which is the part that actually matters to V5: **tying is closed to us.**
# S7 replaced the dense input table with a fixed byte codec plus one projection. A `[V, D]`
# matrix cannot be tied to a thing that has no rows.

# %%
def head_accounting(V_, D_, label):
    tied, untied = V_ * D_, 2 * V_ * D_
    return {"label": label, "V": V_, "D": D_, "untied": untied, "tied": tied,
            "saved": untied - tied, "saved_pct": 100 * (untied - tied) / untied}


configs = [head_accounting(V, cfg.d_model, f"this notebook (V={V:,}, D={cfg.d_model})"),
           head_accounting(V5_V, V5_D, f"V5 target (V={V5_V:,}, D={V5_D:,})")]

print(f"{'configuration':<38}{'untied':>14}{'tied':>14}{'saved':>14}{'':>8}")
print("-" * 88)
for c in configs:
    print(f"{c['label']:<38}{c['untied']/1e6:>12.1f}M{c['tied']/1e6:>13.1f}M"
          f"{c['saved']/1e6:>13.1f}M{c['saved_pct']:>7.0f}%")


def count(m):
    """Sum over *distinct* tensors — a tied weight appears once, which is the whole point."""
    return sum({id(p): p.numel() for p in m.parameters()}.values())


torch.manual_seed(0)
m_untied, m_tied = Model(Config()), Model(Config())
m_tied.heads[0].weight = m_tied.embed.weight                  # the tie
c_untied, c_tied = count(m_untied), count(m_tied)
print(f"\nmeasured on the real module — untied {c_untied:,} params · tied {c_tied:,} params")
print(f"difference {c_untied - c_tied:,} = V x D = {V*cfg.d_model:,}")
GATE_TYING = c_untied - c_tied == V * cfg.d_model
print(f"GATE tying_saves_exactly_VxD: {'PASS' if GATE_TYING else 'FAIL'}")
del m_untied, m_tied

print(f"\nFor V5: the head is {V5_V*V5_D/1e6:.1f}M and there is no input table to tie it to.")
print(f"S7 replaced a {V5_V*V5_D/1e6:.1f}M input table with a 33.55M projection (93.75% saved on the front door).")
print(f"Keep a dense head and that saving across both ends becomes "
      f"{100*(V5_V*V5_D - 33.55e6)/(2*V5_V*V5_D):.1f}%.")

EVIDENCE["exp6_tying"] = {
    "configs": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()} for c in configs],
    "measured_untied_params": c_untied, "measured_tied_params": c_tied,
    "difference": c_untied - c_tied, "V_times_D": V * cfg.d_model,
    "gate_pass": GATE_TYING,
    "v5_head_params": V5_V * V5_D,
    "v5_two_sided_saving_pct": round(100 * (V5_V * V5_D - 33.55e6) / (2 * V5_V * V5_D), 1),
}


# %% [markdown]
# ## Experiment 7 — ordinary cross-entropy against a chunked one
#
# §10.3: process the tokens in blocks, compute their logits, get their loss, throw the logits
# away. The backward pass recomputes each chunk's logits when it needs them — a little extra
# arithmetic for a great deal of memory. The loss is unchanged.
#
# Written as a real `autograd.Function`, because the recomputation *is* the technique. For
# chunk logits `z` with softmax `p`, the gradient is `(p − onehot) / N`, and from that one
# tensor both `dh = dz @ W` and `dW += dzᵀ @ h` fall out, chunk by chunk.

# %% [markdown]
# The implementation is written to `out/chunked_ce.py` and imported back, rather than defined
# directly in a cell. The reason is mechanical: the memory measurement below runs each variant
# in a **subprocess**, and that subprocess has to get this exact class. `inspect.getsource`
# cannot recover a class defined in a notebook cell — inside a kernel it raises
# `TypeError: … is a built-in class` — so a module on disk is what keeps the measured
# implementation and the verified implementation provably the same object.

# %%
CHUNKED_CE_SRC = r'''
"""Chunked cross-entropy — generated by s9_loss_harness. Imported by the notebook AND by the
memory probe subprocess, so both measure the same code."""
import torch
import torch.nn.functional as F


class ChunkedCrossEntropy(torch.autograd.Function):
    """Never materialises [N, V]. Peak logit memory is chunk x V instead of N x V."""

    @staticmethod
    def forward(ctx, h, W, targets, chunk, ignore_index):
        total = h.new_zeros(())
        n_valid = 0
        for i in range(0, h.shape[0], chunk):
            z = h[i:i + chunk] @ W.t()                       # [chunk, V] -- the only big tensor
            t = targets[i:i + chunk]
            keep = t != ignore_index
            if keep.any():
                total += F.cross_entropy(z[keep], t[keep], reduction="sum")
                n_valid += int(keep.sum())
            del z                                            # and it is gone again
        ctx.save_for_backward(h, W, targets)
        ctx.chunk, ctx.ignore_index, ctx.n_valid = chunk, ignore_index, n_valid
        return total / max(n_valid, 1)

    @staticmethod
    def backward(ctx, grad_out):
        h, W, targets = ctx.saved_tensors
        chunk, ignore_index, n_valid = ctx.chunk, ctx.ignore_index, ctx.n_valid
        dh, dW = torch.zeros_like(h), torch.zeros_like(W)
        scale = grad_out / max(n_valid, 1)
        for i in range(0, h.shape[0], chunk):
            hc, t = h[i:i + chunk], targets[i:i + chunk]
            z = hc @ W.t()                                   # recomputed, not stored
            p = torch.softmax(z.float(), dim=-1).to(z.dtype)
            keep = t != ignore_index
            p[torch.arange(len(t), device=t.device), t.clamp(min=0)] -= 1.0
            p[~keep] = 0.0                                   # ignored targets contribute nothing
            dz = p * scale
            dh[i:i + chunk] = dz @ W
            dW += dz.t() @ hc
            del z, p, dz
        return dh, dW, None, None, None
'''

CE_MODULE = OUT / "chunked_ce.py"
CE_MODULE.write_text(CHUNKED_CE_SRC, encoding="utf-8")
sys.path.insert(0, str(OUT))
from chunked_ce import ChunkedCrossEntropy                   # noqa: E402

print(f"wrote and imported {CE_MODULE}")


def chunked_cross_entropy(h, W, targets, chunk=512, ignore_index=PAD):
    return ChunkedCrossEntropy.apply(h, W, targets, chunk, ignore_index)


# Correctness first. A memory win that changes the number is not a win.
N_chk, D_chk = 1024, cfg.d_model
torch.manual_seed(7)
h_t = torch.randn(N_chk, D_chk, requires_grad=True)
W_t = (torch.randn(V, D_chk) * 0.02).requires_grad_(True)     # leaf, so .grad is populated
tg = torch.randint(0, V, (N_chk,))
tg[::13] = PAD                                                # exercise the ignore path too

l_van = F.cross_entropy(h_t @ W_t.t(), tg, ignore_index=PAD)
l_van.backward()
g_h_van, g_W_van = h_t.grad.clone(), W_t.grad.clone()
h_t.grad = None
W_t.grad = None

l_chk = chunked_cross_entropy(h_t, W_t, tg, chunk=512)
l_chk.backward()

d_loss = abs(l_van.item() - l_chk.item())
d_h = (g_h_van - h_t.grad).abs().max().item()
d_W = (g_W_van - W_t.grad).abs().max().item()
print(f"vanilla loss {l_van.item():.10f}")
print(f"chunked loss {l_chk.item():.10f}")
print(f"|Δloss| {d_loss:.3e}   max|Δgrad_h| {d_h:.3e}   max|Δgrad_W| {d_W:.3e}")
GATE_CHUNK_EQ = d_loss < 1e-5 and d_h < 1e-5 and d_W < 1e-4
print(f"GATE chunked_matches_vanilla: {'PASS' if GATE_CHUNK_EQ else 'FAIL'}")
del h_t, W_t, g_h_van, g_W_van


# %% [markdown]
# Now the memory. `torch.cuda.max_memory_allocated` gives exact allocator numbers when there
# is a GPU; on CPU there is no such counter, so each variant runs in its **own subprocess**
# and reports peak RSS. That is a real measurement of a real process, but it is a coarser
# instrument than the allocator — it includes the interpreter and the weights, so the
# analytic logit-tensor figure is printed beside it rather than in place of it.

# %%
PROBE_SRC = textwrap.dedent('''
    import sys, json, resource, platform
    import torch, torch.nn.functional as F
    UNIT = 1 if platform.system() == "Darwin" else 1024   # ru_maxrss: bytes on macOS, KB on Linux
    PAD = 3

    sys.path.insert(0, sys.argv[6])                       # out/, where chunked_ce.py was written
    from chunked_ce import ChunkedCrossEntropy            # the SAME class the notebook verified

    def rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * UNIT

    variant = sys.argv[1]
    N, D, Vv, chunk = (int(a) for a in sys.argv[2:6])
    torch.manual_seed(7)
    h = torch.randn(N, D, requires_grad=True)
    W = (torch.randn(Vv, D) * 0.02).requires_grad_(True)
    t = torch.randint(0, Vv, (N,))
    if torch.cuda.is_available():
        h, W, t = h.cuda().detach().requires_grad_(True), W.cuda().detach().requires_grad_(True), t.cuda()
        torch.cuda.reset_peak_memory_stats()
    base = rss()
    if variant == "vanilla":
        loss = F.cross_entropy(h @ W.t(), t, ignore_index=PAD)
    else:
        loss = ChunkedCrossEntropy.apply(h, W, t, chunk, PAD)
    loss.backward()
    peak = rss()
    cuda = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    print(json.dumps({"variant": variant, "loss": loss.item(),
                      "peak_rss": peak, "baseline_rss": base, "delta": peak - base,
                      "cuda_peak": cuda}))
''')

N_mem, CHUNK = 4096, 512


def run_probe(variant):
    r = subprocess.run([sys.executable, "-c", PROBE_SRC, variant,
                        str(N_mem), str(cfg.d_model), str(V), str(CHUNK), str(OUT)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"probe {variant} failed:\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


print(f"N = {N_mem:,} tokens · D = {cfg.d_model} · V = {V:,} · chunk = {CHUNK}\n")
probes = {v: run_probe(v) for v in ("vanilla", "chunked")}

MB = 1024 ** 2
van, chk = probes["vanilla"], probes["chunked"]
on_cuda = van.get("cuda_peak") is not None
if on_cuda:
    van_mem, chk_mem, unit = van["cuda_peak"] / MB, chk["cuda_peak"] / MB, "CUDA allocator"
else:
    van_mem, chk_mem, unit = van["delta"] / MB, chk["delta"] / MB, "subprocess peak RSS over baseline"
measured_ratio = van_mem / chk_mem
analytic_van, analytic_chk = N_mem * V * 4 / MB, CHUNK * V * 4 / MB

print(f"{'':<34}{'peak RSS':>12}{'over baseline':>16}{'loss':>14}")
print("-" * 78)
for name, p in (("ordinary cross-entropy", van), (f"chunked (chunk={CHUNK})", chk)):
    print(f"{name:<34}{p['peak_rss']/MB:>10.1f} MB{p['delta']/MB:>14.1f} MB{p['loss']:>14.6f}")
print(f"\nMEASURED  {unit}: {van_mem:,.1f} MB vs {chk_mem:,.1f} MB = {measured_ratio:.2f}x")
print(f"ANALYTIC  logits tensor, fp32       : {analytic_van:,.0f} MB vs {analytic_chk:,.0f} MB "
      f"= {analytic_van/analytic_chk:.0f}x")
print(f"loss agreement across the two processes: |Δ| = {abs(van['loss']-chk['loss']):.3e}")
print("\nThe measured ratio is smaller than the analytic one because RSS also counts the")
print("interpreter, torch itself and the [V, D] weight and its gradient — fixed costs that")
print("chunking does not touch. The logits tensor is the part that moved.")

EVIDENCE["exp7_memory"] = {
    "N": N_mem, "D": cfg.d_model, "V": V, "chunk": CHUNK,
    "method": "cuda_allocator" if on_cuda else "subprocess_peak_rss",
    "vanilla_peak_rss_mb": round(van["peak_rss"] / MB, 1),
    "chunked_peak_rss_mb": round(chk["peak_rss"] / MB, 1),
    "vanilla_measured_mb": round(van_mem, 1),
    "chunked_measured_mb": round(chk_mem, 1),
    "measured_ratio": round(measured_ratio, 2),
    "analytic_vanilla_mb": round(analytic_van, 1),
    "analytic_chunked_mb": round(analytic_chk, 1),
    "analytic_ratio": round(analytic_van / analytic_chk, 1),
    "loss_delta_across_processes": abs(van["loss"] - chk["loss"]),
    "correctness": {"loss_delta": d_loss, "grad_h_delta": d_h, "grad_W_delta": d_W,
                    "gate_pass": GATE_CHUNK_EQ},
}


# %% [markdown]
# ---
# # Beyond the seven
#
# The assignment asks for seven numbers. The session says several other things are
# measurable, and they are cheap once the harness exists. §23 lists head stability as
# *genuinely open* for V5 and "worth doing before the real run rather than after a NaN",
# which is the one below that earns its runtime most.
#
# ## Experiment 8 — the gradient sums to zero, and what that lets drift
#
# Differentiate cross-entropy with respect to the logits and almost everything cancels:
#
# `∂L/∂z = softmax(z) − onehot(y)` — what you predicted, minus what was true.
#
# A softmax sums to 1 and a one-hot sums to 1, so **the gradient sums to exactly zero**.
# It can never push the logits up or down *as a group*. That degree of freedom is
# unconstrained by the loss, so `log Z` goes for a walk, the numbers get large, bf16 starts
# losing precision, and a run that looked healthy produces a NaN some thousands of steps
# later. It is not a mysterious instability — it is a direction the loss does not control.

# %%
torch.manual_seed(3)
z_demo = torch.randn(6, V, requires_grad=True)
y_demo = torch.randint(0, V, (6,))
F.cross_entropy(z_demo, y_demo).backward()

g = z_demo.grad
manual = (torch.softmax(z_demo.detach(), -1) - F.one_hot(y_demo, V).float()) / len(y_demo)
print(f"autograd vs softmax(z) - onehot(y):  max|Δ| = {(g - manual).abs().max().item():.3e}")
print(f"\n{'row':>4}{'sum of the gradient':>24}{'max |component|':>19}")
print("-" * 48)
for i in range(len(y_demo)):
    print(f"{i:>4}{g[i].sum().item():>24.3e}{g[i].abs().max().item():>19.3e}")
GATE_GRAD_ZERO = g.sum(dim=-1).abs().max().item() < 1e-6
print(f"\nGATE gradient_sums_to_zero: {'PASS' if GATE_GRAD_ZERO else 'FAIL'}"
      f"  (max |row sum| = {g.sum(dim=-1).abs().max().item():.2e})")
print("Nothing in the loss constrains the overall level of the logits. Hence Section 11.")
del z_demo, y_demo, g, manual


# %% [markdown]
# ### The three fixes, run from one identical initialisation
#
# §11 gives three, and is careful that they do **not** do the same job:
#
# | fix | what it controls | cost |
# |---|---|---|
# | z-loss | penalises `log Z` directly: `L + λ(log Z)²` | one hyperparameter |
# | logit soft-capping | bounds the logits, `z ← c·tanh(z/c)`, so it bounds `log Z` with them | one hyperparameter |
# | output embedding centering | removes the *uniform component*, so the head cannot express a uniform shift | none |
#
# The notes single this out as the place a widget caught the lecturer writing something
# loose, and the correction is the interesting part: **centering pins the mean logit, not
# `log Z`.** Once the mean is fixed, `log Z` is governed by how far the logits *spread*, and
# centering says nothing about spread. This run either reproduces that or it does not.

# %%
SOFT_CAP = 30.0
Z_LAMBDAS = {"z_loss": 1e-4, "z_loss_strong": 1e-2}     # the knob, at two settings


def train_stability(mode, steps=150, bs=4, lr=1e-3):
    """Identical init and identical data for all five. Only the head control differs."""
    torch.manual_seed(1337)
    m = Model(Config()).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    hist = {"step": [], "logZ": [], "mean_logit": [], "loss": []}
    for s in range(steps):
        ids = encode_batch(DOCS, cfg.block_size, bs, offset=s * bs).to(DEVICE)
        z = m(ids)[1][0][:, :-1, :]
        if mode == "soft_cap":
            z = SOFT_CAP * torch.tanh(z / SOFT_CAP)
        tgt = ids[:, 1:]
        loss = F.cross_entropy(z.reshape(-1, V), tgt.reshape(-1), ignore_index=PAD)
        logZ = torch.logsumexp(z.reshape(-1, V), dim=-1)
        if mode in Z_LAMBDAS:
            loss = loss + Z_LAMBDAS[mode] * (logZ ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if mode == "centering":
            with torch.no_grad():   # the head cannot express a uniform shift
                W = m.heads[0].weight
                W -= W.mean(dim=0, keepdim=True)
        hist["step"].append(s)
        hist["logZ"].append(logZ.mean().item())
        hist["mean_logit"].append(z.mean().item())
        hist["loss"].append(loss.item())
    return hist


t0 = time.time()
MODES = ("plain", "z_loss", "z_loss_strong", "soft_cap", "centering")
STAB = {m: train_stability(m) for m in MODES}
print(f"{len(MODES)} x 150 steps in {time.time()-t0:.0f}s\n")

LBL = {"plain": "plain", "z_loss": "z-loss λ=1e-4", "z_loss_strong": "z-loss λ=1e-2",
       "soft_cap": f"soft-cap c={SOFT_CAP:.0f}", "centering": "centering"}
print(f"{'run':<18}{'final log Z':>14}{'final mean logit':>19}{'loss':>10}")
print("-" * 63)
for m in MODES:
    h = STAB[m]
    print(f"{LBL[m]:<18}{h['logZ'][-1]:>14.4f}{h['mean_logit'][-1]:>19.3e}{h['loss'][-1]:>10.4f}")

plainZ = STAB["plain"]["logZ"][-1]
centZ = STAB["centering"]["logZ"][-1]
print(f"\nz-loss attacks the normaliser directly, and the coefficient is the whole control:")
print(f"  λ=1e-4 barely moves it ({plainZ:.2f} -> {STAB['z_loss']['logZ'][-1]:.2f}), "
      f"λ=1e-2 drives it to {STAB['z_loss_strong']['logZ'][-1]:.2f}.")
print(f"  The notes report log Z pinned near 0.07; that is a larger λ over a far longer run")
print(f"  than a 150-step proxy, but the direction and the mechanism are the same.")
print(f"Soft-capping bounds the logits at c={SOFT_CAP:.0f}, so it bounds log Z with them: "
      f"{STAB['soft_cap']['logZ'][-1]:.4f}.")
print(f"\nCentering drives the MEAN LOGIT to {STAB['centering']['mean_logit'][-1]:.2e} — but leaves")
print(f"log Z at {centZ:.4f}, which is {'HIGHER' if centZ > plainZ else 'lower'} than plain's {plainZ:.4f}.")
print("That is §11's correction, reproduced: centering removes the uniform component and")
print("says nothing about the spread, and log Z is governed by the spread. The three fixes")
print("do not do the same job, and only one of them is actually aimed at log Z.")

GATE_ZLOSS = abs(STAB["z_loss_strong"]["logZ"][-1]) < abs(plainZ)
GATE_CENTERING = abs(STAB["centering"]["mean_logit"][-1]) < 1e-4
print(f"\nGATE z_loss_lowers_logZ:            {'PASS' if GATE_ZLOSS else 'FAIL'}")
print(f"GATE centering_pins_mean_logit:     {'PASS' if GATE_CENTERING else 'FAIL'}")

EVIDENCE["exp8_stability"] = {
    "z_lambdas": Z_LAMBDAS, "soft_cap": SOFT_CAP, "steps": 150, "labels": LBL,
    "gradient_sums_to_zero": GATE_GRAD_ZERO,
    "final": {m: {"logZ": round(h["logZ"][-1], 4),
                  "mean_logit": h["mean_logit"][-1],
                  "loss": round(h["loss"][-1], 4)} for m, h in STAB.items()},
    "centering_raises_logZ_vs_plain": bool(centZ > plainZ),
    "curves": {m: {k: ([round(x, 4) for x in v] if k != "step" else v)
                   for k, v in h.items()} for m, h in STAB.items()},
    "gate_z_loss": GATE_ZLOSS, "gate_centering": GATE_CENTERING,
}


# %% [markdown]
# ## Experiment 9 — vocabulary parallelism gives the same number
#
# §10 lists four implementations of one objective. Experiment 7 did two of them
# (materialise, chunk). The fused kernel needs a kernel. The fourth — **sharding the
# vocabulary across devices** — needs N devices, but the *arithmetic* does not: each shard
# computes its own max and its own `Σexp`, and two collectives combine them into the global
# `log Z`. Simulating the shards on one device shows the collective is exact.

# %%
def sharded_cross_entropy(h, W, targets, n_shards, ignore_index=PAD):
    """What each 'device' computes, and the two all-reduces that combine them."""
    Vs = W.shape[0]
    bounds = [(i * Vs // n_shards, (i + 1) * Vs // n_shards) for i in range(n_shards)]

    shard_max = torch.stack([(h @ W[lo:hi].t()).max(dim=-1).values for lo, hi in bounds])
    g_max = shard_max.max(dim=0).values                       # all-reduce #1: max

    shard_sum = torch.stack([torch.exp(h @ W[lo:hi].t() - g_max[:, None]).sum(-1)
                             for lo, hi in bounds])
    g_sum = shard_sum.sum(dim=0)                              # all-reduce #2: sum
    log_Z = g_max + g_sum.log()

    # the true-token logit comes from whichever shard owns that id
    z_y = (h * W[targets.clamp(min=0)]).sum(-1)
    per = log_Z - z_y
    keep = targets != ignore_index
    return per[keep].mean()


torch.manual_seed(11)
h_s = torch.randn(512, cfg.d_model)
W_s = torch.randn(V, cfg.d_model) * 0.02
t_s = torch.randint(0, V, (512,))
t_s[::9] = PAD

ref = F.cross_entropy(h_s @ W_s.t(), t_s, ignore_index=PAD)
print(f"{'implementation':<34}{'loss':>16}{'|Δ| vs reference':>20}")
print("-" * 72)
print(f"{'materialise everything (reference)':<34}{ref.item():>16.10f}{'—':>20}")
shard_losses = {}
for n in (2, 4, 8, 16):
    v = sharded_cross_entropy(h_s, W_s, t_s, n)
    shard_losses[n] = v.item()
    print(f"{f'vocabulary-parallel, {n} shards':<34}{v.item():>16.10f}{abs(v.item()-ref.item()):>20.3e}")
chk = chunked_cross_entropy(h_s, W_s, t_s, chunk=128)
print(f"{'chunked, chunk=128':<34}{chk.item():>16.10f}{abs(chk.item()-ref.item()):>20.3e}")

worst = max(abs(v - ref.item()) for v in shard_losses.values())
GATE_SHARD = worst < 1e-4
print(f"\nGATE sharding_matches_reference: {'PASS' if GATE_SHARD else 'FAIL'}  (worst |Δ| = {worst:.2e})")
print("Peak logit memory per device is B·T·V/N. The loss is the same number to the decimal —")
print("which is §10's point: implementation moves memory by orders of magnitude and moves")
print("the objective not at all.")

EVIDENCE["exp9_sharding"] = {
    "reference_loss": ref.item(), "shard_losses": {str(k): v for k, v in shard_losses.items()},
    "chunked_loss": chk.item(), "worst_delta": worst, "gate_pass": GATE_SHARD,
}
del h_s, W_s, t_s


# %% [markdown]
# ## Experiment 10 — perplexity is a bad cross-tokenizer scoreboard
#
# §7's warning, made concrete. Perplexity is *per token*, and a tokenizer decides what a
# token is. A tokenizer that splits Telugu into three times as many tokens as English for
# the same meaning is being asked an **easier question at each step**, because each step
# covers less meaning — so its perplexity looks better while the model is not better.
#
# The fix is a unit the tokenizer cannot move. **Bits per byte** normalises by the UTF-8
# bytes the tokens actually cover:
#
# `bpb = (Σ nats) / (bytes × ln 2)`
#
# S2 measured exactly this fragmentation on this tokenizer. Here it is again, from the loss
# side, on one model where every language is scored by the same weights.

# %%
def lane_metrics(model_, lane_docs, limit=8):
    """Mean loss per token, plus the bytes AND characters those target tokens cover."""
    nats, n_tok, n_bytes, n_chars = 0.0, 0, 0, 0
    for d in lane_docs[:limit]:
        ids = [BOS] + tok.encode(d["text"]).ids[: cfg.block_size - 1]
        if len(ids) < 16:
            continue
        x = torch.tensor([ids], device=DEVICE)
        with torch.no_grad():
            lg = model_(x)[1][0]
        per = F.cross_entropy(lg[:, :-1, :].reshape(-1, V), x[:, 1:].reshape(-1),
                              reduction="none")
        text = tok.decode(ids[1:], skip_special_tokens=False)
        nats += per.sum().item()
        n_tok += per.numel()
        n_bytes += len(text.encode("utf-8"))
        n_chars += len(text)
    return nats, n_tok, n_bytes, n_chars


by_lang = {}
for d in DOCS:
    by_lang.setdefault(d["lang"], []).append(d)

print(f"{'language':<12}{'tokens':>8}{'bytes':>8}{'chars':>8}{'byte/char':>11}"
      f"{'loss':>9}{'perplexity':>12}{'bits/byte':>11}{'bits/char':>11}")
print("-" * 90)
lang_rows = []
for lang, ds in sorted(by_lang.items()):
    nats, n_tok, n_bytes, n_chars = lane_metrics(base, ds)
    if n_tok == 0 or n_bytes == 0 or n_chars == 0:
        continue
    loss = nats / n_tok
    bpb = nats / (n_bytes * math.log(2))
    bpc = nats / (n_chars * math.log(2))
    row = {"language": lang, "docs": len(ds), "tokens": n_tok, "bytes": n_bytes,
           "chars": n_chars, "bytes_per_char": round(n_bytes / n_chars, 3),
           "chars_per_token": round(n_chars / n_tok, 3),
           "loss": round(loss, 4), "perplexity": round(math.exp(loss), 1),
           "bits_per_byte": round(bpb, 4), "bits_per_char": round(bpc, 4)}
    lang_rows.append(row)
    print(f"{lang:<12}{n_tok:>8,}{n_bytes:>8,}{n_chars:>8,}{n_bytes/n_chars:>11.2f}"
          f"{loss:>9.4f}{math.exp(loss):>12,.1f}{bpb:>11.4f}{bpc:>11.4f}")

rank = lambda key: [r["language"] for r in sorted(lang_rows, key=lambda r: r[key])]
by_ppl, by_bpb, by_bpc = rank("perplexity"), rank("bits_per_byte"), rank("bits_per_char")
print(f"\nranked best-first by perplexity : {' < '.join(by_ppl)}")
print(f"ranked best-first by bits/byte  : {' < '.join(by_bpb)}")
print(f"ranked best-first by bits/char  : {' < '.join(by_bpc)}")

BPB_DISAGREE = by_ppl != by_bpb
BPC_AGREE = by_ppl == by_bpc
print(f"\nPerplexity and bits/byte {'DISAGREE' if BPB_DISAGREE else 'agree'}. "
      f"Perplexity and bits/char {'agree' if BPC_AGREE else 'disagree'}.")
print("\nWorth being careful about what each unit actually removes. Perplexity is per token,")
print("so the tokenizer moves it. Bits per byte removes the tokenizer -- but NOT the")
print("encoding: Devanagari costs about 3 UTF-8 bytes per character where Latin costs 1, so")
print("dividing by bytes hands Indic scripts a discount that has nothing to do with the")
print("model. On this sample that is enough to rank Hindi FIRST by bits/byte while its")
print("perplexity is the worst of the three by a wide margin.")
print("\nBits per CHARACTER removes both, and it agrees with perplexity here. For an")
print("India-first model this is not a footnote: report bits per character, or a script-")
print("aware normaliser, and treat a bits/byte scoreboard across scripts as misleading.")

EVIDENCE["exp10_bits_per_byte"] = {
    "languages": lang_rows,
    "rank_by_perplexity": by_ppl, "rank_by_bits_per_byte": by_bpb,
    "rank_by_bits_per_char": by_bpc,
    "ppl_vs_bpb_disagree": BPB_DISAGREE, "ppl_vs_bpc_agree": BPC_AGREE,
}


# %% [markdown]
# ## Experiment 11 — SFT is the same loss, masked
#
# §15: pre-training is done, and now we want a model that answers questions rather than
# continuing text. **The loss does not change at all.** What changes is which tokens are
# allowed to contribute. The prompt is context, not something the model should learn to
# generate — train on it too and you spend capacity teaching the model to write user
# questions.
#
# S6's corpus already carries the split: segments are tagged `context` and `target`, so the
# prompt/completion boundary here is real rather than staged. It is the same mechanic as
# Experiment 3 and the switch panel — a mask, and a contributing-token count that moves.

# %%
def load_sft_pairs(limit=24):
    """Documents that carry a real context/target split — S6's own segment roles."""
    corpus = ROOT / "s6-dataset-creation" / "corpus"
    pairs = []
    for lane in ("stem_math", "reasoning", "agentic"):
        f = corpus / f"{lane}.jsonl"
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= limit:
                    break
                d = json.loads(line)
                ctx = "\n".join(s["text"] for s in d["segments"] if s.get("role") == "context")
                tgt = "\n".join(s["text"] for s in d["segments"] if s.get("role") == "target")
                if ctx and tgt:
                    pairs.append({"lane": lane, "prompt": ctx, "completion": tgt})
    return pairs


PAIRS = load_sft_pairs()
print(f"{len(PAIRS)} real prompt/completion pairs from "
      f"{sorted({p['lane'] for p in PAIRS})}\n")

rows, tot_prompt, tot_completion = [], 0, 0
sft_ids, sft_promptlen = [], []
for p in PAIRS[:16]:
    p_ids = [BOS] + tok.encode(p["prompt"]).ids
    c_ids = tok.encode(p["completion"]).ids
    seq = (p_ids + c_ids)[: cfg.block_size]
    if len(seq) < 32 or len(p_ids) >= len(seq) - 4:
        continue
    row = torch.full((cfg.block_size,), PAD, dtype=torch.long)
    row[: len(seq)] = torch.tensor(seq)
    sft_ids.append(row)
    sft_promptlen.append(min(len(p_ids), len(seq)))
    tot_prompt += min(len(p_ids), len(seq))
    tot_completion += len(seq) - min(len(p_ids), len(seq))

sft_batch = torch.stack(sft_ids).to(DEVICE)
ex = PAIRS[0]
print(f"example ({PAIRS[0]['lane']}):")
print(f"  prompt     (masked out) : {ex['prompt'][:88]!r}…")
print(f"  completion (trained on) : {ex['completion'][:88]!r}…\n")

sft_per = per_token_loss(base, sft_batch)
sft_tgt = sft_batch[:, 1:].reshape(-1)
is_pad_s = sft_tgt == PAD
is_prompt = torch.zeros_like(is_pad_s)
Tm1 = cfg.block_size - 1
for r, pl in enumerate(sft_promptlen):
    is_prompt[r * Tm1: r * Tm1 + max(pl - 1, 0)] = True

real_s = ~is_pad_s
completion_only = real_s & ~is_prompt
loss_all = sft_per[real_s].mean().item()
loss_completion = sft_per[completion_only].mean().item()
loss_prompt = sft_per[real_s & is_prompt].mean().item()

print(f"{'what contributes':<34}{'tokens':>10}{'loss':>10}{'perplexity':>13}")
print("-" * 68)
print(f"{'prompt + completion (wrong)':<34}{int(real_s.sum()):>10,}{loss_all:>10.4f}{math.exp(loss_all):>13,.1f}")
print(f"{'completion only (SFT)':<34}{int(completion_only.sum()):>10,}{loss_completion:>10.4f}{math.exp(loss_completion):>13,.1f}")
print(f"{'prompt only (for reference)':<34}{int((real_s & is_prompt).sum()):>10,}{loss_prompt:>10.4f}{math.exp(loss_prompt):>13,.1f}")
print(f"\nmasking the prompt drops the contributing count from {int(real_s.sum()):,} to "
      f"{int(completion_only.sum()):,} ({100*int(is_prompt.sum())/int(real_s.sum()):.0f}% was prompt)")
print(f"and moves the loss by {loss_completion - loss_all:+.4f}.")
print("\nSame cross-entropy. Same shift. One mask. That is the whole of SFT, mechanically —")
print("and it is the same lever as padding and document boundaries, pointed somewhere else.")

GATE_SFT = int(completion_only.sum()) < int(real_s.sum())
print(f"\nGATE sft_mask_changes_count: {'PASS' if GATE_SFT else 'FAIL'}")

EVIDENCE["exp11_sft_mask"] = {
    "pairs_used": len(sft_ids), "lanes": sorted({p["lane"] for p in PAIRS}),
    "tokens_prompt_plus_completion": int(real_s.sum()),
    "tokens_completion_only": int(completion_only.sum()),
    "prompt_fraction": round(int(is_prompt.sum()) / int(real_s.sum()), 4),
    "loss_prompt_plus_completion": round(loss_all, 4),
    "loss_completion_only": round(loss_completion, 4),
    "loss_prompt_only": round(loss_prompt, 4),
    "delta": round(loss_completion - loss_all, 4),
    "example": {"lane": PAIRS[0]["lane"], "prompt": PAIRS[0]["prompt"][:200],
                "completion": PAIRS[0]["completion"][:200]},
    "gate_pass": GATE_SFT,
}


# %% [markdown]
# ---
# # Part 2 — one extra head
#
# A second head on the same trunk, predicting token `t+2`. The losses simply add.
#
# The assignment asks what happens to the second head's loss over training compared with the
# first. Measured on the *training* batch that question is hard to answer honestly, because
# batch-to-batch variance is larger than the effect: different documents have different
# intrinsic difficulty, and the two heads move together with it. So both heads are also
# evaluated every 10 steps on **one fixed held-out batch**, which removes that variance and
# leaves only the difference between the heads.

# %%
MTP_STEPS, MTP_BS, MTP_LR, EVAL_EVERY, WARMUP = 500, 8, 3e-4, 10, 50
torch.manual_seed(1337)
mtp = Model(Config(n_extra_heads=1)).to(DEVICE)
opt = torch.optim.AdamW(mtp.parameters(), lr=MTP_LR)


def lr_at(step):
    """Warmup then cosine decay. Without this the held-out curve thrashes hard enough to
    swamp the head-1/head-2 difference, which is the only thing this experiment measures."""
    if step < WARMUP:
        return MTP_LR * (step + 1) / WARMUP
    p = (step - WARMUP) / max(MTP_STEPS - WARMUP, 1)
    return MTP_LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


# Two fixed held-out batches, never trained on. More tokens per eval = less variance in the
# gap, which is small enough to be hidden by a single noisy batch.
eval_ids = torch.cat([encode_batch(DOCS, cfg.block_size, 8, offset=o) for o in (997, 1301)]).to(DEVICE)
print(f"held-out eval batch: {tuple(eval_ids.shape)} = {eval_ids.numel():,} tokens, never trained on")


def eval_heads(m, ids):
    m.eval()
    with torch.no_grad():
        lg1, lg2 = m(ids)[1]
        l1 = F.cross_entropy(lg1[:, :-1, :].reshape(-1, V), ids[:, 1:].reshape(-1), ignore_index=PAD)
        l2 = F.cross_entropy(lg2[:, :-2, :].reshape(-1, V), ids[:, 2:].reshape(-1), ignore_index=PAD)
    m.train()
    return l1.item(), l2.item()


hist = {"step": [], "h1": [], "h2": [], "sum": []}
ev = {"step": [], "h1": [], "h2": []}
t0 = time.time()
for s in range(MTP_STEPS):
    for gp in opt.param_groups:
        gp["lr"] = lr_at(s)
    ids = encode_batch(DOCS, cfg.block_size, MTP_BS, offset=s * MTP_BS).to(DEVICE)
    lg1, lg2 = mtp(ids)[1]
    l1 = F.cross_entropy(lg1[:, :-1, :].reshape(-1, V), ids[:, 1:].reshape(-1), ignore_index=PAD)
    l2 = F.cross_entropy(lg2[:, :-2, :].reshape(-1, V), ids[:, 2:].reshape(-1), ignore_index=PAD)
    loss = l1 + l2                                            # "the losses simply add"
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
    opt.step()
    hist["step"].append(s)
    hist["h1"].append(l1.item())
    hist["h2"].append(l2.item())
    hist["sum"].append(loss.item())
    if s % EVAL_EVERY == 0 or s == MTP_STEPS - 1:
        e1, e2 = eval_heads(mtp, eval_ids)
        ev["step"].append(s)
        ev["h1"].append(e1)
        ev["h2"].append(e2)
        if s % 50 == 0 or s == MTP_STEPS - 1:
            print(f"step {s:>4}  train h1 {l1.item():7.4f} h2 {l2.item():7.4f}  |  "
                  f"held-out h1 {e1:7.4f} h2 {e2:7.4f}  gap {e2-e1:+.4f}")
print(f"\n{MTP_STEPS} steps in {time.time()-t0:.0f}s on {DEVICE}")


# %%
K = 5                                     # the reported figures average the last K evals
h1_final = sum(ev["h1"][-K:]) / K
h2_final = sum(ev["h2"][-K:]) / K
h1_first = sum(ev["h1"][:K]) / K
h2_first = sum(ev["h2"][:K]) / K

half = len(ev["h1"]) // 2
wins_all = sum(1 for a, b in zip(ev["h1"], ev["h2"]) if b > a)
wins_late = sum(1 for a, b in zip(ev["h1"][half:], ev["h2"][half:]) if b > a)
crossings_train = sum(1 for a, b in zip(hist["h1"], hist["h2"]) if b <= a)

print(f"{'':<32}{'head 1 (t+1)':>15}{'head 2 (t+2)':>15}{'gap':>10}")
print("-" * 74)
print(f"{'held-out, first 5 evals':<32}{h1_first:>15.4f}{h2_first:>15.4f}{h2_first-h1_first:>+10.4f}")
print(f"{'held-out, last 5 evals':<32}{h1_final:>15.4f}{h2_final:>15.4f}{h2_final-h1_final:>+10.4f}")
print(f"{'improvement over the run':<32}{h1_first-h1_final:>15.4f}{h2_first-h2_final:>15.4f}")
print(f"{'perplexity, last 5 evals':<32}{math.exp(h1_final):>15,.1f}{math.exp(h2_final):>15,.1f}")
print(f"\nTHE TWO LOSSES : head 1 = {h1_final:.4f}   head 2 = {h2_final:.4f}")
print(f"THEIR SUM      : {h1_final + h2_final:.4f}   (the quantity actually optimised)")
print(f"sum on the last training batch: {hist['sum'][-1]:.4f} · ln(V) reference: {math.log(V):.4f}")

print(f"\nhead 2 above head 1, all evals          : {wins_all}/{len(ev['h1'])}"
      f"  ({100*wins_all/len(ev['h1']):.0f}%)")
print(f"head 2 above head 1, second half of run : {wins_late}/{len(ev['h1'])-half}"
      f"  ({100*wins_late/(len(ev['h1'])-half):.0f}%)")
print(f"same comparison on the noisy TRAIN batch: head 2 not worse {crossings_train}/{len(hist['h1'])}"
      f" times ({100*crossings_train/len(hist['h1']):.0f}%)")

GATE_MTP = h2_final > h1_final
print(f"\nGATE head2_harder_than_head1: {'PASS' if GATE_MTP else 'FAIL'}")

EVIDENCE["part2_mtp"] = {
    "steps": MTP_STEPS, "batch_size": MTP_BS, "lr": MTP_LR, "eval_every": EVAL_EVERY,
    "warmup": WARMUP, "avg_window_evals": K,
    "heldout_head1_first": round(h1_first, 4), "heldout_head2_first": round(h2_first, 4),
    "heldout_head1_final": round(h1_final, 4), "heldout_head2_final": round(h2_final, 4),
    "heldout_sum_final": round(h1_final + h2_final, 4),
    "gap_first": round(h2_first - h1_first, 4),
    "gap_final": round(h2_final - h1_final, 4),
    "improvement_head1": round(h1_first - h1_final, 4),
    "improvement_head2": round(h2_first - h2_final, 4),
    "ppl_head1_final": round(math.exp(h1_final), 1),
    "ppl_head2_final": round(math.exp(h2_final), 1),
    "train_sum_final": round(hist["sum"][-1], 4),
    "head2_above_head1_all_evals": wins_all,
    "head2_above_head1_late_evals": wins_late,
    "n_evals": len(ev["h1"]), "n_late_evals": len(ev["h1"]) - half,
    "crossings_train": crossings_train, "n_train_steps": len(hist["h1"]),
    "head_param_cost": {"one_head": V * cfg.d_model, "two_heads": 2 * V * cfg.d_model,
                        "v5_one_head": V5_V * V5_D, "v5_four_heads": 4 * V5_V * V5_D},
    "gate_pass": GATE_MTP,
    "eval_curves": {k: ([round(x, 4) for x in v] if k != "step" else v) for k, v in ev.items()},
    "train_curves": {k: ([round(x, 4) for x in v] if k != "step" else v) for k, v in hist.items()},
}


# %% [markdown]
# ### The number that actually decides: acceptance rate
#
# §13 is blunt that this is where MTP is usually oversold. *"The extra heads are guesses and
# they get rejected. You do not get four tokens per step. You get some fraction of them, and
# the acceptance rate is what decides whether the technique pays."*
#
# So measure it. Head 2 at position `t` proposes token `t+2` — a draft the model has not
# properly computed. The main path verifies: head 1 at position `t+1` is what the model
# would have produced for `t+2` had it done the work. Accept when they agree.
#
# One caveat stated plainly: this verifies against the **true** prefix rather than a
# generated one, because the sequence is teacher-forced. Real speculative decoding drafts on
# top of its own accepted tokens, where errors compound, so a deployed acceptance rate would
# be no better than this and probably worse. This is the optimistic bound.

# %%
mtp.eval()
with torch.no_grad():
    a_lg1, a_lg2 = mtp(eval_ids)[1]

draft = a_lg2[:, :-2, :].argmax(-1)          # head 2 at t proposes token t+2
verify = a_lg1[:, 1:-1, :].argmax(-1)        # head 1 at t+1 is what the main path would say
truth = eval_ids[:, 2:]                      # the token that actually came next-but-one
valid = truth != PAD

accepted = ((draft == verify) & valid).sum().item()
n_valid = int(valid.sum())
acc_rate = accepted / n_valid
draft_correct = ((draft == truth) & valid).sum().item() / n_valid
verify_correct = ((verify == truth) & valid).sum().item() / n_valid
h1_top1 = ((a_lg1[:, :-1, :].argmax(-1) == eval_ids[:, 1:]) & (eval_ids[:, 1:] != PAD)
           ).sum().item() / int((eval_ids[:, 1:] != PAD).sum())

print(f"{'measurement':<46}{'rate':>10}")
print("-" * 58)
print(f"{'head 1 top-1 accuracy on t+1':<46}{h1_top1:>9.1%}")
print(f"{'head 2 draft matches the main path (ACCEPTANCE)':<46}{acc_rate:>9.1%}")
print(f"{'head 2 draft matches the true token':<46}{draft_correct:>9.1%}")
print(f"{'main path matches the true token at t+2':<46}{verify_correct:>9.1%}")
print(f"\n{accepted:,} of {n_valid:,} drafts accepted.")

speedup = 1 + acc_rate            # one verified token, plus the accepted draft
print(f"\nTokens per forward pass, if every accepted draft is kept: {speedup:.2f}x")
print(f"Head cost to buy it: +{V*cfg.d_model/1e6:.1f}M parameters here, "
      f"+{V5_V*V5_D/1e6:.1f}M at V5's width.")
print("\nThat is the trade in one line, and it is why §13 says acceptance rate rather than")
print("head count is the number to argue about. A head that drafts wrong is pure cost.")

EVIDENCE["part2_acceptance"] = {
    "n_drafts": n_valid, "accepted": accepted,
    "acceptance_rate": round(acc_rate, 4),
    "head1_top1_accuracy": round(h1_top1, 4),
    "draft_matches_truth": round(draft_correct, 4),
    "mainpath_matches_truth": round(verify_correct, 4),
    "tokens_per_pass": round(speedup, 3),
    "caveat": "verified against the true (teacher-forced) prefix; an optimistic bound",
}


# %%
one = V * cfg.d_model
print(f"{'head cost':<30}{'this notebook':>18}{'V5 target':>18}")
print("-" * 68)
print(f"{'1 head':<30}{one/1e6:>16.1f}M{V5_V*V5_D/1e6:>16.1f}M")
print(f"{'2 heads (this experiment)':<30}{2*one/1e6:>16.1f}M{2*V5_V*V5_D/1e6:>16.1f}M")
print(f"{'4 heads (§13)':<30}{4*one/1e6:>16.1f}M{4*V5_V*V5_D/1e9:>15.1f}B")
print("\nEach dense head is another V x D. At V5's width, four of them is 2.1B parameters")
print("of output head alone — which is §13's argument for a factored head, made in numbers.")


# %% [markdown]
# **What happens to head 2's loss over training, and why.**
#
# Both heads fall together and head 2 stays worse — but the part worth reporting is that the
# gap **widens** rather than closing. It starts at `+0.010`, which is indistinguishable from
# noise, and ends at `+0.171`.
#
# At the start the two heads are equally bad because the model has learned nothing: it is
# predicting roughly the unigram distribution, and the unigram distribution is the same one
# step out as two. `t+1` and `t+2` only become different questions once there is some context
# to condition on. So the gap is not present at initialisation and then eroded — it is
# *created* by learning, and it grows as the model gets better at what head 1 is asked to do.
#
# That is also why the early evals are not a clean sweep: head 2 comes out ahead in 4 of the
# 51 evals, all of them in the first half, while the difference is still smaller than the
# measurement noise. Across the second half it is 26 out of 26.
#
# The mechanism is that head 2 never gets to condition on how `t+1` actually resolved. Head 1
# predicts one step into a distribution; head 2 predicts two and has to marginalise over the
# token in between. Its irreducible entropy is genuinely higher, so its loss has a higher
# floor. The gap is information, not an optimisation failure — a perfectly trained pair of
# heads would still show it.
#
# Which is the honest version of §13. The extra head is not free tokens. It is extra
# supervision at training time and a *draft* at inference time, and the draft gets rejected at
# a rate that higher floor predicts — the reason §13 insists acceptance rate, not head count,
# decides whether MTP pays.


# %% [markdown]
# ---
# ## The four switches
#
# §6 names four quiet bugs. Each one moves the loss, and the contributing-token count tells
# you which lie you just told.
#
# Two of them are **training** choices: get them wrong and you train a different model, so the
# honest comparison is between two runs, and the number in the table is the loss that run
# would have printed at you. The other two are **reduction** choices: same model, different
# arithmetic on the way to the scalar. The table says which is which, because conflating them
# is how Experiment 3 was wrong the first time this notebook ran.

# %%
valid = (~pk_is_pad) & (~pk_is_boundary)
baseline = pk_per[valid].mean().item()
n_contrib = int(valid.sum())
tail = lambda c: sum(c[-20:]) / 20

switches = [
    ("correct harness (baseline)", "—", tail(curves["correct"]), n_contrib),
    ("count padding into the loss", "train", tail(curves["pad_counted"]), n_all),
    ("off-by-one: no shift at all", "train", tail(curves["no_shift"]), n_contrib),
    ("predict across the document join", "reduce", pk_per[~pk_is_pad].mean().item(),
     int((~pk_is_pad).sum())),
    ("divide by B·T, not by what counted", "reduce",
     (pk_per[valid].sum() / pk_per.numel()).item(), pk_per.numel()),
]

print(f"{'switch':<38}{'kind':>7}{'loss':>10}{'contributing':>15}{'vs correct':>13}")
print("-" * 83)
ref = switches[0][2]
for name, kind, l, c in switches:
    d = l - ref
    print(f"{name:<38}{kind:>7}{l:>10.4f}{c:>15,}{('—' if abs(d) < 1e-12 else f'{d:+.4f}'):>13}")

better = sum(1 for _, _, l, _ in switches[1:] if l < ref)
print(f"\n{better} of the 4 switches make the loss look BETTER than the correct harness.")
print("That is the whole problem: the number you watch during training is the number that")
print("moves the wrong way when the harness is wrong. Only the contributing-token count,")
print("printed beside it, says which of the four you just did.")

EVIDENCE["four_switches"] = [
    {"switch": n, "kind": k, "loss": round(l, 4), "contributing": c, "delta": round(l - ref, 4)}
    for n, k, l, c in switches]
EVIDENCE["four_switches_n_flattering"] = better


# %% [markdown]
# ---
# ## Page data
#
# `loss-harness.html` recomputes the reduction switches live in the browser, so it needs the
# per-token losses and the masks rather than the summary numbers. Exporting them here means
# the page cannot drift from the run: `build_notebook.py` bakes this straight into the HTML.
#
# The training switches cannot be recomputed from one model's losses — they are different
# runs — so those travel as measured constants and the page labels them as such.

# %%
strip = tok.decode([int(t) for t in row[:26]], skip_special_tokens=False)
EVIDENCE["widget"] = {
    "rows": len(pk_lens), "T": cfg.block_size, "n_targets": int(pk_per.numel()),
    "per_token": [round(x, 3) for x in pk_per.tolist()],
    "is_pad": [int(b) for b in pk_is_pad.tolist()],
    "is_boundary": [int(b) for b in pk_is_boundary.tolist()],
    "alignment_tokens": [tok.decode([int(t)], skip_special_tokens=False) for t in row[:26]],
    "alignment_text": strip,
    "trained_runs": {k: round(sum(c[-20:]) / 20, 4) for k, c in curves.items()},
    "trained_curves": {k: [round(x, 4) for x in c] for k, c in curves.items()},
    # §4's worked example, so the page can teach logits -> softmax -> -log p from zero
    # without the reader needing the class notes open beside it.
    "softmax_demo": {"tokens": ["Delhi", "Mumbai", "Chennai", "banana", "runs"],
                     "logits": [2.0, 1.0, 0.5, -1.0, -1.5], "truth": 2},
}
print(f"page data: {EVIDENCE['widget']['n_targets']:,} per-token losses, "
      f"{sum(EVIDENCE['widget']['is_pad'])} padding, {sum(EVIDENCE['widget']['is_boundary'])} boundary")


# %% [markdown]
# ---
# ## Gates and evidence

# %%
GATES = {
    "tokenizer_hash_verified": tok_sha == FROZEN_TOKENIZER_SHA256,
    "targets_reconstruct_source": bool(targets_are_substring),
    "untrained_loss_at_ln_V": GATE_PERPLEXITY,
    "tying_saves_exactly_VxD": GATE_TYING,
    "chunked_matches_vanilla": GATE_CHUNK_EQ,
    "chunked_uses_less_memory": measured_ratio > 1.0,
    "no_shift_loss_collapses": tail(curves["no_shift"]) < tail(curves["correct"]),
    "batch_actually_has_padding": n_real < n_all,
    "pad_counted_run_flatters_itself": flattery > 0,
    "boundary_costs_more_than_context": boundary_only.item() > without_boundary.item(),
    "head2_harder_than_head1": GATE_MTP,
    "gradient_sums_to_zero": GATE_GRAD_ZERO,
    "z_loss_lowers_logZ": GATE_ZLOSS,
    "centering_pins_mean_logit": GATE_CENTERING,
    "sharding_matches_reference": GATE_SHARD,
    "sft_mask_changes_count": GATE_SFT,
}
EVIDENCE["gates"] = GATES

print(f"{'gate':<40}{'result'}")
print("-" * 52)
for name, ok in GATES.items():
    print(f"{name:<40}{'PASS' if ok else 'FAIL'}")
n_pass = sum(GATES.values())
print(f"\n{n_pass}/{len(GATES)} gates pass")

EVIDENCE["summary"] = {
    "gates_passed": n_pass, "gates_total": len(GATES),
    "seven_numbers": {
        "1_logits_to_hidden_ratio": EVIDENCE["exp1_shapes"]["logits_to_hidden_ratio"],
        "2_no_shift_vs_correct_loss": round(tail(curves["no_shift"]) - tail(curves["correct"]), 4),
        "3_padding_loss_delta": EVIDENCE["exp3_padding"]["delta"],
        "4_boundary_loss_delta": EVIDENCE["exp4_boundary"]["delta"],
        "5_untrained_perplexity": EVIDENCE["exp5_perplexity"]["untrained_perplexity"],
        "6_tying_saving_params": EVIDENCE["exp6_tying"]["difference"],
        "7_memory_ratio_measured": EVIDENCE["exp7_memory"]["measured_ratio"],
    },
    "two_losses": {"head1_t_plus_1": EVIDENCE["part2_mtp"]["heldout_head1_final"],
                   "head2_t_plus_2": EVIDENCE["part2_mtp"]["heldout_head2_final"],
                   "sum": EVIDENCE["part2_mtp"]["heldout_sum_final"]},
}

(OUT / "evidence.json").write_text(json.dumps(EVIDENCE, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'evidence.json'}")

if n_pass != len(GATES):
    raise SystemExit(f"{len(GATES) - n_pass} gate(s) failed")
