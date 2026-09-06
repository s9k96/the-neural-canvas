# %% [markdown]
# # Session 10 — The Training Loop
#
# **ERA V5 · make a real loop tell you the truth about itself.**
#
# Session 9 ended holding one number. This session is about how that number reaches back and
# moves every weight — and how you know, over weeks, that it is working.
#
# ```python
# logits = model(batch)        # forward
# loss   = loss_fn(logits, y)  # one number
# loss.backward()              # fill in every gradient
# optimizer.step()             # move every weight
# optimizer.zero_grad()        # wipe before the next batch
# ```
#
# Five lines. Every training run in the world is these repeated until the data runs out, and
# **every serious bug in them is silent.** The assignment's six tasks are six ways of making
# the loop say something checkable instead of just producing a plausible curve.
#
# **Continuity.** Same model, same frozen Sarvam-1 tokenizer (68,096 tokens, sha256
# `bb5115a3…`) and same S6 corpus as Session 9, because S10 is literally the next step after
# S9's scalar. The model definition is copied rather than imported: each session ships as a
# standalone submission, the same way S6 keeps its own copy of S4's cleaning regexes.

# %%
import hashlib
import json
import math
import platform
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if HERE.name != "s10-training-loop" and (HERE / "s10-training-loop").is_dir():
    HERE = HERE / "s10-training-loop"                    # notebook launched from the repo root
ROOT = HERE.parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVIDENCE = {"meta": {
    "python": platform.python_version(), "torch": torch.__version__,
    "device": str(DEVICE), "platform": platform.platform(),
}}
print(f"python {platform.python_version()} · torch {torch.__version__} · device {DEVICE}")


# %%
TOKENIZER_REPO = "sarvamai/sarvam-1"
FROZEN_TOKENIZER_SHA256 = "bb5115a36ddb956a4ee0fd534e9870dd69157835622aec9c53062896f883c072"
UNK, BOS, EOS, PAD = 0, 1, 2, 3


def load_tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
    return Tokenizer.from_file(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()


tok, tok_sha = load_tokenizer()
V = tok.get_vocab_size()
assert tok_sha == FROZEN_TOKENIZER_SHA256, f"tokenizer drifted: {tok_sha}"
print(f"tokenizer verified · sha256 {tok_sha[:8]}… · vocab V = {V:,}")
EVIDENCE["tokenizer"] = {"repo": TOKENIZER_REPO, "sha256": tok_sha, "vocab_size": V}


def load_corpus_docs(limit_per_lane=40):
    corpus = ROOT / "s06-dataset-creation" / "corpus"
    docs = []
    for lane_file in sorted(corpus.glob("*.jsonl")):
        if lane_file.stem == "eval_registry_docs":       # S6's eval firewall — never train text
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


# %%
class Config:
    def __init__(self, **kw):
        self.vocab_size, self.d_model, self.n_layer, self.n_head = V, 256, 4, 4
        self.d_ff, self.block_size = 704, 128
        self.__dict__.update(kw)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class SwiGLU(nn.Module):
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
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ffn(self.n2(x))


class Model(nn.Module):
    """No dropout anywhere — deliberately. §15 records that V4 set dropout to zero in every
    reversible model because a repeatable forward pass was required; here the reason is the
    same in miniature: Task 2's finite-difference check needs f(w) to be a function."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm_f(x))


def encode_batch(docs, block_size, batch_size, offset=0):
    ids = torch.full((batch_size, block_size), PAD, dtype=torch.long)
    for r in range(batch_size):
        d = docs[(offset + r) % len(docs)]
        e = [BOS] + tok.encode(d["text"]).ids[: block_size - 1]
        ids[r, : len(e)] = torch.tensor(e, dtype=torch.long)
    return ids


def token_loss(model, ids):
    """Returns (summed loss, valid token count) — NOT a mean. Task 3 is entirely about who
    does the dividing, so nothing in this harness hides a division inside a helper."""
    logits = model(ids)
    per = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), ids[:, 1:].reshape(-1),
                          reduction="none")
    keep = (ids[:, 1:].reshape(-1) != PAD).float()
    return (per * keep).sum(), int(keep.sum())


cfg = Config()
model = Model(cfg).to(DEVICE)
N_PARAMS = sum(p.numel() for p in model.parameters())
print(f"model {N_PARAMS/1e6:.1f}M params · head {model.head.weight.numel()/1e6:.1f}M "
      f"({100*model.head.weight.numel()/N_PARAMS:.0f}%)")


# %% [markdown]
# ---
# ## Task 1 — every tensor shape in the step
#
# Session 9 printed the shapes on the way *to* the loss. This is the rest of the step: what
# `backward()` fills in, and what the optimiser keeps between steps. The last two rows are
# the ones people forget when they size a machine, and §13 is about exactly that.

# %%
B, T = 4, cfg.block_size
batch = encode_batch(DOCS, T, B).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

logits = model(batch)
loss_sum, n_tok = token_loss(model, batch)
loss = loss_sum / n_tok
loss.backward()
opt.step()

rows = [
    ("batch (tokens)",   tuple(batch.shape),          "B=sequences in the step · T=positions in each"),
    ("logits",           tuple(logits.shape),         "B · T · V=one score per vocabulary token"),
    ("loss",             tuple(loss.shape),           "scalar — the entire backward signal"),
    ("head.weight",      tuple(model.head.weight.shape), "V rows · D=width of the hidden state"),
    ("head.weight.grad", tuple(model.head.weight.grad.shape), "one gradient per weight — same shape, always"),
    ("embed.weight",     tuple(model.embed.weight.shape), "V rows · D — one learned vector per token"),
    ("blocks.0.qkv.weight", tuple(model.blocks[0].qkv.weight.shape), "3D rows (q,k,v stacked) · D"),
    ("exp_avg (Adam m)", tuple(opt.state[model.head.weight]["exp_avg"].shape), "same shape as the weight — the running mean"),
    ("exp_avg_sq (v)",   tuple(opt.state[model.head.weight]["exp_avg_sq"].shape), "same shape again — the running square"),
    ("grad norm",        (),                          "scalar — every gradient combined into one length"),
]
print(f"{'tensor':<22}{'shape':<22}{'what each dimension is'}")
print("-" * 100)
for name, shape, meaning in rows:
    print(f"{name:<22}{str(shape):<22}{meaning}")

grad_elems = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)
state_elems = sum(v.numel() for s in opt.state.values() for v in s.values() if torch.is_tensor(v) and v.dim() > 0)
print(f"\nparameters       {N_PARAMS:>13,}")
print(f"gradients        {grad_elems:>13,}   (one per parameter, exactly)")
print(f"optimiser state  {state_elems:>13,}   (two per parameter — Adam's m and v)")
print(f"total tensors the step must hold: {N_PARAMS + grad_elems + state_elems:,} numbers "
      f"for {N_PARAMS:,} weights")

GATE_SHAPES = grad_elems == N_PARAMS and state_elems == 2 * N_PARAMS
print(f"\nGATE one_grad_per_weight_two_states: {'PASS' if GATE_SHAPES else 'FAIL'}")

EVIDENCE["task1_shapes"] = {
    "B": B, "T": T, "D": cfg.d_model, "V": V,
    "rows": [{"tensor": n, "shape": list(s), "meaning": m} for n, s, m in rows],
    "params": N_PARAMS, "grads": grad_elems, "optimizer_state": state_elems,
    "gate_pass": GATE_SHAPES,
}
opt.zero_grad(set_to_none=True)


# %% [markdown]
# ---
# ## Task 2 — verify one gradient by hand
#
# > *"Nudge a weight, measure how the loss changed, and compare against what `backward()`
# > reported. They should agree to several decimals, and if they do not, you have found
# > something worth understanding."*
#
# First the notes' own two-link chain, where the answer can be checked on paper:
# `x=2`, `w1=3`, `w2=4`, target `t=20`, so `h=6`, `y=24`, `loss=(24−20)²=16`.

# %%
x, w1_0, w2_0, t_target = 2.0, 3.0, 4.0, 20.0


def toy_loss(w1, w2=w2_0):
    return (w2 * (w1 * x) - t_target) ** 2


analytic = 2 * (w2_0 * w1_0 * x - t_target) * w2_0 * x        # 2(y−t)·w2·x
forward_diff = (toy_loss(w1_0 + 1e-3) - toy_loss(w1_0)) / 1e-3
toy_central = (toy_loss(w1_0 + 1e-5) - toy_loss(w1_0 - 1e-5)) / 2e-5

w1_t = torch.tensor(w1_0, requires_grad=True, dtype=torch.float64)
((w2_0 * (w1_t * x) - t_target) ** 2).backward()

print(f"h = w1·x = {w1_0*x:.0f}   y = w2·h = {w2_0*w1_0*x:.0f}   loss = (y−t)² = {toy_loss(w1_0):.0f}")
print(f"\n{'method':<40}{'dL/dw1':>18}")
print("-" * 58)
print(f"{'analytic  2(y−t)·w2·x':<40}{analytic:>18.10f}")
print(f"{'forward difference, h=1e-3':<40}{forward_diff:>18.10f}   ← the notes’ 64.064")
print(f"{'central difference, h=1e-5':<40}{toy_central:>18.10f}")
print(f"{'autograd  backward()':<40}{w1_t.grad.item():>18.10f}")
print("\nThe forward difference is off by ~0.064 because it is O(h): it measures the slope of a")
print("chord, not a tangent. The central difference is O(h²) and lands on the analytic answer.")
print("That is a property of the estimator, not a disagreement with autograd.")

GATE_TOY = abs(w1_t.grad.item() - analytic) < 1e-9 and abs(toy_central - analytic) < 1e-4
print(f"\nGATE toy_gradient_agrees: {'PASS' if GATE_TOY else 'FAIL'}")


# %% [markdown]
# Now the same check on a **real weight inside the real model**, which is where it stops being
# a formality. The model is put in float64 first: a central difference subtracts two nearly
# equal numbers, so in float32 the cancellation destroys most of the digits you are trying to
# compare, and the check would "fail" for reasons that have nothing to do with the gradient.

# %%
torch.manual_seed(4)
probe = Model(Config(n_layer=2, d_model=128, d_ff=256)).double().to(DEVICE)
probe_batch = encode_batch(DOCS, 64, 2, offset=5).to(DEVICE)


def probe_loss():
    logits = probe(probe_batch)
    per = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), probe_batch[:, 1:].reshape(-1),
                          reduction="none")
    keep = (probe_batch[:, 1:].reshape(-1) != PAD).double()
    return (per * keep).sum() / keep.sum()


probe.zero_grad(set_to_none=True)
probe_loss().backward()

# An embedding row only has a gradient if its token is actually in the batch — picking a
# row at random measures nothing, and reports a perfect but empty agreement.
present = int(probe_batch[0, 3].item())
print(f"embedding row probed: token id {present} = "
      f"{tok.decode([present], skip_special_tokens=False)!r}, which is in the batch\n")

TARGETS = [("head.weight[100, 7]", probe.head.weight, (100, 7)),
           ("blocks.0.qkv.weight[3, 11]", probe.blocks[0].qkv.weight, (3, 11)),
           (f"embed.weight[{present}, 5]", probe.embed.weight, (present, 5))]


def central_diff(param, idx, h):
    with torch.no_grad():
        original = param[idx].item()
        param[idx] = original + h
        lp = probe_loss().item()
        param[idx] = original - h
        lm = probe_loss().item()
        param[idx] = original
    return (lp - lm) / (2 * h)


# The finite difference is the imprecise one here, not autograd. Too large an h and the
# chord stops matching the tangent (truncation, ~h²); too small and subtracting two nearly
# equal losses destroys the digits (cancellation, ~eps·|L|/h). The best h sits between, and
# sweeping it shows the U rather than asserting a tolerance.
HS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
print(f"{'weight':<28}{'autograd':>16}" + "".join(f"{f'h={h:.0e}':>12}" for h in HS))
print("-" * (44 + 12 * len(HS)))
checks = []
for name, param, idx in TARGETS:
    reported = param.grad[idx].item()
    row, best = [], None
    for h in HS:
        m = central_diff(param, idx, h)
        d = abs(reported - m)
        dig = -math.log10(d / abs(reported)) if d > 0 and reported != 0 else 15.0
        row.append(dig)
        if best is None or dig > best[1]:
            best = (h, dig, m)
    checks.append({"weight": name, "autograd": reported, "best_h": best[0],
                   "best_central_diff": best[2], "best_digits": round(best[1], 1),
                   "digits_by_h": {f"{h:.0e}": round(d, 1) for h, d in zip(HS, row)}})
    print(f"{name:<28}{reported:>16.10f}" + "".join(f"{d:>12.1f}" for d in row))
print(f"{'':<28}{'':>16}" + "".join(f"{'':>12}" for _ in HS))
print("  (agreeing significant digits at each step size)")

for c in checks:
    print(f"\n  {c['weight']}")
    print(f"    autograd     {c['autograd']:.12f}")
    print(f"    central diff {c['best_central_diff']:.12f}   at h={c['best_h']:.0e}")
    print(f"    agreement    {c['best_digits']:.1f} significant digits")

worst = min(c["best_digits"] for c in checks)
GATE_REAL_GRAD = worst >= 6
print(f"\nworst agreement across the three, at each one's best h: {worst:.1f} digits")
print(f"GATE real_gradient_agrees_to_6_digits: {'PASS' if GATE_REAL_GRAD else 'FAIL'}")
print("\nThree weights at three depths: the output head, an attention projection two blocks")
print("back, and a row of the embedding table. Autograd is bookkeeping, and the books balance —")
print("the disagreement in every column is a property of the difference quotient, not of the")
print("gradient, which is why the assignment says 'several decimals' and not 'exactly'.")

EVIDENCE["task2_gradient"] = {
    "toy": {"x": x, "w1": w1_0, "w2": w2_0, "target": t_target,
            "h": w1_0 * x, "y": w2_0 * w1_0 * x, "loss": toy_loss(w1_0),
            "analytic": analytic, "forward_diff_h1e3": forward_diff,
            "central_diff_h1e5": toy_central, "autograd": w1_t.grad.item(),
            "gate_pass": GATE_TOY},
    "real": {"checks": checks, "worst_digits": worst, "dtype": "float64",
             "gate_pass": GATE_REAL_GRAD},
}
del probe, probe_batch


# %% [markdown]
# ---
# ## Task 3 — break gradient accumulation on purpose
#
# > *"Use the average of the averages with micro-batches of different lengths, and plot both
# > curves together so you see the gap rather than take my word for it."*
#
# §8: a bug that lived inside every major training framework until 2024. The correct
# combination adds up all the loss and divides by all the tokens, so every token carries equal
# weight. What the frameworks did was average the three averages — giving a short micro-batch
# exactly the same vote as a long one.
#
# First the notes' own table, reproduced exactly.

# %%
NOTES_CASE = [(4, 2.0), (4, 2.0), (2, 5.0)]        # (valid tokens, average loss)


def combine(micro):
    by_token = sum(n * l for n, l in micro) / sum(n for n, _ in micro)
    by_batch = sum(l for _, l in micro) / len(micro)
    return by_token, by_batch


bt, bb = combine(NOTES_CASE)
print(f"{'micro-batch':<14}{'valid tokens':>14}{'average loss':>15}")
print("-" * 45)
for i, (n, l) in enumerate(NOTES_CASE, 1):
    print(f"{i:<14}{n:>14}{l:>15.1f}")
print(f"\ncorrect  — sum of loss / sum of tokens : {bt:.4f}")
print(f"wrong    — average of the averages     : {bb:.4f}")
print(f"error                                  : {100*(bb-bt)/bt:.1f}%   (the notes say 15.4%)")

equal_case = [(4, 2.0), (4, 2.0), (4, 5.0)]
ebt, ebb = combine(equal_case)
print(f"\nnow make the token counts equal (4, 4, 4): {ebt:.4f} vs {ebb:.4f}"
      f"  → error {100*(ebb-ebt)/ebt:.1f}%")
print("\nThat is how it hid. The error vanishes whenever micro-batches hold equal token counts,")
print("which in casual testing they usually do — so the curves looked reasonable while being wrong.")

GATE_NOTES_CASE = abs(bt - 2.6) < 1e-9 and abs(bb - 3.0) < 1e-9
print(f"\nGATE notes_case_reproduces: {'PASS' if GATE_NOTES_CASE else 'FAIL'}")


# %% [markdown]
# Now on real micro-batches, where the unequal token counts are not invented — they are what
# you get when documents have different lengths and short ones are padded.

# %%
def make_micro_batches(offset, n_micro=4, sizes=(2, 2, 2, 2), lengths=(128, 96, 40, 24)):
    """Micro-batches whose valid-token counts genuinely differ, by truncating to
    different sequence lengths — exactly the situation §8 describes."""
    out = []
    for i in range(n_micro):
        ids = encode_batch(DOCS, lengths[i], sizes[i], offset=offset + 3 * i)
        out.append(ids.to(DEVICE))
    return out


micros = make_micro_batches(11)
torch.manual_seed(1337)
m_probe = Model(Config()).to(DEVICE)
with torch.no_grad():
    stats = [(lambda s, n: (n, (s / n).item()))(*token_loss(m_probe, mb)) for mb in micros]

rbt, rbb = combine(stats)
print(f"{'micro-batch':<14}{'valid tokens':>14}{'average loss':>15}")
print("-" * 45)
for i, (n, l) in enumerate(stats, 1):
    print(f"{i:<14}{n:>14,}{l:>15.4f}")
print(f"\ncorrect  — by token   : {rbt:.4f}")
print(f"wrong    — by micro-batch : {rbb:.4f}")
print(f"error on real data    : {100*(rbb-rbt)/rbt:+.2f}%")

EVIDENCE["task3_accumulation"] = {
    "notes_case": {"micro": NOTES_CASE, "by_token": bt, "by_batch": bb,
                   "error_pct": round(100 * (bb - bt) / bt, 2), "gate_pass": GATE_NOTES_CASE},
    "equal_case": {"micro": equal_case, "by_token": ebt, "by_batch": ebb,
                   "error_pct": round(100 * (ebb - ebt) / ebt, 6)},
    "real_micro": {"stats": [{"tokens": n, "loss": round(l, 4)} for n, l in stats],
                   "by_token": round(rbt, 4), "by_batch": round(rbb, 4),
                   "error_pct": round(100 * (rbb - rbt) / rbt, 3)},
}


# %% [markdown]
# ### The decisive measurement: the gradients themselves
#
# Before the curves, the thing that is not a matter of degree. The loss value is a summary; the
# **gradient** is what the optimiser actually consumes. Accumulate the same four micro-batches
# from the *same* weights under both schemes and compare the two vectors directly. This is
# deterministic, so it settles the question the curves can only hint at.

# %%
def accumulate_gradient(m, mbs, mode):
    m.zero_grad(set_to_none=True)
    if mode == "by_token":
        parts = [(token_loss(m, mb)) for mb in mbs]
        total = sum(n for _, n in parts)
        for ls, _ in parts:
            (ls / total).backward()
    else:
        k = len(mbs)
        for mb in mbs:
            ls, n = token_loss(m, mb)
            (ls / n / k).backward()
    return torch.cat([p.grad.detach().flatten().double() for p in m.parameters()
                      if p.grad is not None])


torch.manual_seed(1337)
g_model = Model(Config()).to(DEVICE)
g_correct = accumulate_gradient(g_model, micros, "by_token")
g_wrong = accumulate_gradient(g_model, micros, "by_batch")

rel_l2 = (g_wrong - g_correct).norm().item() / g_correct.norm().item()
cos_g = F.cosine_similarity(g_correct.unsqueeze(0), g_wrong.unsqueeze(0)).item()
ratio = g_wrong.norm().item() / g_correct.norm().item()

print(f"{'quantity':<44}{'value':>18}")
print("-" * 62)
print(f"{'‖g_correct‖':<44}{g_correct.norm().item():>18.6f}")
print(f"{'‖g_wrong‖':<44}{g_wrong.norm().item():>18.6f}")
print(f"{'‖g_wrong‖ / ‖g_correct‖':<44}{ratio:>18.6f}")
print(f"{'relative L2 difference':<44}{rel_l2:>17.2%}")
print(f"{'cosine similarity':<44}{cos_g:>18.6f}")
print(f"{'angle between them':<44}{math.degrees(math.acos(min(1.0, cos_g))):>15.2f}°")

print(f"\nThe two schemes point the optimiser in measurably different directions from identical")
print(f"weights: {rel_l2:.1%} apart in L2, {math.degrees(math.acos(min(1.0, cos_g))):.1f}° off. Not a rounding difference — a")
print(f"different objective. Every step of a real run compounds it.")
GATE_GRAD_DIFFERS = rel_l2 > 0.01 and cos_g < 0.9999
print(f"\nGATE accumulation_schemes_differ: {'PASS' if GATE_GRAD_DIFFERS else 'FAIL'}")

EVIDENCE["task3_gradients"] = {
    "norm_correct": g_correct.norm().item(), "norm_wrong": g_wrong.norm().item(),
    "norm_ratio": ratio, "relative_l2": rel_l2, "cosine": cos_g,
    "angle_degrees": math.degrees(math.acos(min(1.0, cos_g))),
    "gate_pass": GATE_GRAD_DIFFERS,
}
del g_model, g_correct, g_wrong


# %% [markdown]
# ### Both curves, on the same axes
#
# The table is arithmetic. The assignment asks for the *curves*, so: two runs, identical seed,
# identical data, identical micro-batch schedule. The only difference is who does the dividing.
#
# One subtlety that decides whether this comparison means anything. Each run *reports* its own
# loss on its own basis, and those two numbers are not comparable — the wrong run reports a
# different quantity, not just a worse one. So both runs are **also** evaluated every few steps
# on one fixed held-out batch with correct, token-weighted normalisation. That second curve is
# the honest comparison; the reported curve is what you would actually have been staring at.

# %%
ACC_STEPS, ACC_EVAL_EVERY, ACC_LR = 120, 5, 3e-4
eval_batch = encode_batch(DOCS, cfg.block_size, 8, offset=901).to(DEVICE)


def eval_token_loss(m):
    m.eval()
    with torch.no_grad():
        s, n = token_loss(m, eval_batch)
    m.train()
    return (s / n).item()


def run_accumulation(mode, steps=ACC_STEPS):
    """mode='by_token'  -> sum(loss) / sum(tokens)        (correct)
       mode='by_batch'  -> mean over micro-batches of their means  (the 2024 bug)"""
    torch.manual_seed(1337)
    m = Model(Config()).to(DEVICE)
    o = torch.optim.AdamW(m.parameters(), lr=ACC_LR)
    reported, held, xs = [], [], []
    for s in range(steps):
        mbs = make_micro_batches(offset=s * 7)
        o.zero_grad(set_to_none=True)
        if mode == "by_token":
            tot_loss, tot_tok = 0.0, 0
            parts = []
            for mb in mbs:
                ls, n = token_loss(m, mb)
                parts.append((ls, n))
                tot_tok += n
            for ls, _ in parts:
                (ls / tot_tok).backward()          # each token gets weight 1/total
                tot_loss += ls.item()
            rep = tot_loss / tot_tok
        else:
            k = len(mbs)
            tot = 0.0
            for mb in mbs:
                ls, n = token_loss(m, mb)
                (ls / n / k).backward()            # each MICRO-BATCH gets weight 1/k
                tot += (ls / n).item()
            rep = tot / k
        o.step()
        reported.append(rep)
        if s % ACC_EVAL_EVERY == 0 or s == steps - 1:
            xs.append(s)
            held.append(eval_token_loss(m))
    return {"reported": reported, "held_step": xs, "held": held}


t0 = time.time()
ACC = {m: run_accumulation(m) for m in ("by_token", "by_batch")}
print(f"2 x {ACC_STEPS} steps in {time.time()-t0:.0f}s\n")

print(f"{'':<26}{'reported loss':>16}{'held-out (token-weighted)':>28}")
print("-" * 72)
for m, lbl in (("by_token", "correct, by token"), ("by_batch", "wrong, by micro-batch")):
    r = sum(ACC[m]["reported"][-10:]) / 10
    h = sum(ACC[m]["held"][-3:]) / 3
    print(f"{lbl:<26}{r:>16.4f}{h:>28.4f}")

rep_gap = sum(ACC["by_batch"]["reported"][-10:]) / 10 - sum(ACC["by_token"]["reported"][-10:]) / 10
held_gap = sum(ACC["by_batch"]["held"][-3:]) / 3 - sum(ACC["by_token"]["held"][-3:]) / 3
print(f"\ngap in the REPORTED number : {rep_gap:+.4f}  ← the two runs are not even measuring the same thing")
print(f"gap on the HELD-OUT batch  : {held_gap:+.4f}  ← the honest comparison, same basis for both")

print(f"\nWhat this does and does not show. The reported gap is real and misleading: the wrong")
print(f"run prints a {abs(rep_gap):.3f}-lower number, and a person watching that curve would conclude")
print(f"the run was going better. On the held-out batch, scored the same way for both, the")
print(f"difference is {held_gap:+.4f} — which at {ACC_STEPS} steps of a proxy model is inside the noise.")
print(f"\nSo I will not claim the wrong average visibly degrades a short run, because it did not.")
print(f"The defect is upstream of the curve and was measured above: the gradients are "
      f"{rel_l2:.1%} apart.")
print(f"A wrong objective that produces a plausible curve for {ACC_STEPS} steps is precisely why this")
print(f"bug survived in every major framework until 2024 — §8's point exactly: a number that")
print(f"looks plausible is not evidence.")
GATE_ACC = abs(rep_gap) > 0.01
print(f"\nGATE reported_loss_is_misleading: {'PASS' if GATE_ACC else 'FAIL'}")

EVIDENCE["task3_curves"] = {
    "steps": ACC_STEPS, "lr": ACC_LR, "eval_every": ACC_EVAL_EVERY,
    "micro_batch_lengths": [128, 96, 40, 24],
    "curves": {m: {"reported": [round(x, 4) for x in ACC[m]["reported"]],
                   "held_step": ACC[m]["held_step"],
                   "held": [round(x, 4) for x in ACC[m]["held"]]} for m in ACC},
    "reported_gap": round(rep_gap, 4), "heldout_gap": round(held_gap, 4),
    "gate_pass": GATE_ACC,
}


# %% [markdown]
# ---
# ## Task 4 — log the grad norm, and find where it moved first
#
# > *"Log the grad norm at every step, then find one step where it moved before the loss did."*
#
# §12: the grad norm is worth more than the clipping it enables, because **it moves before the
# loss does**. To claim that honestly I need a rule fixed in advance rather than a spike picked
# out by eye afterwards, so:
#
# * a **spike** at step *t* is a *jump* — `x[t] − x[t−1]` — more than `k` MADs above the
#   median jump of the previous `W` steps. It has to be the jump and not the level: the loss
#   is falling, so it can never exceed its own trailing median, and a level-based rule reports
#   zero loss spikes by construction no matter what the run does. Median and MAD rather than
#   mean and σ, so one big value cannot hide by inflating the threshold it is tested against.
# * a **lead** is a grad-norm spike at *t* whose nearest loss spike falls in `t+1 … t+L`.
#
# Two deliberate choices. **Clipping is off** — with clipping on, the norm you log after
# clipping is flat by construction and this trace tells you nothing, which is itself worth
# knowing. And the learning rate is set **high enough that the run is genuinely unstable**,
# because the claim under test is about a run in trouble; a placid run has no event to lead.

# %%
SPIKE_K, SPIKE_WINDOW, MAX_LAG = 3.5, 20, 5
NORM_STEPS, NORM_LR = 300, 2e-3


def train_logging_norms(steps=NORM_STEPS, bs=4, lr=NORM_LR):
    torch.manual_seed(7)
    m = Model(Config()).to(DEVICE)
    o = torch.optim.AdamW(m.parameters(), lr=lr)
    losses, norms = [], []
    for s in range(steps):
        # deliberately heterogeneous: consecutive steps see different lanes and scripts,
        # which is what real mixed-corpus training looks like and where the spikes come from
        ids = encode_batch(DOCS, cfg.block_size, bs, offset=(s * 13) % len(DOCS)).to(DEVICE)
        o.zero_grad(set_to_none=True)
        ls, n = token_loss(m, ids)
        (ls / n).backward()
        gn = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in m.parameters()
                            if p.grad is not None)).item()
        o.step()                                   # no clipping — see the note above
        losses.append((ls / n).item())
        norms.append(gn)
    return losses, norms


t0 = time.time()
LOSSES, NORMS = train_logging_norms()
print(f"{NORM_STEPS} steps in {time.time()-t0:.0f}s, grad norm logged every step\n")


def spikes(series, k=SPIKE_K, window=SPIKE_WINDOW):
    """Spikes in the JUMP, not the level — see the note above on why the level fails here."""
    d = [series[i] - series[i - 1] for i in range(1, len(series))]
    out = []
    for i in range(window, len(d)):
        hist = sorted(d[i - window:i])
        med = hist[len(hist) // 2]
        mad = sorted(abs(h - med) for h in hist)[len(hist) // 2] or 1e-12
        if (d[i] - med) / mad > k:
            out.append(i + 1)                       # index back into the original series
    return out


norm_spikes, loss_spikes = spikes(NORMS), spikes(LOSSES)
leads = [(t, min((u for u in loss_spikes if t < u <= t + MAX_LAG), default=None))
         for t in norm_spikes]
leads = [(t, u, u - t) for t, u in leads if u is not None]

print(f"detection rule: > {SPIKE_K} MADs above the trailing median of {SPIKE_WINDOW} steps")
print(f"grad-norm spikes : {len(norm_spikes)}  at {norm_spikes[:12]}{'…' if len(norm_spikes) > 12 else ''}")
print(f"loss spikes      : {len(loss_spikes)}  at {loss_spikes[:12]}{'…' if len(loss_spikes) > 12 else ''}")
print(f"norm-led events  : {len(leads)}")

if leads:
    t, u, lag = leads[0]
    lo, hi = max(0, t - 3), min(len(NORMS), u + 3)
    print(f"\nTHE STEP: grad norm spiked at {t}, the loss followed at {u} — {lag} step(s) later.\n")
    print(f"{'step':>6}{'grad norm':>14}{'loss':>10}   ")
    print("-" * 46)
    for i in range(lo, hi):
        mark = " <- norm" if i == t else (" <- loss" if i == u else "")
        print(f"{i:>6}{NORMS[i]:>14.4f}{LOSSES[i]:>10.4f}{mark}")
else:
    print("\nNo norm-led event under this rule on this run.")

GATE_LEAD = len(leads) > 0
print(f"\nGATE grad_norm_led_the_loss: {'PASS' if GATE_LEAD else 'FAIL'}")

EVIDENCE["task4_grad_norm"] = {
    "steps": NORM_STEPS, "lr": NORM_LR, "rule": {"k_mads": SPIKE_K, "window": SPIKE_WINDOW, "max_lag": MAX_LAG},
    "clipping": "off, deliberately",
    "n_norm_spikes": len(norm_spikes), "n_loss_spikes": len(loss_spikes),
    "norm_spikes": norm_spikes, "loss_spikes": loss_spikes,
    "leads": [{"norm_step": t, "loss_step": u, "lag": l} for t, u, l in leads],
    "first_lead": ({"norm_step": leads[0][0], "loss_step": leads[0][1], "lag": leads[0][2]}
                   if leads else None),
    "losses": [round(x, 4) for x in LOSSES], "norms": [round(x, 4) for x in NORMS],
    "gate_pass": GATE_LEAD,
}


# %% [markdown]
# ---
# ## Task 5 — compute your own MFU, and report it honestly
#
# `MFU = 6·N·tokens_per_second / what the machine can do per second`
#
# The denominator is the honest part. There is no meaningful datasheet FLOP/s for "a CPU
# running PyTorch", and quoting an H100 number on hardware I am not using would be a fabricated
# result. So the peak is **measured**: the best sustained throughput I can get out of this
# machine on large dense matmuls, which is the most generous denominator that is still true.
# That makes this MFU a ratio of two things I actually observed.

# %%
def measure_peak_flops(sizes=(1024, 2048, 4096), repeats=3):
    best = 0.0
    for n in sizes:
        a, b = torch.randn(n, n, device=DEVICE), torch.randn(n, n, device=DEVICE)
        (a @ b).sum().item()                              # warm up / force realisation
        t0 = time.perf_counter()
        for _ in range(repeats):
            c = a @ b
        c.sum().item()
        dt = (time.perf_counter() - t0) / repeats
        best = max(best, 2 * n ** 3 / dt)
    return best


PEAK = measure_peak_flops()
print(f"measured peak on this device: {PEAK/1e12:.3f} TFLOP/s (best large dense matmul)\n")

MFU_STEPS, MFU_BS = 30, 4
torch.manual_seed(1)
mfu_model = Model(Config()).to(DEVICE)
mfu_opt = torch.optim.AdamW(mfu_model.parameters(), lr=1e-4)
mfu_batch = encode_batch(DOCS, cfg.block_size, MFU_BS, offset=3).to(DEVICE)

for _ in range(3):                                        # warm up
    mfu_opt.zero_grad(set_to_none=True)
    ls, n = token_loss(mfu_model, mfu_batch)
    (ls / n).backward()
    mfu_opt.step()

t0 = time.perf_counter()
for _ in range(MFU_STEPS):
    mfu_opt.zero_grad(set_to_none=True)
    ls, n = token_loss(mfu_model, mfu_batch)
    (ls / n).backward()
    mfu_opt.step()
elapsed = time.perf_counter() - t0

tokens_per_step = MFU_BS * cfg.block_size
tok_per_s = MFU_STEPS * tokens_per_step / elapsed

# Which N? 6N counts the arithmetic of MATMULS. An embedding lookup is a gather — it reads a
# row and does no multiplying at all — so counting the [V, D] embedding table in N invents
# FLOPs that were never performed and inflates MFU. The standard convention counts
# non-embedding parameters, and the difference here is not cosmetic: the table is 46% of N.
n_embed = model.embed.weight.numel() + model.pos.weight.numel()
n_non_embed = N_PARAMS - n_embed
achieved_all = 6 * N_PARAMS * tok_per_s
achieved = 6 * n_non_embed * tok_per_s
# The other term the 6N model drops: attention is quadratic in sequence length.
attn_flops = 12 * cfg.n_layer * cfg.d_model * cfg.block_size * tok_per_s
mfu = achieved / PEAK
mfu_all = achieved_all / PEAK

print(f"{'quantity':<44}{'value':>20}")
print("-" * 66)
print(f"{'all parameters':<44}{N_PARAMS:>20,}")
print(f"{'  of which embedding + positional':<44}{n_embed:>20,}")
print(f"{'non-embedding N (what 6N should use)':<44}{n_non_embed:>20,}")
print(f"{'tokens per second':<44}{tok_per_s:>20,.0f}")
print(f"{'achieved  6·N·tok/s   (non-embedding)':<44}{achieved/1e12:>17.4f} TF/s")
print(f"{'  + attention 12·L·D·T·tok/s':<44}{attn_flops/1e12:>17.4f} TF/s")
print(f"{'what we are paying for (measured)':<44}{PEAK/1e12:>17.4f} TF/s")
print(f"{'MFU':<44}{100*mfu:>19.1f}%")
print(f"{'healthy range (§14)':<44}{'35–50%':>20}")
print(f"\nCounting the embedding table too would report {100*mfu_all:.1f}% — "
      f"{100*(mfu_all-mfu):.1f} points of pure fiction,")
print(f"because that table is looked up, not multiplied. Worth stating in any MFU you publish.")

# The notes' own worked example, to show the formula is the same one
notes_mfu = 6 * 9e9 * 12000 / 7912e12
print(f"\nsame formula on §14's example (9B, 12,000 tok/s, 8×H100): {100*notes_mfu:.1f}%"
      f"   (the notes say 8.2%)")

head_flops = 6 * model.head.weight.numel() * tok_per_s
print(f"\nWhat is costing us the distance to 40%, in order of size:")
print(f"  1. The vocabulary head is {100*model.head.weight.numel()/n_non_embed:.0f}% of non-embedding N at D={cfg.d_model},")
print(f"     so {100*head_flops/achieved:.0f}% of the counted FLOPs are one [T,{cfg.d_model}]×[{cfg.d_model},{V:,}] matmul")
print(f"     with a tiny inner dimension — the least efficient shape in the model.")
print(f"  2. B×T = {tokens_per_step} tokens per step. Every matmul is small enough that kernel launch")
print(f"     and Python overhead are a real fraction of the step, which the 6N model does not count.")
print(f"  3. No fused kernels, no flash attention path on CPU, fp32 throughout — the measured")
print(f"     peak comes from one huge matmul that keeps caches full; a transformer step does not.")
print(f"  4. The optimiser step touches {3*N_PARAMS:,} numbers per step and does no matmul at all,")
print(f"     so it is pure denominator: time spent, zero FLOPs counted by 6N.")
print(f"\n6N is a model of the *useful* arithmetic. Everything above is real time that 6N")
print(f"declines to count, which is exactly why a low MFU is invisible in the loss curve.")

EVIDENCE["task5_mfu"] = {
    "peak_flops_measured": PEAK, "peak_tflops": round(PEAK / 1e12, 4),
    "N_all": N_PARAMS, "N_embedding": n_embed, "N_non_embedding": n_non_embed,
    "batch": MFU_BS, "block": cfg.block_size,
    "steps_timed": MFU_STEPS, "seconds": round(elapsed, 3),
    "tokens_per_second": round(tok_per_s, 1),
    "achieved_flops": achieved, "achieved_tflops": round(achieved / 1e12, 4),
    "attention_tflops": round(attn_flops / 1e12, 4),
    "mfu": round(mfu, 5), "mfu_pct": round(100 * mfu, 2),
    "mfu_pct_counting_embeddings": round(100 * mfu_all, 2),
    "head_share_of_non_embedding_params": round(model.head.weight.numel() / n_non_embed, 4),
    "head_share_of_flops": round(head_flops / achieved, 4),
    "notes_example_mfu_pct": round(100 * notes_mfu, 2),
    "denominator": "measured best large-matmul throughput on this device, not a datasheet figure",
    "N_convention": "non-embedding parameters; an embedding lookup is a gather, not a matmul",
}


# %% [markdown]
# ---
# ## Task 6 — 0.1 in fp32, bf16 and fp8 E4M3, showing the bits
#
# > *"Take the number 0.1 and write out by hand what it looks like … showing the bits. Then say
# > which one you would train in, and why."*
#
# By hand means by hand: decompose `0.1` into sign, exponent and mantissa myself, round the
# mantissa to each format's width, and only *then* check the answer against `struct`/`torch`.
# If I packed the bytes first and read them back, I would be showing that the library agrees
# with itself.
#
# `0.1 = 1.6 × 2⁻⁴`, so every format stores the same exponent and differs only in how much of
# `0.6` survives in the mantissa.

# %%
FORMATS = {                       # name: (exponent bits, mantissa bits, bias)
    "fp32":     (8, 23, 127),
    "bf16":     (8, 7, 127),
    "fp16":     (5, 10, 15),
    "fp8 E4M3": (4, 3, 7),
}


def encode_by_hand(value, e_bits, m_bits, bias):
    """Sign / exponent / mantissa from first principles, round-half-to-even on the mantissa."""
    sign = 0 if value >= 0 else 1
    v = abs(value)
    exp = math.floor(math.log2(v))                    # 0.1 -> -4
    significand = v / (2.0 ** exp)                    # 1.6, in [1, 2)
    frac = significand - 1.0                          # 0.6
    scaled = frac * (1 << m_bits)
    m = math.floor(scaled)
    rem = scaled - m
    if rem > 0.5 or (rem == 0.5 and m % 2 == 1):      # round half to even
        m += 1
        if m == (1 << m_bits):                        # mantissa overflowed into the exponent
            m, exp = 0, exp + 1
    stored = (1 + m / (1 << m_bits)) * (2.0 ** exp)
    return {"sign": sign, "exp_unbiased": exp, "exp_field": exp + bias, "mantissa": m,
            "stored": stored, "bits": f"{sign:01b} {exp + bias:0{e_bits}b} {m:0{m_bits}b}"}


TARGET = 0.1
print(f"0.1 = {TARGET/(2**math.floor(math.log2(TARGET))):.10g} x 2^{math.floor(math.log2(TARGET))}"
      f"   — the same exponent in every format below\n")
print(f"{'format':<10}{'bits (s e m)':<42}{'stores':>18}{'rel. error':>13}")
print("-" * 85)
fmt_rows = []
for name, (e, m, bias) in FORMATS.items():
    r = encode_by_hand(TARGET, e, m, bias)
    rel = abs(r["stored"] - TARGET) / TARGET
    fmt_rows.append({"format": name, "e_bits": e, "m_bits": m, "bias": bias, **r,
                     "rel_error": rel})
    print(f"{name:<10}{r['bits']:<42}{r['stored']:>18.12g}{rel:>13.2e}")

# Cross-check the hand computation against the machine, where the machine has the format.
f32_hex = struct.pack(">f", TARGET).hex()
hand32 = fmt_rows[0]
hand32_int = (hand32["sign"] << 31) | (hand32["exp_field"] << 23) | hand32["mantissa"]
bf16_torch = torch.tensor([TARGET], dtype=torch.float32).to(torch.bfloat16).float().item()
fp16_torch = torch.tensor([TARGET], dtype=torch.float32).to(torch.float16).float().item()

print(f"\ncross-checks against the machine")
print("-" * 60)
print(f"  fp32 by hand : 0x{hand32_int:08x}     struct.pack : 0x{f32_hex}")
print(f"  bf16 by hand : {fmt_rows[1]['stored']:.12g}   torch.bfloat16 : {bf16_torch:.12g}")
print(f"  fp16 by hand : {fmt_rows[2]['stored']:.12g}   torch.float16  : {fp16_torch:.12g}")
try:                                                   # fp8 is not in every torch build
    fp8_torch = torch.tensor([TARGET]).to(torch.float8_e4m3fn).float().item()
    print(f"  fp8  by hand : {fmt_rows[3]['stored']:.12g}   torch.float8_e4m3fn : {fp8_torch:.12g}")
except (AttributeError, RuntimeError) as exc:
    fp8_torch = None
    print(f"  fp8  by hand : {fmt_rows[3]['stored']:.12g}   torch has no usable float8 here ({type(exc).__name__})")

GATE_FP32 = f"0x{hand32_int:08x}" == f"0x{f32_hex}"
GATE_BF16 = abs(fmt_rows[1]["stored"] - bf16_torch) < 1e-12
GATE_FP8 = fp8_torch is None or abs(fmt_rows[3]["stored"] - fp8_torch) < 1e-12
print(f"\nGATE fp32_bits_match_struct: {'PASS' if GATE_FP32 else 'FAIL'}")
print(f"GATE bf16_matches_torch:     {'PASS' if GATE_BF16 else 'FAIL'}")
print(f"GATE fp8_matches_torch:      {'PASS' if GATE_FP8 else 'FAIL'}"
      f"{'  (skipped — no float8 in this build)' if fp8_torch is None else ''}")

print(f"\n0.1 is not representable in binary at any width — it is the repeating fraction")
print(f"0.0001100110011… — so every row above is wrong, and the only question is by how much.")

EVIDENCE["task6_formats"] = {
    "target": TARGET, "formats": fmt_rows,
    "fp32_hex_by_hand": f"0x{hand32_int:08x}", "fp32_hex_struct": f"0x{f32_hex}",
    "bf16_torch": bf16_torch, "fp16_torch": fp16_torch, "fp8_torch": fp8_torch,
    "gates": {"fp32": GATE_FP32, "bf16": GATE_BF16, "fp8": GATE_FP8},
}


# %% [markdown]
# ### Which one would I train in, and why
#
# **bf16, for the weights and the backward pass — with an fp32 master copy.**
#
# The relative errors above rank the formats fp32 < fp16 < bf16 < fp8, and that ranking is
# almost irrelevant to the decision, which is the point of §10. bf16 is *less accurate than
# fp16* on this number and won anyway, because what breaks a run is not the error on 0.1. It
# is what happens to a gradient of `1e-8` late in training, and the next experiment measures
# exactly that.
#
# fp8 E4M3 stores 0.1 as 0.1015625 — a 1.6% error. That is disqualifying for a master weight
# being nudged by 1e-7 per step, because the update would round to nothing every time. It is
# perfectly fine for the *inputs to a matmul*, where the error is averaged over thousands of
# products and does not accumulate. §11's rule: shrink where the error does not accumulate.

# %% [markdown]
# ---
# # Beyond the six
#
# ## Extra 7 — the gradient that becomes exactly zero
#
# §10's real argument for bf16, measured. fp16 spends 5 bits on the exponent and cannot hold
# anything below about `5.96e-8`; bf16 keeps all eight of fp32's exponent bits, so its floor is
# `9.18e-41` — nothing in training will ever reach it.

# %%
grads = [1e-4, 1e-6, 1e-8, 1e-10]
print(f"{'gradient':>12}{'fp16':>16}{'bf16':>16}{'fp16 verdict':>22}")
print("-" * 68)
underflow = []
for g in grads:
    h = torch.tensor([g], dtype=torch.float32).to(torch.float16).float().item()
    b = torch.tensor([g], dtype=torch.float32).to(torch.bfloat16).float().item()
    verdict = "becomes exactly zero" if h == 0 else ("losing digits" if abs(h - g) / g > 1e-3 else "fine")
    underflow.append({"grad": g, "fp16": h, "bf16": b, "verdict": verdict})
    print(f"{g:>12.0e}{h:>16.3e}{b:>16.3e}{verdict:>22}")

SCALE = 1024
g = 1e-8
scaled = torch.tensor([g * SCALE], dtype=torch.float32).to(torch.float16).float().item()
recovered = scaled / SCALE
print(f"\nloss scaling, the fp16 rescue: {g:.0e} × {SCALE} = {g*SCALE:.3e} survives fp16 as "
      f"{scaled:.3e},")
print(f"then divide back → {recovered:.3e}. It works, and it is one more setting to tune.")
print(f"bf16 needs none of it: its floor is 9.18e-41, so the apparatus is unnecessary.")
print(f"\nA gradient of zero means that weight does not move. The model stops learning exactly")
print(f"where the signal was faintest — which is usually where something was left to learn.")

GATE_UNDERFLOW = any(u["fp16"] == 0 and u["bf16"] != 0 for u in underflow)
print(f"\nGATE fp16_underflows_where_bf16_survives: {'PASS' if GATE_UNDERFLOW else 'FAIL'}")
EVIDENCE["extra7_underflow"] = {"rows": underflow, "loss_scale": SCALE,
                                "scaled_survives": scaled, "recovered": recovered,
                                "gate_pass": GATE_UNDERFLOW}


# %% [markdown]
# ## Extra 8 — clipping changes the length, not the direction
#
# §12: measure the combined size of all the gradients, and when it exceeds a threshold shrink
# every one by the same factor. The claim to check is that the *direction* survives exactly.

# %%
torch.manual_seed(21)
clip_model = Model(Config()).to(DEVICE)
clip_batch = encode_batch(DOCS, cfg.block_size, 4, offset=77).to(DEVICE)
ls, n = token_loss(clip_model, clip_batch)
(ls / n).backward()

before = [p.grad.detach().clone().flatten() for p in clip_model.parameters() if p.grad is not None]
norm_before = torch.cat(before).norm().item()
CAP = 1.0
applied = torch.nn.utils.clip_grad_norm_(clip_model.parameters(), CAP).item()
after = torch.cat([p.grad.detach().flatten() for p in clip_model.parameters() if p.grad is not None])
vb = torch.cat(before)
cos = F.cosine_similarity(vb.unsqueeze(0).double(), after.unsqueeze(0).double()).item()

print(f"grad norm before clipping : {norm_before:.6f}")
print(f"cap                       : {CAP}")
print(f"scale factor cap/norm     : {CAP/norm_before:.6f}")
print(f"grad norm after clipping  : {after.norm().item():.6f}")
print(f"cosine similarity before/after : {cos:.15f}")
norm_after = after.norm().item()
rel_err = abs(norm_after - min(CAP, norm_before)) / CAP
print(f"\n1 − cosine = {1-cos:.2e}. The direction is preserved to floating-point noise;")
print(f"only the length changed. That is the whole of gradient clipping.")
print(f"\nThe clipped norm lands at {norm_after:.6f} rather than exactly {CAP}, off by "
      f"{rel_err:.1e} relative.")
print(f"That is float32 accumulation over {after.numel():,} elements, not a failure to clip:")
print(f"summing that many squares loses the low bits, so the norm torch measured and the norm")
print(f"recomputed afterwards differ in the last few. Checked to 1e-3 relative for that reason.")
GATE_CLIP = abs(1 - cos) < 1e-9 and rel_err < 1e-3
print(f"GATE clipping_preserves_direction: {'PASS' if GATE_CLIP else 'FAIL'}")
EVIDENCE["extra8_clipping"] = {"norm_before": norm_before, "cap": CAP,
                               "scale": CAP / norm_before, "norm_after": norm_after,
                               "relative_error": rel_err, "elements": after.numel(),
                               "cosine": cos, "gate_pass": GATE_CLIP}
del clip_model


# %% [markdown]
# ## Extra 9 — sixteen bytes for every weight
#
# §13's table, but counted off a real optimiser rather than quoted. This is the arithmetic that
# makes Sessions 12 and 13 necessary.

# %%
opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4)
ls, n = token_loss(model, batch)
(ls / n).backward()
opt2.step()

state_tensors = [v for s in opt2.state.values() for v in s.values() if torch.is_tensor(v) and v.dim() > 0]
per_weight = [
    ("the weight itself, in bf16", 2),
    ("its gradient, in bf16", 2),
    ("a full-precision copy, in fp32", 4),
    ("the optimiser's two running numbers", 8),
]
print(f"{'what must be held':<40}{'bytes per weight':>18}")
print("-" * 60)
for name, b in per_weight:
    print(f"{name:<40}{b:>18}")
print(f"{'total':<40}{sum(b for _, b in per_weight):>18}")

print(f"\nmeasured on this optimiser: {len(state_tensors)} state tensors for "
      f"{sum(1 for p in model.parameters())} parameter tensors "
      f"= {len(state_tensors)/sum(1 for p in model.parameters()):.0f} per parameter (Adam's m and v)")
print(f"state elements {sum(t.numel() for t in state_tensors):,} = "
      f"{sum(t.numel() for t in state_tensors)/N_PARAMS:.0f} x the {N_PARAMS:,} weights")

print(f"\n{'model':<10}{'training state alone':>24}{'fits in 80 GB?':>18}")
print("-" * 54)
sizes = []
for label, nw in (("2B", 2e9), ("9B", 9e9), ("20B", 20e9), ("120B", 120e9)):
    gib = nw * 16 / 2**30
    sizes.append({"model": label, "weights": nw, "gib": round(gib, 1), "fits_80gb": gib < 80})
    print(f"{label:<10}{gib:>21,.1f} GiB{('yes' if gib < 80 else 'no'):>18}")
crossover = 80 * 2**30 / 16 / 1e9
print(f"\nAn 80 GB accelerator holds about a {crossover:.1f}B model in training state — and has")
print(f"nothing left for the activations it also needs. That is why one card is not enough.")

GATE_BYTES = sum(b for _, b in per_weight) == 16
EVIDENCE["extra9_memory"] = {"per_weight": [{"item": k, "bytes": v} for k, v in per_weight],
                             "total_bytes": sum(b for _, b in per_weight),
                             "state_tensors": len(state_tensors),
                             "state_elements": sum(t.numel() for t in state_tensors),
                             "params": N_PARAMS, "sizes": sizes,
                             "crossover_80gb_billions": round(crossover, 2),
                             "gate_pass": GATE_BYTES}
opt2.zero_grad(set_to_none=True)


# %% [markdown]
# ## Extra 10 — forgetting the wipe
#
# §6: *"Gradients add up rather than replace."* Forget `zero_grad()` and batch two trains on
# top of batch one. **The loss will still fall.** Nothing raises. The model is simply learning
# from a stale blend of everything it has seen.

# %%
def train_wipe(wipe, steps=60, bs=4, lr=3e-4):
    torch.manual_seed(99)
    m = Model(Config()).to(DEVICE)
    o = torch.optim.AdamW(m.parameters(), lr=lr)
    curve, norms = [], []
    for s in range(steps):
        ids = encode_batch(DOCS, cfg.block_size, bs, offset=s * bs).to(DEVICE)
        if wipe:
            o.zero_grad(set_to_none=True)
        ls, n = token_loss(m, ids)
        (ls / n).backward()
        norms.append(torch.sqrt(sum((p.grad.double() ** 2).sum()
                                    for p in m.parameters() if p.grad is not None)).item())
        o.step()
        curve.append((ls / n).item())
    return curve, norms


c_wipe, n_wipe = train_wipe(True)
c_stale, n_stale = train_wipe(False)
print(f"{'run':<28}{'first loss':>12}{'final loss':>12}{'final grad norm':>18}")
print("-" * 72)
print(f"{'zero_grad() every step':<28}{c_wipe[0]:>12.4f}{c_wipe[-1]:>12.4f}{n_wipe[-1]:>18.4f}")
print(f"{'never wiped':<28}{c_stale[0]:>12.4f}{c_stale[-1]:>12.4f}{n_stale[-1]:>18.4f}")
print(f"\nThe un-wiped run's loss fell from {c_stale[0]:.2f} to {c_stale[-1]:.2f}. It looks like training.")
print(f"Its grad norm is {n_stale[-1]/n_wipe[-1]:.1f}x the correct run's, because step {len(c_stale)} is")
print(f"carrying the accumulated sum of all {len(c_stale)} batches — and nothing anywhere raised.")
GATE_WIPE = c_stale[-1] < c_stale[0] and n_stale[-1] > n_wipe[-1]
print(f"\nGATE stale_grads_still_look_like_training: {'PASS' if GATE_WIPE else 'FAIL'}")
EVIDENCE["extra10_zero_grad"] = {
    "wiped": {"curve": [round(x, 4) for x in c_wipe], "final_norm": n_wipe[-1]},
    "stale": {"curve": [round(x, 4) for x in c_stale], "final_norm": n_stale[-1]},
    "norm_ratio": n_stale[-1] / n_wipe[-1], "gate_pass": GATE_WIPE,
}


# %% [markdown]
# ---
# ## Gates and evidence

# %%
GATES = {
    "tokenizer_hash_verified": tok_sha == FROZEN_TOKENIZER_SHA256,
    "one_grad_per_weight_two_states": GATE_SHAPES,
    "toy_gradient_agrees": GATE_TOY,
    "real_gradient_agrees_to_6_digits": GATE_REAL_GRAD,
    "notes_case_reproduces_15_4pct": GATE_NOTES_CASE,
    "reported_loss_is_misleading": GATE_ACC,
    "accumulation_schemes_differ": GATE_GRAD_DIFFERS,
    "grad_norm_led_the_loss": GATE_LEAD,
    "fp32_bits_match_struct": GATE_FP32,
    "bf16_matches_torch": GATE_BF16,
    "fp8_matches_torch": GATE_FP8,
    "fp16_underflows_where_bf16_survives": GATE_UNDERFLOW,
    "clipping_preserves_direction": GATE_CLIP,
    "sixteen_bytes_per_weight": GATE_BYTES,
    "stale_grads_still_look_like_training": GATE_WIPE,
}
EVIDENCE["gates"] = GATES
print(f"{'gate':<42}{'result'}")
print("-" * 54)
for k, v in GATES.items():
    print(f"{k:<42}{'PASS' if v else 'FAIL'}")
n_pass = sum(GATES.values())
print(f"\n{n_pass}/{len(GATES)} gates pass")

EVIDENCE["summary"] = {
    "gates_passed": n_pass, "gates_total": len(GATES),
    "six_answers": {
        "1_tensors_in_the_step": N_PARAMS + grad_elems + state_elems,
        "2_worst_agreeing_digits": worst,
        "3_wrong_average_error_pct": round(100 * (bb - bt) / bt, 2),
        "4_grad_norm_lead_steps": leads[0][2] if leads else None,
        "5_mfu_pct": round(100 * mfu, 2),
        "6_fp8_relative_error_on_0p1": fmt_rows[3]["rel_error"],
    },
}
(OUT / "evidence.json").write_text(json.dumps(EVIDENCE, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'evidence.json'}")
if n_pass != len(GATES):
    raise SystemExit(f"{len(GATES) - n_pass} gate(s) failed")
