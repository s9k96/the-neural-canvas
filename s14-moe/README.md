# One Becomes Eight

**ERA V5 · Session 14 submission.** A dense transformer is trained, its feed-forward block is
turned into eight experts without changing a single output, and it keeps training: validation
loss **5.603 → 5.302** over the next 4M tokens, from exactly where the dense model stopped.

> **The assignment.** *Train a Linear model and convert that into an MoE! Your call on model
> size and data trained on, but must show they continue to train and reduce loss!*

"Linear model" is read as the **dense** model: a decoder whose feed-forward block is one SwiGLU
network that every token passes through, which is what a mixture of experts replaces.

**Live page:** [`one-becomes-eight.html`](one-becomes-eight.html) · **Notebook:**
[`s14_moe.ipynb`](s14_moe.ipynb)

```bash
python s14_moe.py                         # the harness: 8 training runs, ~60 min cold on CPU
python build_notebook.py                  # .py -> executed .ipynb -> baked page (replays cached runs)
python build_notebook.py --bake-only      # re-inject out/evidence.json into the page only
python check_page.py                      # does the page build in a browser? (needs Chrome)
S14_QUICK=1 python s14_moe.py             # smoke run of the whole harness on a toy, ~2 min
```

Use `.venv-s9/bin/python` (torch 2.2.2, nbclient), plus `datasets` for the one-time token fetch.

**Current state: GATES_TBD gates pass**, on CPU (Apple M5), TRAIN_MIN_TBD minutes of training.

---

## 1 · The one thing to check before any curve means anything

"It continues to train" only means something if training continues *from where the dense model
was*. If the conversion itself moved the model, a falling loss afterwards could just be the MoE
recovering from damage the conversion did, and the curve would look the same.

Sparse upcycling (§15 of the notes) makes the conversion exact by construction. Every expert is a
copy of the dense network `E`, and the router's weights for the chosen pair are rescaled to sum
to one, so `Σ gᵢ·E(x) = E(x)·Σ gᵢ = E(x)`. The harness checks this before a single MoE step:

| | value |
|---|---|
| dense validation loss (262,144 held-out tokens) | 5.602706 |
| converted MoE, same tokens | 5.602706 (\|Δ\| 1.5e-8) |
| largest logit difference, one batch | 4.8e-6 on logits up to 11.8 |
| router weights sum to one, worst token | 1.2e-7 off |

### The consequence the notes don't mention

The same identity means that **at the moment of conversion the router gets no gradient from the
language loss.** Every choice of experts gives the same output, so no choice is better than any
other. Measured: **3.6e-9** for the router, against **0.15** for attention on the same batch. It
is an exact zero, up to float32 rounding.

So the router cannot learn anything until the experts have drifted apart, on the different
tokens its random initial choices sent them. One update is enough to start that: the router's
gradient goes 3.6e-9 → 2.0e-4 after one step → 1.4e-2 by step 25 → 5.6e-2 by the end. This is
also why upcycled experts need protecting while they separate (§15, and §7 below).

---

## 2 · Model and data

| | |
|---|---|
| dense model | 4 layers, d_model 256, 4 heads, SwiGLU width 768, tied head: **5.58M** |
| upcycled MoE | 8 experts, top-2, softmax router in float32 at 1/10 init scale: **22.10M total, 7.94M active** |
| grown MoE (§7) | 32 experts, top-4: **78.75M total, 12.69M active** |
| tokens | 20M from 17,152 FineWeb-Edu documents (sample-10BT), streamed once and cached |
| tokenizer | frozen Sarvam-1 (sha256 `bb5115a3…`), as in S9–S13; vocab capped to 8,191 ids + UNK, 95.7% coverage |
| batch | 32 × 256 = 8,192 tokens per step; AdamW β 0.9/0.95, wd 0.1, lr 1e-3, clip 1.0 |

**Why not the S6 corpus.** It is 656,920 tokens. These runs need 11M, which would be about 17 passes
over it, and §5 of the notes records that MoE models overfit faster than dense ones. A falling loss on
repeated data would be memorisation. Here **no token is read twice**: the stream is cut into
disjoint ranges (dense 0–5M, continuation 5–9M, clone 9–11M, validation = the last 262K), and a gate
checks they don't overlap.

**No learning-rate decay, deliberately.** A cosine decay to the end of the dense phase would make the
continuation start by re-warming, and the resulting bump would be the schedule's, not the conversion's.

---

## 3 · It keeps training — and the control

Four branches start from the same dense checkpoint and read the same next 4M tokens in the same order.
The dense model keeps training too, because a dense model *also* keeps reducing its loss on new
tokens. Without it, "the MoE's loss falls" would only show that nothing broke.

| branch | val at conversion | val at end | reduction | tokens/s |
|---|---|---|---|---|
| dense-continued | 5.6027 | **5.2943** | 0.308 | 11,581 |
| **moe-bias** (main run) | 5.6027 | **5.3022** | 0.301 | 6,996 |
| moe-aux | 5.6027 | 5.3039 | 0.299 | 6,838 |
| moe-none | 5.6027 | 5.2898 | 0.313 | 6,704 |

**What the assignment asks for holds.** The MoE starts at exactly the dense model's loss, and every
validation checkpoint after conversion is below that starting point.

**What it doesn't show is the MoE winning.** At 4M tokens, all four branches end within 0.014 of one
another. With one seed, a gap that size can't separate them. The main run ends 0.008 *above*
the dense control, and the unbalanced MoE 0.005 below it. This is reported, not hidden, and it is what
the notes' own §15 numbers predict: sparse upcycling beat continued dense training only with
10–60% *extra* budget, and here the experts spend the first part of that budget just becoming different
from one another. On top of that, equal tokens is not equal compute. Per token the MoE does twice the
feed-forward work, and on this CPU it runs at 0.60× the dense model's throughput.

### The experts did separate

Relative to the dense network's own weight norm, the experts moved on average 0.52–0.71 (per layer)
away from the network they were copied from, and the experts in each layer ended **0.66–0.95 apart from one another**. The
"MoE" isn't the dense model computed eight times.

---

## 4 · Balancing: measured, not assumed

`MaxVio = (max load − mean load) / mean load`, averaged over the 4 layers and counted per 25-step window:

| branch | balancing | MaxVio, last window | worst window | final val |
|---|---|---|---|---|
| moe-none | none | **1.97** | 1.97 | 5.2898 |
| moe-aux | Switch aux loss, α = 0.01 | 0.31 | 0.37 | 5.3039 |
| moe-bias | bias, γ = 0.001 (aux-loss-free) | **0.18** | 0.39 | 5.3022 |

Without balancing, the load is still diverging at the end: MaxVio's worst window is the last one,
with the busiest expert near 3× an even share. Both cures work, and the bias works better, as Wang
et al. report at 1B–3B. At this scale the loss cost of balancing is about 0.01, which is within the
noise that separates any two branches here. The notes' claim that the bias **leaves the language-loss
gradient untouched** is checked directly in the notebook. The bias is a buffer with no gradient, and a
bias that changes which experts are chosen leaves the weights equal to the raw scores, rescaled. One
artefact of that: moe-aux's router gradient at step 1 is 1.2e-2, not ~0, because the aux term reaches
the router even when the language loss cannot.

The bias counts load over the whole 8,192-token step. On one machine that *is* the whole batch,
which is the scope §14 favours.

---

## 5 · Growing again: clone families

CLONE_TBD

---

## 6 · The notes' figures, recomputed

Most of the session is about a 30.5B model nothing here can train. Every figure the notes quote
about it is arithmetic from its shape, so §1 of the notebook recomputes **23 of them**: total
30.53B, active 3.35B, 455 GiB of training state, 20.1 GFLOP per token, 96 KiB KV cache, the §7 router
example's weights, the §13 bias example, C(128, 8), capacity 640, aux loss 0.01 / 1.28, 45.1 GB of
all-to-all per sequence, and 87.4 / 59.3 GiB per card for the EP layouts. **All 23 reproduce.** The page
marks these `computed` and the trained model `measured`.

---

## 7 · Two decisions that came from measuring, not from defaults

**CPU, not the M5's GPU.** MPS was measured first, on this model, with torch 2.2.2. It ran the dense
model no faster than CPU (9.4K vs 8.6–10.3K tokens/s), and the MoE at **half** CPU speed (2.7K vs
5.2K tokens/s). The routing's indexed gather has a 78 ms backward on MPS, against 2.7 ms for
`index_select`. CPU runs are also deterministic, as S9–S12's were.

**`unbind` the stacked expert weights once per forward.** Experts are stored as `[E, d, f]` tensors, so
upcycling is one `repeat`. Indexing that parameter per expert makes autograd build a full `[E, d, f]`
gradient for every slice, which is E² work. Unbinding once removes that cost; the 32-expert step went
from 6.5 s to 4.7 s.

---

## 8 · Scope limits

* **Scale.** One seed, a 5.6M dense model, 5M + 4M tokens. Nothing here establishes behaviour at 1B+,
  and the gaps between branches are one run each.
* **Equal tokens, not equal compute**, in every dense-vs-MoE comparison.
* **Not implemented:** expert parallelism (§15–17 are arithmetic), capacity limits (the page computes
  what they would have dropped from the measured loads instead), z-loss, shared experts, partition and
  drop-upcycling, and a micro-batch vs whole-batch comparison.

---

## 9 · Files

| file | what | committed |
|---|---|---|
| `s14_moe.py` | the harness, `# %%` cells: source of truth | yes |
| `s14_moe.ipynb` | executed notebook with outputs (build artifact) | yes |
| `build_notebook.py` | `.py` → executed `.ipynb` → baked page | yes |
| `one-becomes-eight.html` | the page, `S14DATA` baked in | yes |
| `check_page.py` | renders the page in headless Chrome, asserts every panel built | yes |
| `out/evidence.json` | every number the page shows | yes |
| `out/tokens_20000000_8192.meta.json` | token cache provenance: source, coverage, sha256 | yes |
| `out/tokens_*.npy` | the 20M-token cache, re-streamed in ~11 s, verified by sha256 | no |
| `out/stages/` | per-run JSON + weight checkpoints, the resume cache | no |

Every training run is a cached stage, so `build_notebook.py` replays them in seconds. To make the
notebook retrain from scratch, delete `out/stages/`.

---

## 10 · Gates

GATE_LIST_TBD
