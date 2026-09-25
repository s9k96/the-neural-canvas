# Forget the Middle

**ERA V5 · Session 13 submission.** A reversible stack keeps the state entering the layers and
the state leaving them, throws away everything in between, and rebuilds it on the way back.
Activation memory stops depending on depth.

> **The assignment.** *Train a 20M LLM for 50M tokens on Google Colab. Fix a batch size that you
> can run. Train again with Reversibility (report which variant worked for you, mid-point,
> euler, etc). Train again with Reversibility, but push it to the maximum batch size. Report
> final loss, speed (token/s), memory peak and other findings.*

**Live page:** [`forget-the-middle.html`](forget-the-middle.html) · **Notebook:**
[`s13_reversibility.ipynb`](s13_reversibility.ipynb)

```bash
python build_notebook.py                  # .py -> Colab-ready .ipynb (not executed here)
python build_notebook.py --smoke          # run the CPU smoke path first
python build_notebook.py --bake-only      # inject out/evidence.json into the page
python s13_reversible.py                  # engine self-check: are the rules actually exact?
python check_page.py                      # does the page build in a browser? (needs Chrome)
S13_QUICK=1 python s13_reversibility.py   # CPU smoke run of the whole harness, ~40s
```

**Current state: 14/14 gates pass**, measured on a Tesla T4 in about 74 minutes.

---

## 1 · The one thing to know before reading any number below

**A reversible stack that reconstructs slightly wrong still trains.** The loss still falls. It
simply trains a different model from the one you wrote down.

So the loss curve proves nothing about correctness, and this submission does not use it as
evidence. §4 compares reversible gradients against ordinary autograd instead, in float64, where
an exact implementation must agree to rounding. Three bugs were found that way while building
this, and every one of them left a perfectly healthy-looking loss curve behind.

---

## 2 · Where this assignment sits

Sections 1–15 of the session are about dividing a 30B model across many GPUs — tensor, sequence,
pipeline and context parallelism, schedules, placement, three DeepSeek case studies. **The
assignment draws on §§16–17 only, and runs on one GPU.** That is not a complaint; it is the
reason the page marks every figure `measured`, `computed` or `reference`, so the parallelism
arithmetic is never mistaken for something that ran here.

What the earlier sections supply is the motivation. ZeRO divides the *training state* across
GPUs but never divides the activations, and one 8,192-token sequence through a 30B model stores
**127.5 GiB** of them. Reversibility does not divide activations. It removes them.

| sequence | stored | reversible | ratio |
|---|---:|---:|---:|
| 8,192 | 127.5 GiB | 1.5 GiB | **85.9×** |
| 32,768 | 510.0 GiB | 5.9 GiB | 85.9× |
| 131,072 | 2,040.0 GiB | 23.8 GiB | 85.9× |

The ratio is constant because the 96 cancels. What remains grows with the token count, not with
depth — and §6 measures exactly that on real hardware.

---

## 3 · The mechanism, in my own words

An ordinary block adds its output to its own input, `p[l] = p[l-1] + f(p[l-1])`. That cannot be
run backwards: recovering the input needs `f` evaluated *at the input being recovered*. The
midpoint rule reaches two layers back and evaluates the block in between, at a state the backward
pass already holds:

```
forward:   p[l+1] = p[l-1] + 2h · f(p[l])
backward:  p[l-1] = p[l+1] − 2h · f(p[l])
```

So the backward pass walks down the stack reconstructing each input from the output it holds, and
nothing in between is ever kept. This is **not** recomputation: recomputation stores the input of
every group of layers and runs them forward again. A reversible stack stores no layer inputs at
all.

Four rules are implemented and compared, all fitted to the same parameter count and the same
residual-stream width:

| rule | reversible | why it is here |
|---|---|---|
| `standard` | no | the baseline |
| `euler` | **no** | `p + h·f(p)` has the same flaw as the residual — a documented negative result |
| `midpoint` | yes | the rule §16 specifies; used for all three required runs |
| `blended` | yes | adds Lightning LM's γ; §8 measures what γ costs |
| `revnet` | yes | channel coupling, exactly invertible with no stability band |

---

## 4 · Is it correct?

Two questions, deliberately asked at different precisions, because conflating them is how a
correct implementation gets mistaken for a broken one.

| rule | float64 — is it exact? | float32 — how much drift? | verdict |
|---|---:|---:|---|
| midpoint | 1.0e-14 | 5.4e-06 | ✓ exact |
| blended | 6.4e-14 | 4.5e-05 | ✓ exact |
| revnet | 1.7e-13 | 9.8e-05 | ✓ exact |
| euler | — | — | ✗ no inverse exists |

Every reversible rule agrees with ordinary autograd to float64 rounding. The float32 column is
not a bug — it is the drift §7 is about. A float32-only check could not tell a wrong
implementation from an exact one accumulating rounding, which is precisely the mistake I made
before adding the float64 arm.

---

## 5 · The three runs

50M tokens each, same tokens, same seed, same schedule. 20M parameters, 10 layers, d_model 288,
sequence 512, frozen Sarvam-1 tokenizer, streamed from `HuggingFaceFW/fineweb-edu` (sample-10BT).
Tesla T4.

| run | batch | steps | final loss | tok/s | peak memory |
|---|---:|---:|---:|---:|---:|
| 1. standard | 108 | 904 | **5.058** | 43,701 | 12,117 MB |
| 2. reversible, same batch | 108 | 904 | 5.241 | 36,516 | **2,528 MB** |
| 3. reversible, max batch | 543 | 179 | 6.069 | 36,763 | 11,250 MB |

**Which variant worked: midpoint.** It is exact, it trains, and its float32 drift is negligible
at this depth. RevNet is equally exact and would be the choice if the step size ever proved hard
to tune, since it has no stability band at all. Euler is not reversible and Blended amplifies
reconstruction error — §8.

### What it costs (run 1 → 2)

**4.8× less memory for 16% throughput.** 12,117 MB → 2,528 MB at an identical batch, at a cost of
43,701 → 36,516 tok/s. The paper estimates the compute overhead at 30–50%; measured here it is
16%, which is better than advertised.

### What it buys (run 2 → 3), and this is the interesting one

**Nothing.** Pushing to the largest batch that fits gained **0.7% throughput** — 36,763 against
36,516 — and made the loss materially **worse**, 6.069 against 5.241.

The reason is in the steps column. 50M tokens at batch 543 is **179 optimizer updates** against
904. Same tokens, a fifth of the learning. And §9's frontier shows why the throughput did not move
either: this GPU is already saturated at **batch 4**, so the memory reversibility frees has
nothing to spend itself on.

That is the session's own caveat — *"reversibility also slows a run down whenever memory is not
the binding constraint"* — reproduced rather than quoted. On a larger model, a longer sequence, or
a card where the stored path could not reach saturation, the same measurement would come out the
other way. **The honest conclusion for this configuration is that reversibility is a memory
technique that this hardware did not need.**

### Where this disagrees with the paper

| | the paper | its setting | this run | its setting |
|---|---:|---|---:|---|
| batch the memory buys | **9.9×** | 26 → 257, 80 GB H100 | **5.0×** | 108 → 543 |
| throughput gain from it | **2.01×** | 56.96 → 114.49 samples/s | **1.007×** | 36,516 → 36,763 tok/s |
| compute overhead | **30–50%** | authors' estimate | **16%** | measured at equal batch |

Three differences, two causes, and neither is a disagreement about the method.

**Depth.** The paper measured at **96 layers**; this model has 10. Reversibility removes *per-layer*
storage, so its advantage grows with depth — §6 measures exactly that here, 27.9× at 4 layers rising
to 218.9× at 32. At 96 the batch ratio would be far closer to theirs. A 20M model also spends much of
its memory on an 8,192-row loss head and on optimizer state, neither of which reversibility touches.

**Saturation.** Their 2.01× came from batch 874 against 52 — a regime where the baseline could not
fill the GPU, so extra batch converted directly into throughput. §9 shows this T4 saturating at
**batch 4**, so the same saving converts into nothing. **The two results agree about the mechanism
and disagree about whether the mechanism pays, which is a property of the hardware rather than of
the method.** The 16% overhead against their 30–50% estimate points the same way: at 10 layers one
extra forward evaluation is a smaller share of a step than it is at 96.

---

## 6 · Memory that does not depend on depth

The session's central claim, asserted in the notes and never shown: *"a 96-layer model and a
20-layer model store the same two boundary states."* Width and `d_ff` are held fixed and only
depth moves, so the parameter count is free to vary.

| layers | params | stored | reversible | ratio |
|---:|---:|---:|---:|---:|
| 4 | 10.9M | 165 MB | **5,923,968 B** | 27.9× |
| 8 | 17.0M | 327 MB | **5,923,968 B** | 55.2× |
| 16 | 29.1M | 650 MB | **5,923,968 B** | 109.8× |
| 32 | 53.3M | 1,297 MB | **5,923,968 B** | 218.9× |

Not merely flat — **byte-identical**. Over 8× the depth, stored activations grew **7.85×** and
reversible grew **1.000×**.

**This table is the second version.** The first held *parameters* fixed while varying depth, which
sounds like the tighter control and is not: parameters go as `n_layer × d_model²`, so pinning them
forces `d_model ∝ 1/√depth`. Activations go as `n_layer × d_model`, so they then grow as √depth
rather than linearly — measured 2.44× against a predicted 2.83× — and the reversible column
*fell*, passing a depth-independence gate for entirely the wrong reason. A gate an artifact can
satisfy is not checking anything.

---

## 7 · Reconstruction drift, which is not in the notes

Reversal *subtracts*, so each rebuilt state carries the error of the one above it and the error
compounds down the stack.

| rule | dtype | 2 layers | 8 | 32 |
|---|---|---:|---:|---:|
| midpoint | float32 | 9.1e-08 | 4.3e-06 | 1.7e-05 |
| midpoint | **float16** | 5.1e-04 | 1.1e-02 | **0.113** |
| revnet | float32 | 3.9e-06 | 1.6e-04 | 4.0e-04 |
| revnet | **float16** | 2.4e-02 | 0.394 | **1.82** |

In float32 the drift is negligible. In float16 midpoint reaches **11% gradient error** at 32
layers and revnet passes **100%**, which is no longer a gradient.

**This is why the training runs use autocast rather than `model.half()`** — fp16 matmuls with the
residual stream kept in fp32, so the reconstruction subtracts in fp32 and the error has nothing to
compound in. Choosing half precision for the ~2× speed would have produced a run that trains,
reports a falling loss, and computes the wrong gradients. It is the same failure mode Session 12
measured when a bfloat16 ring summed 1…32 to 524 instead of 528.

---

## 8 · The blend coefficient, and an arithmetic constraint on it

§16 records Lightning LM running `h = 0.25` and a **blend coefficient of 0.5**, and says these
must be set explicitly because library defaults differ. It never gives the formula. Under the
natural reading the inverse divides by `(1−γ)`, so it amplifies any reconstruction error by
`1/(1−γ)` **once per layer**.

| γ | predicted `1/(1−γ)^L` | measured drift | measured amplification |
|---:|---:|---:|---:|
| 0.0 | 1.0× | 4.3e-06 | 1.0× |
| 0.125 | 2.9× | 5.7e-06 | 1.3× |
| 0.25 | 10.0× | 1.3e-05 | 3.0× |
| 0.5 | 256.0× | 8.7e-05 | **20.2×** |

The law is real in direction and **over-predicts in magnitude** — the blend damps the forward
while the inverse amplifies, and the two partly cancel. Treat it as an upper bound.

Either way it constrains what γ = 0.5 can mean: under this reading, 20 layers would amplify by
2²⁰ ≈ 10⁶, so Lightning LM's pairing at depth implies a **different formula from the natural
one**. The notes give the coefficient and not the equation; this is a reading, and the
measurement is what bounds it. I would rather publish that than quietly pick whichever formula
made the numbers agreeable.

---

## 9 · The frontier

The three runs are three points. This is the curve they sit on.

| batch | stored tok/s | stored peak | reversible tok/s | reversible peak |
|---:|---:|---:|---:|---:|
| 1 | 13,159 | 427 MB | 9,717 | 420 MB |
| 4 | **36,382** | 749 MB | 30,036 | 454 MB |
| 16 | 39,012 | 2,052 MB | 33,795 | 695 MB |
| 64 | 38,235 | 7,299 MB | 36,326 | 1,651 MB |
| 256 | — | OOM | 38,489 | 5,494 MB |
| 512 | — | OOM | 35,155 | 10,629 MB |

Both paths reach roughly the same ceiling, ~38–39k tok/s, and the stored path is already there at
**batch 4** on 749 MB. Reversibility moves the curve far to the left — at batch 64 it uses 1,651
MB against 7,299 MB, a **4.4× difference** — but moving left along a flat curve gains nothing.

---

## 10 · Scope limits, stated rather than buried

* **Sections 1–15 of the session are not measured here.** They are about many GPUs; this is one
  T4. The page marks them `computed` or `reference`.
* **fp16 reversibility is measured, not used.** §7 explains why.

### Four experiments identified and not run

Out of GPU time rather than out of interest. Each is a question this submission raises and does not
answer, so each is named rather than omitted.

| not run | what it would answer |
|---|---|
| **a recomputation baseline** | §16 distinguishes reversibility from gradient checkpointing, and checkpointing is what anyone with a memory problem reaches for first. Comparing reversibility only against *store everything* flatters it; the honest three-way is stored / checkpointed / reversible. **This is the most important gap.** |
| **training revnet and blended** | The assignment asks which variant worked. Both are verified exact (§4) and measured for drift (§7), but only midpoint was trained — so "which worked" currently rests on correctness rather than on three loss curves. |
| **a step-size sweep** | §16 calls the rule "only marginally stable" and says h and γ must sit in a narrow range. §8 sweeps γ for *drift*; neither is swept for *trainability*, which is what that sentence is about. h was fixed at 0.25 throughout. |
| **a sequence-length sweep** | Activations scale with tokens, so longer sequences make memory bind harder. It would test whether §5's negative result is specific to sequence 512 or general to this GPU. |
* **The corpus is streamed, not S6's.** Reaching 50M tokens from S6's committed 656,920 would mean
  76 epochs, which trains a memoriser and makes "final loss" mean something other than intended.
  The **tokenizer** is still the frozen Sarvam-1 one from Sessions 9–12.
* **The corpus is `HuggingFaceFW/fineweb-edu` (sample-10BT) — inferred, not recorded, and that
  is a defect worth stating.** The harness tries fineweb-edu, then `wikitext-103-raw-v1`, then
  `roneneldan/TinyStories`, then S6's corpus offline. This run's cache was built before the
  harness wrote a provenance sidecar, so it reported `corpus.source: "cache"` and the dataset
  identity was not recoverable from the run itself — the cache stores ids already remapped to the
  capped vocabulary, so the tokens cannot even be decoded back to text without the mapping, which
  was also not saved. Re-resolving the chain in the same environment selected fineweb-edu on the
  first attempt, 50/50 sampled documents passing the length filter, so no fallback was reached.
  That is a strong inference and it is labelled as one in `evidence.json`.

  **Fixed since.** The cache now writes `tokens_*.meta.json` carrying the dataset, its config, the
  tokenizer sha256, the vocabulary coverage, the kept-id list and a **sha256 of the tokens
  themselves**; reloading verifies that hash and refuses to continue if the corpus changed under a
  resumed run. None of the conclusions depend on the dataset: every run consumed identical tokens
  (gated), so the comparisons are internal. The identity only affects how the absolute loss reads.
* **`meta.wall_seconds` in the evidence is 9.5**, because the final invocation resumed from cached
  stages. The full run cost 74 minutes and is recorded as `wall_seconds_full_run`.

---

## 11 · Files

| file | what it is |
|---|---|
| `s13_reversible.py` | the engine: model, five update rules, the reversible `autograd.Function`, activation accounting, and `gradient_check` |
| `s13_reversibility.py` | the harness and source of truth (`# %%` cells), in nine resumable stages |
| `s13_reversibility.ipynb` | generated for Colab; carries the engine embedded so it needs no clone |
| `build_notebook.py` | `.py` → `.ipynb`, and `--bake-only` to inject returned evidence |
| `make_probe.py` / `s13_probe.ipynb` | the sizing probe that ran before any of this was written |
| `check_page.py` | renders the page in headless Chrome and asserts it actually built |
| `out/evidence.json` | every number on the page |
| `out/stages/` | per-stage checkpoints; gitignored, regenerated |

**Two deviations from the repo's usual pattern**, both forced and both documented in the code.
The notebook is **not executed locally** — the assignment needs peak GPU memory and 150M tokens of
training, and this machine has no CUDA; it is generated here, run on Colab, and its evidence baked
back. And the notebook **embeds the engine** rather than cloning the repo, because
`s13_reversible.py` was not committed when the runs happened. A cell writes it to disk, evicts any
stale copy from `sys.modules`, and asserts that what it loaded has the functions the harness needs.

---

## 12 · Gates

All 14 are asserted by the harness, which exits non-zero if any fails.

**Correctness** — every reversible rule matches ordinary autograd in float64; Euler has no
inverse; there is no dropout anywhere in the model; all four rules have the same parameter count.

**The three runs** — reversibility uses less memory, **costs** throughput (asserted in that
direction, because the honest expectation is that it is slower), buys batch size, every run saw
the same tokens, and every run trained.

**Depth** — reversible activation bytes stay within [0.9, 1.1] across 8× the depth, and stored
bytes track the depth.

**Blending** — γ amplifies drift.

---

## 13 · What I would do next

**Overlap the reconstruction with the backward pass.** The reversible backward recomputes a block
and then waits for it; on a memory-bound run that recompute could hide behind the gradient
computation of the layer above.

**Run it where memory actually binds.** Every interesting conclusion here is limited by a GPU that
saturates at batch 4. The same harness on a longer sequence or a larger model would test the
claim that reversibility *helps*, which this configuration could not.

**Settle the blend formula.** §8 shows the natural reading cannot be what Lightning LM ran at 20
layers. Reading their implementation would replace a bounded inference with a fact.
