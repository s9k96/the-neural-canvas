# One Step, and What It Costs

**ERA V5 · Session 10 submission.** A real training loop made to tell the truth about itself.

**Live page:** [`training-step.html`](training-step.html) · **Notebook:**
[`s10_training_loop.ipynb`](s10_training_loop.ipynb) (committed with its outputs)

```bash
python build_notebook.py        # convert .py -> .ipynb, execute it, bake the page  (~2 min, CPU)
python s10_training_loop.py     # or just run the harness directly
python check_page.py            # does the page actually build in a browser?
```

`build_notebook.py` exits non-zero if any gate fails. That exit code is the pass/fail signal,
not the printed summary. **Current state: 15/15 gates pass.**

---

## 1 · The problem

Session 9 ended holding one scalar. This session is how that number reaches back and moves
every weight, and how you know — over weeks — that it is working.

```python
logits = model(batch)        # forward
loss   = loss_fn(logits, y)  # one number
loss.backward()              # fill in every gradient
optimizer.step()             # move every weight
optimizer.zero_grad()        # wipe before the next batch
```

Every training run in the world is these five lines repeated until the data runs out. The
assignment's closing line is the whole brief: *"Print things and check things. Every serious
training bug is silent, and the loss curve is not going to be the one that tells you."*

---

## 2 · Method, step by step

Same frozen Sarvam-1 tokenizer (68,096 tokens, sha256 `bb5115a3…`), same S6 corpus and same
4-layer decoder as Session 9 — this is literally the next step after that session's scalar. The
model definition is copied rather than imported, because each session ships standalone, the way
S6 keeps its own copy of S4's cleaning regexes.

| # | Step | What it does |
|---|---|---|
| 0 | Verify and load | Hash-check the tokenizer against S6's frozen constant; load real Hindi/code/math/agentic documents. |
| 1 | Shape ledger | Every tensor in the *step* — not just the forward pass, but what `backward()` fills in and what the optimiser keeps. |
| 2 | Gradient by hand | The notes' two-link chain on paper, then three real weights at three depths, in float64, with the step size swept. |
| 3 | Break accumulation | The notes' own table, then real unequal micro-batches, then the **gradients** under both schemes, then two full training runs. |
| 4 | Grad norm | Log it every step at a deliberately unstable learning rate, with a spike rule fixed in advance, and search for a lead. |
| 5 | MFU | Measure tokens/s and an empirical machine peak, then divide honestly. |
| 6 | 0.1 in three formats | Decompose by hand, then check against `struct`/`torch`. |
| 7–10 | Beyond the six | fp16 underflow and the loss-scaling rescue; clipping; 16 bytes/weight on a real optimiser; forgetting `zero_grad()`. |

One structural note: **Task 2 requires float64.** A central difference subtracts two nearly
equal losses, so in float32 the cancellation destroys the digits you are trying to compare and
the check "fails" for reasons that have nothing to do with the gradient.

---

## 3 · Findings — the six tasks

### 1 · Every tensor in the step

`152,445,952` numbers are held for `38,111,488` weights — **4× the model**, before a single
activation. One gradient per parameter exactly, and two optimiser states per parameter (Adam's
`m` and `v`), both verified by count rather than assumed.

### 2 · Verify one gradient by hand

The notes' chain first: `x=2, w₁=3, w₂=4, t=20` → `h=6, y=24, loss=16`. Analytic
`2(y−t)·w₂·x = 64`; forward difference at `h=1e-3` gives **64.064** (the notes' number);
central difference gives **64.0000**; autograd gives **64**.

The forward difference is not wrong — it is `O(h)`, measuring the slope of a *chord*. That is a
property of the estimator, and it is the whole reason the next table sweeps `h`.

Then three real weights, in float64, agreeing significant digits at each step size:

| weight | autograd | h=1e-2 | h=1e-3 | h=1e-4 | h=1e-5 | h=1e-6 | h=1e-7 |
|---|---|---|---|---|---|---|---|
| `head.weight[100,7]` | 0.0000038477 | 4.5 | **6.4** | 6.3 | 4.8 | 4.4 | 2.7 |
| `blocks.0.qkv.weight[3,11]` | 0.0000772910 | 6.2 | **7.7** | 7.3 | 6.5 | 5.0 | 4.0 |
| `embed.weight[12856,5]` | −0.0069339785 | 3.7 | 5.7 | 7.7 | **8.2** | 7.5 | 6.3 |

**Worst agreement, each at its own best h: 6.4 significant digits.** The U is the finding — too
large an `h` and truncation dominates, too small and cancellation does. Note the optimum moves:
the embedding row's gradient is three orders of magnitude larger than the head weight's, so it
tolerates a smaller `h` before cancellation bites.

### 3 · Break gradient accumulation on purpose

The notes' case reproduces exactly: micro-batches of 4, 4, 2 valid tokens with losses 2.0, 2.0,
5.0 give `26.0/10 = 2.6000` by token against `(2+2+5)/3 = 3.0000` by micro-batch — **15.4%
wrong**. Set the counts equal and the error is exactly zero, which is how it hid.

But the decisive measurement is not the loss value, it is the **gradient**, accumulated from
identical weights under both schemes:

| quantity | value |
|---|---|
| relative L2 difference | **24.3%** |
| cosine similarity | 0.9733 |
| angle between them | **13.3°** |
| ‖g_wrong‖ / ‖g_correct‖ | 1.052 |

**What the curves do and do not show.** The wrong run *reports* a 0.146-lower loss, and anyone
watching that curve would conclude the run was going better. On a held-out batch scored
identically for both, the difference is **−0.0083** — inside the noise at 120 steps of a proxy
model.

So I am not claiming the wrong average visibly degrades a short run, because it did not. The
defect is upstream of the curve: the optimiser is being pointed 13.3° in the wrong direction
every step. A wrong objective that produces a plausible curve for 120 steps is precisely why
this survived in every major framework until 2024.

### 4 · The grad norm moved before the loss did

Rule fixed before looking: a **spike** is a jump more than 3.5 MADs above the median jump of the
previous 20 steps; a **lead** is a norm spike whose nearest loss spike lands within 5 steps.

It has to be the *jump* and not the level. The loss is falling, so it can never exceed its own
trailing median — a level-based rule reports zero loss spikes no matter what the run does, which
is exactly what my first version did.

**The step: the grad norm spiked at step 38, the loss followed at step 41 — three steps later.**

| step | grad norm | loss | |
|---|---|---|---|
| 37 | 1.9797 | 7.7542 | |
| 38 | **2.9530** | 6.8980 | ← norm |
| 39 | 3.0018 | 6.8767 | |
| 40 | 3.0763 | 6.8935 | |
| 41 | 1.8845 | **8.3334** | ← loss |

8 norm-led events out of 19 norm spikes. Two deliberate choices: **clipping is off** (a clipped
norm is flat by construction and the trace tells you nothing), and the learning rate is set high
enough that the run is genuinely unstable, because the claim under test is about a run in
trouble. The event was **found, not injected**.

### 5 · MFU, reported honestly

| quantity | value |
|---|---|
| tokens per second | 3,992 |
| non-embedding N | 20,646,144 |
| achieved `6·N·tok/s` | 0.495 TFLOP/s |
| this machine's measured peak | 1.507 TFLOP/s |
| **MFU** | **32.8%** |
| healthy range (§14) | 35–50% |

Two things stated rather than hidden.

**The denominator.** There is no meaningful datasheet FLOP/s for "a CPU running PyTorch", and
quoting an H100 number on hardware I am not using would be a fabricated result. The peak is the
best sustained throughput this machine actually reaches on large dense matmuls — the most
generous denominator that is still true.

**The numerator.** `6N` counts the arithmetic of *matmuls*, and an embedding lookup is a gather
that does no multiplying at all. Counting the 17,465,344-element embedding table in `N` would
report **60.6%** instead of 32.8% — **27.8 points of pure fiction.** My first version did exactly
that and produced a suspicious 61%, which is what caught it.

The same formula on the notes' own example (9B, 12,000 tok/s, 8×H100) returns **8.2%**, which is
what they report.

**What is costing the distance to 40%:**

1. The vocabulary head is 84% of non-embedding `N`, so **84% of counted FLOPs** are one
   `[T,256]×[256,68096]` matmul — a tiny inner dimension against an enormous output, the least
   efficient shape in the model.
2. Only 512 tokens per step, so kernel launch and Python overhead are a real fraction of it —
   time `6N` does not count.
3. No fused kernels, no flash-attention path on CPU, fp32 throughout. The measured peak comes
   from one huge matmul that keeps caches full; a transformer step does not.
4. The optimiser touches every weight three times and does no matmul at all: pure denominator.

### 6 · 0.1 in fp32, bf16 and fp8 E4M3

`0.1 = 1.6 × 2⁻⁴`, so every format stores the same exponent and differs only in how much of
`0.6` survives the mantissa. Decomposed by hand with round-half-to-even, then checked against
the machine.

| format | bits (s / e / m) | stores | relative error |
|---|---|---|---|
| fp32 | `0 01111011 10011001100110011001101` | 0.100000001 | 1.5e-08 |
| bf16 | `0 01111011 1001101` | 0.100097656 | 9.8e-04 |
| fp16 | `0 01011 1001100110` | 0.099975586 | 2.4e-04 |
| **fp8 E4M3** | `0 0011 101` | **0.1015625** | **1.6e-02** |

Cross-checks: the hand-computed fp32 word is `0x3dcccccd`, identical to `struct.pack`; bf16 and
fp8 match `torch.bfloat16` and `torch.float8_e4m3fn` exactly. 0.1 is not representable in binary
at any width — it is the repeating fraction `0.0001100110011…` — so every row is wrong and the
only question is by how much.

**Which would I train in? bf16, with an fp32 master copy.**

The relative errors rank fp32 < fp16 < bf16 < fp8, and that ranking is almost irrelevant to the
decision — which is §10's point. **bf16 is less accurate than fp16 on this number and won
anyway**, because what breaks a run is not the error on 0.1. It is what happens to a gradient of
`1e-8`:

| gradient | in fp16 | in bf16 | fp16 verdict |
|---|---|---|---|
| 1e-4 | 1.000e-04 | 9.998e-05 | fine |
| 1e-6 | 1.001e-06 | 1.001e-06 | fine |
| **1e-8** | **0.000e+00** | 9.996e-09 | **becomes exactly zero** |
| **1e-10** | **0.000e+00** | 1.000e-10 | **becomes exactly zero** |

A gradient of zero means that weight does not move. The model quietly stops learning exactly
where the signal was faintest — which is usually where something was left to learn. fp16's
rescue is loss scaling (×1024 before backward, divide after), which works and is one more
setting to tune and eventually get wrong. bf16's floor is `9.18e-41`; nothing in training will
ever reach it, so the apparatus becomes unnecessary.

fp8 E4M3 storing 0.1 as 0.1015625 is disqualifying for a master weight being nudged by 1e-7 per
step — the update would round to nothing every time. It is fine for *matmul inputs*, where the
error is averaged over thousands of products and does not accumulate. §11's rule: shrink where
the error does not accumulate.

---

## 4 · Findings — beyond the six

**Clipping changes the length, not the direction.** Norm 3.662 → cap 1.0 → scale 0.273. Cosine
similarity before/after: **0.999999999999887**, i.e. `1 − cos = 1.1e-13`. The clipped norm lands
at 0.999683 rather than exactly 1.0 — that is float32 accumulation over 38 million elements, not
a failure to clip, which is why it is checked to 1e-3 relative.

**Sixteen bytes for every weight**, counted off a real optimiser rather than quoted: 2 (bf16
weight) + 2 (bf16 gradient) + 4 (fp32 master) + 8 (Adam's two running numbers). An 80 GB
accelerator holds about a **5.4B** model in training state and has nothing left for activations.

**The wipe is not optional.** Two runs, one never calling `zero_grad()`:

| run | first loss | final loss | final grad norm |
|---|---|---|---|
| `zero_grad()` every step | 11.19 | 7.79 | 2.13 |
| never wiped | 11.19 | 7.81 | **57.77** |

The un-wiped run's loss fell from 11.19 to 7.81. **It looks like training.** Its grad norm is
27× the correct run's because the last step carries the accumulated sum of every batch before
it — and nothing anywhere raised.

---

## 5 · Conclusions

**On the loop.** Four of the failures demonstrated here are invisible in the loss curve, and
three of them make it look *better* or unchanged: the wrong average reports a lower number, the
un-wiped run falls convincingly, a low MFU is indistinguishable from a high one, and an fp16
gradient that underflows to zero simply stops that weight learning. The loss is the one trace
that cannot audit the loop it came from.

**On what to log.** The grad norm earned its place: it moved three steps before the loss on a
real run, under a rule fixed in advance. Tokens/second and MFU catch the failure that has no
loss signature at all. That is §14's four traces, and this run is a small argument for all four.

**On measurement discipline.** Three of my own results were wrong before they were right, and
in each case the error was measuring a real quantity against the wrong reference — a level-based
spike rule on a falling series, `6N` over parameters that do no arithmetic, a finite difference
at a step size the gradient's magnitude could not support. None of them raised; all three
produced confident numbers. The gates caught them because they assert *relationships* — a norm
spike must precede a loss spike, agreement must survive a swept `h` — rather than pinning values.

**For V5**, this session settles two things and leaves four open:

| settled | |
|---|---|
| Loss normalisation | **By token, never by micro-batch.** Not a style preference: 13.3° of gradient direction on this proxy. |
| Grad norm | Logged from step one, clipping on from step one — but log the norm *before* clipping, or the trace is flat by construction and worthless. |

| open | what would settle it |
|---|---|
| Training precision | bf16 is safe, fp8 is proven, NVFP4 is faster and needs Blackwell. A short run in each on the real architecture, comparing loss *and* throughput. |
| Clip threshold | The grad-norm distribution over the first thousand steps. Choose it from data, not habit. |
| MFU floor | A number agreed **before** the run begins, so nobody is negotiating it at three in the morning. |
| Activation checkpointing | Everywhere or only some layers — a memory-against-throughput sweep on the cluster actually rented. |

One thing this run adds to that list: **publish the `N` convention with any MFU.** The same run
is 32.8% or 60.6% depending on whether the embedding table is counted, and only one of those is
a real number.

---

## 6 · What went wrong first

Recorded because in each case the harness produced a confident number that meant nothing.

**The MFU was 61% — above the healthy band, which should never have looked plausible.** I had
put all 38.1M parameters into `6N`, including the 17.5M embedding table. An embedding lookup is
a gather; it does no multiplying. Counting it invented FLOPs that were never performed. Using
non-embedding `N` gives 32.8%, which is both correct and interesting, because it leaves a real
distance to 40% to explain.

**The spike rule could not detect a loss spike at all.** I compared each value to its trailing
median — but the loss is *falling*, so it can never exceed its own trailing median, and the rule
returned 30 norm spikes and 0 loss spikes by construction. Comparing first differences instead
treats a trending series fairly and found 26 loss spikes and 8 norm-led events.

**The gradient check was limited by my step size, not by autograd.** At `h=1e-6` against a
gradient of 4e-6, cancellation error is `≈ ε·|L|/h ≈ 2.4e-9` — about 4 digits, and the gate
demanded 6. The fix was to sweep `h` and report the U, which is a better demonstration than the
tolerance it replaced. In the same pass I found `embed.weight[42,5]` had a gradient of *exactly
zero* — token 42 never appears in the batch, so that check was measuring nothing while reporting
perfect agreement.

**And a name collision that only the JSON encoder caught.** A float `central_diff` in the toy
section was shadowed by a function `central_diff` defined later, so `evidence.json` tried to
serialise a function object. Every gate passed; only the write failed. A reminder that a green
suite is not the same as a correct program.

---

## 7 · Files

| Path | What | In git? |
|---|---|---|
| `s10_training_loop.py` | the source of truth, `# %%` cell-delimited | yes |
| `s10_training_loop.ipynb` | the Colab deliverable, committed **with outputs** | yes |
| `build_notebook.py` | convert → execute → bake; the gate runner | yes |
| `check_page.py` | renders the page in headless Chrome and asserts it built | yes |
| `validate_palette.py` | Python twin of the dataviz palette validator | yes |
| `training-step.html` | the page, with `evidence.json` baked in as a JS blob | yes |
| `s10-assignment.md` | §17, transcribed — no separate assignment file was distributed | yes |
| `out/evidence.json` | every number on the page and in this README | yes |

Following the repo's generate-then-bake pattern (S4, S6, S7, S9): the `.py` is the source of
truth, and **the notebook and the page must both be rebuilt when it changes**.

The notebook runs top to bottom on Colab (first cell installs `tokenizers` and
`huggingface_hub`; the second clones this repo for S6's corpus). Locally it needs torch, numpy,
tokenizers and huggingface_hub. Run time is about **2 minutes on CPU**.

Two numbers here are measurements rather than computations and will move slightly between runs:
**MFU** (it depends on wall-clock timing) and the **measured machine peak**. Every gradient,
loss and bit pattern is deterministic under the fixed seed.
