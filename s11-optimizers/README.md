# Setting the Distance

**ERA V5 · Session 11 submission.** A gradient says which way a weight should move. It says
nothing about how far. Every method in this session is a different rule for that second
decision — and the assignment's closing line is the standard every comparison here is held to:

> Tune both sides before accepting a comparison. Almost every optimizer claim that failed to
> replicate was a well tuned method measured against a badly tuned one.

**Live page:** [`setting-the-distance.html`](setting-the-distance.html) · **Notebook:**
[`s11_optimizers.ipynb`](s11_optimizers.ipynb) (committed with its outputs)

```bash
python build_notebook.py               # .py -> executed .ipynb -> baked page   (~55 min, CPU)
python s11_optimizers.py               # or just the harness, without rebuilding the notebook
python build_notebook.py --bake-only   # re-inject out/evidence.json into the page only
python check_page.py                   # does the page actually build in a browser? (needs Chrome)
python validate_palette.py "#3987e5,#d95926,#199e70" --mode dark --surface "#0a0a0c"
```

`build_notebook.py` exits non-zero if any gate fails. That exit code is the pass/fail signal,
not the printed summary. **Current state: 24/24 gates pass.**

---

## 1 · The problem

Session 10 ended at `optimizer.step()` and left the inside of it unspecified. This session is
that inside: five tasks that each make the step size say something checkable, plus a sixth
section that turns the brief's closing warning into an experiment instead of a quotation.

---

## 2 · Method

Same frozen Sarvam-1 tokenizer (68,096 tokens, sha256 `bb5115a3…`) and the same Session 6
corpus as Sessions 9 and 10 — **656,920 tokens from 514 documents across 7 lanes**, tokenized
once into a single stream so a batch is a deterministic function of the step index alone. Same
step, same tokens, in every run in this file; that is what makes any two runs here comparable,
and it is checked rather than assumed (see the `both_schedules_start_identically` gate).

The model is the S9/S10 four-layer decoder, copied rather than imported — each session ships
standalone, the way S6 keeps its own copy of S4's cleaning regexes — with two deliberate
changes, both forced by this session:

| change | why |
|---|---|
| initialization is `1/√fan_in`, not a flat 0.02 | §9 derives the update-to-weight ratio *from* `1/√fan_in`, and Task 3 checks that derivation. A flat 0.02 would make the check measure the wrong thing at every width but one. |
| `d_head` held at 64, head count scales with width | Task 5 changes the width. If `d_head` moved with it, the `1/√d_head` attention scale would move too and the sweep would confound two effects. |

**One scope limit, stated up front.** Tasks 5 and 6 need many short runs at three widths. At
width 1,024 an untied 68,096-row head is 70M weights and would dominate both the parameter
count and the FLOPs, so the sweep would be measuring the head rather than the width. The sweep
model therefore caps the vocabulary to the **top 8,192 Sarvam-1 ids** by frequency in this
corpus and maps the rest to `UNK` — 93.24% of token occurrences, of the 22,882 distinct ids that
appear at all. The tokenizer and the text are untouched, and **no experiment that is compared
against the notes' own numbers uses the capped vocabulary**: Tasks 1–4 all run on the full
68,096-token model.

---

## 3 · The five tasks

### Task 1 — reproduce Adam by hand

The five gradients are §6's own worked example, so the arithmetic has two independent things to
satisfy: the table printed in the notes, and `torch.optim.AdamW`.

| | worst absolute disagreement |
|---|---|
| hand (float64) vs PyTorch, over `m`, `v`, `m̂`, `v̂`, step and `w` | **1.11e-16** |
| hand vs §6's printed table, at the notes' own precision | **0** — exact at every printed digit |

Float64 throughout, deliberately: in float32 the comparison bottoms out around 1e-8 and reports
the precision of the container rather than the agreement of the two implementations.

The two formulas look different and are not. The notes write `η·m̂/(√v̂ + ε)`; PyTorch computes
`(η/bc₁)·m/(√v/√bc₂ + ε)`. Multiply numerator and denominator through and they are the same
expression, ε included — which is why the disagreement is machine epsilon rather than merely
small.

**One correction to the notes.** §6 says every step "falls within half a percent of 0.001". Its
own printed table does not: step 2 is 0.000988, which is **1.2%** below η. My reproduction
agrees with the table to every printed digit, so the claim is right and the bound is 1.2%.

### Task 2 — turn bias correction off

The ratio of the two step sizes has a closed form, and writing it down settles the question
before any plotting:

```
step_with_bc      m/(1−β₁ᵗ)          √v            √(1−β₂ᵗ)
────────────  =  ─────────────  ·  ──────  =  ──────────────
 step_no_bc      √(v/(1−β₂ᵗ))        m           (1−β₁ᵗ)
```

**The gradients cancel.** The ratio depends on `t`, `β₁` and `β₂` and on nothing else — not the
data, not the model, not the learning rate. Checked numerically on twenty random gradients: the
measured ratios sit on the closed form to **2.2e-07**.

The criterion was fixed before looking: the difference stops mattering at the first step where
the two step sizes are within **1%**.

| | |
|---|---|
| stops mattering at | **3,916 steps** (5%: 2,327) |
| at step 1 | uncorrected steps are 3.16× larger |
| **widest gap, step 12** | **6.57×** |
| at step 20 | still 6.24× |
| weight displacement after 20 steps | 5.99× further |

Two things here are worth more than the headline number. First, **the answer is not 20** — it is
set by `β₂ = 0.999` and is the same for every run anyone does with those betas; the twenty-step
plot the brief asks for shows the gap at its *widest*, not closing. Second, **the gap peaks at
step 12, not step 1**, so the common reading that bias correction "only matters for the first
step or two" has the story backwards.

**And on a real model — where the comparison is a trap.** Turning bias correction off multiplies
the early step size by up to 6.6×, which is a learning-rate change wearing a different name. So
there are three runs, not two: the third keeps bias correction on and raises η by that same
factor.

| 150 steps, identical seed and batches | held-out loss |
|---|---|
| bias correction on, η = 3e-4 | 6.9556 |
| bias correction **off**, η = 3e-4 | 6.6391 (−0.3164) |
| bias correction on, η = 3e-4 × 6.57 | **6.6023** (−0.0368 against the uncorrected run) |

Turning bias correction off looked like an improvement of 0.316 nats. Matching the step size
with the correction left on recovered 0.353 — so the "improvement" was the learning rate. The
first experiment of the session is already an instance of the brief's closing warning.

The per-step factor between the two runs is 3.16 at step 1 — exactly the closed form — and then
drifts *below* it (4.16 vs 4.25, 4.59 vs 4.95, 4.77 vs 5.44). That is not an error: the closed
form assumes both runs have seen the same gradients, and after one step they have not.

### Task 3 — the update-to-weight ratio, per layer

§9 gives a first-step number for a 4,096-wide model: 0.0192. That is not an observation, it is a
**prediction**, and it applies per tensor. At step 1 every element of `m̂/√v̂` is exactly ±1, so

```
‖Δw‖     η·√numel
──── = ────────────── = η·√fan_in
‖w‖     √numel/√fan_in
```

Measured across all 32 tensors of the real model, the worst disagreement with that prediction is
**0.73%**:

| layer | shape | step-1 measured | predicted | why |
|---|---|---|---|---|
| `blocks.0.qkv.weight` | (768, 256) | 4.807e-3 | 4.800e-3 | `η·√fan_in` |
| `blocks.0.down.weight` | (256, 704) | 7.976e-3 | 7.960e-3 | `η·√704` |
| `blocks.0.n1.g` | (256,) | 3.000e-4 | 3.000e-4 | gain, init 1.0 → η exactly |
| `pos.weight` | (128, 256) | 1.505e-2 | 1.500e-2 | init 0.02 → η/0.02 |
| `embed.weight` | (68096, 256) | 1.215e-3 | 1.214e-3 | sparse: only 552 of 68,096 rows get a gradient |

The notes' 19.2e-3 is this same formula at fan-in 4,096; at our width it reads 4.8e-3, and that
is what the run produced. The embeddings are not exceptions — both are predicted too, one from
its init scale and one from how many rows a batch actually touches.

**When does warmup stop changing it?** The question needs a definition before it has an answer,
and the two reasonable definitions disagree.

Define `ρ(t)` = the median layer's ratio ÷ `η(t)`: the ratio with the schedule divided out. It
starts at **15.98 = √256 = √fan_in** — §9's correlated-gradient regime, measured — falls as the
gradients decorrelate, and bottoms out at **step 76** for a 50-step ramp. Past that the step size
is set by the model's own gradients and not by the ramp. **That is the answer: step 76.**

The other reading — when do a warmup run and a no-warmup run agree? — is less tidy. Their median
ratios come within 20% at step 407 and **never within 10% inside 500 steps**, because after
warmup they are different models on different trajectories. Warmup does not stop mattering; it
stops *controlling*.

| | with warmup | without |
|---|---|---|
| peak ratio over the run | 4.339e-3 | 1.505e-2 (**3.47× larger**) |
| the notes' figures, at width 4,096 | 2.83e-3 | 19.2e-3 |

Post-warmup the median ratio stays in the 1e-3 band §9 asks for.

### Task 4 — cosine against WSD, both stopped at step 200

Two schedules spend their budget differently, so a peak learning rate that suits one need not
suit the other. Both were swept over the same three peaks and each judged at its own best (both
chose 1.2e-3). Held-out loss on 16 fixed batches neither run ever trains on.

| model | steps | held-out loss | what it cost |
|---|---|---|---|
| cosine(300), stopped at 200 | 200 | 6.4223 | the horizon was fixed before step 1 |
| WSD(300), stopped at 200 — still at peak η | 200 | 6.3299 | no decay yet |
| **WSD checkpoint at 200, decayed over 20 steps** | 220 | **6.2996** | 10% more compute |
| cosine planned for 200 from the start | 200 | 6.5926 | a different run, not a stopping point |

**Which model would I keep: the WSD one.** At the stopping point it is 0.0923 nats ahead, and
decaying its checkpoint buys a further 0.0303 for 20 steps of compute — an option cosine does
not have, because its shape was fixed to a horizon of 300 before the first step. The reason to
keep it is the ability to stop, branch and continue, not the third decimal place of a loss on a
300-step run.

**A gate I set in advance, and failed.** §10 says a run stopped early "is worse than one trained
to that shorter length deliberately". I gated that, and the early-stopped cosine came out
**0.1703 nats better**. The reason is in the schedules rather than the models: truncating a
300-step cosine at 200 leaves the learning rate high for most of those steps, so the truncated
run receives substantially more integrated learning rate over the same budget, and at 200 steps
this model is nowhere near the regime where finishing the anneal is what matters. I replaced the
failed prediction with a gate on that *mechanism* — the integrated learning rate — and report the
failure here rather than quietly reversing the inequality. It is not a refutation of §10 at
scale; it is a demonstration of why WSD defines its decay as a fraction of the run rather than a
fixed number of steps.

### Task 5 — sweep the learning rate at three widths

Seven learning rates a factor of two apart, 2.5e-4 to 1.6e-2, at widths 256/512/1,024, under
both the standard parameterization and muP. Two rules fixed before looking: a minimum is only
reported if it is **interior to the grid** (an edge minimum means the grid was wrong), and the
sub-grid position comes from a parabola through the three points around the best one. All six
minima are interior.

| parameterization | width | parabola minimum | §12's table |
|---|---|---|---|
| standard | 256 | 3.58e-3 | 3.0e-3 |
| standard | 512 | 1.91e-3 | 1.5e-3 |
| standard | 1,024 | 6.21e-4 | 7.5e-4 |
| muP | 256 | 3.58e-3 | — |
| muP | 512 | 2.89e-3 | — |
| muP | 1,024 | 3.59e-3 | — |

Under SP the minimum moves by **5.76×** across a 4× width range and the fitted exponent is
**η\* ∝ width^−1.26** (§12's table is exactly −1.00). Under muP it moves by **1.25×** — less than
one grid step, i.e. it did not move. A consistency check that falls out for free: at the base
width the two parameterizations are the same model by construction, and the two curves agree to
the last digit.

**The value I would use at width 4,096, and how confident I am.**

Under muP the sweep at width 256 transfers, so the answer is **η = 3.6e-3 nominal**, which at
width 4,096 means an effective hidden-layer rate of `3.6e-3 × 256/4096 =` **2.2e-4**. §12's table
says 1.9e-4 — an 18% disagreement, from a sweep that cost minutes at width 256. Extrapolating
the SP fit instead gives 1.2e-4, which is 1.6× *below* the notes' value, and that gap is the
argument for muP in one number: the extrapolation is doing worse than the transfer.

*Confidence: moderate for the shape, low for the third digit.* The grid is a factor of 2, so no
minimum is located better than about ±40% even after the parabola fit; each point is one seed of
120 steps on a 3-layer model with a capped vocabulary; and short runs are known to prefer larger
learning rates than long ones, which biases every number here high. The honest form of the answer
is a range — roughly **1.8e-3 to 7.2e-3 nominal** — with muP's transfer, not the extrapolation, as
the reason to believe it.

---

## 4 · Beyond the five tasks

### Muon against AdamW, with both sides tuned

§13's own numbers are the prior: 1.4× at 0.1B falling to 1.1× at 1.2B, *against a well tuned
AdamW*. Both sides here get the same seven-point grid, the same data, the same schedule and the
same step budget. The AdamW half of the Muon hybrid — embeddings, head and gains, which §13 says
must stay on AdamW — is pinned to the baseline's own best rate, so the hybrid does not get a
second free knob the baseline never had.

| | best η | final held-out loss |
|---|---|---|
| AdamW | 2.0e-3 | 5.9164 |
| Muon on 2-D matrices, AdamW on the rest | 4.0e-3 | **5.8157** (−0.1007) |

Both minima interior to the grid — **that** is the gate that matters, because until it holds a
comparison of optimizers is a comparison of two grid choices. Muon reaches AdamW's final loss at
step 240 of 300, a **1.25× speedup** at matched tuning, against §13's 1.4× prior at a model three
orders of magnitude larger than this one.

A second observation the sweep gives away for free: Muon's loss-against-η curve is markedly
flatter than AdamW's (5.83–6.11 across the whole grid, against 5.94–6.58), so it is less
sensitive to the choice — which is itself a practical argument, and one that a single-point
comparison would never have surfaced.

One seed, 300 steps and a width-256 model is not enough to claim a speedup at any scale that
matters. It is enough to show what claiming one requires.

### L2 against decoupled decay, on real second moments

§7's table is two invented parameters with `√v̂` of 1.00 and 0.01. The real model is worse than
the invention: across the parameters that have gradients at all, `√v̂` spans **1,791×**
(8.79e-07 to 1.57e-03), so the same decay lands 1,791× more heavily on some weights than others
for a reason unconnected to how large they are. AdamW's decoupled shrinkage is 3.0e-5 of the
weight per step, identical for every weight; the L2 route ranges up to 34 per step, which is not
a shrinkage but a divergence. And **37.7% of parameters** — rare tokens' embedding rows — have
`v̂` exactly 0, where the divisor is ε and it is not a small effect but a different equation.

*One honest limit:* these second moments come from an AdamW run, so this is the instantaneous
misallocation — how unequally the same decay would land, which is exactly §7's claim. It is not a
simulation of an L2 run, where the `λw` term would join the gradient, enter `v̂` itself and damp
the largest of these numbers.

### `ηλ` is one setting — exactly where, and exactly not where

With no gradient at all, AdamW's update is pure decay, `w ← w(1 − ηλ)`, so the steps to fall to
`1/e` must be `1/(ηλ)` for every pair with the same product. Measured over four (η, λ) pairs,
the worst error is **0.002%**, and at η = 0.0003, λ = 0.1 the timescale is **33,333 steps** —
§7's number exactly.

What this does *not* license is treating η and λ as interchangeable in general: they are
interchangeable for the decay timescale and for nothing else, because η also scales the update
that the decay is competing with.

### What the optimizer costs

Counted, not quoted: **2.00** optimizer state tensors per weight over the real model's 38,111,488
parameters.

| optimizer | bytes/weight | a 9B model | fits an 80GB card? |
|---|---|---|---|
| gradient descent | 8 | 67.1 GiB | yes |
| with momentum | 12 | 100.6 GiB | no |
| AdamW | 16 | 134.1 GiB | no |
| 8-bit AdamW | 10 | 83.8 GiB | no |

---

## 5 · Limits

Worth being explicit about what this submission does and does not support.

* **Scale.** Every experiment is a 4-layer, 38M-parameter model (Tasks 1–4) or a 3-layer capped-
  vocabulary model (Tasks 5–6), at 120–500 steps. Nothing here establishes behaviour at 1B+.
* **One seed.** No experiment is repeated across seeds, so differences below roughly 0.05 nats on
  the held-out batches should be read as ties. The Task 4 schedule gap (0.0923) is close to that
  line; the Task 6 optimizer gap (0.1007) is above it but not by much.
* **Short runs bias learning rates high**, which affects every number in Task 5 in the same
  direction.
* **The capped vocabulary** (Tasks 5–6) changes the head's size, and the head is where a large
  share of a language model's gradient signal lives. The width effect survives it; a claim about
  absolute loss would not.
* **Task 2's real-model comparison** is three runs at one learning rate each. The LR-matched
  baseline is matched at the *widest-gap* step (12), which is a coarse match, not an optimum.

---

## 6 · Files

| file | what it is |
|---|---|
| `s11_optimizers.py` | the harness — source of truth, `# %%` cell-delimited |
| `s11_optimizers.ipynb` | build artifact, committed **with its outputs** |
| `build_notebook.py` | `.py` → executed `.ipynb` → bakes `S11DATA` into the page |
| `setting-the-distance.html` | the page; every panel is built by JS from the baked blob |
| `check_page.py` | renders the page in headless Chrome and asserts it actually built |
| `validate_palette.py` | the dataviz palette checks, computed rather than eyeballed |
| `out/evidence.json` | every number on the page, written by the harness |

---

## 7 · Gates

24 gates, each one a claim that could have come out the other way. `build_notebook.py` exits
non-zero if any fails.

| gate | what it holds to |
|---|---|
| `tokenizer_hash_verified` | the frozen Sarvam-1 sha256, as in S6/S9/S10 |
| `hand_adam_matches_torch_to_1e12` | hand Adam vs `torch.optim.AdamW`, float64 |
| `notes_adam_table_reproduces` | §6's printed table, to its own precision |
| `bias_ratio_is_gradient_free` | measured step ratio == closed form on random gradients |
| `twenty_steps_is_not_enough` | at step 20 the two step sizes still differ by >5% |
| `gap_peaks_after_step_one` | the widest gap is not at t = 1 |
| `no_bias_correction_inflates_the_ratio` | uncorrected peak ratio > 3× corrected |
| `lr_matched_baseline_erases_the_gain` | a tuned baseline recovers the "improvement" |
| `step1_ratio_matches_prediction` | all 32 tensors within 2% of `η·√fan_in` |
| `warmup_caps_the_ratio` | no-warmup peak > 3× warmup peak |
| `ratio_stays_in_the_1e3_band` | post-warmup median ratio inside 1e-4 … 1e-2 |
| `both_schedules_start_identically` | same seed, same batches: step-1 losses agree to 1e-12 |
| `wsd_branch_beats_its_checkpoint` | decaying from the checkpoint improves on it |
| `truncated_cosine_got_more_learning_rate` | the mechanism behind the failed §10 prediction |
| `every_minimum_interior_to_the_grid` | all six sweep minima bracketed |
| `sp_minimum_moves_with_width` | monotone decreasing under SP |
| `sp_exponent_near_minus_one` | fitted exponent in [−1.6, −0.4] |
| `mup_minima_move_less_than_sp` | muP transfers better than SP does |
| `both_optimizers_tuned_on_the_same_grid` | AdamW and Muon minima both interior |
| `muon_hybrid_trains` | the Newton–Schulz hybrid runs without diverging |
| `notes_decay_table_reproduces` | §7's L2-vs-decoupled worked example |
| `real_vhat_spread_exceeds_100x` | the measured spread beats the notes' invented 100× |
| `eta_lambda_sets_the_timescale` | measured 1/e decay == `1/(ηλ)` for every pair |
| `adamw_is_sixteen_bytes_two_states` | counted, not assumed |
