# Four Ways to Lie About a Loss

**ERA V5 · Session 9 submission.** A loss harness made correct and observable, and one extra
head predicting `t+2`.

**Live page:** [`loss-harness.html`](loss-harness.html) · **Notebook:**
[`s9_loss_harness.ipynb`](s9_loss_harness.ipynb) (committed with its outputs)

```bash
python build_notebook.py        # convert .py -> .ipynb, execute it, bake the page  (~9 min, CPU)
python s9_loss_harness.py       # or just run the harness directly
python check_page.py            # does the page actually build in a browser?
```

`build_notebook.py` exits non-zero if any gate fails. That exit code is the pass/fail signal,
not the printed summary. **Current state: 16/16 gates pass.**

---

## 1 · The problem

Four lines stand between the model's output and the scalar the optimiser pushes down:

```python
hidden = model(tokens)
logits = output_head(hidden)
loss   = cross_entropy(logits[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1))
```

Every bug that lives in them is silent. None raise. **Three of the four make the loss look
better.** The whole submission is one idea: turn each of those failures into a number you can
read, and put the contributing-token count next to it so you can tell them apart.

---

## 2 · Method, step by step

Everything runs on the **real frozen Sarvam-1 tokenizer** (68,096 tokens, sha256 `bb5115a3…` —
the same hash `s6-dataset-creation/tds/shards.py:32` pins) over **S6's committed corpus**. The
model is a small pre-norm decoder (4 layers, `D=256`, SwiGLU, RMSNorm) with the **real
68,096-wide output head**, which is 46% of its parameters.

| # | Step | What it does |
|---|---|---|
| 0 | Verify and load | Hash-check the tokenizer against S6's frozen constant; load real Hindi/Telugu/code/reasoning documents. A mismatch is a hard gate — different ids would make every number below refer to different tokens. |
| 1 | Shape ledger | Print every tensor and what each dimension is, from `tokens` to the scalar. |
| 2 | Shift audit | Decode the **strings**, inputs beside targets, under three alignments. Then *train* a model under each, because an untrained model scores all three the same. |
| 3 | Padding | Compare two **training runs** — one masked, one not — on a batch built with deliberately varied row lengths so padding exists. |
| 4 | Document boundary | Pack two documents per row with an `<eos>` join, 32 rows from different lanes, and score the join against in-document targets. |
| 5 | The `ln(V)` anchor | Measure a fresh model's loss and check it against `ln(V)`, *and* check the miss is explained by the logit spread. |
| 6 | Tied vs untied | Count parameters on the real module, both ways, at this width and V5's. |
| 7 | Chunked cross-entropy | Write a real `autograd.Function` that recomputes logits in backward; verify it against vanilla, *then* measure peak memory in separate subprocesses. |
| 8 | Head stability | Show the gradient sums to zero, then run five variants from one identical init tracking `log Z`. |
| 9 | Vocabulary parallelism | Simulate 2/4/8/16 shards with two collectives; check the loss is unchanged. |
| 10 | Bits per byte/char | Score three languages with one model in three units. |
| 11 | SFT masking | Mask the prompt on real `context`/`target` pairs from S6's corpus. |
| P2 | Two heads | Train a shared trunk with heads at `t+1` and `t+2`; evaluate on a fixed held-out batch; measure draft acceptance. |

Two ordering constraints made the results mean anything, and both were learned the hard way
(see §6): **steps 3, 4 and the switch table reuse the trained model from step 2**, because on an
untrained model every token costs `ln(V)` and nothing is distinguishable. Step 5 is the
exception — it *must* use a fresh model, because being untrained is its entire point.

---

## 3 · Findings — the assignment

### The seven numbers

| # | Bullet | The number |
|---|---|---|
| 1 | Every tensor shape | logits are **266×** the hidden state here (`V/D` = 68,096/256); **32×** at V5's width, where one logits tensor at 256K context is **64 GiB** |
| 2 | Verify the shift with strings | training under **no shift** reaches **3.6628** against the correct harness's **6.3490** — **2.6863 nats lower, and wrong** |
| 3 | Mask padding | contributing tokens **4,064 → 2,744** (32.5% was padding); a run trained with padding counted reports **6.3347** but is really at **6.4249** — **0.0902 nats too kind** |
| 4 | Pack two documents, mask the join | boundary predictions cost **6.6427** against **6.4607** in-document — **1.028×**; masking 32 of 2,744 targets moves the mean by **−0.0021** |
| 5 | Perplexity of an untrained model | **74,762** against a vocabulary of **68,096** — 109.8%, loss 11.2221 vs `ln(V)` = 11.1287 |
| 6 | Tied vs untied head | tying saves **17,432,576** parameters = exactly `V×D`; **536.9M** at V5's width, where **tying is unavailable** |
| 7 | Chunked vs ordinary cross-entropy | **3,183.3 MB → 485.3 MB** over baseline, a **6.56×** reduction, loss identical to 1.9e-06 |

### The two losses

Held-out batch, mean of the last 5 evaluations, 500 steps:

| head | predicts | loss | perplexity |
|---|---|---|---|
| head 1 | `t+1` | **5.2887** | 198.1 |
| head 2 | `t+2` | **5.4595** | 235.0 |
| **sum** | the quantity actually optimised | **10.7482** | — |

**What happens to head 2's loss over training.** Both heads fall together and head 2 stays
worse, but the reportable finding is that the gap **widens** rather than closing: `+0.0099` over
the first five evaluations, `+0.1709` over the last five — a 17× increase.

At the start the two heads are equally bad because the model has learned nothing; it is
predicting roughly the unigram distribution, and that distribution is the same one step out as
two. `t+1` and `t+2` only become *different questions* once there is context to condition on. So
the gap is not present at initialisation and then eroded — it is **created by learning**, and it
grows as the model gets better at what head 1 is asked to do. That is also why the early
evaluations are not a clean sweep: head 2 comes out ahead in 4 of 51, all in the first half,
while the difference is smaller than the measurement noise. Across the second half it is 26/26.

The mechanism: head 2 never gets to condition on how `t+1` actually resolved. Head 1 predicts one
step into a distribution; head 2 predicts two and must marginalise over the token in between. Its
irreducible entropy is genuinely higher, so its loss has a **higher floor**. The gap is
information, not an optimisation failure — a perfectly trained pair would still show it.

### The four switches

One batch, one trained model per row, contributing-token count beside every loss.

| switch | kind | loss | contributing | vs correct |
|---|---|---|---|---|
| correct harness (baseline) | — | 6.3490 | 2,712 | — |
| count padding into the loss | train | 6.2069 | 4,064 | **−0.1421** |
| off-by-one: no shift at all | train | 3.6628 | 2,712 | **−2.6863** |
| predict across the document join | reduce | 6.4628 | 2,744 | +0.1138 |
| divide by B·T, not by what counted | reduce | 4.3114 | 4,064 | **−2.0376** |

**Three of the four make the loss look better.** The only thing on the panel that says *which*
lie you told is the contributing-token count.

---

## 4 · Findings — beyond the seven

The session says several other things are measurable, and they are cheap once the harness
exists. §23 flags the first as *genuinely open* for V5 and "worth doing before the real run
rather than after a NaN".

### 8 — the gradient sums to zero, and what that lets drift

`∂L/∂z = softmax(z) − onehot(y)`. A softmax sums to 1 and a one-hot sums to 1, so the gradient
sums to **exactly zero** — it can never move the logits up or down *as a group*, and that degree
of freedom is unconstrained by the loss. So `log Z` walks. Five runs, one identical init:

| run | final `log Z` | final mean logit | what it controls |
|---|---|---|---|
| plain | 8.19 | −6.474 | nothing — the control |
| z-loss λ=1e-4 | 7.84 | −6.946 | `log Z`, weakly at this λ |
| z-loss λ=1e-2 | **6.73** | −7.830 | `log Z`, directly |
| soft-cap c=30 | 8.12 | −5.863 | the logits, so `log Z` with them |
| centering | **14.09** | **3.1e-09** | the uniform component only |

The last row is the point, and it reproduces the correction the notes make about *themselves*.
Centering does exactly what it promises — the mean logit goes to `3.1e-09` — and leaves `log Z`
at **14.09, higher than plain's 8.19**. Once the mean is pinned, `log Z` is governed by how far
the logits *spread*, about which centering says nothing. **The three fixes are not
interchangeable, and only one is aimed at `log Z` at all.**

### 9 — vocabulary parallelism gives the same number

The fourth implementation in §10 needs N devices, but the *arithmetic* does not: each shard
computes its own max and `Σexp`, and two collectives combine them into the global `log Z`.
Simulated at 2, 4, 8 and 16 shards, worst disagreement with the reference is **9.5e-07** —
floating-point reassociation, not a different objective.

### 10 — perplexity is a bad cross-tokenizer scoreboard, and bits-per-byte is not the fix

One model, three languages, same weights:

| language | perplexity | bits/byte | bits/char | bytes/char |
|---|---|---|---|---|
| eng_Latn | 447 | 2.527 | 2.527 | 1.00 |
| code | 535 | 3.382 | 3.382 | 1.00 |
| hin_Deva | **27,873** | **1.782** | 4.340 | **2.44** |

Hindi has by far the **worst** perplexity and the **best** bits-per-byte. The reason is not the
model: Devanagari costs 2.44 UTF-8 bytes per character where Latin costs 1.00, so dividing by
bytes hands Indic scripts a discount that has nothing to do with how well anything was
predicted. Bits per *character* removes the encoding too, and it agrees with perplexity again.

### 11 — SFT is the same loss, masked

The loss does not change at all; only which tokens contribute. S6's corpus already tags segments
`context` and `target`, so the prompt/completion boundary is real rather than staged. Masking the
prompt drops the contributing count from **1,980 to 834** (57.9% was prompt) and moves the loss
by **−0.3994**. Same cross-entropy, same shift, one mask — the same lever as padding and document
boundaries, pointed somewhere else.

### Part 2 — acceptance rate, the number that actually decides

Head 2 at position `t` proposes token `t+2`; the main path verifies (head 1 at `t+1` is what the
model would have produced had it done the work); accept when they agree.

| measurement | rate |
|---|---|
| **acceptance rate** | **47.3%** — 953 of 2,016 drafts |
| tokens per forward pass, if every accepted draft is kept | **1.47×** |
| head 1 top-1 accuracy on `t+1`, for reference | 24.2% |
| head 2 draft matches the *true* token | 21.1% |

This verifies against the **true** prefix, because the sequence is teacher-forced. Real
speculative decoding drafts on top of its own accepted tokens, where errors compound, so a
deployed rate would be no better than 47.3% and probably worse. **This is the optimistic bound.**

---

## 5 · Conclusions

**On the harness itself.** The loss is a poor supervisor of its own correctness: three of the
four failure modes move it *downward*. Nothing in a loss curve distinguishes "the model is
learning" from "the harness is lying", which is why the contributing-token count belongs beside
the loss in every training log — it is the one quantity that changes distinctly for each bug.
The `ln(V)` check costs one forward pass and would catch the worst of them before a run starts.

**On the shape of the experiments.** Two of the four §6 bugs are *training* choices and two are
*reduction* choices, and conflating them produces confident nonsense. Scoring a correctly-trained
model with padding counted measures the opposite of what the warning is about. Any claim of the
form "this mistake makes the loss look better" has to be a comparison between two **runs**.

**For V5 specifically:**

| question | what this run says |
|---|---|
| Output head | The head is **536.9M** and **tying is closed** — S7's byte codec has no rows to tie to. Keeping it dense drops S7's 93.75% front-door saving to **46.9% across both ends**. A factored head is the real open question, and it is now the largest single unresolved parameter decision. |
| Logits tensor | 64 GiB at 256K context is an implementation problem, not a tuning one. Chunking is **exact** (1.9e-06) and sharding is **exact** (9.5e-07). Specify the loss *and* how it is computed; they are independent decisions. |
| MTP | The trade is now explicit rather than rhetorical: **47.3% acceptance → 1.47× tokens/pass, for +536.9M parameters per head** at V5's width. Four dense heads is 2.1B of output head alone. Argue about acceptance rate, not head count. |
| Head stability | If the concern is `log Z` drift, **centering does not address it** — it raised `log Z` here. z-loss attacks the normaliser directly and its coefficient is the whole control. Pick on the basis of which failure you are actually preventing. |
| Evaluation | Report **bits per character**, not bits per byte, across scripts. For an India-first model a bits-per-byte scoreboard is misleading in the *flattering* direction, which is the direction least likely to be checked. |

**The methodological conclusion**, which is the one I would carry into the next session: three of
my own experiments produced confident numbers that meant nothing, and in every case the failure
was measuring the right quantity on the wrong object — an untrained model, a batch with no
padding, a single run where a comparison was needed. The gates caught them because the gates
assert *relationships* between numbers, not the numbers themselves. Asserting `boundary_cost >
in_document_cost` catches a vacuous experiment; asserting `boundary_cost == 6.64` would not.

---

## 6 · Three things the first run got wrong

Recorded because in each case the harness produced a confident number that meant nothing.

**The padding experiment was backwards.** Masking padding at *scoring* time, on a model trained
with `ignore_index=PAD`, makes the loss look 2.94 nats **worse** — the opposite of what §6 warns
about. The model had never learned padding, so scoring it there scored it on a task it was never
taught. The warning is about a run *trained* the wrong way that then reported the number it saw.
Fixing it meant training a fourth model. The switch table now labels every switch `train` or
`reduce` for exactly this reason.

**The boundary experiment was vacuous.** On an untrained model a cross-document join is no more
surprising than anything else — every token costs `ln(V)`. It only measures something once the
model knows what an in-document transition looks like, so steps 3, 4 and the switch table all
reuse the trained model from step 2.

**The `ln(V)` gate was mis-specified.** The untrained loss missed `ln(V)` by 0.0934 nats and the
first tolerance called that a failure. It is not slack — for logits `z ~ N(0, σ²)` the expected
loss is `ln(V) + σ²/2`, because the log-sum-exp picks up `ln E[e^z] = σ²/2` while the true-token
term averages to zero. Measured σ is 0.3197, predicting an excess of 0.0511. The gate now
requires **both** that the loss is within 0.25 nats of `ln(V)` *and* that the miss is explained by
the logit spread — a stronger check than the one it replaced.

A fourth, caught before it shipped: `inspect.getsource` **cannot** recover a class defined in a
notebook cell (`TypeError: … is a built-in class`). The memory probe relied on it to hand the
chunked implementation to a subprocess, so it worked locally and would have broken on Colab. The
class is now written to `out/chunked_ce.py` at runtime and imported by both.

---

## 7 · On the memory measurement

Experiment 7's numbers are a **measurement**, and it is worth being precise about what was
measured. There is no CPU equivalent of `torch.cuda.max_memory_allocated`, so each variant runs
in its own subprocess and reports peak RSS; the harness takes the CUDA allocator path
automatically when a GPU is present and records which one it used in `evidence.json`.

Peak RSS is real but coarser than the allocator: it also counts the interpreter, torch itself,
and the `[V, D]` weight and its gradient — fixed costs chunking does not touch. That is why the
measured **6.56×** is smaller than the analytic **8×** on the logits tensor alone. Both are
reported; neither is presented as the other.

Because it is a measurement rather than a deterministic computation, this is the one number here
that moves between runs: 6.05×, 6.38×, 6.52×, 6.55× and 6.56× across five runs of identical code.
If it disagrees with this README by a few hundredths, that is the instrument, not a change.
**Every loss number is deterministic under the fixed seed and does not move at all.**

The chunked implementation is verified before it is measured: `|Δloss| = 9.5e-07`,
`max|Δgrad_h| = 2.2e-11`, `max|Δgrad_W| = 6.5e-09`. A memory win that changes the number is not a
win.

---

## 8 · Configuration, files, reproducing

The V5 target configuration from the class notes (`V=131,072`, `D=4,096`) appears wherever the
arithmetic is the point rather than the run: the parameter counts and the logits-tensor bill.
S6's `eval_registry_docs` lane is excluded from all data loading, because it sits behind that
session's evaluation firewall.

| Path | What | In git? |
|---|---|---|
| `s9_loss_harness.py` | the source of truth, `# %%` cell-delimited | yes |
| `s9_loss_harness.ipynb` | the Colab deliverable, committed **with outputs** | yes |
| `build_notebook.py` | convert → execute → bake; the gate runner | yes |
| `check_page.py` | renders the page in headless Chrome and asserts it built | yes |
| `validate_palette.py` | Python twin of the dataviz palette validator (no node on this machine) | yes |
| `loss-harness.html` | the page, with `evidence.json` baked in as a JS blob | yes |
| `out/evidence.json` | every number on the page and in this README | yes |
| `out/chunked_ce.py` | generated at runtime; the class the memory probe imports | yes |

Following the repo's generate-then-bake pattern (S4, S6, S7): the `.py` is the source of truth,
and **the notebook and the page must both be rebuilt when it changes** — editing `evidence.json`
alone does nothing.

The notebook runs top to bottom on Colab (first cell installs `tokenizers` and
`huggingface_hub`; the second clones this repo for S6's corpus). Locally it needs torch, numpy,
tokenizers and huggingface_hub; the tokenizer downloads once from the HF hub and is cached, so
only the first run needs network. Run time is about **9 minutes on CPU** — training at the full
68,096-wide softmax dominates it: four alignment variants at 200 steps, five stability variants
at 150, and the 500-step two-head run.

### What the page covers that the assignment did not ask for

[`loss-harness.html`](loss-harness.html) is meant to stand on its own without the class notes
open beside it, so it also builds the spine the assignment assumes you have — what a logit is,
what softmax does, why the loss is `−log p`, how perplexity reads — using the notes' own
five-token worked example, made draggable. It closes with the loss map placing next-token
cross-entropy among the losses the rest of the course covers. **Sections 01, 07, 09, 10 and 11
are that material; sections 02–06 and 08 are the assignment.**
