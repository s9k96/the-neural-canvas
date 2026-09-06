# Four Ways to Lie About a Loss

**ERA V5 · Session 9 submission.** A loss harness made correct and observable, and one extra
head predicting `t+2`.

**Live page:** [`loss-harness.html`](loss-harness.html) · **Notebook:**
[`s9_loss_harness.ipynb`](s9_loss_harness.ipynb) (committed with its outputs)

```bash
python build_notebook.py        # convert .py -> .ipynb, execute it, bake the page  (~6 min, CPU)
python s9_loss_harness.py       # or just run the harness directly
python check_page.py            # does the page actually build in a browser?
```

`build_notebook.py` exits non-zero if any gate fails. That exit code is the pass/fail signal,
not the printed summary. **Current state: 11/11 gates pass.**

## The seven numbers

| # | Bullet | The number |
|---|---|---|
| 1 | Every tensor shape | logits are **266×** the hidden state here (`V/D` = 68,096/256); **32×** at V5's width, where one logits tensor at 256K context is **64 GiB** |
| 2 | Verify the shift with strings | training under **no shift** reaches **3.6628** against the correct harness's **6.3490** — **2.6863 nats lower, and wrong** |
| 3 | Mask padding | contributing tokens **4,064 → 2,744** (32.5% was padding); a run trained with padding counted reports **6.3347** but is really at **6.4249** — **0.0902 nats too kind** |
| 4 | Pack two documents, mask the join | boundary predictions cost **6.6427** against **6.4607** in-document — **1.028×**, and masking 32 of 2,744 targets moves the mean by **−0.0021** |
| 5 | Perplexity of an untrained model | **74,762** against a vocabulary of **68,096** — 109.8%, loss 11.2221 vs `ln(V)` = 11.1287 |
| 6 | Tied vs untied head | tying saves **17,432,576** parameters = exactly `V×D`; **536.9M** at V5's width, where **tying is unavailable** |
| 7 | Chunked vs ordinary cross-entropy | **3,182.3 MB → 485.6 MB** over baseline, a **6.55×** reduction, with the loss identical to 1.9e-06 |

## The two losses (Part 2)

Held-out batch, mean of the last 5 evaluations, 500 steps:

| head | predicts | loss | perplexity |
|---|---|---|---|
| head 1 | `t+1` | **5.2887** | 198.1 |
| head 2 | `t+2` | **5.4595** | 235.0 |
| **sum** | the quantity actually optimised | **10.7482** | — |

**What happens to head 2's loss over training.** Both heads fall together and head 2 stays
worse, but the reportable finding is that the gap **widens** rather than closing: `+0.0099`
over the first five evaluations, `+0.1709` over the last five, a 17× increase.

At the start the two heads are equally bad because the model has learned nothing — it is
predicting roughly the unigram distribution, and that distribution is the same one step out as
two. `t+1` and `t+2` only become *different questions* once there is context to condition on.
So the gap is not present at initialisation and then eroded; it is **created by learning**, and
it grows as the model gets better at what head 1 is asked to do. That is also why the early
evaluations are not a clean sweep: head 2 comes out ahead in 4 of 51 evaluations, all in the
first half, while the difference is still smaller than the measurement noise. Across the second
half it is 26 out of 26.

The mechanism is that head 2 never gets to condition on how `t+1` actually resolved. Head 1
predicts one step into a distribution; head 2 predicts two and must marginalise over the token
in between. Its irreducible entropy is genuinely higher, so its loss has a higher floor — the
gap is information, not an optimisation failure, and a perfectly trained pair of heads would
still show it. Which is why §13's insistence that *acceptance rate*, not head count, decides
whether MTP pays is the operationally correct framing.

The cost is honest: each dense head is another `V×D`. At V5's width four heads is **2.1B
parameters of output head alone**.

## Configuration

Not a toy. The tokenizer is the **real frozen Sarvam-1** this course's data pipeline already
uses — 68,096 tokens, sha256 `bb5115a3…`, the same hash `s6-dataset-creation/tds/shards.py:32`
pins and `s7-model-internals` measured against. The harness verifies it and treats a mismatch
as a hard gate: different ids would mean every number here refers to different tokens.

The text is S6's committed `corpus/*.jsonl` — real Hindi, Telugu, code, reasoning and agentic
documents — so the shift audit prints actual Devanagari and the document-packing experiment
joins genuinely unrelated documents from different lanes. S6's `eval_registry_docs` lane is
excluded, because it is behind that session's evaluation firewall.

The model is a small pre-norm decoder (4 layers, `D=256`, SwiGLU, RMSNorm) — the trunk is
deliberately small because the session is about the last layer, but **the output head is the
real 68,096-wide one**, which is 46% of the model's parameters.

The V5 target configuration from the class notes (`V=131,072`, `D=4,096`) appears wherever the
arithmetic is the point rather than the run: the parameter counts and the logits-tensor bill.

## Three things the first run got wrong

Worth recording, because in each case the harness produced a confident number that meant
nothing, and only the gates caught it.

**The padding experiment was backwards.** Masking padding at *scoring* time, on a model trained
with `ignore_index=PAD`, makes the loss look 2.94 nats **worse** — the opposite of what §6
warns about. The model had never learned padding, so scoring it on padding was scoring it on a
task it was never taught. The warning is about a run that was *trained* the wrong way and then
reported the number it saw, which is a comparison between two runs, not two ways of scoring
one. Fixing it meant training a fourth model. The switch table now labels every switch `train`
or `reduce` for exactly this reason.

**The boundary experiment was vacuous.** On an *untrained* model a cross-document join is no
more surprising than anything else — every token costs `ln(V)`. The experiment only measures
something once the model has learned what an in-document transition looks like, so Experiments
3, 4 and the switch table all reuse the trained model from Experiment 2.

**The `ln(V)` gate was mis-specified.** The untrained loss missed `ln(V)` by 0.0934 nats and the
first tolerance called that a failure. It is not slack — for logits `z ~ N(0, σ²)` the expected
loss is `ln(V) + σ²/2`, because the log-sum-exp picks up `ln E[e^z] = σ²/2` while the true-token
term averages to zero. Measured σ is 0.3197, predicting an excess of 0.0511. The gate now
requires both that the loss is within 0.25 nats of `ln(V)` **and** that the miss is explained by
the logit spread, which is a stronger check than the one it replaced.

## The four switches

One batch, one trained model per row, and the contributing-token count printed beside every
loss. Three of the four make the loss look better.

| switch | kind | loss | contributing | vs correct |
|---|---|---|---|---|
| correct harness (baseline) | — | 6.3490 | 2,712 | — |
| count padding into the loss | train | 6.2069 | 4,064 | **−0.1421** |
| off-by-one: no shift at all | train | 3.6628 | 2,712 | **−2.6863** |
| predict across the document join | reduce | 6.4628 | 2,744 | +0.1138 |
| divide by B·T, not by what counted | reduce | 4.3114 | 4,064 | **−2.0376** |

The only thing on the panel that says *which* lie you just told is the contributing-token
count. That is the argument for printing it beside the loss on every run.

## On the memory measurement

Experiment 7's numbers are a **measurement**, and it is worth being precise about what was
measured. There is no CPU equivalent of `torch.cuda.max_memory_allocated`, so each variant runs
in its own subprocess and reports peak RSS; the harness takes the CUDA allocator path
automatically when a GPU is present and records which one it used in `evidence.json`.

Peak RSS is real but coarser than the allocator: it also counts the interpreter, torch itself,
and the `[V, D]` weight and its gradient — fixed costs chunking does not touch. That is why the
measured **6.55×** is smaller than the analytic **8×** on the logits tensor alone. Both are
reported; neither is presented as the other.

Because it is an RSS measurement rather than a deterministic computation, this is also the one
number here that moves between runs: 6.38×, 6.52× and 6.55× across three runs of identical
code. If it disagrees with this README by a few hundredths, that is the instrument, not a
change. Every loss number is deterministic under the fixed seed and does not move at all.

The chunked implementation is a real `torch.autograd.Function` that recomputes each chunk's
logits in the backward pass, and it is verified against ordinary cross-entropy before it is
measured: `|Δloss| = 9.5e-07`, `max|Δgrad_h| = 2.2e-11`, `max|Δgrad_W| = 6.5e-09`. A memory win
that changes the number is not a win.

It lives in `out/chunked_ce.py`, written by the notebook at runtime and imported back, because
the subprocess has to get the *same* class — `inspect.getsource` cannot recover a class defined
in a notebook cell, where it raises `TypeError: … is a built-in class`.

## Files

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
and **the notebook and the page must both be rebuilt when it changes** — editing
`evidence.json` alone does nothing.

## Reproducing

The notebook runs top to bottom on Colab (first cell installs `tokenizers` and
`huggingface_hub`; the second clones this repo for S6's corpus). Locally it needs torch,
numpy, tokenizers and huggingface_hub; the tokenizer downloads once from the HF hub and is
cached thereafter, so only the first run needs network.

Run time is about 6 minutes on CPU — six training runs at the full 68,096-wide softmax
(four alignment variants at 200 steps, plus the 500-step two-head run) dominate it.
