# The Field Changes Its Mind

**ERA V5 · Session 8 submission.** Thirty attention mechanisms in the order they actually
launched, each presented as an answer to a bill the previous one sent.

**Live page:** [`attention-timeline.html`](attention-timeline.html)

The assignment cares about three things, in its own words: *"be right about the dates, right
about the trade-offs, and clear about the story."* This README is mostly about the first, because
that is the one that is easiest to get wrong and easiest to check.

## Run the verification

```bash
python verify_sources.py
```

Exits 0 only if every date on the page survives contact with its primary source. It re-fetches
each entry from the arXiv API and compares our claimed date against the `<published>` field —
the **v1 submission timestamp** — then checks that the chronology is ordered and that the cards
on the page agree with the evidence file. Writes `verification.log`. Needs network; the page
itself is fully static and needs nothing.

Current state: **26/26 unique arXiv records verified, 31 evidence entries in order, 3 claims
declared exempt with reasons.**

## Where the dates come from

| Source type | Count | How it is dated | Machine-checked |
|---|---|---|---|
| arXiv paper | 26 unique IDs, 27 cards | `<published>` from the arXiv Atom API = v1 submission | **yes** |
| Community post | 1 | No timestamp reachable — month precision only | no, declared |
| Model release | 2 | Official release announcement | no, declared |

[`sources.json`](sources.json) is the authority. [`mechanisms.js`](mechanisms.js) carries the
prose. They are separate files so they can drift, which is why `verify_sources.py` cross-checks
them and fails if a card ever shows a date the evidence file does not hold.

**The rule this project runs on: no date is written from memory.** The assistant that built this
has a January 2026 knowledge cutoff and the work was done in August 2026 — a seven-month blind
spot covering the most recent third of the timeline. Everything dated 2026 had to be looked up,
and one of the lookups reversed a conclusion (see below).

## Four places this was easy to get wrong

**1. YaRN is August, not September.** Its arXiv ID is `2309.00071`, so it is very widely cited as
September 2023. The v1 submission is **2023-08-31**. The ID prefix is the announcement month, not
the submission date. Off by one month, and completely invisible once written into prose.

**2. There are two papers one capital letter apart.**

| | What it is | |
|---|---|---|
| **DRoPE** | Directional Rotary Position Embedding — autonomous-driving trajectory modelling. Unrelated. | [arXiv:2503.15029](https://arxiv.org/abs/2503.15029) |
| **DroPE** | Sakana AI. Use RoPE as a training scaffold, then drop positional embeddings entirely and recalibrate briefly. | [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) |

The first is the top search result for the query "DroPE". It is not the one §9 of the class notes
is describing.

**3. NTK-aware scaling has no paper.** It is a
[community post on r/LocalLLaMA by u/bloc97](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
— no arXiv ID, no DOI, no timestamp we could reach from a primary source. It is dated here to the
**month only**, and marked exempt from the automated check rather than given a fabricated
citation. Its position *after* Position Interpolation is not asserted from a calendar: YaRN's own
text describes NTK-aware interpolation as addressing PI's loss of high-frequency information,
which establishes the causal order independently. That is a stronger claim than a date would have
been.

**4. Learned absolute positions predate the Transformer.** They appear in *Convolutional Sequence
to Sequence Learning* on **2017-05-08**, five weeks before *Attention Is All You Need*, and
position-dependent encoding goes back to *End-To-End Memory Networks* (2015). Only **sinusoidal**
originates in the Transformer paper. Collapsing both into one 2017 bucket is the default mistake.

## Two notes on the class notes

Offered in the spirit of *"if you catch me in another one, tell me"* — both are dating artifacts,
not errors of understanding.

**DroPE's mechanism is now public.** §9 says *"the available record does not establish the exact
DroPE algorithm or which rotary dimensions it changes."* Both halves have been superseded: the
method is published as [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) with
[reference code](https://github.com/SakanaAI/DroPE), and it does not change rotary dimensions at
all — it **removes positional embeddings outright** after pretraining and briefly recalibrates.
The "which rotary dimensions" framing presupposes a mechanism DroPE does not use. The notes
appear to predate publication.

**DeepSeek-V4 checked out, and the suspicion was mine.** While verifying, §12's reference to
"DeepSeek-V4's Compressed Sparse Attention" was initially flagged as a probable error, on the
assumption that no such model existed. It does:
[arXiv:2606.19348](https://arxiv.org/abs/2606.19348), v1 2026-04-26, and CSA and HCA are exactly
as described. **The notes were right and the challenge was wrong.** It is recorded here because
the near-miss is the whole lesson of the assignment: a confident correction is just as
checkable — and just as embarrassing — as a confident error.

## What was added beyond the required list

The brief asks for seventeen mechanisms and offers credit for relevant ones it missed. Five were
added because the causal chain breaks without them, not for volume:

| Added | Why the story needs it |
|---|---|
| **Position Interpolation** (2023-06-27) | NTK-aware exists specifically to fix PI's high-frequency loss. The RoPE → NTK → YaRN chain is incoherent without it. |
| **FlashAttention** (2022-05-27) | The counter-example. The bill got 2–4× cheaper with **zero** change to the mathematics — proof that not every saving costs quality, and the reason the 2020 approximation wave mostly did not survive. |
| **Transformer-XL** (2019-01-09) | The first mainstream answer to "what crosses a chunk boundary", and the direct ancestor of §14's Memory Stream. |
| **Mamba / SSD** (2023-12-01, 2024-05-31) | The fixed-state layers in a hybrid schedule descend from this line as much as from linear attention; SSD proved the two families are one structure. |
| **PagedAttention** (2023-09-12) | §10 frames KV cache as a per-user serving cost. This is the systems-layer answer to that exact framing — a reminder that not every answer is architectural. |
| **Qwen3.6** (2026-04-27) | Independent corroboration for §13: a different lab shipping the same 3:1 fixed-state-to-attention ratio as `DDDGDDDG`. |

Also split apart rather than merged: **sliding window** is three distinct moments (Sparse
Transformer 2019, Longformer 2020, Mistral 2023), and **the delta rule** is three (fast-weight
formulation 2021, parallelisation 2024, gating 2024). Treating either as one dated event would
hide the actual chronology.

## How each mechanism is drawn

The brief asks for every mechanism shown **visually**. Thirty bespoke animations would be thirty
chances to draw something decorative that does not match the mathematics, so instead every
mechanism declares one of six primitives — and each primitive **renders from the actual rule**,
not from a picture of it. The pattern grids call `readMask()`, the position charts call
`posCurve()`, the state panels run the real write rule. A wrong rule produces a wrong picture
rather than a pretty one.

| Primitive | Count | What it draws |
|---|---|---|
| **pattern** | 10 | A 32×32 grid: row *i* is a query, column *j* a key it could read. Black is the future. You see the access shape directly — a window, a stride, sinks, block compression, three-branch NSA. |
| **pos** | 9 | Positional signal against distance, with the training boundary marked and everything past it shaded. The learned table visibly *stops*; ALiBi rides down cleanly; PI squeezes; NTK stretches unevenly. |
| **heads** | 3 | Query heads wired to what is actually cached. MQA collapses 8→1, GQA 8→2, MLA 8→one latent. |
| **state** | 5 | One fixed state, written twice. Add-only reads back 95 instead of 55; the delta rule reads 55. This is §5→§6 of the notes, executed. |
| **sys** | 2 | A systems change with no effect on the mathematics — FlashAttention's tiling, PagedAttention's block allocation. |
| **sched** | 1 | Fixed-state vs attention layers through depth, at the `DDDG` ratio. |

Two of these needed correcting after the first pass, both caught by running the renderers
headless and inspecting the numbers rather than by looking at the page:

- **DeepSeek-V4's CSA showed 72% of pairs touched** — visually barely sparse, because a 26-token
  grid has too few blocks for top-k selection to bite. Widened to 32 tokens and tightened the
  selection; it now reads 49%, with 93 exact reads against 166 through compressed entries.
- **SSD's decay band rendered at flat opacity**, which hid the decay — and the decay *is* the
  mechanism. `readMask` now returns the real weight for that case and the renderer shades by it,
  so the equivalence between a recurrent state and a decaying triangular matrix is visible.

## How the trade-off numbers work

Every cost figure on the page is computed live from the KV-cache formula in §10 of the notes:

```
cache bytes = 2 × layers × kv_heads × head_dim × T × batch × bytes_per_number
```

against the notes' own yardstick — 48 layers, head dim 128, bf16. At 8 KV heads and 32,768
tokens this reproduces the notes' published figures exactly: **6.44 GB for one user, 51.54 GB for
eight**. Nothing is copied; if the formula were wrong those two numbers would not land.

Each mechanism declares a cache model (`mha`, `heads`, `window`, `compress`, `latent`, `state`)
which is evaluated at whatever context length the reader selects. That is what makes the verdicts
move when you change the workload, and it is why the page can say a mechanism is right for a 2K
chatbot and wrong for a 1M agent instead of just asserting it.

**What the numbers do not claim.** This compares mechanisms against each other on one fixed
reference shape. It does not predict any real deployment's memory, which also pays for weights,
activations, attention workspaces and allocator headroom. And nothing here measures *quality*:
every quality statement on a card is attributed to that paper's own reported results, not
re-measured here.

## Files

| File | What |
|---|---|
| `attention-timeline.html` | The page. Static, no build step, no network. |
| `mechanisms.js` | Thirty mechanisms: narrative, trade-offs, cost models. |
| `sources.json` | Date evidence. The authority for every date on the page. |
| `verify_sources.py` | Re-fetches arXiv, checks dates, ordering, and page/evidence agreement. |
| `verification.log` | Generated. The last run's full output. |
| `s8-class-notes/` | The session material this responds to. |
