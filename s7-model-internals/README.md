# Dynamic Kronecker: computing the byte position instead of storing it

**ERA V5 · Session 7 assignment — Problem 3.**

> *"Today Kronecker is limiting to presenting 32 position for every work (even "apple" or "a" as
> well). That's a waste of space. What can we do? How can it be dynamic and doesn't force us to
> crop a word (currently we cannot have a word of len more than 32)."*

Measured against the released design in *Kronecker Embeddings: Byte-Level Structured Token
Representations for Parameter-Efficient Language Models* (Shravan, 2026) — `knonecker_paper.pdf`
in this folder — and the Session 7 class notes.

## Run it

```bash
.venv/Scripts/python s7-model-internals/run_demo.py    # 8 experiments, 13 gates, ~70 s, CPU
.venv/Scripts/python s7-model-internals/dynkron.py     # codec self-check
```

Exits 0 only if all thirteen gates pass. Writes `submission_artifacts/evidence.json` and `run.log`.

## The result

Measured on the real frozen tokenizer this course's data pipeline already uses — **Sarvam-1,
68,096 tokens**, sha256 `bb5115a3…`, the same hash s6 pinned in `tds/shards.py:32`. Special
tokens excluded (4,099), matching the paper's Table 3 methodology; census vocabulary 63,997.

| input path | code_dim | tokens it **cannot represent** | projection params @ D=8096 |
|---|---|---|---|
| dense table (V×D) | — | 0 | 1,061,158,912 |
| kronecker one-hot `d_p=32` **(shipped)** | 8192 | **336** | 66,322,432 |
| kronecker one-hot `d_p=64` (the paper's stated mitigation) | 16384 | 0 | 132,644,864 |
| **dynamic `m=8` (this work)** | **2048** | **0** | **16,580,608** |

**Zero collisions, no length limit, 4× smaller than the shipped codec and 8× smaller than the
widening the paper proposes to fix the crop.**

> **Naming.** The paper already uses *`gpu_dynamic`* for a memory-layout variant that computes the
> codec on the fly. That is an implementation detail of the same mathematics. "Dynamic" here means
> something different and mathematically distinct: the **position factor** becomes dynamic.

## The diagnosis

The crop and the waste look like two complaints. They are one cause.

The paper's Equation 1 builds each token's code from `c[b_p] ⊗ p_p`, where — in the paper's own
words — "`p_p ∈ R^{d_p}` is a **one-hot** vector encoding the byte position `p`". A one-hot is a
*stored* signal: a lookup table indexed by position. Every consequence follows from that:

- **The crop.** A one-hot over 32 slots is *undefined* at position 32, so the module truncates and
  two tokens sharing a 32-byte prefix become the same vector permanently. Nothing errors.
- **The waste.** One slot per position fixes `code_dim` at `256 × 32 = 8192` whether the token is
  `"a"` or a 36-byte Tamil cluster. The projection pays for 32 positions on every token.

Session 7 §11 diagnoses precisely this failure for *sequence* position — a learned table "cannot
extrapolate… because they were only ever independent rows in a lookup table" — and §12 names the
cure: **stop storing position and start computing it.** That cure was never applied one axis down,
to byte position inside the codec. This submission applies it.

## The fix

```
κ(t) = (1/√L) · vec( Σ_p  c[byte_p] ⊗ φ(p) )        L = full byte length, no truncation
```

`φ(p) ∈ R^m` is sinusoidal — a *function* of position, defined for every `p ∈ ℕ`. Nothing else
changes: same `1/√L` scale, same z-normalisation, same single shared `Linear(code_dim, d_model)`
as the only trainable object, same `[B,T] → [B,T,D]` seam. The one-hot is the only variable in
every experiment.

Two things fall out at once:

1. **No crop is possible.** There is no window to fall off. A 6,144-byte token encodes.
2. **`code_dim = 256·m` with `m ≪ d_p`.** Position is *packed* into `m` channels instead of
   spending a slot each, so 8 channels carry what 32 slots could not.

**`base` is not inherited, it is calibrated.** The Transformer's 10000 was chosen for sequence
positions running to the thousands. Byte positions inside a token run to about 36, and at
base=10000 the low-frequency channels barely turn across that range — they contribute
near-constant mass that dilutes the channels actually doing the discriminating. Measured, this
makes separation get *worse* as `m` grows, which is the opposite of what more dimensions should
buy. `DEFAULT_BASE = 10.0` restores monotonicity. See E7 — this is the single largest quality
improvement in the design, and it came from measuring a constant nobody questions.

## Three findings against the paper

**1. "Truncated tokens still receive distinct embeddings" — measured false.**
The paper states this twice (§4.2 and §8.4: *"The truncated tokens still receive unique embeddings
from their first 32 bytes; only the post-byte-32 byte structure is lost"*). On Sarvam-1, **336
tokens in 155 groups are bit-identical** at `d_p=32`. Truncation *rate* and collision *rate* are
different quantities, and only the first was reported. Coverage is the paper's metric; collisions
are the consequence.

**2. Coverage on an Indic-first tokenizer is 12× worse than the paper's worst case.**
Reproducing the paper's Table 3 metric exactly:

| | `d_p=16` | `d_p=32` | `d_p=64` |
|---|---|---|---|
| paper, mean of six tokenizers | 98.64% | 99.86% | 99.97% |
| paper, worst (Gemma-3-SP) | 95.98% | 99.82% | 99.97% |
| **Sarvam-1 (this course's tokenizer)** | **69.65%** | **97.90%** | **100.00%** |

The paper's six tokenizers are Llama-3.2, Qwen3, Gemma-3, DeepSeek, o200k and GPT-2 — none is
Indic-first. `d_p=32` truncates 2.10% of this vocabulary against ≤0.18% there. The paper's
recommendation ("for multilingual production we recommend `d_p = 32`") was calibrated on
tokenizers that do not look like the one V5 is actually built on.

**3. "Collisions are mostly absorbed by the transformer body" — true for *similar*, provably
false for *identical*.** The paper's future-work section says input-side collisions are "mostly
absorbed by the transformer body." E3 sharpens this: when two codes are merely *close*, the body
can separate them; when they are **exactly equal**, no body can, at any depth or width, because
the gradient reaching the two tokens is identical by construction. The 336 tokens above are in the
second category.

**And one defect neither the paper nor the notes mention.** Under the paper's byte-fallback
convention (`<0xNN>` encodes the single byte it names), a byte-fallback token and the literal
character token for the same byte are **byte-identical**: `'\t' == '<0x09>'`, `'\n' == '<0x0A>'`.
That is **95 groups / 190 tokens** on Sarvam-1, and *no position scheme can fix it* — including
this one. It is a convention problem, not a codec problem. It is reported separately and excluded
from every count above, because counting it as something this work fixes would be dishonest.

## How it is proved

Six experiments, ten gates, all on the real vocabulary. `run_demo.py` recomputes every number
below and fails loudly if any of them moves. **The control arm is the paper's codec, faithfully:**
UTF-8-safe truncation (back off to the previous codepoint boundary, §3.2), byte-fallback handling,
`1/√L` scaling, per-token z-normalisation. Reproducing the safe truncation *raised* the collision
count from 124 to 336, because backing off a split codepoint leaves only 10 Devanagari characters
rather than 10⅔ — the faithful baseline is worse than a naïve byte cut, not better.

### E1 — Collision census (codec-induced only)

| codec | code_dim | groups | tokens lost |
|---|---|---|---|
| one-hot `d_p=16` | 4096 | 5,278 | 16,963 |
| one-hot `d_p=24` | 6144 | 983 | 2,417 |
| **one-hot `d_p=32` (shipped)** | 8192 | **155** | **336** |
| one-hot `d_p=48` | 12288 | 0 | 0 |
| **dynamic `m=8`** | **2048** | **0** | **0** |
| dynamic `m=12` | 3072 | 0 | 0 |
| dynamic `m=16` | 4096 | 0 | 0 |

**Every one of the 336 is Indic.** Devanagari 69, Bengali 57, Tamil 54, Kannada 45, Malayalam 41,
Telugu 39, Gujarati 20, Gurmukhi 4, Oriya 4, other 3. Zero Latin at any setting — the longest
Latin token here is 23 bytes. This is §8's sovereign risk as a number rather than an argument.

Real casualties, all ordinary morphology:

```
বিদ্যালয়ে == বিদ্যালয়ের              (Bengali, locative vs genitive)
ப்பட்டுள்ள == ப்பட்டுள்ளது == ப்பட்டுள்ளன   (Tamil, three verb endings collapsed to one vector)
ിക്കുന്നതിന == ിക്കുന്നതിന് == ിക്കുന്നതിൽ  (Malayalam)
```

### E2 — Separation margin

Exact-distinctness is a weak claim; two codes differing by 1e-9 are not separable either. Measured
over the 336 adversarial tokens plus a 4,000-token random sample:

| codec | max cosine within the 155 groups | max cosine over the pool |
|---|---|---|
| one-hot `d_p=32` | **0.999997** (identical to float precision) | 1.000000 |
| dynamic `m=8` | 0.995985 | 0.995986 |

0.996 is high — these are near-identical strings and it *should* be — but it is a real distance,
and E3 shows it suffices. Both dynamic columns are the same number: the closest pair anywhere in
the pool is one of the adversarial pairs. Nothing else is closer.

### E3 — Learnability (the actual proof)

A margin in code space is not yet a capability. So: a real transformer, real gradients, measured.

Built the way §10's experiment was rigged. Each of the 336 tokens carries a deterministic label,
and **inside every collision group the labels disagree**. A model whose input path cannot
distinguish two tokens is capped at the majority share of the group — regardless of training time
or the size of the layers above. Everything above the embedding is identical across arms; only the
input path changes.

| arm | accuracy | input-path params (demo scale) |
|---|---|---|
| one-hot `d_p=32` | **52.1%** | 524,288 |
| **dynamic `m=8`** | **100.0%** | **131,072** |
| dense table (control) | 100.0% | 21,504 |

The theoretical ceiling for a codec that collides these tokens is **52.1%**. The shipped codec
scores **52.1%** — not underperforming, but exactly at its ceiling, and no amount of training moves
it. The dynamic codec reaches 100% **with 4× fewer input-path parameters**, matching the dense
table that keeps a private row per token.

A capacity test, so train and eval are the same set by design: the question is whether the
representation *can* hold the distinction, not whether it generalises.

### E4 — Unbounded length

Tokens of 96 / 192 / 384 / 1,536 / 6,144 bytes, each compared against itself-plus-one-byte. The
one-hot codec collides at every size (it stopped reading at byte 32); the dynamic codec at none.
`code_dim` is constant in both — length costs nothing extra.

### E5 — The parameter bill

At V5's reference shape (D=8096), with the 16-bytes-per-parameter AdamW accounting from §3:

| input path | params | training memory | vs dense |
|---|---|---|---|
| dense table V×D | 1,061,158,912 | 16.98 GB | — |
| one-hot `d_p=32` | 66,322,432 | 1.06 GB | 93.75% |
| one-hot `d_p=64` | 132,644,864 | 2.12 GB | 87.50% |
| **dynamic `m=8`** | **16,580,608** | **0.27 GB** | **98.44%** |

This is what makes the submission more than a bug report. The paper's limitations section says
*"Increasing `d_p` at the cost of `D` is the obvious mitigation"* — **doubling** the projection to
133M. Computing the position instead reaches zero collisions at 16.6M: **8× smaller than that
mitigation, 4× smaller than the shipped codec**, while removing the length limit entirely.

### E7 — Choosing `m` by measurement, and the miscalibrated constant

Exact-collision counts hit **zero at every `m` from 2 to 16**, so the count alone cannot choose
`m` — a code that is merely non-identical can still be unlearnable. Margin decides. Sweeping `m`
against `base` (max adversarial cosine, lower is better; every cell has zero collisions):

| m | code_dim | base=10 | base=100 | base=1000 | base=10000 |
|---|---|---|---|---|---|
| 2 | 512 | 0.99002 | 0.99002 | 0.99002 | 0.99002 |
| 4 | 1024 | 0.97373 | 0.98587 | 0.99546 | 0.99951 |
| 8 | 2048 | **0.96443** | 0.98944 | 0.99536 | 0.99599 |
| 16 | 4096 | 0.96270 | 0.99156 | 0.99627 | 0.99714 |

Read down the last column: at the inherited base, **more dimensions make separation worse**. Read
across the `m=8` row: fixing the constant improves the margin more than quadrupling `m` does.

`m=8, base=10` is the chosen configuration. Note that `m=4, base=10` (0.97373) beats the naïve
`m=8, base=10000` (0.99599) at **half the code_dim** — the calibration is worth more than the
capacity. `m=2` reaches zero collisions at code_dim 512 (a 16× saving) but its margin is the worst
in the table, so it is reported and rejected rather than claimed.

### E8 — Corpus impact: does anyone actually hit these tokens?

336 of 63,997 tokens is 0.5% of the vocabulary, which invites a shrug. The question that matters
is what share of a real *stream* it is. Tokenizing s6's committed corpus (665 documents, ~690K
tokens) and counting occurrences that land on a collided embedding:

| lane | tokens | **content-word** collisions | whitespace-run collisions |
|---|---|---|---|
| **indic** | 199,332 | **385 (0.193%)** | 61 (0.031%) |
| eval_registry_docs | 33,726 | 10 (0.030%) | 0 |
| code | 103,922 | **0 (0.000%)** | 681 (0.655%) |
| general_web | 74,383 | 0 (0.000%) | 0 |
| reasoning | 96,033 | 0 (0.000%) | 5 (0.005%) |
| stem_math | 20,997 | 0 (0.000%) | 1 (0.005%) |
| long_context | 76,463 | 0 (0.000%) | 157 (0.205%) |
| agentic | 85,202 | 0 (0.000%) | 3 (0.004%) |

Two unlike failures hide in one number, so they are counted apart. **333 of the 336 collided
tokens are content words; only 3 are whitespace runs** — but those 3 are frequent, which is why
the code lane lights up (Python indentation of 10, 11 and 12 spaces collapsing to one vector).
That failure is real but comparatively benign: a run of indentation loses its exact width.

The content-word failure is not benign, and it is **not spread across the corpus at all**. Every
Latin-dominant lane measures exactly 0.000%. The indic lane measures 0.193% — **one Indic token in
518 is routed through an embedding that is mathematically incapable of representing it.** That is
the sovereign risk of §8, measured on a real stream rather than argued.

(This experiment refuted its own first hypothesis. The initial gate asserted the indic lane would
show the highest *total* collision rate; it does not — code does, on whitespace. Splitting the two
classes is what made the real finding visible, and the failed gate is left in the history rather
than quietly retuned.)

### E6 — Regression: the property that must not be lost

§7 lists "similar spellings start out similar" as something Kronecker buys, and a different
position factor could destroy it. Checked rather than assumed:

```
related    train / training  cos=+0.841     unrelated  train / भारत     cos=-0.005
related    train / trainer   cos=+0.844     unrelated  apple / తెలుగు    cos=-0.003
related    भारत  / भारतीय     cos=+0.781     unrelated  zebra / quilt    cos=-0.008
```

Preserved with a wide gap. The calibrated base lowers these similarities somewhat (they ran
0.86–0.96 at base=10000) — the same sharper position discrimination that separates the adversarial
pairs also makes prefix-sharing count for slightly less. The gap to unrelated pairs remains two
orders of magnitude, so the property holds; this is the trade being made, stated rather than
buried. The other two §7 properties survive by construction: cost still has no `V` in it, and
unseen tokens still have bytes.

## What this costs, honestly

- **Codes are dense, not sparse.** The paper's codec has "at most `L` nonzero coordinates"; the
  sinusoid touches all `256·m`. FLOPs are unchanged (it is a matmul either way), but the paper's
  `gpu_table`/`gpu_dynamic` sparse-gather implementations would need reworking, and their compact
  `uint8` byte buffer no longer suffices as the stored form.
- **The margin is 0.996, not 0.5.** Long tokens with long shared prefixes are genuinely close. E3
  says a model learns from that gap at this scale; a much larger vocabulary should re-run E1/E2
  rather than assume it, and `base` is the knob if it tightens.
- **It does not fix §9.** A computed position factor still leaves the input path with exactly one
  adaptive object. Every §9 argument for keeping the projection trainable applies unchanged —
  harder, if anything, since there are now 4× fewer parameters to adapt with.
- **Verified at 68K, claimed at 131K.** The census is measured on Sarvam-1 because that is the real
  frozen artifact in this repo. The V5 BrahmicTokenizer at 131K has longer tokens and deserves its
  own census before these numbers are trusted at that scale — exactly the standard §13 sets: every
  number is a hypothesis until a proxy run tests it.
- **No training-quality claim.** This work proves *representability*, not perplexity. E8 bounds how
  often the defect is exercised (1 Indic token in 518), which is the input a loss estimate would
  need, but the loss experiment itself needs the paper's nanoGPT setup and is the obvious next step.
- **The margin is measured on one tokenizer.** E7 chooses `m=8, base=10` against Sarvam-1's byte
  length distribution (max 36 bytes). A vocabulary with much longer tokens shifts the useful
  frequency band, and `base` should be re-derived rather than inherited from here — which is the
  same mistake, one level down, that E7 caught in the Transformer's 10000.

## Files

| file | what |
|---|---|
| `dynkron.py` | the codec (both variants), embedding layer, tiny transformer, self-check |
| `run_demo.py` | six experiments, ten gates, evidence bundle |
| `submission_artifacts/evidence.json` | every number above, machine-readable |
| `submission_artifacts/run.log` | the full run |

## What changed on the second pass

The first version of this submission had a defensible result and three soft spots. All three moved:

1. **"336 tokens is 0.5% of the vocabulary, who cares?"** — E8 answers it. Every Latin-dominant
   lane is at 0.000%; Indic is at 0.193%. The defect is not small, it is *concentrated*.
2. **`m=8` was picked, not derived.** — E7 derives it, and in doing so found that the sinusoid
   `base` inherited from the Transformer is miscalibrated for byte positions by three orders of
   magnitude. Fixing that constant improved separation more than quadrupling `m`.
3. **A latent bug.** Odd `m` silently broadcast one `sin` channel across two slots instead of
   erroring — it looked like it worked while halving the usable rank. Now asserted.

## The one-line version

The 32-byte budget was never a byte budget — it was a **one-hot position table**, carrying the
same defect §11 spends a section diagnosing for sequence position. Replacing it with a computed
function removes the crop, removes the waste, and costs 4× less than the thing it replaces.
