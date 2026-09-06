# A real Fourier alternative to Kronecker: binding by convolution

**ERA V5 · Session 7 assignment — Problem 4.**

> *"What is a REAL Fourier alternative of Kronecker? Why can't I represent each character
> like a fourier wave, and just add them to make a word!!"*

You can. The formal name for it is **circular convolution binding** — Holographic Reduced
Representations (Plate, 1995), in the frequency domain (FHRR). The reason it is the *real*
Fourier counterpart of Kronecker, rather than a Fourier-flavoured tweak, is the convolution
theorem.

## Run it

```bash
.venv/Scripts/python s07-model-internals/fourier/run_demo.py   # 11 experiments, 16 gates, ~7 min
.venv/Scripts/python s07-model-internals/fourier/fourier.py    # codec self-check
```

Exits 0 only if all sixteen gates pass. Writes `submission_artifacts/evidence.json`,
`run.log`, and the `embedding_policy_id.json` ledger record §13 asks for.
Measured against **Sarvam-1**, sha256 `bb5115a3…`, the tokenizer s6 froze — 63,997 tokens after
excluding 4,099 special tokens, matching the paper's Table 3 methodology.

## Why convolution is the answer

The Kronecker codec binds a byte value to a byte position with an **outer product**. An outer
product's dimension is the *product* of its factors' dimensions:

```
kronecker:    c[b] (x) p[pos]        dim = d_c * d_p = 256 * 32 = 8192       FORCED
```

The Fourier counterpart of an outer product is circular convolution, because
`DFT(a ⊛ b) = DFT(a) ⊙ DFT(b)` — convolving two signals is multiplying their spectra
elementwise. Elementwise multiplication does **not** multiply dimensions:

```
fourier:      z[b] (.) rho^pos       dim = d                                  FREE
```

That single structural difference is the whole submission. Kronecker's 8192 is not a tuning
choice, it is arithmetic; here `d` is a genuine free parameter, and the experiments below ask how
low it can go.

Position comes from the **shift theorem**: shifting a signal multiplies its spectrum by a phase
ramp. So "byte `v` at position `p`" is the wave for `v` advanced `p` steps along a fixed ramp, and
a word is the sum of its characters' waves — literally the question as posed.

```
z_v ∈ C^(d/2+1),  |z_v[k]| = 1          one frozen wave per byte value
ρ   ∈ R^(d/2+1)                          one frozen position ramp
κ(t) = irfft( (1/√L) · Σ_p exp(i·(z_{b_p} + p·ρ)) )
```

Nothing is learned. Phases are drawn once from a seeded generator and frozen, so the codec is a
deterministic function of the byte string exactly as Kronecker is. **The seed now joins the
tokenizer hash as part of the artifact's identity** — a new field for `embedding_policy_id`.

## The result

| input path | code_dim | tokens it cannot represent | projection @ D=8096 | invertible? |
|---|---|---|---|---|
| dense table (V×D) | — | 0 | 1,061,158,912 | no |
| kronecker one-hot `d_p=32` (shipped) | 8192 | **336** | 66,322,432 | no |
| dynamic kronecker `m=8` (problem 3) | 2048 | 0 | 16,580,608 | no |
| **fourier `d=64`** | **64** | **0** | **518,144** | via nearest-code |
| **fourier `d=2048`** | 2048 | 0 | 16,580,608 | **yes, byte-exact** |

**Zero collisions at d=64 — 128× smaller than the shipped Kronecker codec, and a 99.95%
reduction against a dense table.** Or spend the same budget as dynamic Kronecker and get full
invertibility instead.

And the result that matters most for an India-first model, from F8: on tokens the model was
**never trained on even once**, a dense table scores 22% against a 20% chance floor — its rows
never received a single gradient — while this codec scores **99%**. §2's per-row learning-rate
asymmetry, the mechanism the course blames for low-resource languages lagging, **does not exist
in a scheme with no rows.**

## How it is proved

Eleven experiments. The census, margin and learnability harness is **imported from the problem-3
submission rather than reimplemented**, so both codecs are judged by identical instruments.

### F1 — Discrimination: how small can `d` go?

| d | collisions | projection @ D=8096 |
|---|---|---|
| 64 | **0** | 0.5M |
| 128 | 0 | 1.0M |
| 256 | 0 | 2.1M |
| 512 | 0 | 4.1M |
| 1024 | 0 | 8.3M |

Zero at every dimension tested, down to **d=64** — while the shipped Kronecker codec loses 336
tokens at code_dim 8192. The dimension that forced Kronecker's whole parameter bill turns out not
to be needed at all; it was an artifact of the binding operator, not of the problem.

### F2 — Separation margin (the honest weak spot)

Max cosine among the 155 pairs the shipped codec destroys:

| d | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| max adversarial cosine | 0.98202 | 0.97741 | 0.97708 | 0.97684 | 0.97392 |

**Essentially flat in `d`.** These are near-identical strings differing in a final byte or two, so
their superpositions genuinely overlap; extra dimensions add room but not distance. A tuned
dynamic Kronecker codec reaches **0.964**, which is better. If margin is the thing you care about,
problem 3's answer wins; this one wins on dimension and on invertibility.

### F3 — Learnability: does the lossiness survive a real model?

This is the experiment the approach could have failed. Kronecker's code is exact and sparse; this
one superposes waves with crosstalk. Same rig as problem 3 — 336 tokens, labels forced to disagree
inside each collision group, everything above the embedding held identical.

| arm | accuracy | input-path params |
|---|---|---|
| kronecker one-hot `d_p=32` | **52.1%** (its exact ceiling) | 524,288 |
| **fourier `d=256`** | **100.0%** | **16,384** |
| fourier `d=2048` | 100.0% | 65,536 |
| dense table (control) | 100.0% | 21,504 |

**100% with 32× fewer input-path parameters than the codec it replaces.** The crosstalk is real
but it is far below the threshold that matters for discrimination.

### F4 — Invertibility: the property Kronecker does not have

This is what makes the scheme *Fourier* rather than a re-parameterisation. Correlating the code
against a candidate `(byte, position)` wave recovers whether that pair is present, so a token can
be decoded by argmax over 256 candidates per position — no table consulted, nothing stored per
token.

Two regimes, costing very differently:

**Byte-argmax decode** (vocabulary-*free*: can emit strings never seen in training)

| d | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| whole-token exact | 25.5% | 56.0% | 87.8% | 99.0% | **100.0%** |
| byte accuracy | 57.1% | 84.6% | 98.1% | 99.9% | **100.0%** |

Degradation is by length, exactly as the capacity law predicts (`d ≳ 20·L`) — at d=1024, tokens
of 0–23 bytes decode at 100.0% and only the 32–39 byte bucket drops to 99.0%.

**Nearest-code decode** (vocabulary-*bounded*: match the predicted vector against known codes)

| d | 64 | 128 | 256 | 1024 |
|---|---|---|---|---|
| correct token retrieved | **100.00%** | 100.00% | 100.00% | 100.00% |

Perfect wherever discrimination works — which F1 showed is d=64.

**Why this matters beyond problem 4.** The paper's future-work Hypothesis A wants an output head
that emits a `D`-vector and decodes byte composition from it, removing the `d_model × |V|` matrix
and with it the fixed vocabulary. That needs precisely this property, and the two regimes price it:
a cheap tied head needs only nearest-code decoding (d=64); a genuinely vocabulary-free model needs
byte-argmax (d=2048). The paper flags collisions in Kronecker space as the obstacle — here the
collision count is zero and the decode rate is measured, not assumed.

### F5 — The position ramp, and a window that tries to come back

`ramp="shift"` uses the canonical `2πk/d`, making binding an exact circular shift — the textbook
shift theorem, and more interpretable. But a circular shift **wraps**:

```
ramp=random   |<phase(p=0), phase(p=d)>| = 0.0271   <- no period
ramp=shift    |<phase(p=0), phase(p=d)>| = 1.0000   <- wraps: the window is back, at d
```

Position `p` and `p+d` are identical under the canonical ramp, so the 32-byte window returns as a
`d`-byte window. Frozen random phases have no period, so length stays genuinely unbounded. This is
the same stored-vs-computed trade §12 describes, appearing a third time.

### F6 — Parameter bill at V5 reference shape (D=8096, 16 bytes/param AdamW)

| input path | params | training memory | vs dense |
|---|---|---|---|
| dense table V×D | 1,061,158,912 | 16.98 GB | — |
| kronecker one-hot `d_p=32` | 66,322,432 | 1.06 GB | 93.75% |
| kronecker one-hot `d_p=64` | 132,644,864 | 2.12 GB | 87.50% |
| dynamic kronecker `m=8` | 16,580,608 | 0.27 GB | 98.44% |
| **fourier `d=2048`** (decodable) | 16,580,608 | 0.27 GB | 98.44% |
| **fourier `d=64`** (discrimination only) | **518,144** | **0.01 GB** | **99.95%** |

### F7 — Regressions that must not be lost

**Order sensitivity.** A bag of waves would score 1.000; the phase ramp prevents that:

```
abc vs cba            cos=+0.361
dog bites man vs man bites dog   cos=+0.538
ab  vs ba             cos=+0.027
```

**Prefix similarity** (§7's property) survives — shared prefixes contribute identical terms to the
sum:

```
related    train / training  cos=+0.798     unrelated  train / भारत   cos=-0.015
related    train / trainer   cos=+0.849     unrelated  apple / తెలుగు  cos=+0.016
related    भारत  / भारतीय     cos=+0.830     unrelated  zebra / quilt  cos=+0.010
```

### F8 — The Zipf asymmetry, removed rather than mitigated

§2 is the course's sharpest claim about embeddings: a dense table "is a hundred and thirty-one
thousand small objects whose effective learning rates are set by the corpus rather than by the
optimizer, and the slow end of that range is where the low-resource languages live." A rare row is
visited rarely, stays near its initialisation, and "contributes noise to every sequence it appears
in."

**This codec has no rows.** Every token's gradient lands in the same projection, and the code is
already meaningful because it was computed from bytes. So the spread should not exist at all.

Test: 650 real tokens across 5 scripts, labelled by script — a property that generalises across
tokens. Trained on a Zipfian stream where the most frequent token is seen **~550×** more often
than the rarest trained one. 100 tokens are held out of training entirely. Evaluated uniformly.

| frequency band | dense table | fourier d=256 |
|---|---|---|
| head (top 10%) | 100.0% | 100.0% |
| middle | 99.5% | 100.0% |
| tail (bottom 50%) | 97.1% | 100.0% |
| **never seen once** | **22.0%** | **99.0%** |

Chance is 20%. The dense table's unseen rows are **at chance** — they never received a single
gradient, so they still hold their random initialisation, exactly as §2 describes. The Fourier arm
reaches 99% on tokens it has never been trained on, because their codes come from bytes and the
projection that reads them was trained by every *other* token.

This is the paper's own future work answered at small scale — it lists both *"Multilingual
evaluation… test whether Kronecker's byte-level prior helps on low-resource languages"* and
*"Token-bucket NLL analysis… Kronecker should plausibly help most on rare/long tokens."* The
mechanism is confirmed; the loss claim still needs a real training run.

### F9 — §9's adaptation boundary, which applies *harder* to this design

§9 warns that "a compressed input path is a less capable adapter by construction." At d=64 the
projection is 128× smaller than the shipped Kronecker one, which by the notes' own argument makes
this the most fragile input path in the course. So it is measured, not argued away.

A mixture shift is staged mid-run (stream flips from 90% Latin to 70% Devanagari) and the gradient
norm of the layers **above** the embedding is tracked — §9's leading indicator.

| input path | baseline | peak after shift | ratio |
|---|---|---|---|
| dense table | 0.444 | 3.881 | 8.75× |
| dense table, frozen | 1.376 | 5.723 | 4.16× |
| fourier d=2048 | 3.800 | 14.498 | 3.82× |
| fourier d=64 | 1.054 | 4.587 | 4.35× |
| fourier d=64, frozen | 1.831 | 5.633 | 3.08× |

**Read this table down the pairs, not across the rows.** The ratio normalises by each arm's own
baseline, and those baselines differ for reasons unrelated to adaptation — a dense table idles the
layers above it, a wider projection changes the activation scale. Comparing dense against Fourier,
or one `d` against another, is confounded, and **no ranking between them is claimed.** The first
version of this experiment did claim one, on the ratio column, and it was wrong.

What *is* controlled is a matched pair — identical architecture, identical scale, the only
difference being whether the input path may adapt:

| pair | baseline | peak |
|---|---|---|
| dense: trainable → **frozen** | 0.444 → **1.376** | 3.881 → **5.723** |
| fourier d=64: trainable → **frozen** | 1.054 → **1.831** | 4.587 → **5.633** |

In both pairs freezing raises the load on the layers above — not only at the shift but throughout
the run. §9's mechanism reproduces on a codec 128× smaller than the one the V4 scar was first seen
on, and its operational conclusion carries over unchanged: **the projection stays trainable, and
freezing is a scheduled, logged decision or it does not happen.**

### F10 — Concatenation is addition, exactly

Because binding is a convolution, the codec is a **homomorphism**: joining two strings is adding
their codes, once the second is rotated forward by the length of the first.

```
κ(xy) = [ √Lx·κ(x) + √Ly·rot^Lx(κ(y)) ] / √(Lx + Ly)
```

`Lx`, `Ly` are **byte** counts (not characters); `rot` advances a code by one byte position — the
circular convolution binding is built from. Nothing re-reads the bytes of the joined string.

**Worst residual over 400 real token pairs: 1.94e-15** — exact to machine precision. The Kronecker
grid has no law of this kind: its code is a set of marked cells, so joining two tokens means
re-marking the grid from scratch at new positions.

Two things this experiment forced:

- **It found a real defect.** The law held at 1e-15 in the frequency domain but missed by **1e-2**
  in the time domain — the space the model actually sees. `irfft` requires the DC and Nyquist
  channels to be real, and with arbitrary phases there it was silently discarding
  `Im(spec[Nyquist]) = 0.65`. Those two channels now carry ±1 phases; round-trip error went from
  0.65 to 4e-16 and every downstream number was re-measured.
- **Byte-fallback tokens are excluded, and not as a hedge.** `token_bytes('<0x1B>')` is one byte
  *by convention*, but `token_bytes('<0x1B>' + 'x')` is the literal string, because the joined text
  no longer matches the `<0xNN>` form. The byte *mapping* is not a homomorphism over concatenation;
  the codec still is. Exactly 4 of 400 pairs were affected, all of them byte-fallback.

The identity is exact on the **raw** code. The shipped codec adds a per-token z-normalisation — an
affine rescale — so downstream it holds up to that constant. Compose in raw space and normalise
last if you want to use it.

### F11 — The crosstalk model, checked rather than asserted

Every capacity claim here rests on `SNR ≈ √(d/L)`, which had been stated and never tested.
Unbinding sums `nf` channels: the matching term adds coherently, the other `L−1` are sums of
random unit phasors. So `signal = 1.0`, `noise std = √((L−1)/2nf)`, `SNR = √(2nf/(L−1)) ≈ √(d/L)`.

| d | SNR predicted | SNR measured | decodable? |
|---|---|---|---|
| 128 | 2.98 | 2.83 | **no** |
| 256 | 4.20 | 4.02 | yes |
| 512 | 5.93 | 5.76 | yes |
| 1024 | 8.38 | 8.03 | yes |
| 2048 | 11.84 | 11.32 | yes |

**Worst disagreement: 5.2% across a 16× range of `d`.**

The cross-check matters more than the fit. Decoding is an argmax over 256 candidates, so the true
byte must beat the *largest* of 255 noise draws — for Gaussian noise, ≈ `√(2 ln 255)` = **3.33σ**.
That threshold falls between d=128 (2.83, below) and d=256 (4.02, above). F4 independently
measured byte accuracy at those dimensions as **57% and 84%**: the collapse lands where this model
says it should, from an experiment that knew nothing about it.

So `d` is not a knob found by trial. **Per-byte decoding needs roughly `d ≳ 11·L`**, and
whole-token exactness is stricter because every byte must land — which is why d=2048 covers a
36-byte vocabulary while d=64 suffices for discrimination alone.

Reported honestly: measured noise runs a few percent **above** prediction at every `d`, never
below. That is a bias, not scatter. The derivation assumes the `L−1` interfering terms are
independent and in a real token they are not quite — repeated byte values correlate, and the two
real-valued channels contribute different variance. A good model, not an exact one.

### The ledger record

§13 requires a checkpoint to be able to say what its input path was doing, so `run_demo.py` emits
`submission_artifacts/embedding_policy_id.json`:

```json
{
  "embedding_type": "fourier_convolution_binding_v1",
  "code_dim": 2048,
  "position_ramp": "random_frozen_phases",
  "codec_seed": 20260815,
  "codec_trainable": false,
  "projection": { "shape": [2048, 8096], "trainable": true, "unfreeze_schedule": null },
  "tokenizer_hash": "bb5115a3…",
  "tying": { "tied": false, "reason": "code_dim != d_model …" },
  "position_policy": "deferred_to_session_8"
}
```

**`codec_seed` is a field the course has not needed before.** §9 argues the tokenizer hash matters
more here than anywhere else, because changing the tokenizer changes every token's bytes and so
every code. The PRNG seed has exactly the same property: change it and every wave, and therefore
every embedding, changes — while the tokenizer hash stays identical and the ledger looks fine. It
has to be pinned or the record is incomplete.

**On weight tying** (§5): the paper notes tying is *impossible* for Kronecker because
`code_dim ≠ d_model`, and that applies here identically — there is no transpose of a
`2048 × 8096` projection that produces vocabulary logits. This costs nothing, because §13 already
commits V5 to an untied head on the grounds that at V5's scale the saving is small and the
constraint on the geometry is not.

## What this costs, honestly

- **It is lossy where Kronecker is exact.** Kronecker's code is sparse with exactly `L` nonzeros;
  this superposes with crosstalk, and capacity is statistical (`SNR ~ √(d/L)`) rather than
  guaranteed. F3 shows the noise is far below what discrimination needs, but it is noise.
- **Margin does not improve with `d`** (F2). On near-identical strings, dynamic Kronecker
  separates better. Two submissions, two different strengths — this is not strictly dominant.
- **Byte-value similarity is arbitrary.** Random phases mean `a` and `b` get unrelated waves.
  Kronecker's one-hot is equally arbitrary there, so it is parity, not a regression — but the
  "waves" framing invites the hope that similar bytes get similar waves, and they do not. Choosing
  structured phases (by Unicode block, say) is an obvious and untested extension.
- **A new frozen artifact.** The PRNG seed determines every code. Change it and every embedding
  changes, exactly as changing the tokenizer would — so `embedding_policy_id` must carry it, and
  §9's argument about a checkpoint being unable to describe its own input path applies with one
  more field.
- **Dense codes.** The paper's `gpu_table`/`gpu_dynamic` sparse-gather implementations do not
  carry over; the compact `uint8` byte buffer no longer suffices as the stored form.
- **No training-quality claim.** This proves representability, decodability, parameter cost and
  the rare-token mechanism. Whether any of it converts into better validation loss needs the
  paper's nanoGPT setup and a GPU.
- **F8 tests a byte-derivable label.** Script is recoverable from bytes, which is what lets the
  shared projection generalise to unseen tokens. A label that is *arbitrary* per token — an
  identity the bytes do not predict — would still need that token's own gradient, and there the
  dense table's per-row capacity is an advantage, not a defect. The honest claim is narrower than
  "rare tokens are solved": what is removed is the *noise-from-initialisation* failure §2
  describes, not the need to learn token-specific facts.
- **One vocabulary.** All numbers are Sarvam-1. The capacity law `d ≳ 20·L` is the thing to
  re-derive on a vocabulary with longer tokens, rather than inheriting `d` from here.

## Files

| file | what |
|---|---|
| `fourier.py` | the codec, embedding layer, decode/unbind, self-check |
| `run_demo.py` | eleven experiments, sixteen gates, evidence bundle |
| `submission_artifacts/evidence.json` | every number above, machine-readable |
| `submission_artifacts/embedding_policy_id.json` | the §13 ledger record for this input path |
| `submission_artifacts/run.log` | the full run |
| `../fourier-embeddings.html` | the interactive explainer — the mechanism, drawn |

## The one-line version

The Fourier counterpart of the Kronecker product is circular convolution, and swapping the outer
product for it makes the code dimension stop being a product: **zero collisions at d=64 instead of
8192**, unbounded token length, and — uniquely — a code you can read the word back out of.
