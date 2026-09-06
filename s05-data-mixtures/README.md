# V5 Data Mixture & Curriculum Plan

**ERA V5 · Session 5 submission.** A pretraining mixture is a set of trade-offs against a fixed
token budget, composed *backward* from the benchmarks we intend to win. This document fixes a
defended share for every capability lane, sizes each lane against the real supply from the
inventory, declares the protected floor and the anneal reserve, lays out the curriculum and its
difficulty/reasoning bands, and states the 1B/3B proxy that must confirm the numbers before any of
them is trusted at full scale.

Every share here is a **hypothesis**, not a constant. The proxy in §9 is what earns it the right to
shape the full run.

---

## 1. Targets → benchmarks (what we are composing backward from)

Three capabilities justify this project; each maps to a measurable lane. A capability with no
benchmark is a capability nobody can verify we built.

| Capability we want | Lane(s) that buy it | Benchmarks it must win |
|---|---|---|
| Agentic / coding assistant (Codex-style) | Code, Agentic/tool-use | LiveCodeBench, Aider Polyglot, Codeforces · SWE-bench Verified/Live, τ²-bench, BFCL v3, GAIA, BrowseComp, OSWorld |
| Controllable reasoning (effort dial) | Reasoning traces, STEM/math | AIME, GPQA-Diamond, FrontierMath, HLE |
| Native Indic fluency (**the differentiator**) | Indic | MILU (AI4Bharat), IndicGenBench (29 langs) |
| Broad world knowledge (substrate) | General web | MMLU |
| Long-horizon context | Long-context | long-eval (needle / multi-doc) |

Reading each benchmark at the **loss-map** level fixes the *training shape*: SWE-bench evaluates a
patch → loss on the generated diff only; a tool-use benchmark evaluates the emitted function call →
loss on the JSON call, tool returns stay masked; RL-graded tasks carry a reward, no token target.
The mixture is a list of these shapes, weighted.

---

## 2. Main pretraining mixture (share of total pretraining tokens)

This is the **general-web-heavy main run**. Scarce lanes (agentic, long reasoning, verified Indic)
are deliberately held *small* here and concentrated in the short anneal (§6). A lane needs **≥ 8 %**
to meaningfully move its benchmarks; lanes below that line are *seeded* in the main run and *funded*
in the anneal.

| Lane | Share | One-line defense |
|---|---:|---|
| General web | **34 %** | Largest lane because it is the most abundant data (~4.5–4.8T available) and carries MMLU-breadth; already falling vs a naive 60 %+ web crawl. |
| Code | **24 %** | Core product capability; funded (≥8 %) so LiveCodeBench/Aider actually move. Real supply is deep (1.1T), so we can afford weight here. |
| Indic | **16 %** | Protected differentiator, funded well above its 12 % floor. Split across 4 provenance tiers in §4. |
| STEM / math | **12 %** | Feeds AIME/GPQA and is the substrate the later reasoning stage builds on. |
| Reasoning traces | **6 %** | *Seeded* below the 8 % line — the model learns the *structure* of worked reasoning now; the real reasoning capability is trained later (SFT + RLVR, Sessions 17–18) from data reserved here. |
| Long-context | **6 %** | Seeded; the dedicated long-context capability is a late curriculum stage (§7), not a flat share. |
| Agentic / tool-use | **2 %** | At the protected **floor**, not an ambition — real supply is 0.63B, so this share is almost entirely synthetic (see §3). Funded to 8 % only in the anneal. |

Sum = 100 %. This is the widget's `pretrain` preset, adopted as the defensible baseline and defended
line-by-line above.

---

## 3. Supply reconciliation — where shares need repetition or synthesis

Sizing must happen in **tokens**, not samples: a few thousand agentic trajectories can outweigh
millions of short function-call samples. Verdict rule (from the inventory tool): `demand ≤ supply` →
**covered**; `demand ≤ 4× supply` → **needs repetition**; `demand > 4× supply` → **must synthesize**.

We report both the widget-default **2T** run and a **10T** stress case, because run size is itself a
data decision — bigger runs flip "covered" lanes into scarcity.

| Lane | Real supply | Demand @2T | Verdict @2T | Demand @10T | Verdict @10T |
|---|---:|---:|---|---:|---|
| General web | 4.5T | 680B | covered (0.15×) | 3.4T | covered (0.76×) |
| Code | 1.1T | 480B | covered (0.44×) | 2.4T | **needs repetition** (2.2×) |
| STEM / math | 250B | 240B | covered (0.96×) | 1.2T | **must synthesize** (4.8×) |
| Indic | 276B | 320B | **needs repetition** (1.2×) | 1.6T | **must synthesize** (5.8×) |
| Long-context | 100B | 120B | **needs repetition** (1.2×) | 600B | **must synthesize** (6.0×) |
| Reasoning traces | 85B | 120B | **needs repetition** (1.4×) | 600B | **must synthesize** (7.1×) |
| **Agentic / tool-use** | **0.63B** | 40B | **must synthesize** (63×) | 200B | **must synthesize** (317×) |

**Honest accounting — the three things a reviewer will push on:**
- **Agentic is the binding constraint at every scale.** The 2 % floor buys 40B tokens against 0.63B
  real supply, so **≥98 % of the agentic lane is generated, not collected.** Synthesis plan in §8.
- **Scale is not free.** At 10T, code, STEM, Indic, long-ctx and reasoning all cross into
  repetition-or-synthesis. Our default target is **2T** precisely so the scarce lanes are met with
  ≤~1.4× repetition + bounded synthesis; going to 10T is a decision to 5–7× the synthetic budget, not
  a free win. We commit to 2T for the first production run and re-derive this table before scaling.
- **Repetition is capped.** No lane is repeated beyond ~2× unique tokens in the main run (epoch cap);
  demand above that is met by synthesis, not silent re-epoching, so we never inflate a lane's real
  weight on paper.

---

## 4. Indic slot — split across the four provenance tiers (Session 3)

The 16 % Indic slot is **not one number.** Sorting the inventory by *verified native tokens* shows how
little genuinely native material exists, which is exactly what forces a synthetic budget. Tier shares
are of the Indic slice; token figures shown at the 2T run (Indic = 320B).

| Tier | What it is | Share of Indic | Tokens @2T | Supply reality |
|---|---|---:|---:|---|
| **A — verified native** | Human-written native pages, curated corpora (e.g. Sangraha-verified, IndicCorp-curated), native news/Wikipedia | **40 %** | 128B | The scarce core. Verified-native is a *small fraction* of the 276B Indic total → Tier A itself needs mild repetition (~1.5–2×) and is the pool the anneal reserve (§6) is drawn from. |
| **B — unverified crawl** | Language-ID'd web crawl in Indic scripts, not human-verified | **25 %** | 80B | Abundant but noisier; carries breadth, not trust. Quality-filtered per Session 4. |
| **C — translated** | High-quality English→Indic MT of curated English (textbooks, QA, instructions) | **20 %** | 64B | Fills breadth cheaply; capped at 20 % so the model does not learn "translationese" as native register. |
| **D — synthetic** | Model-generated native Indic (native-prompt generation, back-translation round-trips, instruction-synth) | **15 %** | 48B | Deliberately created to close the gap the target defines. Filtered for fluency + factuality before use. |

**Why not more Tier A?** Because it does not exist in unique form at 128B. Pretending it does is the
"wishful accounting" this session exists to prevent. The 40/25/20/15 split is the honest statement of
*how much Indic we must build* to hit 16 %.

---

## 5. Protected always-on floor (outside the OPUS selector)

OPUS retains only ~40 % of candidate batches (≈6× effective-token value in V4) by scoring each batch
against a proxy direction. That proxy is English-heavy, so it **starves native Indic and unfamiliar
agentic trajectories toward zero** unless we wall them off. V4 solved this with an 8 % always-on Indic
lane; **V5 extends the protection to three scarce lanes.**

| Protected lane | Always-on floor (share of *every* batch, selector cannot cross) |
|---|---:|
| Indic | **12 %** |
| Reasoning traces | **4 %** *(our extension — the notes say V5 protects reasoning too; a small floor keeps the trace-structure signal alive under an English proxy)* |
| Agentic / tool-use | **2 %** |
| **Total protected** | **≈18 %** — the aggressive selector operates on the remaining ~82 %. |

The floor is a **guarantee, not a large share**: long-tail capability is a decision, not a residue.
Everything above the floor in each lane is still subject to selection.

---

## 6. Anneal reserve (held back for the cooldown)

The final anneal is the highest-leverage stage: a short phase (~**3 % of total tokens**), LR decayed,
run on a concentrated mix of the *best* data. Its gain is only possible if that data **survives the
main run** — so we quarantine it now, before the selector can spend it.

**Anneal mixture** (widget `anneal` preset — scarce lanes concentrated):
Indic 28 %, Code 20 %, Reasoning 18 %, STEM 10 %, Long-ctx 8 %, Agentic 8 %, Web 8 %.

At 2T total, anneal ≈ **60B tokens**; the resulting scarce-lane demand and the reserve we quarantine
up front:

| Lane | Anneal demand @60B | Reserve quarantined from main run |
|---|---:|---|
| Indic (Tier A native) | 16.8B | ~20B best verified-native (highest-quality Sangraha/IndicCorp, native news) |
| Reasoning | 10.8B | ~12B longest *verified* reasoning traces (checked final answers) |
| Code | 12.0B | ~12B highest-signal repo-edit / diff data |
| STEM | 6.0B | ~7B hardest *verified* math (competition, proof-checked) |
| Long-context | 4.8B | ~5B genuine long documents (books, multi-file repos) |
| Agentic | 4.8B | **all ~0.63B real trajectories** + ~4.5B top-filtered synthetic |
| Web | 4.8B | drawn live from the covered web pool (not scarce, not reserved) |

The reserve is tagged at ingestion and excluded from OPUS candidacy until the anneal begins. Reserve
sizing carries ~15 % margin over anneal demand so selection inside the anneal still has choice.

---

## 7. Curriculum — the order the model learns in

Proportions decide *how much*; the curriculum decides *when*. Difficulty and capability both ramp;
long context enters late; the best data is spent last.

| Stage | ~% of tokens | Mixture emphasis | Purpose |
|---|---:|---|---|
| 0 · Seed | ~5 % | Broad general web only | Establish language, script, basic structure |
| 1 · General foundation | ~45 % | Web-heavy, code/STEM ramping in | World knowledge + factual substrate |
| 2 · Capability ramp | ~32 % | Shift toward code, STEM, reasoning (mirrors V4: web 72→18, code 13→35, STEM 7→39) | Build the target capabilities on the foundation |
| 3 · Long-context | ~15 % | Introduce long sequences (books, multi-file repos, long-eval-shaped) | Learn to preserve info across long context *after* it can read/reason |
| 4 · Anneal | ~3 % | Anneal preset + reserved Tier-A data, LR decayed | Disproportionate final capability gain |

**Stability rule (non-negotiable):** never change the mixture in one hard step. Every transition is
**warmup-blended across several-B tokens**. V4 saw the gradient norm jump **~150×** when a sudden
Hindi-share increase hit frozen embeddings — an event that can destroy a run. Architecture and mixture
are frozen before the main run; transitions are planned, infrequent, and monitored like an
architectural change (grad-norm + loss-spike alerts on every boundary).

### Difficulty bands (each stage climbs this ladder)

| Band | Description | Concrete example |
|---|---|---|
| L1 · Basic | Single-step, no planning | `reverse("cat")` → `"tac"`; `12 × 8 = 96` |
| L2 · Intermediate | Multi-step, standard patterns | Implement binary search with tests; a GSM8K word problem |
| L3 · Advanced | Planning / domain depth required | A LeetCode-hard problem; an AIME 2024 problem |
| L4 · Frontier | Research-level | An SWE-bench Verified repo patch; a FrontierMath / GPQA-Diamond item |

### Reasoning-length bands (a *distribution*, not one depth)

The effort dial (low/medium/high/ultra) is not created at inference — it *selects* behaviours the
model was trained on. The mixture must reserve traces across the full range **in math, code, and
general problem-solving** so the behaviour is not tied to one domain.

| Effort | ~think-token budget | Concrete example |
|---|---:|---|
| Low | 0–128 | "15 % of 200?" → "30." Direct answer, no visible trace. |
| Medium | 128–512 | GSM8K: lay out the arithmetic steps, then answer. |
| High | 512–2k | AIME: try an approach, check an intermediate result, adjust. |
| Ultra | 2k–8k+ | Olympiad proof / hard SWE task: explore alternatives, verify, self-correct before answering. |

Short traces are the foundation; progressively longer traces teach the model to sustain, verify, and
correct. The later RLVR stage teaches it to *obey the requested effort setting*.

---

## 8. Building the agentic lane (the ≥98 % that must be synthesized)

Because real supply is 0.63B against 40B demand, this lane is *built*. Each synthetic trajectory is a
long chain — plan → tool-call(args) → observation → failure → recovery → final answer — with the
**masking rule enforced**: loss on the model's planning / tool-calls / final answer (green); tool
observations and user turns are context (grey). Applying loss to observations would teach the model to
*hallucinate tool results instead of calling tools*.

Sources & method: seed from the real trajectories (SWE-Gym, τ-bench-style, ToolBench/BFCL, APIGen);
generate the rest via a strong teacher operating real tools in sandboxes (code exec, retrieval,
browser), keeping only trajectories whose final outcome the environment verifies (tests pass / task
state reached). The verified real trajectories are reserved for the anneal (§6); synthetic fills the
main-run 2 %.

---

## 9. Proxy experiment — the hypothesis test (1B & 3B)

**No share here is trusted until a cheap run confirms it.** A data decision is a hypothesis until an
experiment tests it.

**Setup.** Train a **1B** model on the candidate mixture for ~20–30B tokens, and a **3B** for ~60B
tokens (small but Chinchilla-reasonable). Two baselines:
(a) the **naive web-heavy** preset; (b) the candidate mixture with **protected floors disabled**.

**Named metrics (per lane):** held-out per-lane validation loss **plus** downstream —
MMLU (web) · HumanEval + LiveCodeBench-lite (code) · GSM8K + MATH-subset (STEM/reasoning) ·
**MILU (Indic)** · BFCL tool-call accuracy on a held-out set (agentic) · long-eval needle@16k (long-ctx).

**Confirm / refute criteria (the line that makes this defensible):**
- **Mixture confirmed** iff, at equal tokens vs the naive baseline, **MILU ↑ ≥ 5 pts** and **BFCL
  tool-call accuracy ↑ ≥ 8 pts** with **MMLU regression ≤ 1 pt**. Otherwise refuted → rebalance.
- **Protected floor confirmed** iff disabling floors (baseline b) drops MILU / BFCL by **> 3 pts** —
  i.e. we reproduce, in miniature, the English-proxy starvation OPUS would cause at scale.
- **Anneal confirmed** iff running the final 3 % on the reserve vs on ordinary sampled data yields a
  scarce-lane (MILU + AIME-subset) gain **larger than its 3 % token cost** would predict linearly.
- **Scale-stability gate:** keep only shares whose advantage **holds in rank from 1B → 3B.** A share
  that helps at 1B but not 3B is removed before full scale.

Only shares that clear these gates enter the production recipe.

---

## 10. Cleaning follow-through

The mixture names the **starved lanes** — the cleaning effort now points there, not at already-deep
lanes. Priority order for continued Session-4 cleaning/verification:
1. **Verified-native Indic (Tier A)** — the binding sub-constraint; every verified token here reduces
   repetition on the anneal reserve.
2. **Real agentic trajectories** — verify/clean every one; each is Tier-A scarce and irreplaceable.
3. **Long verified reasoning traces** — with checked final answers (they feed the reserve and the
   later RLVR stage).

A mixture is only as trustworthy as the cleaned, documented tokens standing behind it.

---

### One-line summary

Compose backward from the benchmarks; fund code + Indic + STEM in a web-heavy main run; hold agentic /
long-reasoning / verified-Indic small on purpose and pay them off in a 3 % anneal from a
pre-quarantined Tier-A reserve; protect ~18 % of every batch from an English-biased selector; blend
every transition to keep the run stable; and trust no number until a 1B/3B proxy confirms it.
