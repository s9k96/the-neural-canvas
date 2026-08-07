# V5 Training Data Execution System

**ERA V5 · Session 6 submission.** Session 5 produced a mixture *plan*. This is the system that
**executes** it and then **proves what it did**: immutable tokenized shards with manifests, an
evaluation firewall, per-step mixture quotas with protected floors, OPUS selection with a full
audit trail, a consumption ledger, a token-level learning trace, and a crash / resume / replay /
fork cycle in which the batches are bit-identical.

The point is not scale. The point is that every claim in `submission_artifacts/evidence.md` was
**recomputed from the artifacts on disk**, and that a reviewer can delete the whole bundle,
re-run one command, and get the same hashes back.

## Run it

```bash
.venv/Scripts/python s6-dataset-creation/run_demo.py            # the complete demonstration
.venv/Scripts/python s6-dataset-creation/tests/test_invariants.py   # 13 invariants
```

`run_demo.py` is offline and deterministic; it takes ~50 s on a CPU and exits 0 only if all nine
evidence rows pass. **That one command produces everything**, from a fresh clone:

| It writes | What |
|---|---|
| `submission_artifacts/` | the mandated bundle — 62 files: `run.log`, `evidence.json`, `evidence.md`, `performance.json`, `manifests/`, `ledgers/`, `checkpoints/` |
| `shards/` | the immutable shard payloads (`.npz`), kept outside the bundle so manifests can be revalidated against bytes that live elsewhere |
| `out/s6data.js` | the data behind `dataset-creation.html` — presentation only, so a failure here is logged and cannot change the exit code |
| `corpus/` | **only if missing** — it then invokes `prepare_corpus.py` first, which needs the network |

`corpus/` is committed, so a normal run never touches the network. To refresh the snapshot
deliberately:

```bash
.venv/Scripts/python s6-dataset-creation/prepare_corpus.py
```

Note that `corpus/` is the reproducibility anchor: hashes are byte-stable across runs *given the
same snapshot*. Delete it and the re-fetch may legitimately return different rows, so the new
artifacts will be internally consistent and still 9/9, but their hashes will not match the
committed ones.

### What is in git and what is not

| Path | In git? | Why |
|---|---|---|
| `corpus/` | **yes** (3.3 MB) | An input, not an output. It is what makes the run offline and the hashes stable; re-fetching is not equivalent. |
| `submission_artifacts/` | **yes** (23.7 MB) | The brief requires the generated manifests, ledgers, checkpoints and reports to *be* the submission. A reviewer sees the evidence without running anything. |
| `out/s6data.js` | **yes** (116 KB) | Page data, so `dataset-creation.html` renders from a clone. Same convention as `s4-data-cleaning/out/`. |
| `shards/` | **no** (2.8 MB) | Pure derived data: tokenizing the committed corpus with the frozen tokenizer reproduces it byte-for-byte, and `run_demo.py` rebuilds it every run. |

Because the artifacts are committed but the shard payloads are not, `test_invariants.py` checks for
both up front and says which one is missing rather than failing deep inside the immutability test.

Requires `numpy`, `tokenizers` and `huggingface_hub`. No torch, no `datasets`, no pytest.

## Architecture

```
corpus/*.jsonl                 committed snapshot of real HF data + provenance
      │
      ▼  tds/shards.py         clean → tokenize (frozen sarvam-1) → immutable shards + §6 manifests
      │                        admission gate: license, tokenizer hash, cleaning lineage, overlap
      ├─ tds/firewall.py       test/validation registry, never-train flag, 8-gram fingerprints
      │
      ▼  tds/mixture.py        S5 stages → per-step lane quotas, protected floors, anneal reserve
      │                        OPUS: score → accept / reject / defer / protected-floor override
      ▼  tds/packing.py        6 policies → sequences: token ids, loss mask, segment mask, pos ids
      │                        plan_batch() is a PURE function of (seed, branch, step)
      ▼  tds/train.py          numpy causal LM, Adam, checkpoints tied to ledger offsets,
      │                        crash → resume → replay → fork → audit
      ├─ tds/ledger.py         append-only consumption ledger, token trace, learning ledger
      ▼  tds/evidence.py       re-reads the artifacts and recomputes all 9 requirements
submission_artifacts/
```

Seven modules, one per rubric area — a reviewer looking for *Evaluation and validation firewall*
finds `tds/firewall.py`.

## The four decisions that matter

**1. The planned stream is a pure function, not a stateful iterator.**
`plan_batch(schedule, pool, seed, branch, step)` derives everything from the step index plus RNG
seeded on `(seed, branch, step, lane, slot)`. No cursor, no generator state, no dataloader RNG
survives between steps. Resume, replay and fork are therefore *the same call with the same
arguments*, and "no skipped or repeated batch" becomes a property of the design rather than of
careful state restoration. The protected-floor signal OPUS uses is pure for the same reason
(`Schedule.floor_slots(step)`): if it depended on how much of a lane the process had served so
far, replay would make different decisions.

**2. Two commands, one demo command.** Fetching live data inside the demo would make the hashes
unreproducible, so the network is quarantined in `prepare_corpus.py` and its output is committed.

**3. Evidence is derived, never declared.** `tds/evidence.py` knows nothing about the run. It
opens the manifests, ledgers, proofs and reports and recomputes each claim — shard content hashes
from the shard bytes, lane shares from the ledger's per-sequence lane counts, throughput from the
effective stream, and every hash in `resume_proof.json` / `replay_proof.json` / `fork_proof.json`
cross-checked against `consumption.jsonl`. That last cross-check is what makes the proofs
falsifiable: a proof file that agrees only with itself proves nothing. Test 13 corrupts one batch
hash in a copy of the ledger and asserts the Replay row flips to FAIL.

**4. The model is small, but the masks are real.** A numpy causal LM (masked-mean attention +
linear readout, Adam, float64) exists to produce real per-token losses, real gradient norms and
real optimizer state — and to genuinely *consume* the block-diagonal attention mask. That is what
makes test 5 a proof: two samples packed into one window produce **exactly** the per-token losses
they produce when packed alone (`atol=1e-12`). A system that merely recorded the mask would fail
that test.

Checkpoints store float64 because replay restores the model and then re-scores OPUS candidates
with it; a float32 round-trip could flip a decision and break the bit-exact replay proof. That
constraint is what sizes the model (`d_model=32`, model vocab 2048) — it keeps a checkpoint at
~4 MB.

## Data

Fetched through the HF datasets-server rows API with stdlib `urllib` (~100 rows per lane); each
fetch records dataset, config, split, offset, length, license and a sha256 of the retrieved text
into `corpus/sources.json`.

| Lane | Source | License |
|---|---|---|
| general_web | `HuggingFaceFW/fineweb-edu` sample-10BT | ODC-By-1.0 |
| code | `codeparrot/github-code-clean` all-all | per-row |
| indic (Tier A) | `wikimedia/wikipedia` 20231101.hi / .mr / .te | CC-BY-SA-4.0 |
| stem_math | `openai/gsm8k` main/train | MIT |
| reasoning | `open-thoughts/OpenThoughts-114k` | Apache-2.0 |
| agentic | `NousResearch/hermes-function-calling-v1` | Apache-2.0 |
| long_context | `emozilla/pg19` | Apache-2.0 |
| **test — never-train** | `cais/mmlu` all/test, `openai/gsm8k` main/test | MIT |
| **validation — non-gradient** | fineweb-edu and wikipedia-hi held-out row ranges | as above |

525 training documents → **24 admitted shards**, 560,071 tokens. Sangraha (S4's source) is not used
here: its rows API 500s and the parquet is 378 MB per language. Wikipedia hi/mr/te is what the S5
plan itself names for Tier A ("native news/Wikipedia").

**Tokenizer:** `sarvamai/sarvam-1`, frozen at
`bb5115a36ddb956a4ee0fd534e9870dd69157835622aec9c53062896f883c072`. `run_demo` re-hashes the file
and refuses to continue on a mismatch. Shards store true 68k sarvam ids — that is what manifests
and audits reference; the model softmaxes over the 2048 most frequent ids plus an UNK bucket
(77.2 % token coverage), and the remap table's sha256 is in `tokenizer_manifest.json` so the
scoping of every reported loss is stated rather than hidden.

Cleaning is deliberately **not** uniform: the ghost-tag strip runs on prose lanes only. Applying
it to code or agentic trajectories would eat legitimate `</s>`-like sequences and tool-call
markers — the S4 lesson that an English-prose-tuned cleaner destroys structured data.

## What the run demonstrates

**Admission gate — 3 real rejections, not staged ones.** `codeparrot/github-code-clean` carries a
per-row license, and the shard containing its `gpl-2.0` row is rejected as `unsafe_license`.

**Firewall — two deliberate attacks, both caught.** A never-train MMLU shard offered to the gate
is rejected on its `never_train_flag`; a real MMLU test item spliced into a fineweb document is
caught by 8-gram fingerprint overlap and the poisoned shard is rejected with its parent recorded.
A validation shard is separately refused as `validation_only_not_gradient_bearing`, and
`materialize` calls `firewall.assert_trainable` on every shard id at serve time, so a never-train
shard cannot reach a loss-bearing batch even if the gate is bypassed.

Fingerprints with a document frequency above 3 are dropped as boilerplate (9 of 23,692). Without
that, Hindi Wikipedia's reference and category furniture flagged three honest indic shards for
overlap they did not have — the false-positive side of the same trade-off S4 makes for cleaning.

**Mixture — S5's plan, compiled.** Five stages (seed → foundation → capability ramp →
long-context → anneal) with warmup-blended transitions, integrating to within 1.5 points of the
S5 headline mixture on every lane. Floors (indic 12 %, reasoning 4 %, agentic 2 %) are enforced
over rolling 10-step windows rather than per batch: at 8 sequences per step a 4 % floor would
round up to one whole sequence, i.e. 12.5 %, silently inflating the scarce lane it is meant to
merely protect. Six shards are quarantined for the anneal and **zero** are consumed before step
78. Realized lane shares match the compiled quotas to within 0.0001.

**Packing — the policy trade-off is measured, not asserted.** All six policies run over the same
data in `packed_batch_report.json`. For the code lane: `pad_only` 0.777 utilization, `best_fit`
0.989, `structure_preserving` 0.982 — the ~0.7 points structure safety costs. Structured lanes use
target-centric units: one unit per model turn, preceded by as much of its immediately preceding
context as the window allows, so the S5 masking rule holds (loss on the model's planning, tool
calls and answer; tool observations and user turns carried as context). Zero truncations across
the run; overall utilization 0.982.

Mask invariants, checked on **every** materialized batch (500 of them): 0 loss positions on
padding, 0 on context, 0 cross-segment visible attention pairs, 0 non-causal pairs, 0 position-id
violations.

**OPUS — 1,520 decisions with reasons.** Proxy is the model's own mean loss on the candidate's
loss-bearing tokens under the current checkpoint (high loss = high remaining learning value),
threshold at the 60th percentile of a rolling score buffer, so ~40 % is retained as S5 assumes.
Rejections carry reasons, near-misses in scarce lanes are *deferred* with a
`deferred_until_stage`, and floor-protected slots are rescued with
`protected_floor_override: true` — 14 of them, only ever on indic, reasoning and agentic.
Per-lane rejection rates are the honest asymmetry: general_web 0.47, stem_math 0.44 and code 0.42
against agentic 0.0 and reasoning 0.04, because the scarce lanes sit behind the floor.

The OPUS score buffer travels **in the checkpoint**. Without it, replay would restore the weights
but not the selector's state, and the decisions would drift.

**Crash → resume.** The run crashes deliberately after step 54; the last checkpoint is step 40, so
15 steps (60 microbatches) are in flight. Resume appends a `rollback` record superseding exactly
those 60 events — the append-only log is never rewritten — and re-serves from step 40 under
attempt 2. The step-40 batch hashes on attempt 2 are **identical** to attempt 1, and they match
`expected_next_batch.plan_digest`, which was written into the checkpoint *before* the crash. The
effective stream is 320 events over steps 0–79 with 0 gaps and 0 duplicates. Recovery itself took
0.035 s (checkpoint load + rollback + re-derivation); the training that follows is not recovery
cost.

**Replay.** Steps 20–40 re-run from `ckpt_step0020`: **80/80** batch hashes, token span ids and
loss-mask hashes identical to the original ledger events.

**Fork.** `fork-1` from `ckpt_step0040` with a new data-branch identity: every batch at the
divergence step differs from main, the divergence point and ledger offset are recorded, and
re-running the fork from the same checkpoint reproduces it exactly.

**Learning ledger.** 21 shard report cards + 7 lane roll-ups: average token loss, high-perplexity
clusters, loss delta between first and last exposure, gradient norm and gradient alignment (cosine
to the EMA direction), OPUS score, repeated-pass effect, model phase, and a
useful / neutral / harmful classification. The token trace uses §11's three tiers — full 17-field
rows for one flagged step, compressed rows for sampled steps, aggregates by shard / lane /
language / position bucket for the whole run.

**Throughput** (from the committed run; rates move with the machine, the counts do not).
8,436 raw tok/s → 3,552 accepted after OPUS → 3,336 useful loss-bearing tok/s.
The gaps are the point: raw includes the candidates OPUS rejected and the 30,720 positions the
crash discarded; useful is what actually bore loss. Cache hit rate 0.993 over 24 real shard reads
at 0.84 ms mean — measured, not estimated.

## Artifacts

Every file below is generated by the run. `manifests/shards/*.json` carry §6's 14 fields,
`consumption.jsonl` §8's 17, `opus_decisions.jsonl` §10's 11, `token_trace.jsonl` §11's 17,
`learning_ledger.jsonl` §12's 11, `eval_registry.json` §13's 6, `performance.json` §15's 10.

| Requirement (evidence row) | Artifact |
|---|---|
| Tokenizer integrity | `manifests/tokenizer_manifest.json`, `manifests/manifest_validation.json` |
| Evaluation firewall | `manifests/eval_registry.json`, `manifests/shards_index.json` |
| Packing correctness | `manifests/packed_batch_report.json` |
| Mixture compliance | `manifests/mixture_schedule.json`, `ledgers/mixture_compliance.json` |
| OPUS audit trail | `ledgers/opus_decisions.jsonl` |
| Crash recovery | `checkpoints/resume_proof.json`, `checkpoints/checkpoints_index.json` |
| Replay | `checkpoints/replay_proof.json`, `checkpoints/fork_proof.json` |
| Learning trace | `ledgers/learning_ledger.jsonl`, `ledgers/token_trace.jsonl`, `ledgers/token_stats.json` |
| Throughput | `performance.json` |

Plus `run.log` (the 13 mandated events in order, with `[PASS]` lines), `ledgers/audit.json`,
`ledgers/branches.json`, `ledgers/consumption.jsonl` and `shards/*.npz` (the shard payloads, kept
outside the bundle so the manifests can be revalidated against the bytes).

## Verifying it

**Execute.** Delete `submission_artifacts/` (leaving `corpus/` in place), re-run, and compare.
Every manifest, ledger, proof and checkpoint array is byte-identical across runs. The one
exception is the Throughput row of
`evidence.json`, which reports measured rates — its token counts and packing utilization are
identical, only the wall-clock-derived rate moves. (Checkpoint `.npz` files are zip archives with
embedded timestamps, so compare the arrays inside, not the file bytes.)

**Verify evidence.** `run.log` contains all 13 mandated events (checked by
`all_required_events_logged` at the end of the log) and `evidence.md` has the brief's 9 rows, each
naming a file, a JSON pointer, the recomputed value and the method used to get it. One event pair
is transposed against the brief's listing: `evaluation data blocked` precedes
`manifests validated`, because the admission verdict is itself a manifest field, so the firewall
and the gate have to settle before a manifest can be revalidated against its final content. The
log says so at the top rather than leaving it to be noticed.

**Inspect code.** Corrupt one `batch_hash` in `ledgers/consumption.jsonl` and re-run the evidence
builder alone:

```bash
.venv/Scripts/python -m tds.evidence submission_artifacts    # from s6-dataset-creation/
```

The Replay row turns FAIL on `proof_hashes_match_ledger` and the exit code is 1. Test 13 automates
exactly this.

## Honest limitations

- **Two ranks are simulated in-process.** Rank partitioning is real in the plan and the ledger
  (each microbatch records its rank), but there is no distributed loader, no real sharded IO
  across processes.
- **The model is a linear readout over a masked causal mean.** It has no learned QK attention, so
  the mask is *honoured* rather than *learned*; the `ponytail:` comment in `tds/train.py` names the
  upgrade path. Loss falls 7.62 → 5.00 over 80 steps (minimum 4.57, rising into the anneal stage as
  the mixture shifts), which is enough to make the learning ledger and the loss-spike audit real,
  and nothing more is claimed for it.
- **At `seq_len` 256 a long agentic trajectory keeps only the context nearest its model turn.**
  Left truncation is standard SFT treatment, but a 2,445-token tool preamble does not fit in a
  256-token window. Raising `seq_len` costs O(T²) in the attention mask.
- **Deduplication is exact-match only.** S4 already has the MinHash/LSH near-duplicate pass; this
  submission records `dedup_status` from content hashes and does not re-implement it.
- **Loss delta is measured between a shard's first and last exposure**, not against a held-out
  probe, so it mixes the shard's own effect with the run's overall progress. Stated in the field
  name rather than smoothed over.
- **The demo's token budget is 163,840 tokens.** Every supply lane reads "covered" at this scale —
  the scarcity verdicts in `mixture_schedule.json` are computed by S5's `demand ≤ supply` /
  `≤ 4×` rule and would flip to `needs_repetition` / `must_synthesize` at production scale, which
  is exactly the table S5 §3 argues about.
