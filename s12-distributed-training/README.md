# The Redundancy Tax

**ERA V5 · Session 12 submission.** Eight GPUs running data parallelism hold 3,576 GiB of
memory in order to store 447 GiB of information. ZeRO is the arithmetic of getting that back.

> **The assignment.** *Work with your agents and create a simple 32 virtual GPUs (can be your
> CPU threads or Colab GPU). Then write a demo model that runs on top of these. Simulate ZeRO,
> ZeRO1, ZeRO2, and ZeRO3. Show how the memory and computation changes.*

**Live page:** [`the-redundancy-tax.html`](the-redundancy-tax.html) · **Notebook:**
[`s12_distributed.ipynb`](s12_distributed.ipynb) (committed with its outputs)

```bash
python build_notebook.py               # .py -> executed .ipynb -> baked page   (~11 min, CPU)
python s12_distributed.py              # or just the harness, without rebuilding the notebook
python build_notebook.py --bake-only   # re-inject out/evidence.json into the page only
python check_page.py                   # does the page actually build in a browser? (needs Chrome)
```

`build_notebook.py` exits non-zero if any gate fails. That exit code is the pass/fail signal,
not the printed summary. **Current state: 47/47 gates pass**, in about 12.5 minutes of CPU.

---

## 1 · The position this submission takes on the word "simulate"

The assignment says *simulate*. I read that as permission, not as an instruction, and went the
other way: **nothing here is simulated.**

The 32 virtual GPUs are 32 real operating-system processes in a `gloo` process group, talking
over loopback TCP. The collectives are real sockets. Every byte reported as crossing the wire
was handed to `dist.isend` by code in this repository and counted at that line. Every byte of
memory reported is the sum of `untyped_storage().size()` over the tensors a rank was actually
holding.

The reason matters more than the choice. This session is a set of claims about **quantities** —
16 bytes per weight, then 5.50, 3.75, 2.00; 2P, 2P, 2P, 3P. A simulation that derives those
numbers from the same formula that predicted them has proved nothing at all; it is the formula
printing itself. So the rule for this submission is: *measure first, compare with the formula
afterwards, and report the gap when there is one.* There is a gap, three times, and all three are in §8
below.

---

## 2 · The arithmetic, in my own words

A weight is not one number. To train it you have to keep four things:

| what is kept for one weight | bytes | dtype in this harness |
|---|---|---|
| the weight, in the 16-bit format the arithmetic uses | 2 | `bfloat16` |
| its gradient | 2 | `bfloat16` |
| a 32-bit copy of the weight, kept for accuracy | 4 | `float32` |
| Adam's two running averages, m and v | 8 | `float32` × 2 |
| **total** | **16** | |

Those dtypes are not a description of the model — they *are* the model. `s12_ranks.Engine`
holds `bfloat16` parameters and gradients with a `float32` master copy and `float32` moments,
so the sixteen bytes are bytes that get allocated.

30 × 10⁹ weights × 16 bytes = 480 GB = **447 GiB**, which is six 80 GB cards before a single
calculation. That is the whole problem, and everything below is a way of not paying it
thirty-two times over.

**Why the 32-bit copy exists** is worth stating because I ended up measuring it by accident.
Repeatedly adding very small updates to a 16-bit number loses them to rounding. The harness
sums the integers 1…32 around its own ring twice, in `float32` and in `bfloat16`:

```
exact          528
float32 ring   528
bfloat16 ring  524      ← off by 4, or 0.76%
```

A weight update is exactly that kind of accumulation, repeated every step for the length of
training. Keep the authoritative value at 32 bits, make a 16-bit copy for the arithmetic, and
the error stops compounding. Twelve of the sixteen bytes exist for that reason, which is also
why ZeRO-1 — which splits only those twelve — gets most of the saving for none of the cost.

---

## 3 · Data parallelism, and the two things it promises

Before any of ZeRO, the arrangement it improves on. Every GPU gets a complete copy of the model
and a different portion of the data; the gradients differ because each GPU saw different text;
they are averaged, and the same average is applied everywhere. Two promises fall out of that,
and I checked both rather than assuming them.

**One: the copies stay identical.** They started identical and applied an identical update, so
they must remain identical — with no drift, ever. This is what the whole arrangement rests on,
because the moment two ranks disagree about the weights they are no longer training one model.
Every rank hashes the entire model it can see, so the check is whether 32 hex strings are the
same string. After 12 steps: **one distinct hash across all 32 ranks** (`9673f067dd…`).

**Two: the global batch is a product, and only the product matters.**

> global batch = sequences per GPU × GPUs × accumulation steps

Accumulation steps are repeats of the forward and backward pass before the weights move, used
when the batch you want is larger than what fits at once. If that formula is real, any
factorization of the same global batch is the same training run. So I ran the same 32 sequences
two ways:

| | seq/GPU | GPUs | accum | global batch | final loss |
|---|---:|---:|---:|---:|---:|
| this submission | 1 | 32 | 1 | 32 | 5.870162 |
| the previous production run's shape | 2 | 8 | 2 | 32 | 5.870005 |

Relative difference **2.7e-05**, largest weight difference 3.9e-03. Not bitwise, and it should not be: 32 ranks summing one
sequence each around a ring and 8 ranks summing four apiece are different orders of the same
addition. The point is that *the arrangement of the hardware does not change what the model
learns* — only the product does. That is what makes a recipe portable from 8 GPUs to 32, and it
is also why the second row was worth running: it is exactly the configuration §12 of the notes
records the previous run using.

---

## 4 · What I had to build, and why it turned out to be the useful part

The first thing I tried did not exist:

```
dist.reduce_scatter(out, ins)
RuntimeError: ProcessGroupGloo does not support reduce_scatter
```

`gloo` is the CPU backend — the only one available without GPUs — and it implements
`all_reduce` and `all_gather` but **not `reduce_scatter`**, which is the one operation both
ZeRO-2 and ZeRO-3 are built on.

This was the most useful accident in the assignment. Writing the ring by hand on
`isend`/`irecv` meant every byte crossing a socket passes through fourteen lines of my own
code, past a counter. That is the only reason the communication figures here are *measured*.
Had `reduce_scatter` worked, I would have had to take the library's word for the traffic, or
quote the notes back at them.

The ring reduce-scatter is `world - 1` hops. At each hop a rank sends one chunk to its right
neighbour and adds the chunk arriving from its left:

```python
for step in range(self.world - 1):
    send_idx = (self.rank - step - 1) % self.world
    recv_idx = (self.rank - step - 2) % self.world
    self._exchange(chunks[send_idx], tmp)
    chunks[recv_idx] += tmp
```

**That `- 1` offset is a bug I had to find.** The textbook indexing (`send_idx = rank - step`)
is correct as an algorithm and leaves rank *r* owning chunk *r+1*. Everything else in the
engine — the optimizer shards, the master copy, the gather — assumes rank *r* owns chunk *r*.
The mismatch does not crash. It silently trains the wrong slice, and the loss still goes down,
because averaging the wrong permutation of gradients is still averaging something. It showed up
only as rank 0 holding a partially-reduced chunk when I traced four ranks by hand. The lesson I
take from it: in distributed code the failure mode is not a crash, it is a plausible number.

The ring is checked against two independent references before anything uses it:

1. **Against `gloo` itself.** Hand-written `all_reduce` vs `dist.all_reduce`, relative
   difference **2.9e-07** — `float32` rounding, not a wrong algorithm. They cannot be bitwise
   equal, because floating-point addition is not associative and the two use different orders.
2. **Against the session's own claim.** *A reduce-scatter followed by an all-gather is an
   all-reduce.* Both sides run through the same ring in the same order, so here the standard is
   **exact bitwise equality**, and anything less would be a bug.

---

## 5 · The three stages, and how each one is actually implemented

### ZeRO-1 — split the optimizer state

Twelve of the sixteen bytes are the master copy and Adam's two moments. During the update every
rank applies the same average to the same weights and gets the same answer, so thirty-one of
thirty-two are repeating work already being done. Give each rank one slice to own: it
reduce-scatters the gradients, keeps only its own slice of the result, updates only its slice
of the master state, and then all-gathers the updated weights so everyone has the full model
again.

`2 + 2 + 12/N`. At 32 GPUs, **4.375 bytes per weight**.

### ZeRO-2 — split the gradients too

A rank only ever needs the gradients for the slice it is responsible for updating. So instead
of holding the whole gradient buffer until the end of backward, each gradient is reduced to its
owner *the moment backward finishes producing it* and discarded everywhere else. In the engine
that is a `register_post_accumulate_grad_hook` per parameter:

```python
def hook(p):
    self._reduce_one(name, p)     # ring reduce-scatter; keep my slice
    p.grad = None                 # discarded as soon as it has been sent where it is needed
```

`2 + (2 + 12)/N`. At 32 GPUs, **2.4375 bytes per weight**.

### ZeRO-3 — split the weights themselves

Each rank stores one slice of the model. When the forward pass reaches a layer, the ranks
collect that layer's weights from each other, use them, and discard them again; on the way back
they collect them a second time.

This is the part that took real work, and three things about it are not obvious:

**(a) Freeing the weight has to free the storage, not the tensor.** Setting `p.data` to an
empty tensor breaks backward, because autograd saved the parameter *object* for the matrix
multiply's backward and reads it later. What works is what FSDP does: keep the same tensor and
resize its storage to zero bytes, then resize it back and refill before backward needs it.

```python
p.untyped_storage().resize_(0)                              # release
p.untyped_storage().resize_(p.numel() * p.element_size())   # re-gather
p.data.copy_(flat[:p.numel()].view_as(p))
```

This is also what makes ZeRO-3's memory *measurable*: a released weight is still a live tensor,
but its storage reports zero bytes, so summing storage sizes gives the truth for free.

**(b) The refill has to write through `.data`.** A plain `p.copy_(src)` ticks the autograd
version counter, and backward then refuses the tensor it saved:
`one of the variables needed for gradient computation has been modified by an inplace
operation`. Writing through `p.data` does the same memory write without the tick.

**(c) `register_full_backward_hook` fires too early on a residual block, and this one cost me
an afternoon.** The post-backward hook fires when the gradient *with respect to the module's
input* is complete. In a block with a skip connection, `x + f(x)`, the input gradient is
complete as soon as the skip path delivers it — which is **before** the block's own weights
have been touched. Releasing there frees weights that backward still needs, and the error you
get is `storage size 0`, several layers away from the cause. The fix is to release on a count
instead: a unit is discarded when the last of its parameters has received a gradient. That is
also what real implementations do, and now I know why.

`(2 + 2 + 12)/N` — the whole sixteen bytes divided. At 32 GPUs, **0.5 bytes per weight**.

---

## 6 · Results — memory

Four runs on 32 processes. Same model, same seed, same tokens in the same order; the only thing
that changes is what each rank is allowed to keep. The *measured* column is the sum of storage
sizes a rank was holding at the instant `backward()` returned.

| arrangement | measured | the ledger says | peak | 30B per GPU | load balance |
|---|---:|---:|---:|---:|---:|
| data parallelism | **16.0000** | 16.0000 ✓ | 16.61 | 447.0 GiB | 1.0000× |
| ZeRO-1 | **4.3750** | 4.3750 ✓ | 4.69 | 122.2 GiB | 1.0000× |
| ZeRO-2 | **2.4375** | 2.4375 ✓ | 3.05 | 68.1 GiB | 1.0000× |
| ZeRO-3 | **0.5000** | 0.5000 ✓ | 1.41 | 14.0 GiB | 1.0000× |

The load-balance column is the heaviest rank over the lightest. At 1.0000× it says every rank
received a different slice and none is carrying more than its share — the part of the idea that
is easy to state and easy to get wrong.

**Two numbers per stage, because the notes' table has one.** The *ladder* is the state a rank
holds with no collective in flight. The *peak* is that plus the largest buffer a single
collective needs. No implementation escapes the second one, and it is worth naming that the gap
is wider here than it would be on a real model: the buffer is sized by the largest single
tensor, and in a 2.6M-parameter model with a 2,048-token vocabulary the embedding is a far
bigger share of the whole than it is at 30B. **The gap is an artifact of the toy. The ladder is
not.**

---

## 7 · Results — communication

| arrangement | measured | the ring predicts | the notes say | 30B, per GPU per step |
|---|---:|---:|---:|---:|
| data parallelism | **1.938P** | 1.938P ✓ | 2P | 116 GB |
| ZeRO-1 | **1.938P** | 1.938P ✓ | 2P | 116 GB |
| ZeRO-2 | **1.938P** | 1.938P ✓ | 2P | 116 GB |
| ZeRO-3 | **2.906P** | 2.906P ✓ | 3P | 174 GB |

**Stages 1 and 2 really are free.** ZeRO-1 and ZeRO-2 move the same bytes as plain data
parallelism while holding 6.6× less state, and the reason is the equivalence from §3: data
parallelism's all-reduce already *is* a reduce-scatter followed by an all-gather. Stages 1 and
2 run those same two phases and simply keep the intermediate slice instead of discarding it.
They pay nothing because they were already doing the work.

ZeRO-3 costs exactly 1.5× — it adds a gather of the weights on the way forward and another on
the way back — and buys 32× less state for it.

---

## 8 · Three places where the measurement corrected me

This is the section I would keep if I had to throw the rest away.

### A ring does not move P. It moves P·(N−1)/N.

Each rank sends every chunk but its own: N−1 of the N. At 32 GPUs that is 0.969P, so data
parallelism's real cost is **1.94P, not 2P**, and ZeRO-3's is **2.91P, not 3P**. The notes' 2P
and 3P are the large-N limit of this. At 32 GPUs the limit is 3% away; at 4 GPUs it is 25%
away, which is the sort of error that would swallow a real comparison between two small runs.

I would rather report 1.94P and explain it than report 2P and be quietly wrong. Note that the
*ratio* between stages is unaffected — ZeRO-3 is 1.5× data parallelism at every world size,
because the (N−1)/N factor divides out — which is why the notes can round it away safely for
the conclusion they draw, and why I cannot for the number I print.

### Rank 0's loss is not the run's loss.

My first equivalence check reported the 32-GPU run at loss 7.17 against the single-GPU run's
5.87 and I nearly went looking for a bug in the sharding. There was no bug. Each rank trains on
a different sequence, so rank 0's loss is the loss on one thirty-second of the batch — and at
the last step the 32 ranks report losses **4.07 apart** from each other. The run's loss is the
*mean* of them, and taking that mean is the same averaging the all-reduce performs on the
gradient. Averaged properly: **5.870162** against **5.870350**.

It is a small mistake and an instructive one. The distributed run does not have a loss until
you decide to average it, in the same way it does not have a gradient until the all-reduce.

### The cost is not the bytes. It is the messages.

I spent the whole of §7 counting bytes, because the session's units are multiples of P and P is
a quantity of bytes. Then §11's bucketing run moved **exactly the same 10.0 MB per rank per
step** and spent **8.2× less time doing it** — 1.439s down to 0.175s — purely by sending it as
62 messages instead of 1,984.

So "2P on the wire" is a statement about volume, and volume is only half of what a step pays
for. The other half is the fixed cost of starting a transfer, 0.73 ms per send here, which the
multiples-of-P framing does not see at all. This reframed overlap and bucketing for me: they are
not tuning knobs bolted onto a finished design, they are the reason the design's own cost model
is incomplete without them.

---

## 9 · Results — is it still the same training run?

A memory saving that changed the answer would be worthless. Two claims, two different standards
of proof:

**Across the stages, the standard is bitwise.** All four run the identical ring in the identical
order on identical data; only the question of which rank keeps which slice differs. Every
arithmetic operation therefore sees the same inputs in the same order, so the final weights must
agree to the last bit. A tolerance here would be hiding something.

| | vs data parallelism | max weight difference |
|---|---|---:|
| ZeRO-1 | ✓ bit-identical | 0.0 |
| ZeRO-2 | ✓ bit-identical | 0.0 |
| ZeRO-3 | ✓ bit-identical | 0.0 |

Loss curves are identical across all four stages for all 12 steps.

**Against a single GPU, the standard is a tolerance, and its size is the finding.** The session
says a distributed run is *mathematically* identical to a single-GPU run on a batch 32 times
larger. Mathematically, yes. In floating point, no — one rank computing the gradient of 32
sequences in one matrix multiply, and 32 ranks computing one sequence each and summing around a
`bfloat16` ring, are different summation orders of the same quantity. Measured against a
one-rank run on the same 32 sequences: final loss **5.870162 vs 5.870350** (relative 3.2e-05),
largest weight difference **2.4e-03**, which is 2.4e-03 of the largest weight and exactly
`bfloat16`'s resolution. Small, and not zero.

---

## 10 · Results — computation, and the memory wall

| arrangement | step | compute | communication | comm / compute |
|---|---:|---:|---:|---:|
| data parallelism | 4.213s | 2.774s | 1.439s | 52% |
| ZeRO-1 | 5.198s | 3.520s | 1.678s | 48% |
| ZeRO-2 | 5.412s | 2.757s | 2.655s | 96% |
| ZeRO-3 | 6.109s | 2.726s | 3.384s | 124% |

**Read the ordering, not the seconds.** Thirty-two processes are contending for 10 physical
cores, so the absolute times describe this laptop. §11 also shows most of this column is message
count rather than volume. What transfers is that ZeRO-3 moves 1.5× the
bytes and spends 2.35× the time doing it, and that none of it is overlapped with the backward
pass — which is the entire answer to this column and is scoped out here (§11).

The memory wall, from measured bytes-per-parameter multiplied out to 30B:

| GiB per GPU | 4 GPUs | 8 | 16 | 32 | |
|---|---:|---:|---:|---:|---|
| data parallelism | 447.0 | 447.0 | 447.0 | 447.0 | never fits |
| ZeRO-1 | 195.6 | 153.7 | 132.7 | 122.2 | never fits |
| ZeRO-2 | 153.7 | 104.8 | 80.3 | 68.1 | fits from 32 GPUs |
| ZeRO-3 | 111.8 | 55.9 | 27.9 | 14.0 | fits from 8 GPUs |

A card holds 74.5 GiB. These reproduce the session's table exactly, from measurements rather
than from the formula. The projection is a ratio argument, and the assumption it rests on —
that bytes-per-parameter is independent of parameter count — is exactly what the 4/8/16/32
sweep checks.

**The floor.** Data parallelism and ZeRO-1 both leave the weight and the gradient — four bytes —
replicated on every card at every world size. Four bytes across 30 billion weights is 111.8 GiB
whether you have one card or a thousand, and four bytes fills a 74.5 GiB card exactly at **20
billion parameters**. A 30B model is past that line, so no GPU count rescues those two
arrangements. That is the real content of the wall: not that ZeRO-1 is inefficient, but that it
is *unavailable*.

---

## 11 · The rest of the session, and what I could and could not measure of it

Five topics remain from the notes. Two of them I found a way to measure; three are arithmetic,
and the README says so rather than letting them read like results.

### Bucketing, measured — and the reason overlap exists

Everything above reduces **one tensor at a time**: each parameter tensor gets its own ring, and
a ring across 32 ranks is 31 hops each way. That is 1,984 separate sends per step of about 5 KB
each. The notes say gradients are instead collected into *buckets*, and that the bucket size
sets a balance. Rather than quote that, I added a mode that takes it to the limit — every
gradient in the model packed into one buffer and reduced once — and measured both ends of the
balance:

| | messages/step | bytes/message | waiting | step | peak B/param |
|---|---:|---:|---:|---:|---:|
| one ring per tensor | 1,984 | 5.0 KB | 1.439s | 4.213s | 16.61 |
| one ring for everything | 62 | 161 KB | **0.175s** | 2.980s | **20.00** |

**Identical bytes on the wire** — 10.0 MB per rank per step either way — 32× fewer messages, and
**8.2× less time waiting**. Effective throughput goes from 7.0 MB/s to 57.2 MB/s on the same
link, which is the whole finding: the small-message case was never moving bytes, it was paying
0.73 ms of fixed cost per send. That is *why* bucketing exists, derived from my own numbers
instead of taken on faith.

And the other side, which the notes state and this makes concrete: one buffer holding every
gradient is a second copy of them, so peak memory goes from 16.61 to 20.00 bytes per weight
while the state actually held is unchanged at 16.00. **A bucket buys latency with memory**,
which is exactly why production settings are a few hundred megabytes and not "everything".
Bucketing also changes the summation order, so it is not bitwise identical to the per-tensor
run — largest weight difference 3.9e-03. Grouping tensors is a numerical choice as well as a
performance one.

This is also the argument for **overlap**, which I did not implement. The backward pass runs
last layer to first, so the last layer's gradients are done long before the pass reaches the
first and can start moving immediately; a bucket is sent as soon as it fills, part way through.
Smaller buckets give more room to overlap, larger ones amortise the fixed cost measured above,
and on fast hardware there is a third effect — past some point the transfers queue behind each
other and step time climbs again. ZeRO-3 needs the same treatment in reverse, prefetching the
next layer's weights. **Not implemented, and the first thing I would add.**

### What this costs on hardware that is not a laptop — arithmetic, not measurement

My seconds describe 32 processes on 10 cores. The useful question is whether the transfer is
small compared to the work on real hardware:

| path | bandwidth | 2P = 120 GB | 3P = 180 GB |
|---|---:|---:|---:|
| NVLink, inside one node | 450 GB/s | 0.27s | 0.40s |
| InfiniBand, between nodes | 50 GB/s | 2.40s | 3.60s |

| card | compute/step | 2P on InfiniBand | comm / compute |
|---|---:|---:|---:|
| 64 × H100 | 7.10s | 2.40s | **34%** |
| 64 × B200 | 3.12s | 2.40s | **77%** |

The volume does not move — 120 GB crosses either way. What moves is the compute it has to hide
behind, and a B200 step is less than half an H100 step. **Faster GPUs raise the ratio of
communication to compute.** Both are still below 1, so a fully overlapped transfer is still
fully hidden, and that margin is the entire reason overlap stops being an optimization and
becomes a requirement. The nine-fold gap between the two links is also the whole content of the
"how many GPUs per node" question below.

### Offload — arithmetic, not measurement

The optimizer state is the natural thing to move to system memory: twelve of the sixteen bytes,
touched once per step. Two versions — park it there and copy it to the GPU for the update, or
run the update on the CPU too, so the state never moves. The cost is PCIe at ~60 GB/s, far below
NVLink and comparable to a network cable, so **offload converts a memory problem into a
bandwidth problem** and only pays when memory is the binding constraint.

Per step per GPU at 30B on 32 GPUs: the CPU-side update moves **3.75 GB** over PCIe — the
gradient shard out, the updated weight shard back — which is 0.06s. The GPU-side one moves
**22.50 GB**, the whole twelve-byte state shard in both directions, which is 0.38s. **6× less**,
because on the CPU side the twelve bytes never leave system memory at all. That is why the
CPU-side version is the one that earns its place. Not implemented here.

### DeepSpeed and FSDP2 — reference

ZeRO is the design; these are two implementations of it. DeepSpeed is Microsoft's, released
alongside the paper, configured by a single JSON file, with the most complete offload support
and all three stages. FSDP2 is PyTorch's own, the recommended path from 2.6 onward, and it
corresponds to stage 3 — applied with `fully_shard()` on parts of the model, with each parameter
split along its first dimension as a DTensor that knows which part of itself lives where.

Worth naming the overlap with §5: my hand-rolled ZeRO-3 hit exactly the problem FSDP2 solves
with storage resizing. A released weight has to stay the *same tensor object* with a zero-byte
storage, because autograd saved that object and will read it again in the backward pass. I found
that out by breaking it.

### What V4 ran, and the four questions still open — reference

The previous run, LightningLM v0.1, used DeepSpeed at ZeRO stage 2 on 8 GPUs: bf16,
`overlap_comm` enabled, `round_robin_gradients` enabled as an out-of-memory fix (it is in the
config's filename), 2×10⁸-byte buckets, weight decay deliberately 0.0, peak LR 3e-4 with 500
warmup steps. **Its global batch of 32 came from 2 sequences × 8 GPUs × 2 accumulation steps —
the second row of §3's table**, which is why that comparison was worth running.

It never needed stage 3. At 30B, §10 says that is no longer a choice. Four questions stay open,
and what this harness can honestly say about each:

| question | what would settle it | what this gives |
|---|---|---|
| ZeRO-2 on 32 GPUs, or ZeRO-3 on 8? | measured step time for both on the real architecture, activations included | the memory side exactly — 68.1 GiB vs 14.0 GiB per GPU — and the 1.50× communication ratio. Not step time on hardware I do not have, and activations are excluded. |
| How many GPUs per node, how many nodes? | how much traffic can be kept inside a node | nothing: one loopback link, no node boundary. The 9× above is the whole argument. |
| Is 8-bit committed from the start? | a short bf16 vs MXFP8 run on the same architecture | the storage arithmetic (12.1%) and nothing else. bf16 is as low as this CPU goes. |
| Does any state go to system memory? | whether the run is memory- or communication-bound | the PCIe arithmetic above only. |

---

## 12 · The 8-bit ledger

| | bytes per weight | 30B model |
|---|---:|---:|
| 16-bit weights and gradients | 16.0000 | 447.0 GiB |
| 8-bit weights and gradients (MXFP8) | 14.0625 | 392.9 GiB |
| saving | 1.9375 | **12.1%** |

Two bytes out of sixteen, less the 0.0625 bytes per parameter the per-32-block scales add back
across the two tensors. The twelve bytes of master copy and moments are untouched, because the
update arithmetic still needs the accuracy.

This section is computed, not measured, and that is a deliberate line. `bfloat16` is as low as
this CPU goes, and claiming to have measured MXFP8 on hardware that cannot multiply it would be
exactly the thing the rest of this submission exists to avoid. The real contribution of 8-bit is
elsewhere anyway: faster matrix multiplication, smaller activations, and **every multiple of P
on this page halves**, because P is the parameters in the compute format.

---

## 13 · Scope limits, stated rather than buried

* **The model is small.** Four layers, `d_model` 192, vocabulary capped to the top 2,048
  Sarvam-1 ids by frequency in the S6 corpus (76% of token occurrences) — 2,582,208 parameters.
  It has to be: 32 processes hold 32 copies of everything under data parallelism, on one laptop.
  Nothing measured depends on the size, because bytes *per parameter* and traffic *in multiples
  of P* are both ratios — but the peak-vs-ladder gap in §5 does, and is flagged there.
* **12 steps per stage.** Enough to show the loss falling and the stages agreeing bitwise; not a
  training run.
* **Offload to CPU and NVMe is not implemented** — §11 is its arithmetic only.
* **Communication is not overlapped with compute.** §11 measures the bucketing half of the
  problem and stops there. Given §11's hardware numbers this is the single change that would
  most improve the run, and **it is the first thing I would add**. Left out rather than
  half-done.
* **The hardware and precision sections are arithmetic, not measurement**, and are labelled as
  such everywhere they appear — on the page with an explicit marker. I do not have an H100, a
  B200, or a card that can multiply MXFP8.
* **Gradient accumulation is implemented for data parallelism only.** ZeRO-2 and ZeRO-3 reduce
  each gradient from a backward hook the instant it appears, so accumulating across micro-batches
  needs that reduce deferred to the final one — which is what real implementations spend a
  `no_sync` context on. §3's formula is a statement about data parallelism, so this does not
  weaken the measurement, but it is a real limit of the engine.
* **Absolute timings describe this laptop**, with 32 processes on 10 cores.

---

## 14 · Files

| file | what it is |
|---|---|
| `s12_ranks.py` | the rank worker: the model, the hand-written ring collectives, and the four stages. A module of its own because `mp.spawn` pickles by module path, and a function defined in a notebook cell has no module path to pickle. |
| `s12_distributed.py` | the harness and source of truth (`# %%` cell-delimited): launches every configuration, collects the evidence, runs the gates. |
| `s12_distributed.ipynb` | build artifact, committed **with its outputs** so a reviewer sees the numbers without running anything. |
| `the-redundancy-tax.html` | the page, with `out/evidence.json` baked in as a literal JS blob. |
| `build_notebook.py` | `.py` → executed `.ipynb` → baked page. Non-zero exit on any failed gate. |
| `check_page.py` | renders the page in headless Chrome and asserts it actually built. |
| `out/evidence.json` | every number on the page. |
| `out/runs/` | per-rank scratch, ~175 MB, gitignored — regenerated byte-for-byte on every run. |

Sections 1–2 of `s12_distributed.py` also verify the frozen Sarvam-1 tokenizer
(sha256 `bb5115a3…`) and read S6's committed corpus, so this session is continuous with
Sessions 9, 10 and 11 and needs no network beyond the tokenizer's first download.

---

## 15 · Gates

All 47 are asserted by the harness; `build_notebook.py` returns non-zero if any fails.

**The collectives** — the hand-written ring agrees with `gloo`; reduce-scatter + all-gather is
bitwise an all-reduce; the `bfloat16` ring loses bits and the `float32` ring does not.

**Memory** — each of the four stages matches its formula to within 0.1%; the ladder is
monotonic; every rank holds an equal share.

**Communication** — each stage matches `phases · P · (N−1)/N` to within 0.1%; ZeRO-1 and ZeRO-2
cost exactly what data parallelism costs; ZeRO-3 costs exactly 1.5× that.

**Correctness** — all three ZeRO stages are bitwise identical to data parallelism, with
identical loss curves; the 32-GPU run matches the 1-GPU run within `bfloat16` error, in weights
and in loss; the model actually trained.

**The wall** — ZeRO-3 scales as 16/N at every world size; data parallelism and ZeRO-1 never fit
a card; ZeRO-2 fits from 32 GPUs; ZeRO-3 from 8.

**Data parallelism** — the 32 replicas hash to one value; the two factorizations of the same
global batch agree.

**Bucketing** — one bucket moves the same bytes as per-tensor reduction, in >10× fewer messages,
in less time, for more peak memory, with the state held unchanged.

**The hardware arithmetic** — the faster card raises the communication ratio; both ratios stay
below 1; the interconnect gap is 9×; NVLink carries 2P in under a third of a second.

**Offload** — the CPU-side update moves less over PCIe than the GPU-side one.

**Precision** — the MXFP8 ledger is 14.0625 bytes per weight and saves 12.1%.
