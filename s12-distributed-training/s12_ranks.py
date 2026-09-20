"""The rank worker: what one of the 32 virtual GPUs actually runs.

This is a module of its own, and that is not a style choice. `mp.spawn` pickles the worker
by module path, so a function defined in a notebook cell lives in the kernel's `__main__`
and cannot be sent to a spawned process. The harness (`s12_distributed.py`, and therefore
the notebook built from it) imports this file instead.

Nothing here is simulated. Each rank is a real OS process, the collectives are real sockets,
and every byte counted in `Ring.bytes_sent` was handed to `dist.isend`.
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

BF16 = torch.bfloat16
FP32 = torch.float32

# The 16-byte ledger of §1, as the dtypes that produce it:
#   weight 2 (bf16) · gradient 2 (bf16) · master copy 4 · Adam m 4 · Adam v 4
MASTER_BYTES_PER_PARAM = 12          # master + m + v, the part ZeRO-1 shards
COMPUTE_BYTES_PER_PARAM = 2          # the bf16 weight, the part only ZeRO-3 shards
GRAD_BYTES_PER_PARAM = 2             # the bf16 gradient, the part ZeRO-2 shards


# --------------------------------------------------------------------------------------
# The model. Copied from S9/S10/S11 rather than imported — each session ships standalone —
# with the muP switch dropped, since nothing here changes width.
# --------------------------------------------------------------------------------------
class Config:
    def __init__(self, **kw):
        self.vocab_size, self.d_model, self.n_layer, self.d_head = 2048, 192, 4, 64
        self.block_size = 128
        self.embed_std = 0.02
        self.__dict__.update(kw)
        self.n_head = max(1, self.d_model // self.d_head)
        self.d_ff = round(self.d_model * 8 / 3 / 64) * 64


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.g


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.d_head = cfg.n_head, cfg.d_head
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.n_head * cfg.d_head, bias=False)
        self.proj = nn.Linear(cfg.n_head * cfg.d_head, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        H = self.n_head * self.d_head
        q, k, v = self.qkv(self.n1(x)).split(H, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, H))
        y = self.n2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 1.0 / math.sqrt(m.weight.shape[1]))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, cfg.embed_std)

    def forward(self, idx):
        x = self.embed(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm_f(x))

    def units(self):
        """The granularity at which ZeRO-3 gathers weights: §6's 'when the forward pass
        reaches a layer, the GPUs collect that layer's weights from each other'."""
        return ([("embed", self.embed), ("pos", self.pos)]
                + [(f"block{i}", b) for i, b in enumerate(self.blocks)]
                + [("norm_f", self.norm_f), ("head", self.head)])


# --------------------------------------------------------------------------------------
# Collectives, written out by hand and counting their own bytes.
#
# gloo has no reduce_scatter ("ProcessGroupGloo does not support reduce_scatter"), which is
# the operation ZeRO-2 and ZeRO-3 are built on. So the ring from §4 is written here on
# isend/irecv. That is the reason the communication figures in this submission are measured
# rather than quoted: every byte passes through this class.
# --------------------------------------------------------------------------------------
class Ring:
    def __init__(self, rank, world):
        self.rank, self.world = rank, world
        self.right, self.left = (rank + 1) % world, (rank - 1) % world
        self.bytes_sent = 0
        self.msgs = 0            # §10: the fixed cost of a transfer is paid per message
        self.seconds = 0.0
        self.live = []           # transient buffers, so the memory probe can see them
        self.probe = lambda: None

    def _drop(self, t):
        """Remove by identity. `list.remove` compares with `==`, which on tensors is a
        broadcast comparison, not a search."""
        self.live = [x for x in self.live if x is not t]

    def _exchange(self, send, recv):
        """One hop around the ring: send right, receive from left, at the same time."""
        self.bytes_sent += send.numel() * send.element_size()
        self.msgs += 1
        reqs = [dist.isend(send.contiguous(), self.right), dist.irecv(recv, self.left)]
        for r in reqs:
            r.wait()

    def reduce_scatter(self, flat):
        """`flat` is world-divisible and identical in shape on every rank. Returns this
        rank's chunk of the element-wise sum. Sends (world-1)/world of `flat` — one P."""
        if self.world == 1:
            return flat.clone()
        t0 = time.perf_counter()
        chunks = list(flat.chunk(self.world))
        tmp = torch.empty_like(chunks[0])
        self.live.append(tmp)
        for step in range(self.world - 1):
            # offset by -1 so that after world-1 hops rank r owns chunk r exactly; the
            # textbook indexing leaves rank r owning chunk r+1, which silently mismatches
            # the shard layout everything else here uses.
            send_idx = (self.rank - step - 1) % self.world
            recv_idx = (self.rank - step - 2) % self.world
            self._exchange(chunks[send_idx], tmp)
            chunks[recv_idx] += tmp
            self.probe()
        self._drop(tmp)
        self.seconds += time.perf_counter() - t0
        return chunks[self.rank].clone()

    def all_gather(self, chunk):
        """The reverse. Each rank starts with its own chunk and ends with all of them."""
        if self.world == 1:
            return chunk.clone()
        t0 = time.perf_counter()
        flat = torch.empty(chunk.numel() * self.world, dtype=chunk.dtype)
        self.live.append(flat)
        chunks = list(flat.chunk(self.world))
        chunks[self.rank].copy_(chunk)
        for step in range(self.world - 1):
            send_idx = (self.rank - step) % self.world
            recv_idx = (self.rank - step - 1) % self.world
            self._exchange(chunks[send_idx], chunks[recv_idx])
            self.probe()
        self._drop(flat)
        self.seconds += time.perf_counter() - t0
        return flat

    def all_reduce(self, flat):
        """§4: a reduce-scatter followed by an all-gather is an all-reduce, and the pair
        costs the same as the whole. This function is that sentence, executed."""
        return self.all_gather(self.reduce_scatter(flat))


# --------------------------------------------------------------------------------------
# One rank's share of a training run.
# --------------------------------------------------------------------------------------
def _pad_to(n, world):
    return ((n + world - 1) // world) * world


class Engine:
    """Holds whatever the chosen stage says this rank is responsible for, and nothing else.

    mode is one of: dp, zero1, zero2, zero3.
      dp     — every rank holds all 16 bytes per weight, reduced one tensor at a time
      dp_flat— the same, but every gradient is packed into a single buffer and reduced once,
               which is §10's bucketing taken to its limit
      zero1  — the 12 optimizer bytes are split
      zero2  — the gradient is split as well, reduced away the moment it is finished
      zero3  — the weight is split too, gathered per layer and discarded after use
    """

    def __init__(self, mode, cfg, rank, world, ring, lr=3e-4, betas=(0.9, 0.95), eps=1e-8):
        self.mode, self.rank, self.world, self.ring = mode, rank, world, ring
        self.dp_like = mode in ("dp", "dp_flat")     # holds all 16 bytes; differs only in bucketing
        self.lr, self.betas, self.eps = lr, betas, eps
        self.t = 0
        ring.probe = self.mark

        torch.manual_seed(1337)               # every rank starts from the same weights
        self.model = Model(cfg).to(BF16)
        self.named = list(self.model.named_parameters())

        self.shard_of = {}                    # name -> (padded numel, shard numel)
        self.master, self.m, self.v = {}, {}, {}
        self.gshard = {}                      # name -> this rank's slice of the summed grad
        self.pshard = {}                      # name -> this rank's slice of the weight (ZeRO-3)

        for name, p in self.named:
            npad = _pad_to(p.numel(), world)
            s = npad // world
            self.shard_of[name] = (npad, s)
            flat32 = self._padded(p.detach().float(), npad)
            if self.dp_like:
                self.master[name] = flat32.clone()
                self.m[name] = torch.zeros(npad, dtype=FP32)
                self.v[name] = torch.zeros(npad, dtype=FP32)
            else:
                lo = rank * s
                self.master[name] = flat32[lo:lo + s].clone()
                self.m[name] = torch.zeros(s, dtype=FP32)
                self.v[name] = torch.zeros(s, dtype=FP32)

        self.params_bytes_full = sum(_pad_to(p.numel(), world) for _, p in self.named) * 2
        self.peak = 0
        self.after_backward = 0
        self.gathered = {}                    # ZeRO-3: which units are resident right now
        self._unit_state = {}
        if mode == "zero3":
            self._install_zero3()
        if mode in ("zero2", "zero3"):
            self._install_grad_hooks()

    # -- helpers ------------------------------------------------------------------------
    @staticmethod
    def _padded(t, npad):
        flat = torch.zeros(npad, dtype=t.dtype)
        flat[:t.numel()] = t.reshape(-1)
        return flat

    def resident_bytes(self):
        """Training state this rank is holding, in bytes actually allocated.

        Reading `untyped_storage().size()` rather than `numel()*element_size()` is what makes
        ZeRO-3 measurable: a released weight is a live tensor whose storage is zero bytes.
        Activations are deliberately excluded — §7 says they are extra, and they are the same
        on every stage here because every stage runs the same micro-batch.
        """
        seen, total = set(), 0
        pools = [[p for _, p in self.named], list(self.pshard.values()),
                 list(self.master.values()), list(self.m.values()), list(self.v.values()),
                 list(self.gshard.values()), self.ring.live,
                 [p.grad for _, p in self.named if p.grad is not None]]
        for pool in pools:
            for t in pool:
                st = t.untyped_storage()
                if st.data_ptr() in seen:
                    continue
                seen.add(st.data_ptr())
                total += st.size()
        return total

    def mark(self):
        self.peak = max(self.peak, self.resident_bytes())

    # -- ZeRO-3: weights appear for one layer and are gone again ------------------------
    def _install_zero3(self):
        """Gather on the way in, discard on the way out, in both directions.

        Two details that cost an afternoon each. The refill writes through `p.data` so the
        autograd version counter does not tick — a plain `copy_` makes backward refuse the
        tensor it saved. And a unit is released when the last of its parameters has received
        a gradient, not from `register_full_backward_hook`: a residual block's input gradient
        is complete as soon as the skip connection delivers, which is before the block's own
        weights have been touched, so the post-backward hook fires too early and frees
        weights that backward still needs.
        """
        for name, p in self.named:
            npad, s = self.shard_of[name]
            lo = self.rank * s
            self.pshard[name] = self._padded(p.detach(), npad)[lo:lo + s].clone()

        for unit_name, mod in self.model.units():
            owned = [(n, p) for n, p in self.named if self._unit_of(n) == unit_name]
            self._bind_unit(unit_name, mod, owned)

        for _, p in self.named:                   # start the run fully sharded
            p.untyped_storage().resize_(0)

    def _unit_of(self, pname):
        if pname.startswith("blocks."):
            return f"block{pname.split('.')[1]}"
        return pname.split(".")[0]

    def _bind_unit(self, unit_name, mod, owned):
        state = {"pending": 0}

        def fill(*_):
            if self.gathered.get(unit_name):
                return
            for n, p in owned:
                npad, _ = self.shard_of[n]
                flat = self.ring.all_gather(self.pshard[n])
                p.untyped_storage().resize_(p.numel() * p.element_size())
                p.data.copy_(flat[:p.numel()].view_as(p))
            self.gathered[unit_name] = True
            state["pending"] = len(owned)
            self.mark()

        def free(*_):
            if not self.gathered.get(unit_name):
                return
            for _, p in owned:
                p.untyped_storage().resize_(0)
            self.gathered[unit_name] = False

        mod.register_forward_pre_hook(lambda m, i: fill())
        mod.register_forward_hook(lambda m, i, o: free())
        mod.register_full_backward_pre_hook(lambda m, go: fill())
        self._unit_state[unit_name] = (state, free)

    # -- ZeRO-2 and ZeRO-3: reduce each gradient the moment it exists -------------------
    def _install_grad_hooks(self):
        for name, p in self.named:
            p.register_post_accumulate_grad_hook(self._make_grad_hook(name))

    def _make_grad_hook(self, name):
        def hook(p):
            self.mark()
            self._reduce_one(name, p)
            p.grad = None                      # §6: 'discarded as soon as they have been sent'
            if self.mode == "zero3":
                unit = self._unit_of(name)
                state, free = self._unit_state[unit]
                state["pending"] -= 1
                if state["pending"] <= 0:
                    free()
        return hook

    def _reduce_one(self, name, p):
        npad, _ = self.shard_of[name]
        buf = self._padded(p.grad, npad)
        self.ring.live.append(buf)
        mine = self.ring.reduce_scatter(buf)
        self.ring._drop(buf)
        mine /= self.world
        self.gshard[name] = self.gshard[name] + mine if name in self.gshard else mine

    # -- one training step --------------------------------------------------------------
    def step(self, micro_batches):
        """One optimizer step over an accumulation group.

        `micro_batches` is the list of forward/backward passes to run before the weights move
        — the third factor in §3's global batch = sequences per GPU x GPUs x accumulation
        steps. More than one is allowed only under data parallelism: ZeRO-2 and ZeRO-3 reduce
        each gradient from a backward hook the instant it appears, so accumulating across
        micro-batches would need that reduce deferred to the final one, which is what real
        implementations spend a `no_sync` context on and what this engine does not implement.
        """
        if len(micro_batches) > 1 and not self.dp_like:
            raise ValueError(f"{self.mode} cannot accumulate: its reduce fires from a hook")
        self.model.train()
        total = 0.0
        for x, y in micro_batches:
            logits = self.model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
            (loss / len(micro_batches)).backward()
            total += loss.item() / len(micro_batches)
        # The ladder of §6 is the state resident at exactly this instant: weights, whatever
        # gradient the stage has kept, and the optimizer. No collective is in flight, so this
        # is the number the notes' table is about. `peak` below is that plus the largest
        # buffer any one collective needs, which the table does not mention and every real
        # implementation still pays.
        self.after_backward = max(getattr(self, "after_backward", 0), self.resident_bytes())
        self.mark()

        if self.mode == "dp_flat":
            # Every gradient in one buffer, one collective. Same bytes, far fewer messages —
            # and a whole extra copy of the gradients resident while it happens, which is
            # exactly why production bucket sizes are a few hundred MB and not "everything".
            segs = [self._padded(p.grad, self.shard_of[n][0]) for n, p in self.named]
            flat = torch.cat(segs)
            del segs
            self.ring.live.append(flat)
            reduced = self.ring.all_reduce(flat) / self.world
            self.ring._drop(flat)
            off = 0
            for name, p in self.named:
                npad, _ = self.shard_of[name]
                p.grad = reduced[off:off + p.numel()].view_as(p).clone()
                off += npad
            self.mark()
        elif self.mode in ("dp", "zero1"):
            for name, p in self.named:
                npad, _ = self.shard_of[name]
                buf = self._padded(p.grad, npad)
                self.ring.live.append(buf)
                if self.mode == "dp":
                    p.grad = (self.ring.all_reduce(buf)[:p.numel()].view_as(p) / self.world)
                else:
                    self.gshard[name] = self.ring.reduce_scatter(buf) / self.world
                    p.grad = None
                self.ring._drop(buf)
            self.mark()

        self._update()
        self.t += 1
        return total

    def _update(self):
        """AdamW with weight decay 0, which is what V4 ran (§12). Identical arithmetic on
        every stage — only the length of the vectors it runs on changes."""
        b1, b2 = self.betas
        bc1, bc2 = 1 - b1 ** (self.t + 1), 1 - b2 ** (self.t + 1)
        for name, p in self.named:
            g = (p.grad.reshape(-1).float() if self.dp_like
                 else self.gshard[name].float())
            if self.dp_like:
                npad, _ = self.shard_of[name]
                g = self._padded(g, npad)
            m, v, w = self.m[name], self.v[name], self.master[name]
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            w.addcdiv_(m / bc1, (v / bc2).sqrt().add_(self.eps), value=-self.lr)
            if self.dp_like:
                p.data.copy_(w[:p.numel()].view_as(p).to(BF16))
                p.grad = None
            elif self.mode == "zero3":
                self.pshard[name] = w.to(BF16)
            else:
                flat = self.ring.all_gather(w.to(BF16))
                p.data.copy_(flat[:p.numel()].view_as(p))
        self.gshard.clear()
        self.mark()

    # -- what this rank reports back ----------------------------------------------------
    def weights_hash(self):
        """The full model as this rank can see it, for the cross-stage equality gate."""
        parts = []
        for name, p in self.named:
            if self.mode == "zero3":
                npad, _ = self.shard_of[name]
                flat = self.ring.all_gather(self.pshard[name])
                parts.append(flat[:p.numel()].float())
            else:
                parts.append(p.detach().reshape(-1).float())
        return torch.cat(parts)


# --------------------------------------------------------------------------------------
# What each spawned process runs.
# --------------------------------------------------------------------------------------
def _batch(stream, step, batch, block, offset, micro):
    """The global batch for `step`, of which this rank takes rows [offset, offset+micro).

    Deterministic in the step index alone, so every stage and every world size trains on
    exactly the same tokens in exactly the same order.
    """
    g = torch.Generator().manual_seed(90_000 + step)
    ix = torch.randint(len(stream) - block - 1, (batch,), generator=g)
    ix = ix[offset:offset + micro]
    x = torch.stack([stream[i:i + block] for i in ix])
    y = torch.stack([stream[i + 1:i + 1 + block] for i in ix])
    return x, y


def collective_selfcheck(ring, rank, world):
    """Two things worth proving before any of the numbers above are believed.

    1. The hand-written ring agrees with gloo's own all_reduce.
    2. §4's claim, exactly: reduce_scatter then all_gather *is* all_reduce.
    """
    torch.manual_seed(500 + rank)
    n = 4096 * world
    x = torch.randn(n)
    mine = ring.all_reduce(x.clone())
    theirs = x.clone()
    dist.all_reduce(theirs)
    two_phase = ring.all_gather(ring.reduce_scatter(x.clone()))
    bf = (torch.arange(world, dtype=FP32) + 1)[rank].repeat(1024).to(BF16)
    bf_sum = ring.all_reduce(bf.clone())[0].float().item()
    fp_sum = ring.all_reduce(bf.float().clone())[0].item()
    return {
        "vs_gloo_max_abs": (mine - theirs).abs().max().item(),
        "vs_gloo_rel": ((mine - theirs).abs().max() / theirs.abs().max()).item(),
        "two_phase_identical": bool(torch.equal(mine, two_phase)),
        "bf16_ring_sum": bf_sum,
        "fp32_ring_sum": fp_sum,
        "exact_sum": float(world * (world + 1) // 2),
    }


def worker(rank, world, mode, opts, result_dir):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(opts["port"]))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)

    stream = torch.load(opts["stream_path"])
    cfg = Config(**opts["cfg"])
    ring = Ring(rank, world)

    check = collective_selfcheck(ring, rank, world) if opts.get("selfcheck") else None
    ring.bytes_sent, ring.seconds = 0, 0.0

    eng = Engine(mode, cfg, rank, world, ring, lr=opts["lr"])
    micro, steps = opts["micro"], opts["steps"]
    accum = opts.get("accum", 1)
    gbatch = micro * world * accum           # §3: sequences per GPU x GPUs x accumulation steps

    losses, per_step = [], []
    for s in range(steps):
        # Rank r's a-th micro-batch is rows [a*world*micro + r*micro, +micro) of the global
        # batch for this step. Every factorization of the same global batch therefore draws
        # the same sequences, which is what makes the two arrangements comparable at all.
        mbs = [_batch(stream, s, gbatch, cfg.block_size, a * world * micro + rank * micro, micro)
               for a in range(accum)]
        b0, m0, t0 = ring.bytes_sent, ring.msgs, time.perf_counter()
        c0 = ring.seconds
        losses.append(eng.step(mbs))
        per_step.append({"wall": time.perf_counter() - t0,
                         "comm": ring.seconds - c0,
                         "bytes": ring.bytes_sent - b0,
                         "msgs": ring.msgs - m0})

    w = eng.weights_hash()
    n_params = sum(p.numel() for _, p in eng.named)
    # §3: "because they started identical and applied an identical update, they remain
    # identical". Every rank hashes the whole model it can see, so that sentence becomes a
    # comparison of 32 hex strings rather than an assumption.
    w_sha = hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest()
    out = {
        "rank": rank, "mode": mode, "world": world,
        "losses": losses, "per_step": per_step,
        "peak_bytes": eng.peak,
        "ladder_bytes": eng.after_backward,
        "params": n_params,
        "P_bytes": eng.params_bytes_full,
        "bytes_sent_total": ring.bytes_sent,
        "msgs_total": ring.msgs,
        "micro": micro, "accum": accum, "global_batch": gbatch,
        "weight_sha256": w_sha,
        "weight_checksum": float(w.double().sum().item()),
        "weight_absmax": float(w.abs().max().item()),
        "selfcheck": check,
    }
    tag = opts.get("tag", "")
    if rank == 0:
        out["weights_head"] = w[:16].tolist()
        torch.save(w, Path(result_dir) / f"weights_{mode}{tag}_w{world}.pt")
    Path(result_dir, f"rank_{mode}{tag}_w{world}_{rank}.json").write_text(json.dumps(out))
    dist.barrier()
    dist.destroy_process_group()


# --------------------------------------------------------------------------------------
# Launcher. The harness shells out to this rather than calling `mp.spawn` itself.
#
# spawn re-imports the parent's `__main__` in every child. From a plain script that needs an
# `if __name__ == "__main__"` guard, which a `# %%` cell file does not have, and from a
# notebook it is worse still. Putting the spawn behind its own entry point means the caller
# — script or notebook, identically — only ever runs a subprocess.
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import torch.multiprocessing as mp

    spec = json.loads(Path(sys.argv[1]).read_text())
    mp.spawn(worker,
             args=(spec["world"], spec["mode"], spec["opts"], spec["result_dir"]),
             nprocs=spec["world"], join=True)
