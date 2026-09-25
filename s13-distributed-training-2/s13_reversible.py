"""The reversible stack: the thing §16 of the session describes, built so it can be checked.

A standard residual block adds its output to its own input, `p = p + f(p)`, and that cannot be
run backwards: recovering the input would need `f` evaluated *at the input being recovered*.
The rules here reach further back so that the block is always evaluated at a state the
backward pass already holds, which makes the stack invertible and lets the forward pass throw
away every intermediate activation.

Nothing here is approximate. The whole point of the exercise is that a reversible stack which
is subtly wrong still trains — the loss still falls — so `gradient_check()` below compares
against ordinary autograd rather than trusting the loss curve.

Imported by `s13_reversibility.py`; kept separate so it can be tested without running the
harness, and so the Colab notebook has one file to carry.
"""

import math
from contextlib import contextmanager

import torch.utils.checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F

def _amp_decorators():
    """`custom_fwd`/`custom_bwd` moved namespace between torch versions.

    These matter more than they look. Under autocast the forward runs its matmuls in fp16;
    the reversible backward re-runs the same blocks to rebuild activations, and if it does so
    under a *different* autocast state it computes different numbers and the reconstruction
    silently stops matching. The decorators restore the forward's autocast state in backward,
    which is exactly the guarantee this stack needs.
    """
    try:                                                   # torch >= 2.4
        from torch.amp import custom_bwd, custom_fwd
        return (lambda f: custom_fwd(f, device_type="cuda"),
                lambda f: custom_bwd(f, device_type="cuda"))
    except (ImportError, TypeError):
        try:                                               # torch < 2.4
            from torch.cuda.amp import custom_bwd, custom_fwd
            return custom_fwd, custom_bwd
        except ImportError:
            return (lambda f: f), (lambda f: f)


_custom_fwd, _custom_bwd = _amp_decorators()

RULES = ("standard", "euler", "midpoint", "blended", "revnet")
REVERSIBLE = ("midpoint", "blended", "revnet")       # euler and standard are not invertible


# --------------------------------------------------------------------------------------
# Model. The S9-S12 decoder, with one change forced by this session: no dropout anywhere.
# §16 makes that a correctness requirement rather than a tuning choice — the backward pass
# recomputes each block, and a random mask would make the reconstruction differ from what
# the forward pass actually used.
# --------------------------------------------------------------------------------------
class Config:
    def __init__(self, **kw):
        self.vocab_size, self.d_model, self.n_layer, self.d_head = 256, 384, 10, 64
        self.block_size = 512
        self.embed_std = 0.02
        self.rule, self.h, self.gamma = "standard", 1.0, 0.0
        self.__dict__.update(kw)
        self.n_head = max(1, self.d_model // self.d_head)
        self.d_ff = getattr(self, "d_ff", None) or round(self.d_model * 8 / 3 / 64) * 64

    def __repr__(self):
        return (f"Config(rule={self.rule}, d_model={self.d_model}, n_layer={self.n_layer}, "
                f"d_ff={self.d_ff}, vocab={self.vocab_size}, h={self.h}, gamma={self.gamma})")


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # Accumulate in float32 for the low-precision dtypes, but never DOWN-cast: writing
        # `x.float()` unconditionally computes the norm in float32 even for a float64 input,
        # which quantises the block into steps of ~1e-7. Any reconstruction error smaller
        # than a step can then flip one, so the block behaves discontinuously and the
        # reversible path looks broken when it is not.
        acc = x.float() if x.dtype in (torch.bfloat16, torch.float16) else x
        return x * torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.g


class Block(nn.Module):
    """Attention followed by the feed-forward network — §16's `f`, the whole transformer block.

    `width` is the channel count this block reads and writes. It equals d_model for every rule
    except `revnet`, where the stream is split in half and each coupling function sees d_model/2.
    """

    def __init__(self, cfg, width=None):
        super().__init__()
        d = width or cfg.d_model
        self.n_head = max(1, d // cfg.d_head)
        self.d_head = cfg.d_head
        H = self.n_head * self.d_head
        # Always honour cfg.d_ff. Deriving it from the block's own width instead would make
        # revnet's size unsteerable, and matching parameter counts across rules is the only
        # way the throughput comparison means anything.
        d_ff = cfg.d_ff
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * H, bias=False)
        self.proj = nn.Linear(H, d, bias=False)
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        H = self.n_head * self.d_head
        q, k, v = self.qkv(self.n1(x)).split(H, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.proj(a.transpose(1, 2).reshape(B, T, H))
        y = self.n2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


# --------------------------------------------------------------------------------------
# The five update rules, each written once so the forward pass and the inverse cannot drift
# apart. `step` advances the stack; `invert` recovers the state two layers back.
# --------------------------------------------------------------------------------------
def step(rule, p_prev, p, fout, h, gamma):
    """One layer of the stack. `fout` is f(p), already computed."""
    if rule == "standard":
        return p + fout                       # the ordinary residual; p_prev unused
    if rule == "euler":
        return p + h * fout                   # a step size, still not invertible
    if rule == "midpoint":
        return p_prev + 2 * h * fout          # §16's rule
    if rule == "blended":
        return (1 - gamma) * p_prev + gamma * p + 2 * h * fout
    raise ValueError(rule)


def invert(rule, p_next, p, fout, h, gamma):
    """Recover p_prev from p_next and p. `fout` is f(p) — a state the backward pass holds."""
    if rule == "midpoint":
        return p_next - 2 * h * fout
    if rule == "blended":
        # The division is the whole problem with this rule: it multiplies any error in
        # p_next by 1/(1-gamma), once per layer. See the gamma sweep in the harness.
        return (p_next - gamma * p - 2 * h * fout) / (1 - gamma)
    raise ValueError(f"{rule} has no inverse")


def _grad_recursion(rule, A, B, jvp, h, gamma):
    """Push the gradient pair (dL/dp_next, dL/dp) down one layer.

    p_next = c1*p_prev + c2*p + k*f(p), so dL/dp_prev = c1*A and dL/dp = B + c2*A + k*J^T A.
    `jvp` is the J^T A term, already scaled by k.
    """
    if rule == "midpoint":
        return B + jvp, A                     # c1 = 1, c2 = 0
    if rule == "blended":
        return B + gamma * A + jvp, (1 - gamma) * A
    raise ValueError(rule)


# --------------------------------------------------------------------------------------
# Stored-activation path: ordinary autograd, every intermediate kept. The reference.
# --------------------------------------------------------------------------------------
def run_stored(blocks, x, rule, h, gamma):
    if rule == "revnet":
        x1, x2 = x.chunk(2, dim=-1)
        for l in range(0, len(blocks) - 1, 2):
            y1 = x1 + blocks[l](x2)
            y2 = x2 + blocks[l + 1](y1)
            x1, x2 = y1, y2
        return torch.cat([x1, x2], dim=-1)
    p_prev = x
    p = p_prev + h * blocks[0](p_prev)         # bootstrap: the first step has no p_prev to use
    for l in range(1, len(blocks)):
        p_prev, p = p, step(rule, p_prev, p, blocks[l](p), h, gamma)
    return p


# --------------------------------------------------------------------------------------
# Reversible path: the forward keeps only the boundary states.
# --------------------------------------------------------------------------------------
class _RevStack(torch.autograd.Function):
    """Forward under no_grad, keeping x, p_L and p_{L-1}. Backward rebuilds the rest.

    The input state x is kept because the bootstrap step is a plain Euler step and is not
    itself invertible — which is exactly §16's "the state that enters the stack".
    """

    @staticmethod
    @_custom_fwd
    def forward(ctx, x, blocks, rule, h, gamma, *params):
        ctx.blocks, ctx.rule, ctx.h, ctx.gamma = blocks, rule, h, gamma
        with torch.no_grad():
            p_prev = x
            p = p_prev + h * blocks[0](p_prev)
            for l in range(1, len(blocks)):
                p_prev, p = p, step(rule, p_prev, p, blocks[l](p), h, gamma)
        ctx.save_for_backward(x, p, p_prev)
        return p

    @staticmethod
    @_custom_bwd
    def backward(ctx, g_out):
        x, p, p_prev = ctx.saved_tensors
        blocks, rule, h, gamma = ctx.blocks, ctx.rule, ctx.h, ctx.gamma
        p_hi, p_lo = p, p_prev                        # (p_{l+1}, p_l)
        A, B = g_out, torch.zeros_like(g_out)         # dL/dp_{l+1}, dL/dp_l
        pgrads = [torch.zeros_like(q) for b in blocks for q in b.parameters()]
        sizes = [len(list(b.parameters())) for b in blocks]
        offs, acc = [], 0
        for n in sizes:
            offs.append(acc)
            acc += n

        for l in range(len(blocks) - 1, 0, -1):
            plist = list(blocks[l].parameters())
            pl = p_lo.detach().requires_grad_(True)
            with torch.enable_grad():
                fout = blocks[l](pl)
            with torch.no_grad():
                p_down = invert(rule, p_hi, p_lo, fout.detach(), h, gamma)
            g = torch.autograd.grad(fout, [pl] + plist, grad_outputs=2 * h * A)
            for i, gv in enumerate(g[1:]):
                pgrads[offs[l] + i] += gv
            A, B = _grad_recursion(rule, A, B, g[0], h, gamma)
            p_hi, p_lo = p_lo, p_down

        # the bootstrap layer, whose input is the x we kept
        x0 = x.detach().requires_grad_(True)
        plist = list(blocks[0].parameters())
        with torch.enable_grad():
            f0 = blocks[0](x0)
        g = torch.autograd.grad(f0, [x0] + plist, grad_outputs=h * A)
        for i, gv in enumerate(g[1:]):
            pgrads[offs[0] + i] += gv
        # p_1 = p_0 + h*f_0(p_0), so p_0 receives A through the identity path and h*J^T A
        # through the block — on top of B, which layer 1 already accumulated into it. The
        # B term is easy to drop, and dropping it leaves every block gradient correct while
        # silently corrupting only the embedding gradients.
        gx = B + A + g[0]
        return (gx, None, None, None, None, *pgrads)


class _RevNetStack(torch.autograd.Function):
    """Channel-coupling reversibility: y1 = x1 + F(x2), y2 = x2 + G(y1).

    Each layer inverts on its own, exactly, with no step size and no stability band — which
    is why this is the variant that still trains when the midpoint rule will not.
    """

    @staticmethod
    @_custom_fwd
    def forward(ctx, x, blocks, *params):
        ctx.blocks = blocks
        with torch.no_grad():
            x1, x2 = x.chunk(2, dim=-1)
            for l in range(0, len(blocks) - 1, 2):
                x1 = x1 + blocks[l](x2)
                x2 = x2 + blocks[l + 1](x1)
        out = torch.cat([x1, x2], dim=-1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    @_custom_bwd
    def backward(ctx, g_out):
        (out,) = ctx.saved_tensors
        blocks = ctx.blocks
        y1, y2 = out.chunk(2, dim=-1)
        g1, g2 = g_out.chunk(2, dim=-1)
        g1, g2 = g1.contiguous(), g2.contiguous()
        pgrads = [torch.zeros_like(q) for b in blocks for q in b.parameters()]
        sizes = [len(list(b.parameters())) for b in blocks]
        offs, acc = [], 0
        for n in sizes:
            offs.append(acc)
            acc += n

        for l in range(len(blocks) - 2, -1, -2):
            # undo y2 = x2 + G(y1)
            y1d = y1.detach().requires_grad_(True)
            with torch.enable_grad():
                gy = blocks[l + 1](y1d)
            with torch.no_grad():
                x2 = y2 - gy.detach()
            gg = torch.autograd.grad(gy, [y1d] + list(blocks[l + 1].parameters()),
                                     grad_outputs=g2)
            for i, gv in enumerate(gg[1:]):
                pgrads[offs[l + 1] + i] += gv
            g1 = g1 + gg[0]
            # undo y1 = x1 + F(x2)
            x2d = x2.detach().requires_grad_(True)
            with torch.enable_grad():
                fx = blocks[l](x2d)
            with torch.no_grad():
                x1 = y1 - fx.detach()
            gf = torch.autograd.grad(fx, [x2d] + list(blocks[l].parameters()),
                                     grad_outputs=g1)
            for i, gv in enumerate(gf[1:]):
                pgrads[offs[l] + i] += gv
            g2 = g2 + gf[0]
            y1, y2 = x1, x2
        return (torch.cat([g1, g2], dim=-1), None, *pgrads)


def run_reversible(blocks, x, rule, h, gamma):
    params = [q for b in blocks for q in b.parameters()]
    if rule == "revnet":
        return _RevNetStack.apply(x, blocks, *params)
    return _RevStack.apply(x, blocks, rule, h, gamma, *params)


# --------------------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, cfg, reversible=False):
        super().__init__()
        self.cfg, self.reversible = cfg, reversible
        width = cfg.d_model // 2 if cfg.rule == "revnet" else cfg.d_model
        n = cfg.n_layer + (cfg.n_layer % 2 if cfg.rule == "revnet" else 0)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, width if cfg.rule == "revnet" else None)
                                    for _ in range(n))
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 1.0 / math.sqrt(m.weight.shape[1]))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, cfg.embed_std)

    def hidden(self, idx):
        """Everything up to the loss head. Split out so the head can be applied in chunks —
        see `chunked_ce`, and the reason it exists."""
        c = self.cfg
        x = self.embed(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        runner = run_reversible if self.reversible else run_stored
        return self.norm_f(runner(self.blocks, x, c.rule, c.h, c.gamma))

    def head_for_loss(self):
        return self.head

    def forward(self, idx):
        return self.head(self.hidden(idx))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def chunked_ce(hidden, head, targets, chunks=8):
    """Cross-entropy without ever materialising the full B x T x V logits tensor.

    The probe showed why this is not a micro-optimisation. At batch 64, seq 512 and a
    vocabulary of 8,192, the logits alone are 2.15 GB of a 15.6 GB card — so the term that
    limits batch size is the loss head, not the residual stack, and reversibility (which only
    removes the stack) looks far less useful than it is. Each chunk is checkpointed, so its
    logits are rebuilt during the backward pass instead of being held.
    """
    B, T, D = hidden.shape
    flat_h = hidden.reshape(B * T, D)
    flat_t = targets.reshape(-1)
    n = flat_t.numel()
    total = flat_h.new_zeros((), dtype=torch.float32)

    def piece(h, t):
        return F.cross_entropy(head(h).float(), t, reduction="sum")

    for h, t in zip(flat_h.chunk(chunks), flat_t.chunk(chunks)):
        if torch.is_grad_enabled() and h.requires_grad:
            total = total + torch.utils.checkpoint.checkpoint(piece, h, t, use_reentrant=False)
        else:
            total = total + piece(h, t)
    return total / n


def count_params(cfg):
    """Parameter count in closed form, so configuration search does not build models.

    Instantiating a Model per candidate is thousands of tensor allocations and turns a
    sub-second search into minutes.
    """
    d, V = cfg.d_model, cfg.vocab_size
    w = d // 2 if cfg.rule == "revnet" else d
    n_blocks = cfg.n_layer + (cfg.n_layer % 2) if cfg.rule == "revnet" else cfg.n_layer
    H = max(1, w // cfg.d_head) * cfg.d_head
    per_block = 2 * w + w * 3 * H + H * w + 2 * w * cfg.d_ff + cfg.d_ff * w
    return V * d + cfg.block_size * d + d + d * V + n_blocks * per_block


def match_params(target, rule, n_layer, d_model, vocab, block_size):
    """Pick d_ff so every rule reaches the same parameter count.

    A revnet block runs on half the channels, so at equal d_model it has roughly a quarter
    of a standard block's parameters and a quarter of its FLOPs. Benchmarking the two at the
    same width compares a model against a smaller model — which is how the probe made revnet
    look 1.8x faster than it is.
    """
    best, best_err = None, float("inf")
    for d_ff in range(64, 8193, 64):
        cfg = Config(rule=rule, n_layer=n_layer, d_model=d_model, d_ff=d_ff,
                     vocab_size=vocab, block_size=block_size)
        err = abs(count_params(cfg) - target)
        if err < best_err:
            best, best_err = d_ff, err
    return best


def fit_config(target, rule, n_layer, vocab, block_size, widths=range(128, 769, 32)):
    """The (d_model, d_ff) nearest `target` parameters for this rule at this depth."""
    best = None
    for d in widths:
        d_ff = match_params(target, rule, n_layer, d, vocab, block_size)
        cfg = Config(rule=rule, n_layer=n_layer, d_model=d, d_ff=d_ff,
                     vocab_size=vocab, block_size=block_size)
        n = count_params(cfg)
        if best is None or abs(n - target) < abs(best[2] - target):
            best = (d, d_ff, n)
    return best


# --------------------------------------------------------------------------------------
# Measurement. Two different quantities, and the difference matters.
#   * activation bytes — what the forward pass kept for the backward pass. This is the
#     quantity §1's 127.5 GiB and §17's 1.5 GiB are about, and it is device-independent.
#   * peak memory — what the assignment asks for: everything resident at the high-water
#     mark, weights and optimizer included. CUDA only.
# --------------------------------------------------------------------------------------
@contextmanager
def activation_bytes():
    """Sum the unique storages autograd saves for backward, via saved-tensor hooks."""
    seen, total = {}, [0]

    def pack(t):
        st = t.untyped_storage()
        key = st.data_ptr()
        if key and key not in seen:
            seen[key] = st.size()
            total[0] += st.size()
        return t

    def unpack(t):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield total


def peak_bytes_reset(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def peak_bytes_read(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device)
    return None


# --------------------------------------------------------------------------------------
def gradient_check(cfg, batch=2, seq=32, seed=0, device="cpu", dtype=torch.float32):
    """The gate that matters: do reversible gradients equal stored-activation gradients?

    A reversible stack that reconstructs slightly wrong still trains, and its loss still
    falls, so the loss curve is not evidence of anything. This is.
    """
    device = torch.device(device)
    torch.manual_seed(seed)
    ref = Model(cfg, reversible=False).to(device=device, dtype=dtype)
    torch.manual_seed(seed)
    rev = Model(cfg, reversible=True).to(device=device, dtype=dtype)

    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    idx = torch.randint(cfg.vocab_size, (batch, seq), generator=g).to(device)
    tgt = torch.randint(cfg.vocab_size, (batch, seq), generator=g).to(device)

    def loss_of(model, counter=False):
        logits = model(idx)
        return F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), tgt.reshape(-1))

    # Count only what the FORWARD pass saved. Leaving the hook installed during backward
    # counts the one-layer graphs the reversible path rebuilds, which is the memory it
    # explicitly does not hold on to, and makes the saving look far smaller than it is.
    with activation_bytes() as a_ref:
        l_ref = loss_of(ref)
    l_ref.backward()
    with activation_bytes() as a_rev:
        l_rev = loss_of(rev)
    l_rev.backward()

    worst, worst_name = 0.0, ""
    for (n, a), (_, b) in zip(ref.named_parameters(), rev.named_parameters()):
        if a.grad is None or b.grad is None:
            continue
        scale = max(a.grad.abs().max().item(), 1e-12)
        rel = (a.grad - b.grad).abs().max().item() / scale
        if rel > worst:
            worst, worst_name = rel, n
    return {
        "rule": cfg.rule, "n_layer": cfg.n_layer,
        "worst_rel_grad_error": worst, "worst_param": worst_name,
        "activation_bytes_stored": a_ref[0], "activation_bytes_reversible": a_rev[0],
        "activation_ratio": a_ref[0] / max(a_rev[0], 1),
        "params": ref.n_params(),
    }


if __name__ == "__main__":
    # Runnable self-check. Two separate questions, and conflating them is how a correct
    # implementation gets mistaken for a broken one:
    #
    #   1. Is the implementation exact?  Ask in float64. An exact reversible stack differs
    #      from ordinary autograd only by rounding, so the error must collapse when the
    #      precision rises. If it does not, the arithmetic is wrong.
    #   2. How much does it drift in practice?  Ask in float32. This error is real and it
    #      grows with depth, because every reconstruction feeds the next one.
    import torch

    print("1. exactness  (float64 — must collapse to rounding)")
    print(f"   {'rule':<10}{'layers':>7}{'worst rel grad err':>21}{'':>4}")
    bad = []
    for rule in REVERSIBLE:
        for n_layer in (4, 12):
            cfg = Config(rule=rule, n_layer=n_layer, d_model=128, vocab_size=512,
                         block_size=64, h=0.25 if rule != "revnet" else 1.0,
                         gamma=0.5 if rule == "blended" else 0.0)
            e = gradient_check(cfg, batch=2, seq=16, dtype=torch.float64)["worst_rel_grad_error"]
            ok = e < 1e-9
            bad += [] if ok else [(rule, n_layer, e)]
            print(f"   {rule:<10}{n_layer:>7}{e:>21.2e}{'  ok' if ok else '  FAIL':>4}")

    print("\n2. drift and memory  (float32 — the practical numbers)")
    print(f"   {'rule':<10}{'layers':>7}{'drift':>12}{'act bytes stored':>19}"
          f"{'reversible':>12}{'ratio':>9}")
    for rule in REVERSIBLE:
        for n_layer in (4, 12):
            cfg = Config(rule=rule, n_layer=n_layer, d_model=128, vocab_size=512,
                         block_size=64, h=0.25 if rule != "revnet" else 1.0,
                         gamma=0.5 if rule == "blended" else 0.0)
            r = gradient_check(cfg, batch=2, seq=16)
            print(f"   {rule:<10}{n_layer:>7}{r['worst_rel_grad_error']:>12.2e}"
                  f"{r['activation_bytes_stored']:>19,}{r['activation_bytes_reversible']:>12,}"
                  f"{r['activation_ratio']:>8.1f}x")

    print("\n3. euler is not reversible, which is the point of including it")
    cfg = Config(rule="euler", n_layer=8, d_model=128, vocab_size=512, block_size=64, h=0.25)
    try:
        invert("euler", torch.zeros(1), torch.zeros(1), torch.zeros(1), 0.25, 0.0)
        print("   FAIL: euler claimed an inverse")
        bad.append(("euler", 8, float("inf")))
    except ValueError as exc:
        print(f"   {exc}")

    print("\nOK" if not bad else f"\nFAILED: {bad}")
