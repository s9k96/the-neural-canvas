"""Dynamic Kronecker byte codec -- Session 7 assignment, problem 3.

The shipped codec marks a 256 x pos_dim grid: cell (byte_value, byte_position).
The position factor is a ONE-HOT over pos_dim slots, and that single choice causes
both defects problem 3 names:

  * the crop   -- a one-hot over 32 slots is undefined at position 32, so bytes past
                  it are dropped and two tokens sharing a 32-byte prefix collide
                  permanently (L = min(len(byte_seq), pos_dim) in the released module).
  * the waste  -- every token reserves all 32 slots whether it is "a" or a 36-byte
                  Tamil conjunct cluster, so code_dim is 256*32 = 8192 regardless.

The fix is the move Session 7 section 12 already names for sequence position, applied
one axis down to BYTE position: stop storing position, start computing it. Replace the
one-hot with phi(p), a vector-valued function defined for every p in N:

    kappa(t) = (1/sqrt(L)) * vec( sum_p  c[byte_p] (x) phi(p) ),   L = full byte length

Nothing is truncated, because phi is defined at every position. code_dim becomes
256*m with m << pos_dim, because phi packs position into m dimensions instead of
spending one slot per position.

Both codecs below are identical in every other respect -- same 1/sqrt(L) scale, same
z-normalisation, same single shared projection -- so the only variable in every
experiment is the position factor.
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn as nn

BYTE_VALUES = 256

# Calibrated to the byte-position range a token actually spans (tens, not thousands).
# The Transformer's 10000 is wrong here; E7 measures why.
DEFAULT_BASE = 10.0

# Byte-fallback tokens (<0xNN>) encode the single byte they name, not their literal
# surface form -- paper section 3.2, "Byte-fallback tokens".
_BYTE_FALLBACK = re.compile(r"^<0[xX]([0-9a-fA-F]{2})>$")


def token_bytes(text: str) -> bytes:
    """The byte sequence the codec sees for a token, per the paper's conventions."""
    m = _BYTE_FALLBACK.match(text)
    if m:
        return bytes([int(m.group(1), 16)])
    return text.encode("utf-8")


def utf8_safe_truncate(b: bytes, n: int) -> bytes:
    """Paper section 3.2: 'if d_p falls in the middle of a multi-byte codepoint, we back
    off to the previous codepoint boundary'. Reproduced so the control arm is the real
    shipped codec and not a strawman."""
    if len(b) <= n:
        return b
    cut = n
    while cut > 0 and (b[cut] & 0xC0) == 0x80:   # walked onto a continuation byte
        cut -= 1
    return b[:cut]


# --------------------------------------------------------------------------
# position factors
# --------------------------------------------------------------------------
def phi_onehot(positions: np.ndarray, pos_dim: int) -> np.ndarray:
    """The shipped factor: one slot per position, undefined past pos_dim."""
    out = np.zeros((len(positions), pos_dim), dtype=np.float64)
    keep = positions < pos_dim
    out[np.arange(len(positions))[keep], positions[keep]] = 1.0
    return out


def phi_sinusoid(positions: np.ndarray, m: int, base: float = DEFAULT_BASE) -> np.ndarray:
    """Computed factor: defined for every position, no window, no crop.

    `base` is the calibration knob, and the default is NOT the Transformer's 10000.
    That value was chosen for sequence positions running to the thousands; byte
    positions inside a token run to about 36. At base=10000 the low-frequency channels
    barely turn across that range, so they contribute near-constant mass that dilutes
    the channels doing the discriminating -- measured, it makes separation *worse* as m
    grows. At base=10 every channel completes a useful rotation over the real byte
    range, and margin improves monotonically with m again. See E7.
    """
    # Odd m would silently broadcast one sin channel across two slots instead of
    # erroring, which looks like it works and quietly halves the usable rank.
    assert m % 2 == 0, f"m must be even, got {m}"
    j = np.arange(m // 2, dtype=np.float64)
    w = 1.0 / (base ** (2 * j / m))
    ang = positions[:, None].astype(np.float64) * w[None, :]
    out = np.empty((len(positions), m), dtype=np.float64)
    out[:, 0::2] = np.sin(ang)
    out[:, 1::2] = np.cos(ang)
    return out


# --------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------
class ByteCodec:
    """Turns token text into a fixed, untrained code. Shared by both variants.

    mode="onehot"  -> shipped behaviour, crops at pos_dim, code_dim = 256*pos_dim
    mode="dynamic" -> computed phi, never crops,           code_dim = 256*m
    """

    def __init__(self, mode: str = "dynamic", m: int = 8, pos_dim: int = 32,
                 base: float = DEFAULT_BASE, znorm: bool = True):
        assert mode in ("onehot", "dynamic")
        self.mode = mode
        self.pos_dim = pos_dim
        self.m = m
        self.base = base
        self.znorm = znorm
        self.width = pos_dim if mode == "onehot" else m
        self.code_dim = BYTE_VALUES * self.width

    def truncates(self) -> bool:
        return self.mode == "onehot"

    def encode_one(self, text: str) -> np.ndarray:
        """[code_dim] float64. The grid is (byte value, position-factor channel)."""
        raw = token_bytes(text)
        if self.mode == "onehot":
            raw = utf8_safe_truncate(raw, self.pos_dim)   # the crop, stated explicitly
        b = np.frombuffer(raw, dtype=np.uint8)
        L = max(len(b), 1)
        pos = np.arange(len(b))
        f = (phi_onehot(pos, self.pos_dim) if self.mode == "onehot"
             else phi_sinusoid(pos, self.m, self.base))

        grid = np.zeros((BYTE_VALUES, self.width), dtype=np.float64)
        np.add.at(grid, b, f)             # c[byte] (x) phi(pos), summed over positions
        code = grid.reshape(-1) / math.sqrt(L)
        if self.znorm:
            s = code.std()
            code = (code - code.mean()) / (s if s > 1e-12 else 1.0)
        return code

    def encode_many(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.code_dim), dtype=np.float64)
        for i, t in enumerate(texts):
            out[i] = self.encode_one(t)
        return out

    def collision_key(self, text: str, ndigits: int = 9):
        """Exact-collision identity. Sparse: only byte values the token actually uses."""
        b = token_bytes(text)
        if self.mode == "onehot":
            kept = utf8_safe_truncate(b, self.pos_dim)
            # one-hot codes are equal iff the (kept bytes, scale) agree
            return ("onehot", len(kept), kept)
        arr = np.frombuffer(b, dtype=np.uint8)
        grid = np.zeros((BYTE_VALUES, self.m), dtype=np.float64)
        np.add.at(grid, arr, phi_sinusoid(np.arange(len(arr)), self.m, self.base))
        used = np.flatnonzero(grid.any(axis=1))
        return ("dyn", len(b), used.tobytes(),
                np.round(grid[used], ndigits).tobytes())

    def __repr__(self):
        tag = f"onehot pos_dim={self.pos_dim}" if self.mode == "onehot" else \
              f"dynamic m={self.m}"
        return f"ByteCodec({tag}, code_dim={self.code_dim})"


# --------------------------------------------------------------------------
# embedding layer -- the seam is preserved: [B,T] ints in, [B,T,D] floats out
# --------------------------------------------------------------------------
class KroneckerEmbedding(nn.Module):
    """Frozen code table (buffer, untrained) + one shared projection (the only Parameter).

    Same public behaviour as the released module, so this is a drop-in swap: everything
    about how the row was manufactured stays private to the layer.
    """

    def __init__(self, token_texts: list[str], d_model: int, codec: ByteCodec):
        super().__init__()
        self.codec = codec
        self.d_model = d_model
        codes = codec.encode_many(token_texts)
        self.register_buffer("code", torch.tensor(codes, dtype=torch.float32))  # NOT trained
        self.proj = nn.Linear(codec.code_dim, d_model, bias=False)              # all params

    @property
    def n_trainable(self) -> int:
        return self.codec.code_dim * self.d_model

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.code[ids])


class DenseEmbedding(nn.Module):
    """Control arm: the ordinary V x D table."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size

    @property
    def n_trainable(self) -> int:
        return self.vocab_size * self.d_model

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)


# --------------------------------------------------------------------------
# the smallest transformer that can carry the argument
# --------------------------------------------------------------------------
class TinyTransformer(nn.Module):
    """One block, learned absolute positions. Deliberately small: the claim under test
    is about the input path, so everything above it is held fixed across arms."""

    def __init__(self, embed: nn.Module, n_classes: int, d_model: int,
                 n_heads: int = 2, max_len: int = 8):
        super().__init__()
        self.embed = embed
        self.pos = nn.Embedding(max_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                nn.Linear(4 * d_model, d_model))
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ff(self.ln2(x))
        return self.head(x[:, -1])


def demo():
    """Self-check: the codec does what the docstring claims."""
    shipped = ByteCodec("onehot", pos_dim=32)
    dyn = ByteCodec("dynamic", m=8)

    # 1. the crop is real, and the notes' own example pair is one of its victims
    a, b = "अंतर्राष्ट्रीयकरण", "अंतर्राष्ट्रीयता"
    assert len(a.encode()) > 32 and len(b.encode()) > 32
    assert shipped.collision_key(a) == shipped.collision_key(b), "expected shipped collision"
    assert dyn.collision_key(a) != dyn.collision_key(b), "dynamic must separate them"
    assert np.abs(shipped.encode_one(a) - shipped.encode_one(b)).max() < 1e-12
    assert np.abs(dyn.encode_one(a) - dyn.encode_one(b)).max() > 1e-6

    # 2. dynamic is smaller AND unbounded
    assert dyn.code_dim == 2048 and shipped.code_dim == 8192
    long = "x" * 500
    assert np.isfinite(dyn.encode_one(long)).all()
    assert dyn.collision_key(long) != dyn.collision_key(long + "y")

    # 3. short tokens still work and stay distinct
    for x, y in [("a", "b"), ("apple", "apples"), ("train", "trainer")]:
        assert dyn.collision_key(x) != dyn.collision_key(y)

    # 3b. paper conventions: byte fallback and UTF-8-safe truncation
    assert token_bytes("<0xC3>") == b"\xc3" and token_bytes("ab") == b"ab"
    assert utf8_safe_truncate(("क" * 20).encode(), 32) == ("क" * 10).encode(), \
        "must back off to a codepoint boundary, not split one"
    assert len(utf8_safe_truncate(b"abcdefgh", 4)) == 4

    # 4. the seam holds: [B,T] ints -> [B,T,D] floats, code buffer untrained
    emb = KroneckerEmbedding(["a", "bb", "ccc", "dddd"], d_model=16, codec=dyn)
    out = emb(torch.tensor([[0, 1, 2], [3, 0, 1]]))
    assert out.shape == (2, 3, 16)
    trainable = {n for n, p in emb.named_parameters() if p.requires_grad}
    assert trainable == {"proj.weight"}, trainable

    print("dynkron self-check: ok")


if __name__ == "__main__":
    demo()
