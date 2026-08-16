"""Fourier byte codec -- Session 7 assignment, problem 4.

    "What is a REAL Fourier alternative of Kronecker? Why can't I represent each
     character like a fourier wave, and just add them to make a word!!"

The answer is that the Fourier counterpart of an outer product is CIRCULAR CONVOLUTION,
by the convolution theorem:  DFT(a (*) b) = DFT(a) . DFT(b).  Binding by convolution is
the Holographic Reduced Representation (Plate, 1995); doing it directly in the frequency
domain, where each symbol is a set of unit-modulus phases, is FHRR.

Kronecker binds (byte value, byte position) with a tensor product, so the code dimension
is the PRODUCT d_c * d_p = 256 * 32 = 8192. Convolution binds in CONSTANT dimension d.
That is the whole structural difference, and it is why d is a free parameter here while
8192 is forced there.

Position comes from the shift theorem: shifting a signal multiplies its spectrum by a
phase ramp, so "byte v at position p" is the wave for v, advanced by p steps of a fixed
ramp. A word is the sum of its characters' waves. Literally the question as asked.

    z_v in C^(d/2+1),  |z_v[k]| = 1        one frozen wave per byte value
    rho in R^(d/2+1)                       one frozen position ramp
    kappa(t) = irfft( (1/sqrt(L)) * sum_p exp(i * (z_v[b_p] + p*rho)) )

Nothing is learned. The phases are drawn once from a seeded generator and frozen, so the
codec is a deterministic function of the byte string exactly as Kronecker is -- but the
seed now joins the tokenizer hash as part of the artifact's identity.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from dynkron import token_bytes  # noqa: E402  (shared byte conventions)

DEFAULT_SEED = 20260815

# Two different capabilities need two different capacities, measured in F1 and F4:
#   discrimination (no two tokens share a code)  -- d = 128 suffices on this vocabulary
#   decoding (read the byte string back out)     -- d ~ 20*L; 2048 decodes this vocab
#                                                   exactly, 1024 gets 99% of tokens
# The default buys both, because invertibility is the property that distinguishes this
# codec from Kronecker at all. Drop to 128 if the input path is all you need.
DEFAULT_D = 2048


class FourierCodec:
    """Duck-types ByteCodec so the problem-3 experiment harness runs unchanged.

    ramp="random" : position phases are frozen random. No period, so token length is
                    genuinely unbounded.
    ramp="shift"  : position phases are the canonical 2*pi*k/d, making binding an exact
                    circular shift. More interpretable, but it wraps at d -- which
                    reintroduces a window, just a much larger one. Measured in F5.
    """

    def __init__(self, d: int = DEFAULT_D, seed: int = DEFAULT_SEED, ramp: str = "random",
                 znorm: bool = True):
        assert d % 2 == 0, f"d must be even, got {d}"
        assert ramp in ("random", "shift")
        self.d = d
        self.code_dim = d
        self.seed = seed
        self.ramp = ramp
        self.znorm = znorm
        self.nf = d // 2 + 1
        rng = np.random.default_rng(seed)
        self.byte_phase = rng.uniform(0, 2 * np.pi, size=(256, self.nf))
        self.pos_phase = (rng.uniform(0, 2 * np.pi, size=self.nf) if ramp == "random"
                          else 2 * np.pi * np.arange(self.nf) / d)
        # The DC and Nyquist channels have no conjugate partner, so irfft requires them
        # to be real -- give them an arbitrary phase and irfft silently discards the
        # imaginary part (measured: 0.65 at Nyquist, real information thrown away). Use
        # +-1 there instead: still unit modulus, still informative, and now the code
        # survives the trip to the time domain intact, which is what makes the
        # concatenation law hold in the space the model actually sees.
        for k in (0, self.nf - 1):
            self.byte_phase[:, k] = np.pi * (rng.random(256) < 0.5)
            self.pos_phase[k] = 0.0

    # -- core -------------------------------------------------------------
    def _spectrum(self, text: str) -> np.ndarray:
        b = np.frombuffer(token_bytes(text), dtype=np.uint8)
        L = max(len(b), 1)
        p = np.arange(len(b))[:, None]
        phase = self.byte_phase[b] + p * self.pos_phase[None, :]   # bind
        return np.exp(1j * phase).sum(axis=0) / np.sqrt(L)         # superpose

    def encode_one(self, text: str) -> np.ndarray:
        c = np.fft.irfft(self._spectrum(text), n=self.d)
        if self.znorm:
            s = c.std()
            c = (c - c.mean()) / (s if s > 1e-12 else 1.0)
        return c

    def encode_many(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.code_dim), dtype=np.float64)
        for i, t in enumerate(texts):
            out[i] = self.encode_one(t)
        return out

    def collision_key(self, text: str, ndigits: int = 6):
        return np.round(self.encode_one(text), ndigits).tobytes()

    def truncates(self) -> bool:
        return False

    # -- the algebra ------------------------------------------------------
    def rotate(self, code: np.ndarray, times: int = 1) -> np.ndarray:
        """Advance a code by `times` byte positions. This is the circular convolution
        that binding is built from; with ramp="shift" it is literally a circular shift."""
        return np.fft.irfft(np.fft.rfft(code) * np.exp(1j * self.pos_phase * times),
                            n=self.d)

    def concat(self, x: str, y: str) -> np.ndarray:
        """kappa(xy) built from kappa(x) and kappa(y) WITHOUT looking at the bytes of
        the joined string -- concatenation becomes addition after rotation.

            kappa(xy) = [ sqrt(Lx) kappa(x) + sqrt(Ly) rot^Lx( kappa(y) ) ] / sqrt(Lx+Ly)

        Exact only without z-normalisation, which is a per-token affine rescale.
        """
        assert not self.znorm, "the algebra is exact on raw codes; use znorm=False"
        lx, ly = max(len(token_bytes(x)), 1), max(len(token_bytes(y)), 1)
        return ((math.sqrt(lx) * self.encode_one(x)
                 + math.sqrt(ly) * self.rotate(self.encode_one(y), lx))
                / math.sqrt(lx + ly))

    # -- the property Kronecker does not have -----------------------------
    def score(self, text: str, value: int, pos: int) -> float:
        """Ask the code: was byte `value` at position `pos`? ~1 yes, ~0 no."""
        spec = self._spectrum(text)
        probe = np.exp(1j * (self.byte_phase[value] + pos * self.pos_phase))
        L = max(len(token_bytes(text)), 1)
        return float(np.real(np.vdot(probe, spec)) / self.nf * np.sqrt(L))

    def decode(self, text: str, length: int | None = None) -> bytes:
        """Read the byte string back out of the code. Unbinding, not a lookup."""
        spec = self._spectrum(text)
        L = length if length is not None else len(token_bytes(text))
        out = bytearray()
        for p in range(L):
            probe = np.exp(1j * (self.byte_phase + p * self.pos_phase[None, :]))
            scores = np.real(probe.conj() @ spec)      # all 256 candidates at once
            out.append(int(np.argmax(scores)))
        return bytes(out)

    def __repr__(self):
        return f"FourierCodec(d={self.d}, ramp={self.ramp}, seed={self.seed})"


class FourierEmbedding(nn.Module):
    """Frozen code table (buffer) + one shared projection (the only Parameter).

    Same seam as every other input path in this course: [B,T] ints -> [B,T,D] floats.
    """

    def __init__(self, token_texts: list[str], d_model: int, codec: FourierCodec):
        super().__init__()
        self.codec = codec
        self.d_model = d_model
        codes = codec.encode_many(token_texts)
        self.register_buffer("code", torch.tensor(codes, dtype=torch.float32))
        self.proj = nn.Linear(codec.code_dim, d_model, bias=False)

    @property
    def n_trainable(self) -> int:
        return self.codec.code_dim * self.d_model

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.code[ids])


def demo():
    """Self-check: the codec does what the docstring claims."""
    c = FourierCodec(d=256)          # discrimination-only capacity

    # 1. deterministic, and order matters (a phase ramp is not a bag of bytes)
    assert np.array_equal(c.encode_one("भारत"), c.encode_one("भारत"))
    assert c.collision_key("abc") != c.collision_key("cba")

    # 2. no window: the pair the shipped Kronecker codec merges stays distinct,
    #    and length is unbounded
    assert c.collision_key("अंतर्राष्ट्रीयकरण") != c.collision_key("अंतर्राष्ट्रीयता")
    long = "क" * 500
    assert np.isfinite(c.encode_one(long)).all()
    assert c.collision_key(long) != c.collision_key(long + "ख")

    # 3. dimension is constant, not a product -- the whole point
    assert c.code_dim == 256          # not 256 * anything

    # 4. the code is readable back out, at the capacity decoding needs (F4)
    big = FourierCodec(d=DEFAULT_D)
    for w in ("a", "train", "भारत", "internationalization", "अंतर्राष्ट्रीयकरण"):
        assert big.decode(w) == token_bytes(w), f"failed to decode {w}"
    assert big.score("भारत", token_bytes("भारत")[0], 0) > 0.5
    assert big.score("भारत", (token_bytes("भारत")[0] + 7) % 256, 0) < 0.3

    # 4b. concatenation is addition after rotation, exactly
    raw = FourierCodec(d=64, znorm=False)
    for x, y in [("a", "b"), ("tra", "in"), ("भा", "रत")]:
        assert np.abs(raw.encode_one(x + y) - raw.concat(x, y)).max() < 1e-12, (x, y)

    # 5. the seam holds, and only the projection is trainable
    emb = FourierEmbedding(["a", "bb", "ccc", "dddd"], d_model=16, codec=c)
    assert emb(torch.tensor([[0, 1, 2], [3, 0, 1]])).shape == (2, 3, 16)
    assert {n for n, p in emb.named_parameters() if p.requires_grad} == {"proj.weight"}

    print("fourier self-check: ok")


if __name__ == "__main__":
    demo()
