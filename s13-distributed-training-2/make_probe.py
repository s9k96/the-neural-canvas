"""Generate `s13_probe.ipynb` — a short notebook that sizes the real run before it is booked.

The S13 harness will ask a Colab T4 for three training runs of 50 million tokens each. At
fp32 a T4 is about 8 TFLOPS, so that could be twenty minutes or it could be four hours, and
the difference decides whether the assignment is one sitting or several. This probe answers
it in a couple of minutes, the same way the gloo probes sized Session 12 before any of its
engine was written.

The notebook carries the engine with it. `s13_reversible.py` is not committed, so a Colab
`git clone` cannot reach it; instead the source is embedded here at build time and written to
disk by the notebook's first cell. One copy, no divergence, nothing to push first.

    python make_probe.py        # writes s13_probe.ipynb
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "s13_reversible.py"
OUT = HERE / "s13_probe.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


ENGINE_CELL = (
    "# The reversible engine, embedded so this notebook needs nothing else. Written to disk\n"
    "# rather than pasted into a cell so the import below is the same module the repo tests.\n"
    "engine_src = r'''" + ENGINE.read_text(encoding="utf-8").replace("'''", "\\'\\'\\'") + "'''\n"
    "open('s13_reversible.py', 'w').write(engine_src)\n"
    "print(f'wrote s13_reversible.py ({len(engine_src):,} bytes)')\n"
)

ENV_CELL = '''import subprocess, torch, platform
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
except FileNotFoundError:
    print("no nvidia-smi on this machine")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | device {dev}")
if dev.type == "cuda":
    cap = torch.cuda.get_device_capability()
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"compute capability {cap[0]}.{cap[1]} | VRAM {total/1e9:.1f} GB")
    # bf16 needs Ampere (8.0+). A T4 is Turing (7.5), so the low-precision work has to be
    # fp16 there, and the harness must record which one it actually used.
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"fp16 supported: True (all CUDA)")
'''

CORRECTNESS_CELL = '''import torch, s13_reversible as R

# Does the custom autograd.Function work on CUDA at all? Everything downstream assumes it.
# float64 is the exactness check; a T4 is slow at it but this is 2x16 tokens.
print(f"{'rule':<10}{'device':>8}{'dtype':>10}{'worst rel grad err':>21}")
for rule in R.REVERSIBLE:
    for dv, dt in ((dev_name, torch.float32), ("cpu", torch.float64)):   # not `dev`: that is the torch.device every later cell uses
        cfg = R.Config(rule=rule, n_layer=6, d_model=128, vocab_size=512, block_size=64,
                       h=0.25 if rule != "revnet" else 1.0,
                       gamma=0.5 if rule == "blended" else 0.0)
        e = R.gradient_check(cfg, batch=2, seq=16, device=dv, dtype=dt)["worst_rel_grad_error"]
        print(f"{rule:<10}{dv:>8}{str(dt).replace('torch.',''):>10}{e:>21.2e}")
'''

SIZE_CELL = '''import s13_reversible as R

# Find the width that puts each depth at ~20M parameters, which is what the assignment asks
# for and what the depth sweep needs to hold fixed.
def params_at(d_model, n_layer, vocab=8192, block=512, rule="standard"):
    return R.Model(R.Config(rule=rule, d_model=d_model, n_layer=n_layer,
                            vocab_size=vocab, block_size=block)).n_params()

TARGET = 20_000_000
print(f"{'layers':>7}{'d_model':>9}{'params':>12}")
FIT = {}
for n_layer in (6, 10, 16, 24):
    best = min(range(128, 769, 64), key=lambda d: abs(params_at(d, n_layer) - TARGET))
    FIT[n_layer] = best
    print(f"{n_layer:>7}{best:>9}{params_at(best, n_layer):>12,}")
BASE_LAYERS = 10
BASE_D = FIT[BASE_LAYERS]
print(f"\\nbase config for the three required runs: {BASE_LAYERS} layers, d_model {BASE_D}, "
      f"{params_at(BASE_D, BASE_LAYERS):,} params")
'''

BENCH_CELL = '''import time, torch, torch.nn.functional as F, s13_reversible as R

def bench(rule, reversible, batch, seq=512, dtype=torch.float32, steps=8,
          d_model=None, n_layer=None):
    """Tokens per second and peak memory for one configuration. Not training — timing."""
    d_model = d_model or BASE_D; n_layer = n_layer or BASE_LAYERS
    cfg = R.Config(rule=rule, d_model=d_model, n_layer=n_layer, vocab_size=8192,
                   block_size=seq, h=0.25 if rule in ("midpoint", "blended", "euler") else 1.0,
                   gamma=0.5 if rule == "blended" else 0.0)
    m = R.Model(cfg, reversible=reversible).to(device=dev, dtype=dtype)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x = torch.randint(8192, (batch, seq), device=dev)
    R.peak_bytes_reset(dev)
    t0 = time.time()
    for i in range(steps):
        if i == 2:                      # skip two warm-up steps, then start the clock
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        loss = F.cross_entropy(m(x).float().reshape(-1, 8192), x.reshape(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt_s = max(time.time() - t0, 1e-9)
    tps = batch * seq * max(steps - 2, 1) / dt_s
    peak = R.peak_bytes_read(dev)
    del m, opt, x
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return tps, peak, loss.item()

BATCH, SEQ = 8, 512
rows = []
print(f"{'path':<24}{'dtype':>9}{'tok/s':>11}{'peak MB':>10}{'50M tokens':>13}")
for label, rule, rev in (("standard (stored)", "standard", False),
                         ("midpoint (reversible)", "midpoint", True),
                         ("revnet (reversible)", "revnet", True)):
    for dtype in (torch.float32, torch.float16):
        try:
            tps, peak, _ = bench(rule, rev, BATCH, SEQ, dtype)
            mins = 50e6 / tps / 60
            rows.append((label, str(dtype).replace("torch.", ""), tps, peak, mins))
            print(f"{label:<24}{str(dtype).replace('torch.',''):>9}{tps:>11,.0f}"
                  f"{(peak or 0)/1e6:>10.0f}{mins:>12.1f}m")
        except Exception as exc:
            print(f"{label:<24}{str(dtype).replace('torch.',''):>9}  FAILED: {str(exc)[:50]}")
'''

MAXBATCH_CELL = '''# How much batch does the saved memory actually buy? This is the assignment's third run.
def max_batch(rule, reversible, seq=512, dtype=torch.float32, cap=2048):
    b, best = 1, 0
    while b <= cap:
        try:
            bench(rule, reversible, b, seq, dtype, steps=3)
            best = b; b *= 2
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            break
    return best

DT = torch.float32
print(f"{'path':<24}{'max batch':>11}{'tokens/step':>14}")
mb = {}
for label, rule, rev in (("standard (stored)", "standard", False),
                         ("midpoint (reversible)", "midpoint", True)):
    mb[label] = max_batch(rule, rev, 512, DT)
    print(f"{label:<24}{mb[label]:>11}{mb[label]*512:>14,}")
if mb.get("standard (stored)"):
    print(f"\\nreversibility buys "
          f"{mb['midpoint (reversible)']/mb['standard (stored)']:.1f}x the batch")
'''

REPORT_CELL = '''import json
probe = {
    "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
    "capability": list(torch.cuda.get_device_capability()) if dev.type == "cuda" else None,
    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9, 1) if dev.type == "cuda" else None,
    "bf16": torch.cuda.is_bf16_supported() if dev.type == "cuda" else False,
    "torch": torch.__version__,
    "base_layers": BASE_LAYERS, "base_d_model": BASE_D,
    "fit": FIT,
    "bench": [{"path": a, "dtype": b, "tok_s": round(c), "peak_bytes": d,
               "minutes_for_50M": round(e, 1)} for a, b, c, d, e in rows],
    "max_batch": mb,
}
print(json.dumps(probe, indent=1))
print("\\n^ paste this back")
'''

cells = [
    md("# S13 probe — how long will the real run take?\\n\\n"
       "Three training runs of 50M tokens are about to be booked on this GPU. This notebook "
       "measures throughput, peak memory and maximum batch size first, so the real harness "
       "can be sized instead of guessed at.\\n\\n"
       "**Run all, then paste the final JSON back.** Two or three minutes."),
    code(ENGINE_CELL),
    md("## 1 · What hardware is this, and what precision can it do?"),
    code(ENV_CELL),
    code("dev_name = 'cuda' if torch.cuda.is_available() else 'cpu'"),
    md("## 2 · Does the reversible stack work on this device?\\n\\n"
       "The custom `autograd.Function` has only ever run on CPU. If reversible gradients do "
       "not match ordinary autograd here, nothing measured below means anything."),
    code(CORRECTNESS_CELL),
    md("## 3 · What width gives 20M parameters at each depth?"),
    code(SIZE_CELL),
    md("## 4 · Throughput and memory\\n\\n"
       "Two paths, two precisions. The last column is the one that decides the plan: how long "
       "50M tokens takes."),
    code(BENCH_CELL),
    md("## 5 · Maximum batch size\\n\\n"
       "The assignment's third run pushes reversibility to the largest batch that fits. This "
       "is how much that is."),
    code(MAXBATCH_CELL),
    md("## 6 · Report"),
    code(REPORT_CELL),
]

nb = {
    "cells": [dict(c, id=f"probe{i:02d}") for i, c in enumerate(cells)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python"},
                 "accelerator": "GPU", "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT.name} · {len(cells)} cells · engine embedded "
      f"({len(ENGINE_CELL):,} bytes)")
