"""Bake s13_reversibility.py into a Colab-ready .ipynb, and bake its results into the page.

This is the repo's usual generate-then-bake pattern with one deliberate break. S9 through S12
execute their notebook locally with nbclient and commit it with its outputs. S13 cannot: the
assignment asks for peak GPU memory and 150M tokens of training, and this machine has no CUDA.
So the notebook is *generated* here, *executed* on a Colab GPU, and the `out/evidence.json` it
produces comes back to be baked into the page.

That also means the notebook has to stand alone. `s13_reversible.py` is not committed to the
repo yet, so a Colab `git clone` cannot reach it; instead its source is embedded at build time
in a cell that writes it to disk. One copy, no divergence, nothing to push first.

    python build_notebook.py               # .py -> s13_reversibility.ipynb  (no execution)
    python build_notebook.py --smoke       # also run the CPU smoke path first
    python build_notebook.py --bake-only   # inject out/evidence.json into the page
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "s13_reversibility.py"
ENGINE = HERE / "s13_reversible.py"
NB = HERE / "s13_reversibility.ipynb"
EVIDENCE = HERE / "out" / "evidence.json"
WIDGET = HERE / "forget-the-middle.html"

BEGIN = "/* BEGIN GENERATED s13data */"
END = "/* END GENERATED s13data */"


def split_cells(text):
    """Split a `# %%` / `# %% [markdown]` delimited script into notebook cells."""
    cells, kind, buf = [], "code", []

    def flush():
        body = "\n".join(buf).strip("\n")
        if body.strip():
            cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
            continue
        buf.append(line)
    flush()
    return cells


def strip_markdown(body):
    out = []
    for line in body.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def setup_cell():
    """Colab has torch; it does not always have the tokenizer or dataset libraries."""
    return ("# Colab setup. A machine that already has these skips straight through.\n"
            "try:\n"
            "    import tokenizers, huggingface_hub, datasets  # noqa: F401\n"
            "except ImportError:\n"
            "    !pip install -q tokenizers huggingface_hub datasets\n")


def engine_cell():
    """Write the engine to disk from a copy embedded at build time.

    Embedding rather than cloning keeps the notebook runnable anywhere, and generating this
    cell from the real file means the two cannot drift apart.
    """
    src = ENGINE.read_text(encoding="utf-8").replace("'''", "\\'\\'\\'")
    return ("# The reversible engine, carried with the notebook so it needs nothing else.\n"
            f"engine_src = r'''{src}'''\n"
            "open('s13_reversible.py', 'w').write(engine_src)\n"
            "\n"
            "# Evict any copy that is already imported. Colab keeps one Python process across\n"
            "# notebooks, so a module imported by an earlier run -- the sizing probe, say --\n"
            "# stays in sys.modules, and a later `import` silently hands back that OLD object\n"
            "# even though the file on disk has just been rewritten. The failure then surfaces\n"
            "# many cells later as a missing attribute, pointing at the wrong thing entirely.\n"
            "import sys\n"
            "sys.modules.pop('s13_reversible', None)\n"
            "import s13_reversible as _engine\n"
            "_need = ('fit_config', 'match_params', 'count_params', 'chunked_ce',\n"
            "         'gradient_check', 'Model', 'Config')\n"
            "_missing = [n for n in _need if not hasattr(_engine, n)]\n"
            "assert not _missing, (\n"
            "    f'stale engine loaded, missing {_missing}. '\n"
            "    'Runtime > Restart session, then Run all.')\n"
            "print(f'wrote and loaded s13_reversible.py ({len(engine_src):,} bytes)')\n")


def to_notebook(cells):
    nb = {"cells": [], "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU", "colab": {"provenance": []}},
        "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, body) in enumerate(cells):
        cid = f"s13cell{i:03d}"
        if kind == "markdown":
            nb["cells"].append({"cell_type": "markdown", "id": cid, "metadata": {},
                                "source": strip_markdown(body)})
        else:
            nb["cells"].append({"cell_type": "code", "id": cid, "metadata": {},
                                "execution_count": None, "outputs": [], "source": body})
    return nb


def bake_widget():
    if not (WIDGET.exists() and EVIDENCE.exists()):
        return False
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    blob = f"{BEGIN}\nconst S13DATA = {json.dumps(ev, indent=2, ensure_ascii=False)};\n{END}"
    html = WIDGET.read_text(encoding="utf-8")
    if BEGIN not in html or END not in html:
        print(f"  [warn] {WIDGET.name} has no generated-data markers; not baking")
        return False
    html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: blob, html, flags=re.S)
    WIDGET.write_text(html, encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake-only", action="store_true",
                    help="skip generation; inject out/evidence.json into the page")
    ap.add_argument("--smoke", action="store_true",
                    help="run the CPU smoke path (S13_QUICK=1) before generating")
    args = ap.parse_args()

    if args.bake_only:
        if not EVIDENCE.exists():
            print("no out/evidence.json — run the notebook on a GPU first")
            return 1
        ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        gates = ev.get("gates", {})
        ok = bake_widget()
        print(f"baked {len(json.dumps(ev)):,} bytes into {WIDGET.name}" if ok else "nothing baked")
        for name, passed in gates.items():
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        print(f"{sum(1 for v in gates.values() if v)}/{len(gates)} gates pass")
        return 0 if ok and gates and all(gates.values()) else 1

    if args.smoke:
        print("running the CPU smoke path…")
        env = dict(os.environ, S13_QUICK="1")
        proc = subprocess.run([sys.executable, str(SRC)], env=env, cwd=str(HERE))
        if proc.returncode:
            print("smoke path failed — not generating the notebook")
            return 1

    cells = split_cells(SRC.read_text(encoding="utf-8"))
    nb = to_notebook(cells)
    nb["cells"].insert(1, {"cell_type": "code", "id": "s13setup", "metadata": {},
                           "execution_count": None, "outputs": [], "source": setup_cell()})
    nb["cells"].insert(2, {"cell_type": "code", "id": "s13engine", "metadata": {},
                           "execution_count": None, "outputs": [], "source": engine_cell()})
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"wrote {NB.name} · {len(nb['cells'])} cells ({code} code), engine embedded")
    print("\nThis notebook is NOT executed here — it needs a CUDA GPU for peak-memory")
    print("measurement. Run it on Colab, then bring out/evidence.json back and run:")
    print("    python build_notebook.py --bake-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
