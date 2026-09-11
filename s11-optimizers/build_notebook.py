"""Bake s11_optimizers.py into an executed .ipynb, then bake the results into the widget.

The repo's generate-then-bake pattern (S4, S6, S7, S9, S10), applied to a notebook: the `.py`
is the source of truth, the `.ipynb` is a build artifact that carries its outputs so a reviewer
sees the numbers without running anything, and `setting-the-distance.html` gets the same
numbers as a literal JS blob.

    python build_notebook.py              # convert, execute, bake   (~55 min, CPU)
    python build_notebook.py --no-exec    # convert only (fast; leaves outputs stale)
    python build_notebook.py --bake-only  # re-inject out/evidence.json into the page only

Exits non-zero if any gate in the harness fails, so the exit code is the pass/fail signal.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "s11_optimizers.py"
NB = HERE / "s11_optimizers.ipynb"
EVIDENCE = HERE / "out" / "evidence.json"
WIDGET = HERE / "setting-the-distance.html"

BEGIN = "/* BEGIN GENERATED s11data */"
END = "/* END GENERATED s11data */"


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
    """`# ` comment prefixes back to plain markdown."""
    out = []
    for line in body.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def to_notebook(cells):
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, (kind, body) in enumerate(cells):
        # nbformat >= 4.5 requires a cell id; without one it warns on every write.
        cid = f"s11cell{i:03d}"
        if kind == "markdown":
            nb["cells"].append({"cell_type": "markdown", "id": cid, "metadata": {},
                                "source": strip_markdown(body)})
        else:
            nb["cells"].append({"cell_type": "code", "id": cid, "metadata": {},
                                "execution_count": None, "outputs": [], "source": body})
    return nb


def install_cell():
    """Colab has torch but not always tokenizers/huggingface_hub. Quiet, idempotent."""
    return {
        "cell_type": "code", "id": "s11setup", "metadata": {}, "execution_count": None, "outputs": [],
        "source": ("# Colab setup. Local runs with the deps already installed skip straight through.\n"
                   "try:\n"
                   "    import tokenizers, huggingface_hub  # noqa: F401\n"
                   "except ImportError:\n"
                   "    !pip install -q tokenizers huggingface_hub\n"),
    }


def corpus_cell():
    """The notebook reads S6's committed corpus. On Colab that means cloning the repo."""
    return {
        "cell_type": "code", "id": "s11corpus", "metadata": {}, "execution_count": None, "outputs": [],
        "source": ("# The harness reads real documents from s06-dataset-creation/corpus/, which lives in\n"
                   "# this repo. On Colab, clone it; locally this is already on disk and does nothing.\n"
                   "import os\n"
                   "if not os.path.isdir('s06-dataset-creation') and not os.path.isdir('../s06-dataset-creation'):\n"
                   "    !git clone -q https://github.com/s9k96/the-neural-canvas.git\n"
                   "    os.chdir('the-neural-canvas/s11-optimizers')\n"),
    }


def execute(path):
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
        failed = None
    except CellExecutionError as exc:
        failed = str(exc)
    nbformat.write(nb, path)
    return failed


def bake_widget():
    """Inject evidence.json into setting-the-distance.html between the generated markers."""
    if not (WIDGET.exists() and EVIDENCE.exists()):
        return False
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    blob = f"{BEGIN}\nconst S11DATA = {json.dumps(ev, indent=2, ensure_ascii=False)};\n{END}"
    html = WIDGET.read_text(encoding="utf-8")
    if BEGIN not in html or END not in html:
        print(f"  [warn] {WIDGET.name} has no generated-data markers; not baking")
        return False
    html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: blob, html, flags=re.S)
    WIDGET.write_text(html, encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true", help="convert only, do not run the notebook")
    ap.add_argument("--bake-only", action="store_true",
                    help="skip conversion and execution; re-inject out/evidence.json into the page")
    args = ap.parse_args()

    if args.bake_only:
        if not EVIDENCE.exists():
            print("no out/evidence.json to bake — run without --bake-only first")
            return 1
        ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        ok = bake_widget()
        gates = ev.get("gates", {})
        print(f"baked {len(json.dumps(ev)):,} bytes into {WIDGET.name}" if ok else "nothing baked")
        return 0 if ok and all(gates.values()) else 1

    cells = split_cells(SRC.read_text(encoding="utf-8"))
    nb = to_notebook(cells)
    nb["cells"].insert(1, install_cell())          # after the title, before the imports
    nb["cells"].insert(2, corpus_cell())
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {NB.name} · {len(nb['cells'])} cells "
          f"({sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} code)")

    if args.no_exec:
        print("--no-exec: skipping execution. Outputs in the notebook are stale.")
        return 0

    print("executing (this runs the whole harness end to end; ~55 minutes on CPU)…")
    failed = execute(NB)
    if failed:
        print(f"\nnotebook execution FAILED:\n{failed[-3000:]}")
        return 1
    print(f"executed {NB.name}")

    if not EVIDENCE.exists():
        print("no evidence.json — the harness did not reach the end")
        return 1
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    gates = ev.get("gates", {})
    passed = sum(1 for v in gates.values() if v)
    for name, ok in gates.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{passed}/{len(gates)} gates pass")

    if bake_widget():
        print(f"baked {len(json.dumps(ev)):,} bytes of evidence into {WIDGET.name}")

    return 0 if passed == len(gates) and gates else 1


if __name__ == "__main__":
    sys.exit(main())
