"""Gera `dfu_tissue_segmentation.ipynb` (e, opcionalmente, um script plano) a partir de `parts/*.py` em formato percent.

Uso:  python build_notebook.py [--flat caminho/script.py]

Cada ficheiro em `parts/` é uma sequência de células separadas por linhas `# %%` (código) ou `# %% [markdown]` (texto; as linhas
seguintes são comentários `# ...` que passam a Markdown). O notebook gerado não contém saídas: estas só existem depois de o executar.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
CELL_RE = re.compile(r"^# %%(?: \[(markdown)\])?\s*$")


def parse_cells(text: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            cells.append(("markdown" if m.group(1) else "code", []))
        elif cells:
            cells[-1][1].append(line)
    out = []
    for kind, lines in cells:
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            body = "\n".join(re.sub(r"^# ?", "", l) for l in lines)
        else:
            body = "\n".join(lines)
        if body.strip():
            out.append((kind, body))
    return out


def build(flat: Path | None = None) -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    flat_lines: list[str] = ["# Gerado por build_notebook.py — script plano equivalente ao notebook (para testes fora do Jupyter)."]
    for part in sorted((HERE / "parts").glob("*.py")):
        for kind, body in parse_cells(part.read_text()):
            nb.cells.append(nbf.v4.new_markdown_cell(body) if kind == "markdown" else nbf.v4.new_code_cell(body))
            if kind == "code":
                flat_lines += ["", f"# ---- {part.name} ----", body]
    path = HERE / "dfu_tissue_segmentation.ipynb"
    nbf.write(nb, path)
    if flat:
        flat.write_text("\n".join(flat_lines) + "\n")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", type=Path, default=None)
    args = ap.parse_args()
    p = build(args.flat)
    print("Escrito:", p, "| células:", len(nbf.read(p, as_version=4).cells))
