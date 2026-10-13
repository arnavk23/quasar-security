#!/usr/bin/env python3
"""
Builds the flat upload package for papers/qmi/qmi.tex:

    papers/qmi/submission/   main.tex, figures, sn-jnl.cls, sn-mathphys-num.bst,
                             references.bib, main.bbl, main.pdf
    papers/qmi/qmi_submission.zip

Run python scripts/fill_papers.py first. Figure paths are flattened, so the
package compiles on its own, which is what the journal's system needs.
"""
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "papers" / "qmi"
OUT = SRC / "submission"


def run(*cmd):
    subprocess.run(cmd, cwd=OUT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    tex = (SRC / "qmi.tex").read_text()
    tex = re.sub(r"\\graphicspath\{.*?\}\n", "", tex)
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    for f in figs:
        flat = f.replace("/", "_")
        shutil.copy(ROOT / "figures" / f, OUT / flat)
        tex = tex.replace("{" + f + "}", "{" + flat + "}")
    tex = tex.replace("\\bibliography{../references}", "\\bibliography{references}")
    (OUT / "main.tex").write_text(tex)
    for f in ("sn-jnl.cls", "sn-mathphys-num.bst"):
        shutil.copy(SRC / f, OUT / f)
    shutil.copy(ROOT / "papers" / "references.bib", OUT / "references.bib")

    run("pdflatex", "-interaction=nonstopmode", "main.tex")
    run("bibtex", "main")
    run("pdflatex", "-interaction=nonstopmode", "main.tex")
    run("pdflatex", "-interaction=nonstopmode", "main.tex")
    log = (OUT / "main.log").read_text(errors="ignore")
    errors = len(re.findall(r"^!", log, flags=re.M))
    for f in OUT.iterdir():
        if f.suffix in (".aux", ".log", ".out", ".blg"):
            f.unlink()

    keep = sorted(p.name for p in OUT.iterdir())
    with zipfile.ZipFile(SRC / "qmi_submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in keep:
            z.write(OUT / name, name)
    print(f"{len(figs)} figures, {errors} LaTeX errors, files: {', '.join(keep)}")
