#!/usr/bin/env python3
"""
Fills the data-derived placeholders (@NAME@) in the paper sources from the
result CSVs. Run after all experiments:

    python scripts/fill_papers.py

Templates live next to each paper as *.template.tex; the filled file is the
paper's main .tex. Missing data leaves a visible [pending: NAME] marker.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
HY, QZ = ROOT / "data" / "hyqad", ROOT / "data" / "qztna"
SCRATCH_SECTIONS = ROOT / "papers" / "combined" / "sections"


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def cell(x, bold=False, digits=3):
    def short(v):
        t = f"{v:.{digits}f}"
        return t[1:] if t.startswith("0") else t
    s, c = short(np.mean(x)), short(ci95(x))
    body = f"{s}{{\\scriptstyle\\pm{c}}}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def load():
    d = {}
    b = pd.read_csv(HY / "benchmark.csv")
    d["hy"] = b[~b.model.str.startswith("mcnemar")]
    d["syn"] = pd.read_csv(QZ / "qztna_detection_benchmark.csv")
    rp = QZ / "real_detection.csv"
    d["real"] = pd.read_csv(rp) if rp.exists() else None
    for k, f in (("hyz", HY / "zeroday.csv"), ("synz", QZ / "qztna_zeroday.csv"),
                 ("realz", QZ / "real_zeroday.csv"), ("reala", QZ / "real_kernel_alignment.csv"),
                 ("realm", QZ / "real_matched_n.csv"), ("reall", QZ / "real_latency.csv")):
        d[k] = pd.read_csv(f) if f.exists() else None
    return d


def indist_rows(d):
    models = ["AE-SVM", "AE-SVM (anchored)", "RBF-SVM", "Laplacian-SVM", "Poly-SVM", "GP", "Hybrid",
              "RF", "RF-isotonic", "XGBoost", "MLP", "DNN", "Autoencoder"]
    cols = []  # (frame, filter, metric)
    hy = d["hy"][d["hy"].strength == 0.3]
    cols += [(hy, None, "mcc"), (hy, None, "auc"), (d["syn"], None, "accuracy"), (d["syn"], None, "auc")]
    real = d["real"]
    for ds in ("UNSW-NB15", "CIC-IDS2017"):
        sub = real[real.dataset == ds] if real is not None else None
        cols += [(sub, None, "f1"), (sub, None, "auc")]
    best = []
    for frame, _, metric in cols:
        if frame is None:
            best.append(None)
            continue
        means = frame.groupby("model")[metric].mean()
        best.append(means.max())
    rows = []
    for m in models:
        cells = []
        for (frame, _, metric), b in zip(cols, best):
            if frame is None or m not in set(frame.model):
                cells.append("--")
                continue
            v = frame[frame.model == m][metric]
            cells.append(cell(v, bold=abs(v.mean() - b) < 5e-4))
        if all(c == "--" for c in cells):
            continue
        rows.append(f"    {m} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def mean_zd(frame, model, **filters):
    sub = frame[frame.model == model]
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    return sub.groupby("held_out").auc.mean().mean() if len(sub) else np.nan


def fill_combined(d):
    out = {"INDIST_ROWS": indist_rows(d)}
    rz, ra, rm = d["realz"], d["reala"], d["realm"]
    real = d["real"]

    if real is not None and "AE-SVM (anchored)" in set(real.model):
        parts = []
        for ds in ("UNSW-NB15", "CIC-IDS2017"):
            sub = real[real.dataset == ds]
            a = sub[sub.model == "AE-SVM (anchored)"].accuracy.mean()
            p = sub[sub.model == "AE-SVM"].accuracy.mean()
            parts.append(f"{ds} ${a:.3f}$ vs.\\ ${p:.3f}$")
        out["ANCHOR_REAL_INDIST"] = ("On the public datasets anchoring leaves in-distribution accuracy "
                                     "essentially unchanged (" + "; ".join(parts) + ").")
    if rm is not None:
        g = rm.groupby(["dataset", "n", "model"]).auc.mean()
        u = (g[("UNSW-NB15", 200, "AE-SVM")], g[("UNSW-NB15", 200, "RBF-SVM")])
        c = (g[("CIC-IDS2017", 200, "AE-SVM")], g[("CIC-IDS2017", 200, "RBF-SVM")])
        out["MATCHED_REAL"] = (f"On the public datasets AE-SVM tracks RBF-SVM on UNSW-NB15 at every $n$ "
                               f"(AUC ${u[0]:.3f}$ vs.\\ ${u[1]:.3f}$ at $n=200$) and trails it on CIC-IDS2017 "
                               f"at small $n$ (${c[0]:.3f}$ vs.\\ ${c[1]:.3f}$); the Laplacian kernel is the "
                               f"strongest kernel at every $n$ on both.")
    if rz is not None:
        txt, abs_txt, conc = [], [], []
        for ds in ("UNSW-NB15", "CIC-IDS2017"):
            sub = rz[rz.dataset == ds]
            ae, rbf = mean_zd(sub, "AE-SVM"), mean_zd(sub, "RBF-SVM")
            ext = {m: mean_zd(sub, m) for m in ("RF", "XGBoost", "DNN", "Autoencoder", "AE-SVM (anchored)",
                                                  "AE-SVM (nonneg)")}
            txt.append((ds, ae, rbf, ext))
        line = "; ".join(f"{ds} ${ae:.3f}$ vs.\\ ${rbf:.3f}$" for ds, ae, rbf, _ in txt)
        out["ZERODAY_REAL"] = (f"On the public benchmarks, where every family is held out in turn, mean "
                               f"held-out AUC for AE-SVM versus RBF-SVM is: {line}. " +
                               " ".join(f"On {ds}, tree ensembles and the DNN reach "
                                        f"${min(e['RF'], e['XGBoost'], e['DNN']):.3f}$--"
                                        f"${max(e['RF'], e['XGBoost'], e['DNN']):.3f}$ and the benign-only "
                                        f"autoencoder ${e['Autoencoder']:.3f}$."
                                        for ds, _, _, e in txt))
        out["ABS_REAL"] = " and ".join(f"${ae:.3f}$ vs.\\ ${rbf:.3f}$ on {ds}" for ds, ae, rbf, _ in txt)
        out["CONC_REAL"] = (" (mean held-out AUC " +
                            "; ".join(f"${ae:.3f}$ vs.\\ ${rbf:.3f}$ for RBF on {ds}" for ds, ae, rbf, _ in txt) +
                            ")")
        if all(not np.isnan(e["AE-SVM (anchored)"]) for _, _, _, e in txt):
            worst = min(e["AE-SVM (anchored)"] - ae for _, ae, _, e in txt)
            out["ANCHOR_SAFE"] = (" and lowered no public-benchmark held-out AUC by more than "
                                  f"${abs(min(worst, 0)):.3f}$" if worst > -0.02 else
                                  f" but lowered public-benchmark held-out AUC by up to ${-worst:.3f}$, so it is "
                                  "a targeted fix rather than a free default")
            out["ANCHOR_REAL_ZD"] = ("On the public benchmarks the anchored encoding gives mean held-out AUC " +
                                     "; ".join(f"${e['AE-SVM (anchored)']:.3f}$ on {ds} (plain ${ae:.3f}$, "
                                               f"non-negative ${e['AE-SVM (nonneg)']:.3f}$)"
                                               for ds, ae, _, e in txt) + ".")
        out["ZD_ABS_MERGED"] = "; ".join(f"${ae:.2f}$ vs.\\ ${rbf:.2f}$ on {ds}" for ds, ae, rbf, _ in txt)
        u, c = txt[0], txt[1]
        out["REAL_TEXT"] = (
            "Table~\\ref{tab:real} reports in-distribution detection. AE-SVM and RBF-SVM are statistically tied "
            "on UNSW-NB15 and AE-SVM is marginally worse on CIC-IDS2017; tree ensembles, the Laplacian kernel and "
            "the DNN lead. On held-out attack families (Fig.~\\ref{fig:realzd}) mean AUC is "
            f"${u[1]:.3f}$ for AE-SVM versus ${u[2]:.3f}$ for RBF-SVM on UNSW-NB15, and ${c[1]:.3f}$ versus "
            f"${c[2]:.3f}$ on CIC-IDS2017; the norm-anchored encoding gives ${u[3]['AE-SVM (anchored)']:.3f}$ and "
            f"${c[3]['AE-SVM (anchored)']:.3f}$, and the benign-only autoencoder ${u[3]['Autoencoder']:.3f}$ and "
            f"${c[3]['Autoencoder']:.3f}$.")
        out["REAL_ZERODAY_README"] = "Public data: " + "; ".join(
            f"{ds} {ae:.2f} vs {rbf:.2f} (anchored {e['AE-SVM (anchored)']:.2f})" for ds, ae, rbf, e in txt) + "."
    if ra is not None:
        g = ra.groupby(["dataset", "kernel"]).alignment.mean()
        out["ALIGN_REAL"] = "; " + "; ".join(
            f"{ds} ${g[(ds, 'AE')]:.2f}$ vs.\\ ${g[(ds, 'RBF')]:.2f}$" for ds in ("UNSW-NB15", "CIC-IDS2017"))
    if d["reall"] is not None:
        lat = d["reall"].groupby("model").batch_ms_per_flow.mean()
        kern = lat[["AE-SVM", "RBF-SVM", "Laplacian-SVM", "Poly-SVM"]]
        fast = lat[["XGBoost", "MLP", "Autoencoder"]]
        out["LATENCY_REAL"] = (" On the public data, however, kernel SVMs with 1000 Nystr\\\"om landmarks are the "
                               f"slowest models in batch inference (${kern.min():.2f}$--${kern.max():.2f}$\\,ms per "
                               f"flow, averaged over both datasets) while XGBoost, the MLP and the autoencoder need "
                               f"${fast.min():.3f}$--${fast.max():.3f}$\\,ms; these timings come from a shared, "
                               f"heavily loaded machine and should be read as an ordering only.")
    return out


def real_rows(d):
    real = d["real"]
    if real is None:
        return None
    models = ["AE-SVM", "AE-SVM (anchored)", "AE-SVM (nonneg)", "RBF-SVM", "Laplacian-SVM", "Poly-SVM",
              "RF", "XGBoost", "MLP", "DNN", "Autoencoder"]
    best = {(ds, m): real[real.dataset == ds].groupby("model")[m].mean().max()
            for ds in ("UNSW-NB15", "CIC-IDS2017") for m in ("f1", "auc")}
    rows = []
    for name in models:
        if name not in set(real.model):
            continue
        cells = []
        for ds in ("UNSW-NB15", "CIC-IDS2017"):
            for metric in ("f1", "auc"):
                v = real[(real.dataset == ds) & (real.model == name)][metric]
                cells.append(cell(v, bold=abs(v.mean() - best[(ds, metric)]) < 5e-4))
        rows.append(f"    {name} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def substitute(path, values):
    text = path.read_text()
    def rep(m):
        key = m.group(1)
        return values.get(key, f"[pending: {key}]")
    new = re.sub(r"@([A-Z_]+)@", rep, text)
    path.write_text(new)
    left = re.findall(r"\[pending: ([A-Z_]+)\]", new)
    print(f"{path.relative_to(ROOT)}: {'complete' if not left else 'pending ' + ', '.join(sorted(set(left)))}")


if __name__ == "__main__":
    d = load()
    vals = fill_combined(d)
    rr = real_rows(d)
    if rr is not None:
        vals["REAL_ROWS"] = rr
    for tpl, out in ((ROOT / "papers/combined/quasar.template.tex", ROOT / "papers/combined/quasar.tex"),
                     (ROOT / "papers/qztna/qcyber.template.tex", ROOT / "papers/qztna/qcyber.tex")):
        if tpl.exists():
            out.write_text(tpl.read_text())
            substitute(out, vals)
    readme = ROOT / "README.md"
    if "@REAL_ZERODAY@" in readme.read_text() and "REAL_ZERODAY_README" in vals:
        readme.write_text(readme.read_text().replace("@REAL_ZERODAY@", vals["REAL_ZERODAY_README"]))
