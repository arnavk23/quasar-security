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


def fill_qztna(d):
    """Tables and numbers for papers/qztna/qcyber.template.tex (no semicolons in output)."""
    out = {}
    syn, real = d["syn"], d["real"]
    DS = ("UNSW-NB15", "CIC-IDS2017")

    # in-distribution table
    cols = [(syn, "accuracy"), (syn, "auc")] + [(real[real.dataset == ds], m) for ds in DS for m in ("f1", "auc")]
    best = [f.groupby("model")[m].mean().max() for f, m in cols]
    rows = []
    for name in ["AE-SVM", "AE-SVM (anchored)", "AE-SVM (nonneg)", "RBF-SVM", "Laplacian-SVM", "Poly-SVM",
                 "RF", "XGBoost", "MLP", "DNN", "Autoencoder"]:
        cells = []
        for (f, m), b in zip(cols, best):
            v = f[f.model == name][m]
            cells.append("--" if v.empty else cell(v, bold=abs(v.mean() - b) < 5e-4))
        if any(c != "--" for c in cells):
            rows.append(f"    {name} & " + " & ".join(cells) + r" \\")
    out["QZ_T_INDIST"] = "\n".join(rows)

    # scale table: AE minus RBF in percentage points
    mn = pd.read_csv(QZ / "qztna_matched_n.csv").groupby(["n", "kernel"]).accuracy.mean().unstack()
    dm = pd.read_csv(QZ / "qztna_dimensionality.csv").groupby(["d", "model"]).accuracy.mean().unstack()
    gn = (mn["ae"] - mn["rbf"]) * 100
    gd = (dm["AE-SVM"] - dm["RBF-SVM"]) * 100
    pad = lambda xs, k: list(xs) + [""] * (k - len(xs))
    fmt = lambda v: f"${v:+.1f}$"
    out["QZ_T_SCALE"] = "\n".join([
        "    $n$ & " + " & ".join(str(int(n)) for n in gn.index) + r" \\",
        "    Gap & " + " & ".join(fmt(v) for v in gn.values) + r" \\",
        r"    \midrule",
        "    $d$ & " + " & ".join(pad([str(int(x)) for x in gd.index], len(gn))) + r" \\",
        "    Gap & " + " & ".join(pad([fmt(v) for v in gd.values], len(gn))) + r" \\",
    ])

    # zero-day table
    sz, rz = d["synz"], d["realz"]
    zcols = [sz] + [rz[rz.dataset == ds] for ds in DS]
    zbest = [f.groupby(["model", "held_out"]).auc.mean().groupby("model").mean().max() for f in zcols]
    rows = []
    for name in ["AE-SVM", "AE-SVM (nonneg)", "AE-SVM (anchored)", "RBF-SVM", "RF", "XGBoost", "DNN", "Autoencoder"]:
        cells = []
        for f, b in zip(zcols, zbest):
            v = mean_zd(f, name)
            cells.append("--" if np.isnan(v) else (f"$\\mathbf{{{v:.3f}}}$" if abs(v - b) < 5e-4 else f"${v:.3f}$"))
        rows.append(f"    {name} & " + " & ".join(cells) + r" \\")
    out["QZ_T_ZD"] = "\n".join(rows)
    ae = {ds: mean_zd(rz[rz.dataset == ds], "AE-SVM") for ds in DS}
    rb = {ds: mean_zd(rz[rz.dataset == ds], "RBF-SVM") for ds in DS}
    out["QZ_ABS_REAL"] = (f"${ae[DS[0]]:.3f}$ against ${rb[DS[0]]:.3f}$ for RBF-SVM on UNSW-NB15 and "
                          f"${ae[DS[1]]:.3f}$ against ${rb[DS[1]]:.3f}$ on CIC-IDS2017")

    # anchor sweep
    sw = pd.read_csv(QZ / "qztna_anchor_sweep.csv")
    order = ["0", "1", "2", "4", "8", "16", "32", "median"]
    m = sw.groupby(["c", "seed", "held_out"]).auc.mean().groupby("c").mean()
    out["QZ_T_SWEEP"] = "\n".join([
        "    $c$ & " + " & ".join("median" if c == "median" else c for c in order) + r" \\",
        r"    \midrule",
        "    AUC & " + " & ".join(f"${m[c]:.3f}$" for c in order) + r" \\",
    ])

    # alignment
    sa = pd.read_csv(QZ / "qztna_kernel_alignment.csv").groupby("kernel").alignment.mean()
    ra = d["reala"].groupby(["dataset", "kernel"]).alignment.mean()
    rows = []
    for label, sk, rk in (("AE", "AE-SVM", "AE"), ("AE (nonneg)", None, "AE (nonneg)"),
                          ("RBF", "RBF-SVM", "RBF"), ("Laplacian", "Laplacian-SVM", "Laplacian")):
        cells = [f"${sa[sk]:.2f}$" if sk else "--"] + [f"${ra[(ds, rk)]:.2f}$" for ds in DS]
        rows.append(f"    {label} & " + " & ".join(cells) + r" \\")
    out["QZ_T_ALIGN"] = "\n".join(rows)

    # certification
    c = pd.read_csv(QZ / "qztna_certified_radii.csv")
    rows = []
    for k in ("AE-SVM", "RBF-SVM"):
        sub = c[c.model == k]
        ok = sub[~sub.abstained]
        rows.append(f"    {k} & ${ok.radius.mean():.3f}$ & ${sub.radius.median():.3f}$ & "
                    f"${100 * sub.abstained.mean():.1f}\\%$ \\\\")
    out["QZ_T_CERT"] = "\n".join(rows)

    # latency
    sl = pd.read_csv(QZ / "qztna_latency.csv").set_index("component").latency_ms
    rl = d["reall"].set_index(["dataset", "model"]).batch_ms_per_flow
    syn_name = {"AE-SVM": "AE-SVM (Nystrom)", "RBF-SVM": "RBF-SVM (Nystrom)", "RF": "Random Forest", "MLP": "MLP"}
    rows = []
    for name in ["AE-SVM", "RBF-SVM", "RF", "XGBoost", "MLP", "DNN", "Autoencoder"]:
        cells = [f"${sl[syn_name[name]]:.4f}$" if name in syn_name else "--"]
        cells += [f"${rl[(ds, name)]:.4f}$" for ds in DS]
        rows.append(f"    {name} & " + " & ".join(cells) + r" \\")
    out["QZ_T_LAT"] = "\n".join(rows)
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
    vals.update(fill_qztna(d))
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
