#!/usr/bin/env python3
"""
Prints the LaTeX table bodies and headline numbers used in papers/*/ from the
result CSVs, so no number in a paper is typed by hand.

    python scripts/make_tables.py            # everything
    python scripts/make_tables.py qztna      # one section
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
HY, QZ = ROOT / "data" / "hyqad", ROOT / "data" / "qztna"


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def pm(x, digits=3):
    return f"${np.nanmean(x):.{digits}f} \\pm {ci95(x):.{digits}f}$"


def section(title):
    print(f"\n% ===== {title} =====")


def hyqad():
    b = pd.read_csv(HY / "benchmark.csv")
    m = b[~b.model.str.startswith("mcnemar")]
    section("HYQAD Table: detection at n=1400 (MCC, AUC, ECE per strength)")
    order = ["RF", "RF-isotonic", "Hybrid", "RBF-SVM", "GP", "AE-SVM", "AE-SVM (anchored)",
             "OC-SVM", "IF", "LOF"]
    for name in order:
        cells = []
        for s in (1.0, 0.3):
            r = m[(m.model == name) & (m.strength == s)]
            if r.empty:
                cells += ["--"] * 3
                continue
            cells += [pm(r.mcc), f"${r.auc.mean():.3f}$",
                      f"${r.ece.mean():.3f}$" if r.ece.notna().any() else "--"]
        print(f"    {name:<18} & " + " & ".join(cells) + r" \\")
    mc = b[b.model.str.startswith("mcnemar")]
    section("HYQAD McNemar (pooled discordant pairs, #seeds with p<0.05)")
    for (s, pair), g in mc.groupby(["strength", "model"]):
        print(f"%  s={s} {pair[8:]}: n01={int(g.n01.sum())} n10={int(g.n10.sum())} "
              f"significant={(g.p < 0.05).sum()}/{len(g)}")
    z = pd.read_csv(HY / "zeroday.csv")
    section("HYQAD zero-day mean AUC over held-out attacks")
    print("% " + z.groupby(["strength", "model"]).auc.mean().unstack(0).round(3).to_string().replace("\n", "\n% "))


def qztna_synthetic():
    d = pd.read_csv(QZ / "qztna_detection_benchmark.csv")
    section("Q-ZTNA Table: synthetic detection (10 seeds)")
    for name in ["AE-SVM", "AE-SVM (anchored)", "RBF-SVM", "Laplacian-SVM", "Poly-SVM", "RF", "XGBoost", "MLP"]:
        r = d[d.model == name]
        print(f"    {name:<18} & " + " & ".join(pm(r[c]) for c in
              ("accuracy", "f1", "auc", "precision", "recall")) + r" \\")
    a = lambda n: d[d.model == n].sort_values("seed")
    for x, y in [("AE-SVM", "RBF-SVM"), ("AE-SVM (anchored)", "RBF-SVM"), ("AE-SVM (anchored)", "AE-SVM")]:
        print(f"%  paired t-test {x} vs {y}: accuracy p={ttest_rel(a(x).accuracy, a(y).accuracy).pvalue:.2g}, "
              f"AUC p={ttest_rel(a(x).auc, a(y).auc).pvalue:.2g}")
    z = pd.read_csv(QZ / "qztna_zeroday.csv")
    section("Q-ZTNA synthetic zero-day AUC")
    t = z.groupby(["model", "held_out"]).auc.mean().unstack()
    t["mean"] = t.mean(1)
    print("% " + t.round(3).to_string().replace("\n", "\n% "))


def qztna_real():
    path = QZ / "real_detection.csv"
    if not path.exists():
        print("\n% real-data results not available yet")
        return
    d = pd.read_csv(path)
    for ds in ("UNSW-NB15", "CIC-IDS2017"):
        section(f"Real detection: {ds} (5 seeds; accuracy, F1, AUC, recall, FPR)")
        for name in ["AE-SVM", "AE-SVM (anchored)", "AE-SVM (nonneg)", "RBF-SVM", "Laplacian-SVM", "Poly-SVM",
                     "RF", "XGBoost", "MLP", "DNN", "Autoencoder"]:
            r = d[(d.dataset == ds) & (d.model == name)]
            if r.empty:
                continue
            print(f"    {name:<18} & " + " & ".join(pm(r[c]) for c in
                  ("accuracy", "f1", "auc", "recall", "fpr")) + r" \\")
    zp = QZ / "real_zeroday.csv"
    if zp.exists():
        z = pd.read_csv(zp)
        for ds in ("UNSW-NB15", "CIC-IDS2017"):
            section(f"Real zero-day AUC: {ds}")
            t = z[z.dataset == ds].groupby(["model", "held_out"]).auc.mean().unstack()
            t["mean"] = t.mean(1)
            print("% " + t.round(3).to_string().replace("\n", "\n% "))
    mp = QZ / "real_matched_n.csv"
    if mp.exists():
        mn = pd.read_csv(mp)
        section("Real matched-n AUC")
        print("% " + mn.groupby(["dataset", "n", "model"]).auc.mean().unstack().round(3).to_string().replace("\n", "\n% "))
    ap = QZ / "real_kernel_alignment.csv"
    if ap.exists():
        al = pd.read_csv(ap)
        section("Real kernel-target alignment (mean over families)")
        print("% " + al.groupby(["dataset", "kernel"]).alignment.mean().unstack().round(3).to_string().replace("\n", "\n% "))


SECTIONS = dict(hyqad=hyqad, qztna=qztna_synthetic, real=qztna_real)

if __name__ == "__main__":
    for name in sys.argv[1:] or SECTIONS:
        SECTIONS[name]()
