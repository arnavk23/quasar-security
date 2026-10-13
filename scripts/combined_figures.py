#!/usr/bin/env python3
"""
Cross-domain figures for papers/combined/quasar.tex. Reads data/hyqad/*.csv and
data/qztna/*.csv only.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HY, QZ = ROOT / "data" / "hyqad", ROOT / "data" / "qztna"
FIGS = ROOT / "figures" / "combined"
FIGS.mkdir(parents=True, exist_ok=True)

COLORS = {"AE-SVM": "#2a78d6", "AE-SVM (nonneg)": "#9fc3ee", "AE-SVM (anchored)": "#e87ba4",
          "RBF-SVM": "#eb6834", "RF": "#6250d6", "DNN": "#e34948", "Autoencoder": "#008300"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 650, "legend.frameon": False,
})


def seed_means(df, group_cols=("seed",)):
    """Mean AUC over held-out attacks per seed, so CIs are over seeds."""
    return df.groupby(list(group_cols)).auc.mean()


def ci95(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def zeroday_domains():
    """Returns {domain label: {model: per-seed mean zero-day AUC}}."""
    out = {}
    hz = pd.read_csv(HY / "zeroday.csv")
    hz = hz[hz.strength == 0.3]
    out["QKD ($s{=}0.3$)"] = {m: seed_means(hz[hz.model == m]) for m in hz.model.unique()}
    ab = pd.read_csv(HY / "zeroday_mahal_ablation.csv")
    ab = ab[(ab.strength == 0.3) & (ab.features == "without Mahalanobis")]
    out["QKD, no Mahal."] = {m: seed_means(ab[ab.model == m]) for m in ab.model.unique()}
    sz = pd.read_csv(QZ / "qztna_zeroday.csv")
    out["Synthetic flows"] = {m: seed_means(sz[sz.model == m]) for m in sz.model.unique()}
    rp = QZ / "real_zeroday.csv"
    if rp.exists():
        rz = pd.read_csv(rp)
        for ds in ("UNSW-NB15", "CIC-IDS2017"):
            sub = rz[rz.dataset == ds]
            out[ds] = {m: seed_means(sub[sub.model == m]) for m in sub.model.unique()}
    return out


def fig_zeroday_all():
    doms = zeroday_domains()
    models = ["AE-SVM", "AE-SVM (nonneg)", "AE-SVM (anchored)", "RBF-SVM", "RF"]
    labels = list(doms)
    fig, ax = plt.subplots(figsize=(7.1, 2.4))
    w = 0.84 / len(models)
    for k, m in enumerate(models):
        mu = [doms[d][m].mean() if m in doms[d] else np.nan for d in labels]
        ci = [ci95(doms[d][m]) if m in doms[d] else 0 for d in labels]
        x = np.arange(len(labels)) + (k - (len(models) - 1) / 2) * w
        ax.bar(x, mu, w * 0.92, yerr=ci, color=COLORS[m], label=m,
               error_kw=dict(lw=0.6, ecolor=MUTED, capsize=1.2))
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Mean AUC on held-out attack types")
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(models))
    fig.savefig(FIGS / "fig_zeroday_all.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_zeroday_all.png")


if __name__ == "__main__":
    fig_zeroday_all()
