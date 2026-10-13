#!/usr/bin/env python3
"""
Q-ZTNA figures for the zero-day mechanism controls and the real-data
(UNSW-NB15, CIC-IDS2017) evaluation. Reads only data/qztna/*.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qztna.data.real_datasets import CICIDS_FAMILIES, UNSW_FAMILIES

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "qztna"
FIGS = ROOT / "figures" / "qztna"

# Same entity -> color map as scripts/hyqad/gen_figures.py.
STYLE = {
    "AE-SVM":            dict(color="#2a78d6", marker="o", ls="-"),
    "AE-SVM (anchored)": dict(color="#e87ba4", marker="X", ls="-"),
    "AE-SVM (nonneg)":   dict(color="#9fc3ee", marker="o", ls=":"),
    "RBF-SVM":           dict(color="#eb6834", marker="s", ls="--"),
    "Laplacian-SVM":     dict(color="#eda100", marker="<", ls="--"),
    "Poly-SVM":          dict(color="#c98500", marker=">", ls=":"),
    "RF":                dict(color="#6250d6", marker="D", ls=":"),
    "XGBoost":           dict(color="#1baf7a", marker="^", ls="-."),
    "MLP":               dict(color="#8a8984", marker="h", ls="-"),
    "DNN":               dict(color="#e34948", marker="v", ls="-"),
    "Autoencoder":       dict(color="#008300", marker="P", ls="--"),
}
SYN_ATTACKS = ["port_scan", "ddos", "exfiltration", "dns_tunneling", "arp_spoofing"]
SYN_LABEL = {"port_scan": "Port scan", "ddos": "DDoS", "exfiltration": "Exfiltration",
             "dns_tunneling": "DNS tunnel", "arp_spoofing": "ARP spoof"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W, PAGE_W = 3.45, 7.1

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "savefig.dpi": 300, "legend.frameon": False,
})


def ci95(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def save(fig, name):
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def grouped_bars(ax, df, groups, group_col, models, value="auc"):
    w = 0.84 / len(models)
    for k, m in enumerate(models):
        g = df[df.model == m].groupby(group_col)[value]
        mu, ci = g.mean().reindex(groups), g.agg(ci95).reindex(groups)
        x = np.arange(len(groups)) + (k - (len(models) - 1) / 2) * w
        ax.bar(x, mu, w * 0.92, yerr=ci, color=STYLE[m]["color"], label=m,
               error_kw=dict(lw=0.6, ecolor=MUTED, capsize=1.2))


def fig6_zeroday():
    df = pd.read_csv(DATA / "qztna_zeroday.csv")
    models = [m for m in ("AE-SVM", "AE-SVM (nonneg)", "AE-SVM (anchored)", "RBF-SVM") if m in set(df.model)]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    grouped_bars(ax, df, SYN_ATTACKS, "held_out", models)
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
    ax.set_xticks(range(len(SYN_ATTACKS)), [SYN_LABEL[a] for a in SYN_ATTACKS], rotation=15)
    ax.set_ylabel("AUC on held-out attack")
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig6_zeroday.png")


def fig16_real_detection():
    df = pd.read_csv(DATA / "real_detection.csv")
    models = [m for m in ("AE-SVM", "AE-SVM (anchored)", "RBF-SVM", "Laplacian-SVM", "Poly-SVM",
                          "RF", "XGBoost", "MLP", "DNN", "Autoencoder") if m in set(df.model)]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.3))
    for ax, metric, label in zip(axes, ("auc", "f1"), ("ROC-AUC", "F1")):
        grouped_bars(ax, df.assign(g=df.dataset), ["UNSW-NB15", "CIC-IDS2017"], "g", models, metric)
        ax.set_xticks([0, 1], ["UNSW-NB15", "CIC-IDS2017"])
        ax.set_ylabel(f"{label} (mean ± 95% CI, 5 seeds)")
        ax.set_ylim(0.0 if metric == "f1" else 0.7, 1.005)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5)
    save(fig, "fig16_real_detection.png")


def fig17_real_zeroday():
    df = pd.read_csv(DATA / "real_zeroday.csv")
    models = [m for m in ("AE-SVM", "AE-SVM (anchored)", "RBF-SVM", "RF", "XGBoost", "DNN", "Autoencoder")
              if m in set(df.model)]
    fig, axes = plt.subplots(2, 1, figsize=(PAGE_W, 4.2))
    for ax, ds, fams in zip(axes, ("UNSW-NB15", "CIC-IDS2017"), (UNSW_FAMILIES, CICIDS_FAMILIES)):
        sub = df[df.dataset == ds]
        grouped_bars(ax, sub, fams, "held_out", models)
        ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
        ax.set_xticks(range(len(fams)), [{"Reconnaissance": "Recon."}.get(f, f) for f in fams])
        ax.set_ylabel("AUC on held-out family")
        ax.set_title(f"{ds}: leave-one-attack-family-out")
        ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(models))
    save(fig, "fig17_real_zeroday.png")


def fig18_real_matched_n():
    df = pd.read_csv(DATA / "real_matched_n.csv")
    models = [m for m in ("AE-SVM", "AE-SVM (nonneg)", "RBF-SVM", "Laplacian-SVM") if m in set(df.model)]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.2))
    for ax, ds in zip(axes, ("UNSW-NB15", "CIC-IDS2017")):
        sub = df[df.dataset == ds]
        for m in models:
            g = sub[sub.model == m].groupby("n").auc
            mu, ci = g.mean(), g.agg(ci95)
            ax.plot(mu.index, mu, label=m, **STYLE[m])
            ax.fill_between(mu.index, mu - ci, mu + ci, color=STYLE[m]["color"], alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("Training flows n (all models)")
        ax.set_title(ds)
    axes[0].set_ylabel("ROC-AUC (mean ± 95% CI)")
    axes[1].legend(loc="lower right")
    save(fig, "fig18_real_matched_n.png")


if __name__ == "__main__":
    for f in (fig6_zeroday, fig16_real_detection, fig17_real_zeroday, fig18_real_matched_n):
        try:
            f()
        except FileNotFoundError as e:
            print(f"  skipped {f.__name__}: {e.filename} missing")
