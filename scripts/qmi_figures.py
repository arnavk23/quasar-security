#!/usr/bin/env python3
"""
Figures for papers/qmi/qmi.tex (the Quantum Machine Intelligence version).
fig_geometry needs no data. fig_feature_maps reads data/qmi/qmi_kernels.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures" / "qmi"
FIGS.mkdir(parents=True, exist_ok=True)

COLORS = {"AE": "#2a78d6", "AE-anchored": "#e87ba4", "RBF": "#eb6834", "ZZ": "#6250d6", "Laplacian": "#1baa7d",
          "Normal": "#2a78d6", "Attack": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "legend.frameon": False,
})
DOMAINS = ["QKD", "QKD, centred Mahal.", "QKD, no Mahal.", "Synthetic flows", "UNSW-NB15", "CIC-IDS2017"]


def ci95(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def fig_geometry():
    """Toy data in R^2. Attacks are normal points pushed outward along the same directions."""
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(400, 2)) * 0.7
    attack = rng.normal(size=(400, 2)) * 0.7
    attack *= (3.0 / np.maximum(np.linalg.norm(attack, axis=1), 0.3))[:, None]
    c = np.median(np.linalg.norm(normal, axis=1))
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.0), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))

    ax = axes[0]
    for X, k in ((attack, "Attack"), (normal, "Normal")):
        ax.scatter(X[:, 0], X[:, 1], s=4, color=COLORS[k], lw=0, alpha=0.7, label=k)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=INK, lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-4.2, 4.2), ax.set_ylim(-4.2, 4.2)
    ax.set_title("(a) Inputs and the unit circle", fontsize=8)

    bins = np.linspace(0, 1, 26)
    ax = axes[1]
    ang = lambda X: (np.arctan2(X[:, 1], X[:, 0]) % np.pi) / np.pi
    for X, k in ((normal, "Normal"), (attack, "Attack")):
        ax.hist(ang(X), bins=bins, color=COLORS[k], alpha=0.6, label=k)
    ax.set_xlabel(r"Direction $\theta/\pi$ (all that $K_A$ sees)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Plain amplitude encoding", fontsize=8)

    ax = axes[2]
    last = lambda X: c / np.sqrt((X ** 2).sum(1) + c ** 2)
    for X, k in ((normal, "Normal"), (attack, "Attack")):
        ax.hist(last(X), bins=bins, color=COLORS[k], alpha=0.6, label=k)
    ax.set_xlabel(r"Anchor amplitude $c/\sqrt{\|\mathbf{x}\|^2+c^2}$")
    ax.set_title("(c) Norm-anchored encoding", fontsize=8)
    ax.legend(loc="upper right")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIGS / "fig_geometry.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_geometry.png")


def fig_feature_maps():
    df = pd.read_csv(ROOT / "data" / "qmi" / "qmi_kernels.csv")
    df = df[df.shots == 0]
    models = ["AE", "AE-anchored", "ZZ", "RBF", "Laplacian"]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.6), sharex=True)
    w = 0.84 / len(models)
    for ax, col, lab in ((axes[0], "auc_indist", "In-distribution AUC"),
                         (axes[1], "auc_heldout", "Held-out attack AUC")):
        for k, m in enumerate(models):
            mu, ci = [], []
            for d in DOMAINS:
                s = df[(df.domain == d) & (df.model == m)].groupby("seed")[col].mean()
                mu.append(s.mean()), ci.append(ci95(s))
            x = np.arange(len(DOMAINS)) + (k - (len(models) - 1) / 2) * w
            ax.bar(x, mu, w * 0.92, yerr=ci, color=COLORS[m], label=m,
                   error_kw=dict(lw=0.6, ecolor=MUTED, capsize=1.2))
        ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
        ax.set_ylabel(lab)
        ax.set_ylim(0.3, 1.02)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(models))
    axes[1].set_xticks(range(len(DOMAINS)), [d.replace(", ", ",\n") for d in DOMAINS])
    fig.tight_layout(h_pad=0.6)
    fig.savefig(FIGS / "fig_feature_maps.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_feature_maps.png")


if __name__ == "__main__":
    fig_geometry()
    if (ROOT / "data" / "qmi" / "qmi_kernels.csv").exists():
        fig_feature_maps()
