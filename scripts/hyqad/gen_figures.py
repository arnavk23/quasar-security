#!/usr/bin/env python3
"""
Figures for the HYQAD paper. Reads only the CSVs written by
scripts/hyqad/run_experiments.py (plus the simulator for the QBER histogram);
no result is hard-coded here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from hyqad.simulation import QKDDataGenerator
from hyqad.simulation.data_generator import ATTACK_TYPES

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "hyqad"
FIGS = ROOT / "figures" / "hyqad"
FIGS.mkdir(parents=True, exist_ok=True)

# Fixed entity -> style map, shared by every figure (color never follows rank).
STYLE = {
    "AE-SVM":      dict(color="#2a78d6", marker="o", ls="-"),
    "RBF-SVM":     dict(color="#eb6834", marker="s", ls="--"),
    "GP":          dict(color="#1baf7a", marker="^", ls="-."),
    "RF":          dict(color="#6250d6", marker="D", ls=":"),
    "Hybrid":      dict(color="#e34948", marker="v", ls="-"),
    "RF-isotonic": dict(color="#008300", marker="P", ls="--"),
    "AE-SVM (anchored)": dict(color="#e87ba4", marker="X", ls="-"),
    "OC-SVM":      dict(color="#8a8984", marker="x", ls=":"),
    "IF":          dict(color="#b0afa8", marker="+", ls=":"),
}
LABEL = {"AE-SVM": "AE-SVM (amplitude kernel)", "AE-SVM (anchored)": "AE-SVM, norm-anchored", "Hybrid": "Hybrid (GP + AE-SVM)",
         "RF-isotonic": "RF + isotonic"}
ATTACK_LABEL = {"pns": "PNS", "detector_blinding": "Det. blinding",
                "trojan_horse": "Trojan horse", "intercept_resend": "Int.-resend"}
SHORT_ATTACK = {"pns": "PNS", "detector_blinding": "Blinding", "trojan_horse": "Trojan", "intercept_resend": "Int.-res."}
SHORT = {"AE-SVM": "AE-SVM", "AE-SVM (anchored)": "AE anch.", "RF-isotonic": "RF-iso", "Hybrid": "Hybrid"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "savefig.dpi": 300,
    "legend.frameon": False,
})
COL_W, PAGE_W = 3.45, 7.1


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def save(fig, name):
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def line_panel(ax, df, x, y, models, logx=True):
    for m in models:
        sub = df[df.model == m].groupby(x)[y]
        mu, ci = sub.mean(), sub.agg(ci95)
        st = STYLE[m]
        ax.plot(mu.index, mu.values, label=LABEL.get(m, m), **st)
        ax.fill_between(mu.index, mu - ci, mu + ci, color=st["color"], alpha=0.12, lw=0)
    if logx:
        ax.set_xscale("log")


def fig_framework():
    fig = plt.figure(figsize=(PAGE_W, 1.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off"); ax.grid(False)
    boxes = [
        (0.1, "Decoy-state BB84\nchannel simulator",
         "Poisson gain, drift in\nloss / misalignment /\nefficiency / background,\nfinite-size QBER noise"),
        (2.9, "Attack models\n(strength s)",
         "PNS, detector blinding,\nTrojan horse,\nintercept-resend;\nperturbations x s"),
        (5.7, "Features (8-dim)",
         "7 standardized\nmeasurements +\nrobust Mahalanobis\ndistance to normal"),
        (8.5, "Detectors",
         "Bayesian GP (Laplace)\nAE-SVM: fidelity kernel,\n3 qubits, exact\nRBF-SVM, RF, stacking"),
        (11.3, "Evaluation",
         "matched-n, strength sweep,\nzero-day, calibration,\nselective prediction,\nevasion, latency"),
    ]
    for x, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.25), 2.5, 3.4, boxstyle="round,pad=0.05",
                                    fc="#f4f3f0", ec=MUTED, lw=0.8))
        ax.text(x + 1.25, 3.15, title, ha="center", va="center", fontsize=7, fontweight="bold", color=INK)
        ax.text(x + 1.25, 1.45, body, ha="center", va="center", fontsize=6, color=MUTED, linespacing=1.35)
    for x in (2.6, 5.4, 8.2, 11.0):
        ax.annotate("", xy=(x + 0.3, 1.95), xytext=(x, 1.95),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    save(fig, "fig1_framework_hyqad.png")


def fig_qber():
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.1), sharey=True)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
    for ax, s in zip(axes, (1.0, 0.3)):
        df = QKDDataGenerator(rng=np.random.default_rng(0), attack_strength=s).generate(20000, 0.5)
        bins = np.linspace(0, 0.4, 61)
        for c, atk in zip(colors, ["normal"] + ATTACK_TYPES):
            v = df.loc[df.attack_type == atk, "qber"]
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.3, color=c,
                    label="Normal" if atk == "normal" else ATTACK_LABEL[atk])
        ax.axvline(0.11, color=MUTED, ls="--", lw=0.8)
        ax.text(0.113, ax.get_ylim()[1] * 0.9, "11% bound", fontsize=6.5, color=MUTED)
        ax.set_title(f"Attack strength s = {s}")
        ax.set_xlabel("QBER")
    axes[0].set_ylabel("Density")
    axes[1].legend(loc="upper right")
    save(fig, "fig2_qber_dist.png")


def fig_matched_n():
    df = pd.read_csv(DATA / "matched_n.csv")
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.3))
    for ax, s in zip(axes, (1.0, 0.3)):
        line_panel(ax, df[df.strength == s], "n_train", "mcc", ["AE-SVM", "RBF-SVM", "GP", "RF", "Hybrid"])
        ax.set_title(f"s = {s}")
        ax.set_xlabel("Training samples n (all models)")
    axes[0].set_ylabel("MCC (mean ± 95% CI, 10 seeds)")
    axes[1].legend(loc="lower right", ncol=1)
    save(fig, "fig3_matched_n.png")


def fig_strength():
    df = pd.read_csv(DATA / "strength_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1]))
    line_panel(axes[0], df[df.attack == "all"], "strength", "mcc", ["AE-SVM", "RBF-SVM", "GP", "RF"], logx=True)
    axes[0].set_xlabel("Attack strength s (log scale)")
    axes[0].set_ylabel("MCC (mean ± 95% CI)")
    axes[0].legend(loc="lower right")

    rec = (df[(df.attack != "all") & (df.model == "GP")]
           .groupby(["attack", "strength"]).recall.mean().unstack())
    rec = rec.loc[ATTACK_TYPES]
    ax = axes[1]
    ax.grid(False)
    im = ax.imshow(rec.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(rec.shape[1]), [f"{s:g}" for s in rec.columns])
    ax.set_yticks(range(rec.shape[0]), [ATTACK_LABEL[a] for a in rec.index])
    for i in range(rec.shape[0]):
        for j in range(rec.shape[1]):
            v = rec.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v > 0.6 else INK)
    ax.set_xlabel("Attack strength s")
    ax.set_title("GP recall per attack type")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    save(fig, "fig4_strength_sweep.png")


def fig_zeroday():
    df = pd.read_csv(DATA / "zeroday.csv")
    models = [m for m in ["AE-SVM", "AE-SVM (anchored)", "RBF-SVM", "GP", "RF", "OC-SVM"] if m in set(df.model)]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.3), sharey=True)
    for ax, s in zip(axes, (1.0, 0.3)):
        sub = df[df.strength == s]
        w = 0.8 / len(models)
        for k, m in enumerate(models):
            g = sub[sub.model == m].groupby("held_out").auc
            mu, ci = g.mean().reindex(ATTACK_TYPES), g.agg(ci95).reindex(ATTACK_TYPES)
            x = np.arange(len(ATTACK_TYPES)) + (k - (len(models) - 1) / 2) * w
            ax.bar(x, mu, w * 0.9, yerr=ci, color=STYLE[m]["color"], label=LABEL.get(m, m),
                   error_kw=dict(lw=0.7, ecolor=MUTED, capsize=1.5))
        ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
        ax.set_xticks(range(len(ATTACK_TYPES)), [SHORT_ATTACK[a] for a in ATTACK_TYPES])
        ax.set_title(f"Held-out attack AUC, s = {s}")
        ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel("ROC-AUC on unseen attack")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(models))
    save(fig, "fig5_zeroday.png")


def fig_calibration_selective():
    bench = pd.read_csv(DATA / "benchmark.csv")
    sel = pd.read_csv(DATA / "selective_prediction.csv")
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.3))
    models = ["GP", "AE-SVM", "RBF-SVM", "RF", "RF-isotonic", "Hybrid"]
    ax = axes[0]
    w = 0.38
    for k, s in enumerate((1.0, 0.3)):
        g = bench[(bench.strength == s) & bench.model.isin(models)].groupby("model").ece
        mu, ci = g.mean().reindex(models), g.agg(ci95).reindex(models)
        ax.bar(np.arange(len(models)) + (k - 0.5) * w, mu, w * 0.9, yerr=ci,
               color=["#2a78d6", "#eb6834"][k], label=f"s = {s}",
               error_kw=dict(lw=0.7, ecolor=MUTED, capsize=1.5))
    ax.set_xticks(range(len(models)), [SHORT.get(m, m) for m in models], fontsize=6.5)
    ax.set_ylabel("ECE (15 bins, lower is better)")
    ax.legend(loc="upper right")
    ax.set_title("Calibration on held-out test set")

    ax = axes[1]
    for m in ("GP", "RF-isotonic", "AE-SVM"):
        g = sel[(sel.strength == 0.3) & (sel.model == m)].groupby("tau")
        cov, acc = g.coverage.mean(), g.accuracy.mean()
        ax.plot(cov, acc, label=LABEL.get(m, m), **STYLE[m])
    ax.set_xlabel("Coverage (fraction not deferred)")
    ax.set_ylabel("Accuracy on retained samples")
    ax.set_title("Selective prediction, s = 0.3")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig6_calibration_selective.png")


def fig_adversarial():
    df = pd.read_csv(DATA / "adversarial.csv")
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    g = df[df.strength == 0.3].groupby("budget")
    for col, m in (("rf_detect", "RF"), ("gp_detect", "GP")):
        mu, ci = g[col].mean(), g[col].agg(ci95)
        ax.plot(mu.index, mu, label=f"{m} detection rate", **STYLE[m])
        ax.fill_between(mu.index, mu - ci, mu + ci, color=STYLE[m]["color"], alpha=0.12, lw=0)
    ax.plot(g.gp_unc.mean().index, g.gp_unc.mean(), color=MUTED, ls="--", marker="o", mfc="white",
            label="GP mean uncertainty u")
    ax.set_xlabel("Evasion budget")
    ax.set_ylabel("Rate / uncertainty")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=6.5)
    save(fig, "fig7_adversarial.png")


def fig_crossover():
    df = pd.read_csv(DATA / "kernel_crossover.csv").rename(columns={"acc": "accuracy"})
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.2), sharey=True)
    for ax, task, title in zip(axes, ("angular", "radial"),
                               ("Label depends on direction only", "Label depends on norm only")):
        line_panel(ax, df[df.task == task], "n_train", "accuracy", ["AE-SVM", "RBF-SVM", "GP", "RF"])
        ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Training samples n")
    axes[0].set_ylabel("Test accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4)
    save(fig, "fig8_kernel_crossover.png")


def fig_mahal_ablation():
    df = pd.read_csv(DATA / "zeroday_mahal_ablation.csv")
    models = ["AE-SVM", "AE-SVM (anchored)", "RBF-SVM"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    groups = [(s, f) for s in (1.0, 0.3) for f in ("with Mahalanobis", "without Mahalanobis")]
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        mu, ci = [], []
        for s, f in groups:
            v = df[(df.strength == s) & (df.features == f) & (df.model == m)].groupby("seed").auc.mean()
            mu.append(v.mean()); ci.append(ci95(v))
        x = np.arange(len(groups)) + (k - 1) * w
        ax.bar(x, mu, w * 0.9, yerr=ci, color=STYLE[m]["color"], label=LABEL.get(m, m),
               error_kw=dict(lw=0.7, ecolor=MUTED, capsize=1.5))
    ax.set_xticks(range(len(groups)), [f"s={s}\n{f.split()[0]}" for s, f in groups], fontsize=6.5)
    ax.set_xlabel("Mahalanobis feature")
    ax.set_ylabel("Mean zero-day AUC")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6)
    save(fig, "fig9_mahal_ablation.png")


if __name__ == "__main__":
    for f in (fig_framework, fig_qber, fig_matched_n, fig_strength, fig_zeroday,
              fig_calibration_selective, fig_adversarial, fig_crossover, fig_mahal_ablation):
        try:
            f()
        except FileNotFoundError as e:
            print(f"  skipped {f.__name__}: {e.filename} missing")
