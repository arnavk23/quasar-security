#!/usr/bin/env python3
"""Generate publication-quality figures for the CCRI paper."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from hyqad.simulation import QKDDataGenerator

sns.set_theme(style="whitegrid", font_scale=1.15, palette="muted")
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})
FIGS = "/home/arnav/Downloads/ccri/figures"
rng = np.random.default_rng(42)
gen = QKDDataGenerator()

# ================================================================
# FIGURE 1: Framework Architecture (professional block diagram)
# ================================================================
def fig1_framework():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor("#fafafa")

    # Define boxes: (x, y, w, h, color, text)
    boxes = [
        (0.5, 5.5, 2.8, 1.8, "#4472C4", "QKD Channel\nSimulation",
         "• Fiber loss model\n• Dark counts / afterpulse\n• Background noise"),
        (0.5, 2.5, 2.8, 1.8, "#5B9BD5", "Attack Vector\nGeneration",
         "• Photon-Number-Splitting\n• Detector Blinding\n• Trojan Horse / Int-Resend"),
        (4.2, 4.0, 3.0, 2.2, "#ED7D31", "Feature Engineering\n(Quantum-Inspired)",
         "• Amplitude Encoding\n  (8-dim → Hilbert space)\n• Covariance Features\n  (Mahalanobis, local stats)"),
        (8.0, 4.0, 3.0, 2.2, "#70AD47", "Hybrid Detector",
         "• Bayesian GP Classifier\n  (RBF/Matérn kernel)\n• Quantum Kernel SVM\n  (PennyLane 3-qubit)\n• Meta-learner stacking"),
        (4.2, 1.0, 3.0, 1.5, "#FFC000", "Parameter Optimization\n(JSO-Inspired)",
         "• L-BFGS-B / DE\n• Basin hopping\n• Grid search CV"),
        (8.0, 1.0, 3.0, 1.5, "#A5A5A5", "Evaluation &\nBenchmarking",
         "• 7 metrics (F1, AUC…)\n• 4 classical baselines\n• Latency profiling\n• Edge deployment"),
    ]
    for (x, y, w, h, color, title, desc) in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              fc=color, ec="white", lw=2.5, alpha=0.92)
        ax.add_patch(box)
        # Shadow
        shadow = FancyBboxPatch((x + 0.08, y - 0.08), w, h, boxstyle="round,pad=0.08",
                                 fc="black", ec="none", lw=0, alpha=0.10)
        ax.add_patch(shadow)
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(x + 0.25, y + h - 0.65, desc, ha="left", va="top",
                fontsize=8.5, color="white", fontfamily="monospace")

    # Arrows with proper edge-to-edge coordinates
    arrows = [
        # QKD Channel -> Feature Engineering: (from right edge of box 1 to left edge of box 3)
        (0.5 + 2.8, 5.5 + 1.8/2, 4.2, 4.0 + 2.2/2),
        # Attack Vectors -> Feature Engineering
        (0.5 + 2.8, 2.5 + 1.8/2, 4.2, 4.0 + 2.2/2),
        # Feature Engineering -> Hybrid Detector
        (4.2 + 3.0, 4.0 + 2.2/2, 8.0, 4.0 + 2.2/2),
        # Feature Engineering -> Optimization
        (4.2 + 3.0/2, 4.0, 4.2 + 3.0/2, 1.0 + 1.5),
        # Optimization -> Hybrid Detector
        (4.2 + 3.0/2, 1.0 + 1.5, 4.2 + 3.0/2, 4.0),
        # Hybrid Detector -> Evaluation
        (8.0 + 3.0/2, 4.0, 8.0 + 3.0/2, 1.0 + 1.5),
    ]
    for (x1, y1, x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=2.5,
                                    connectionstyle="arc3,rad=0.0"))

    # Data flow labels
    ax.text(3.9, 6.0, "channel\nstate", ha="center", va="center",
            fontsize=8, fontstyle="italic", color="#333")
    ax.text(3.9, 3.8, "attack\nsignals", ha="center", va="center",
            fontsize=8, fontstyle="italic", color="#333")
    ax.text(6.5, 5.5, "encoded\nfeatures", ha="center", va="center",
            fontsize=8, fontstyle="italic", color="#333")

    ax.set_title("Hybrid AI-Quantum Anomaly Detection Framework for QKD Systems",
                 fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig1_framework.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig1_framework.png")

# ================================================================
# FIGURE 2: QBER Distributions - Normal vs Attack
# ================================================================
def fig2_qber():
    df = gen.generate(n_samples=5000, attack_ratio=0.3)
    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl, color, label in [(0, "#2E7D32", "Normal Channel"), (1, "#C62828", "Under Attack")]:
        subset = df[df["label"] == lbl]["qber"]
        ax.hist(subset, bins=40, alpha=0.8, color=color, label=label,
                density=True, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.11, color="#333", ls="--", lw=1.5, label="Typical QKD threshold (~11%)")
    ax.set_xlabel("Quantum Bit Error Rate (QBER)", fontsize=13)
    ax.set_ylabel("Probability Density", fontsize=13)
    ax.set_title("QBER Distribution: Normal Operation vs. Attack Scenarios", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.text(0.02, ax.get_ylim()[1]*0.95, f"N = {len(df)} samples\nAttack ratio = 30%",
            fontsize=9, fontstyle="italic", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8))
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig2_qber_dist.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig2_qber_dist.png")

# ================================================================
# FIGURE 3: Attack Radar Signatures
# ================================================================
def fig3_radar():
    radar_metrics = ["QBER", "Photon\nCount Rate", "Detection\nEfficiency",
                     "Dark Count\nRate", "Signal\nAttenuation", "Background\nNoise"]
    attack_profiles = {}
    for atk_name in ["pns", "detector_blinding", "trojan_horse", "intercept_resend"]:
        sub_df = gen.generate(n_samples=1000, attack_ratio=1.0, attack_types=[atk_name])
        attack_profiles[atk_name] = sub_df[["qber","photon_count_rate","detection_efficiency",
                                            "dark_count_rate","signal_attenuation","background_noise"]].mean().values
    normal_df = gen.generate(n_samples=1000, attack_ratio=0.0)
    attack_profiles["normal"] = normal_df[["qber","photon_count_rate","detection_efficiency",
                                            "dark_count_rate","signal_attenuation","background_noise"]].mean().values

    # Normalize relative to normal
    baseline = attack_profiles["normal"].copy()
    for key in attack_profiles:
        attack_profiles[key] = np.where(baseline > 0, attack_profiles[key] / baseline, 1.0)

    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = {"pns": "#2196F3", "detector_blinding": "#FF9800",
              "trojan_horse": "#4CAF50", "intercept_resend": "#F44336", "normal": "#9E9E9E"}
    styles = {"pns": ("o", "-"), "detector_blinding": ("s", "--"),
              "trojan_horse": ("^", "-."), "intercept_resend": ("D", ":"), "normal": ("o", "-")}

    for name in ["normal", "pns", "detector_blinding", "trojan_horse", "intercept_resend"]:
        vals = attack_profiles[name].tolist() + [attack_profiles[name][0]]
        marker, linestyle = styles[name]
        ax.plot(angles, vals, marker=marker, linestyle=linestyle, color=colors[name],
                label=name.replace("_", " ").title(), lw=2.5, markersize=7, alpha=0.85)
        ax.fill(angles, vals, alpha=0.06, color=colors[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=10, fontweight="bold")
    ax.set_ylim(0.5, 1.8)
    ax.set_title("Attack Signature Profiles in QKD Parameter Space\n(Normalized to Baseline)",
                 fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=11,
              frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig3_attack_radar.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig3_attack_radar.png")

# ================================================================
# FIGURE 4: Confusion Matrix + ROC Curve
# ================================================================
def fig4_confusion_roc():
    df = gen.generate(n_samples=3000, attack_ratio=0.3)
    cols = [c for c in df.columns if c not in ("label", "is_attack")]
    X, y = df[cols].values, df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    y_score = rf.predict_proba(X_te)[:, 1]

    fpr, tpr, _ = roc_curve(y_te, y_score)
    roc_auc = auc(fpr, tpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"])
    disp.plot(ax=ax1, cmap="Blues", colorbar=False, values_format="d")
    ax1.set_title("Confusion Matrix (Random Forest)", fontsize=13, fontweight="bold")

    ax2.plot(fpr, tpr, color="#1565C0", lw=3, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random (AUC = 0.5)")
    ax2.fill_between(fpr, tpr, alpha=0.15, color="#1565C0")
    ax2.set_xlabel("False Positive Rate", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    ax2.set_title("Receiver Operating Characteristic", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11, loc="lower right")
    ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Detection Performance (RF Baseline, 3000 samples)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig4_confusion_roc.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig4_confusion_roc.png")

# ================================================================
# FIGURE 5: Benchmark Comparison - grouped bar chart
# ================================================================
def fig5_benchmark():
    models = ["Random\nForest", "Hybrid\nDetector", "One-Class\nSVM", "Isolation\nForest", "LOF"]
    metrics_def = {
        "Accuracy": [0.995, 0.940, 0.894, 0.748, 0.755],
        "F1 Score": [0.992, 0.889, 0.810, 0.269, 0.302],
        "ROC-AUC": [1.000, 1.000, 0.888, 0.999, 0.905],
    }
    x = np.arange(len(models))
    w = 0.22
    colors = ["#2E7D32", "#1565C0", "#E65100"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (mname, vals) in enumerate(metrics_def.items()):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, label=mname, color=colors[i],
                      edgecolor="white", linewidth=0.8, alpha=0.90)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Performance Benchmark: Hybrid Detector vs. Classical Baselines",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", ls="--", lw=1, alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig5_benchmark.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig5_benchmark.png")

# ================================================================
# FIGURE 6: Bayesian GP Uncertainty Scatter
# ================================================================
def fig6_uncertainty():
    from hyqad.models import BayesianAnomalyEstimator
    df = gen.generate(n_samples=2000, attack_ratio=0.3)
    cols = [c for c in df.columns if c not in ("label", "is_attack")]
    X, y = df[cols].values, df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    bae = BayesianAnomalyEstimator(random_state=42)
    bae.fit(X_tr[:400], y_tr[:400])
    preds, proba, uncert = bae.predict_with_uncertainty(X_te)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    sc = ax1.scatter(proba, uncert, c=y_te, cmap="coolwarm", alpha=0.6,
                     s=30, edgecolors="k", linewidth=0.3)
    cbar = plt.colorbar(sc, ax=ax1, shrink=0.8)
    cbar.set_label("Ground Truth (0=Normal, 1=Attack)", fontsize=10)
    ax1.set_xlabel("Anomaly Probability p(y=1|x)", fontsize=12)
    ax1.set_ylabel("Predictive Uncertainty √[p·(1-p)]", fontsize=12)
    ax1.set_title("Uncertainty vs. Anomaly Score", fontsize=13, fontweight="bold")
    ax1.axvline(x=0.5, color="gray", ls="--", lw=1.5, alpha=0.6)
    ax1.grid(True, alpha=0.2)

    # Right: histogram of anomaly scores by class
    ax2.hist(proba[y_te == 0], bins=30, alpha=0.7, color="#2E7D32",
             label="Normal", density=True, edgecolor="white", linewidth=0.5)
    ax2.hist(proba[y_te == 1], bins=30, alpha=0.7, color="#C62828",
             label="Attack", density=True, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Anomaly Score", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Score Distribution by Class", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Bayesian GP: Uncertainty Quantification for QKD Anomaly Detection",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig6_uncertainty.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig6_uncertainty.png")

# ================================================================
# FIGURE 7: Correlation Heatmap
# ================================================================
def fig7_correlation():
    df = gen.generate(n_samples=5000, attack_ratio=0.3)
    cols = [c for c in df.columns if c not in ("label", "is_attack")]
    corr = df[cols].corr()
    labels = ["QBER", "Photon\nCount", "Det.\nEfficiency", "Dark\nCount",
              "Error\nRate", "Signal\nAtt.", "Backgr.\nNoise", "Time"]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap, center=0,
                square=True, linewidths=0.8, ax=ax,
                annot_kws={"fontsize": 9, "fontweight": "bold"},
                cbar_kws={"shrink": 0.75, "label": "Pearson Correlation"})
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=9, rotation=0)
    ax.set_title("Cross-Parameter Correlation Matrix\nQKD Channel Measurements",
                 fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig7_correlation.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig7_correlation.png")

# ================================================================
# FIGURE 8: Amplitude Encoding Visualization
# ================================================================
def fig8_amplitude():
    from hyqad.features import AmplitudeEncoder
    X_raw = rng.normal(0, 1, (8, 8))
    encoder = AmplitudeEncoder(encoding_dim=8)
    X_enc = encoder.transform(X_raw)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    vmax = max(np.abs(X_raw).max(), 1.0)

    ax = axes[0]
    im = ax.imshow(X_raw, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("Raw Classical\nQKD Features", fontsize=12, fontweight="bold")
    ax.set_xlabel("Channel Parameter (1–8)", fontsize=10)
    ax.set_ylabel("Sample", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im = ax.imshow(X_enc, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_title("Amplitude-Encoded\n(Hilbert Space, 3-qubit)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Basis State |i⟩", fontsize=10)
    ax.set_ylabel("Sample", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[2]
    norms_before = np.linalg.norm(X_raw, axis=1)
    norms_after = np.linalg.norm(X_enc, axis=1)
    x_vals = np.arange(len(norms_before))
    ax.bar(x_vals - 0.15, norms_before, 0.3, label="Before encoding", color="#1565C0", alpha=0.8)
    ax.bar(x_vals + 0.15, norms_after, 0.3, label="After (L2=1)", color="#E65100", alpha=0.8)
    ax.set_xlabel("Sample Index", fontsize=10)
    ax.set_ylabel("L2 Norm", fontsize=10)
    ax.set_title("Normalization Effect\nof Encoding", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks(x_vals)
    ax.axhline(y=1.0, color="gray", ls="--", lw=1, alpha=0.5)

    fig.suptitle("Amplitude Encoding: Classical QKD Signals → Quantum Hilbert Space",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig8_amplitude.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig8_amplitude.png")

# ================================================================
# FIGURE 9: Channel Transmittance
# ================================================================
def fig9_transmittance():
    fig, ax1 = plt.subplots(figsize=(8, 5))
    losses = np.linspace(0, 40, 200)
    for alpha, color, ls in [(0.15, "#2E7D32", "-"), (0.20, "#1565C0", "--"), (0.25, "#C62828", "-.")]:
        T = 10 ** (-alpha * losses / 10)
        ax1.plot(losses, T, color=color, ls=ls, lw=2.5, label=f"α = {alpha} dB/km")
    ax1.set_xlabel("Fiber Length (km)", fontsize=12)
    ax1.set_ylabel("Channel Transmittance T", fontsize=12)
    ax1.set_title("Quantum Channel Transmittance vs. Fiber Distance", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, title="Fiber Loss", title_fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-4, 1)

    # Inset: QBER vs signal intensity
    from hyqad.simulation import QKDChannelSimulator
    sim = QKDChannelSimulator(length_km=50)
    mus = np.linspace(0.05, 1.0, 50)
    qbers = [sim._compute_qber(mu) for mu in mus]
    ax_inset = fig.add_axes([0.55, 0.18, 0.32, 0.28])
    ax_inset.plot(mus, qbers, color="#E65100", lw=2.5)
    ax_inset.set_xlabel("Signal Intensity μ", fontsize=8)
    ax_inset.set_ylabel("QBER", fontsize=8)
    ax_inset.set_title("QBER vs μ (50 km)", fontsize=9, fontweight="bold")
    ax_inset.grid(True, alpha=0.2)
    ax_inset.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig9_transmittance.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig9_transmittance.png")

# ================================================================
# FIGURE 10: Latency / Throughput Analysis
# ================================================================
def fig10_latency():
    models = ["Bayesian GP\n(Full)", "Bayesian\nLinear Est.", "Cov. Features\n+ RF", "Quantum\nKernel (4q)", "Full Pipeline\n(Bayesian)"]
    latencies = [7.3, 0.4, 1.2, 50.0, 8.5]
    colors_bar = ["#1565C0", "#2E7D32", "#FF8F00", "#C62828", "#6A1B9A"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.barh(models, latencies, color=colors_bar, edgecolor="white", linewidth=1.2, height=0.6)
    for bar, v in zip(bars, latencies):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{v:.1f} ms", va="center", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Inference Latency (ms/sample)", fontsize=12)
    ax1.set_title("Per-Sample Inference Latency\n(Intel i7, CPU)", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, 70)
    ax1.grid(True, axis="x", alpha=0.2)

    # Right: throughput
    components = ["Feature\nExtraction", "Bayesian GP\nInference", "Meta-Learner\nStacking", "Full\nPipeline"]
    times = [1.2, 5.8, 1.5, 8.5]
    colors_pie = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]
    wedges, texts, autotexts = ax2.pie(times, labels=components, autopct="%1.1f%%",
                                        colors=colors_pie, startangle=90,
                                        textprops={"fontsize": 9},
                                        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(10)
    ax2.set_title("Latency Breakdown\n(Full Pipeline = 8.5 ms)", fontsize=13, fontweight="bold")

    fig.suptitle("Computational Performance Analysis for Edge Deployment",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig10_latency.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig10_latency.png")

# ================================================================
# FIGURE 11: Feature Importance
# ================================================================
def fig11_importance():
    from hyqad.features import CovarianceFeatureExtractor
    df = gen.generate(n_samples=3000, attack_ratio=0.3)
    cols = [c for c in df.columns if c not in ("label", "is_attack")]
    X, y = df[cols].values, df["label"].values

    cov = CovarianceFeatureExtractor(smoothing=1e-6)
    cov.fit(X)
    X_cov = cov.transform(X)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_cov, y)

    feat_names = [f"LocalMean_{i}" for i in range(8)] + [f"LocalStd_{i}" for i in range(8)] + \
                 ["Mahalanobis", "CovNorm", "MeanPairCorr", "MaxPairCorr"]
    importances = rf.feature_importances_
    idx = np.argsort(importances)[-10:]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(range(len(idx)), importances[idx], color=["#E65100" if "Mean" in feat_names[i] or "Std" in feat_names[i]
                                                       else "#1565C0" for i in idx],
            edgecolor="white", linewidth=0.8, height=0.7)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feat_names[i] for i in idx], fontsize=10)
    ax.set_xlabel("Feature Importance (RF)", fontsize=12)
    ax.set_title("Top 10 Most Discriminative Features\n(Covariance + Raw Combined Space)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#E65100", label="Local stat (window)"),
                       Patch(facecolor="#1565C0", label="Global covariance")]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig11_importance.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig11_importance.png")


# ================================================================
# FIGURE 12: GP Accuracy-vs-Training-Size Learning Curve
# ================================================================
def fig12_gp_scaling():
    df = pd.read_csv("/home/arnav/Downloads/ccri/gp_scaling.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sizes = sorted(df.n_train.unique())

    means = [df[df.n_train == s].acc.mean() for s in sizes]
    stds  = [df[df.n_train == s].acc.std() for s in sizes]
    ax1.errorbar(sizes, means, yerr=stds, color="#1565C0", lw=2.5, marker="o",
                 markersize=8, capsize=4, capthick=1.5, label="GP accuracy")
    ax1.axhline(y=0.700, color="gray", ls="--", lw=1.5, alpha=0.6, label="Majority baseline (70%)")
    ax1.axhline(y=0.997, color="#E65100", ls=":", lw=1.5, alpha=0.7, label="Full benchmark (0.997)")
    ax1.fill_between(sizes, [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)], alpha=0.12, color="#1565C0")
    ax1.set_xlabel("Training set size ($n_{\\mathrm{train}}$)", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("GP Accuracy vs. Training Set Size\n(28-dim features, 3 seeds)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(0.65, 1.02); ax1.set_xlim(0, 1100)
    ax1.annotate("Saturates by\n$n \\approx 100$", xy=(200, 0.987), xytext=(500, 0.935),
                 fontsize=9, color="#333",
                 arrowprops=dict(arrowstyle="->", color="#333", lw=1.5, connectionstyle="arc3,rad=-0.3"))

    means_f1 = [df[df.n_train == s].f1.mean() for s in sizes]
    stds_f1  = [df[df.n_train == s].f1.std() for s in sizes]
    ax2.errorbar(sizes, means_f1, yerr=stds_f1, color="#2E7D32", lw=2.5, marker="s",
                 markersize=8, capsize=4, capthick=1.5, label="GP F1 score")
    ax2.fill_between(sizes, [m - s for m, s in zip(means_f1, stds_f1)],
                     [m + s for m, s in zip(means_f1, stds_f1)], alpha=0.12, color="#2E7D32")
    ax2.set_xlabel("Training set size ($n_{\\mathrm{train}}$)", fontsize=12)
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.set_title("GP F1 Score vs. Training Set Size\n(28-dim features, 3 seeds)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_ylim(0.65, 1.02); ax2.set_xlim(0, 1100)

    fig.suptitle("Bayesian GP Learning Curve on 28-Dimensional Engineered Features",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig12_gp_scaling.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig12_gp_scaling.png")


# ================================================================
# FIGURE 13: Matched-n Learning Curves (NEW — fixes confounded ablation)
# ================================================================
def fig13_matched_n_curves():
    import pandas as pd
    csv_path = "/home/arnav/Downloads/ccri/matched_n_results.csv"
    if not os.path.exists(csv_path):
        print("  SKIPPED fig13: matched_n_results.csv not found. Run matched_n_ablation.py first.")
        return
    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    sizes = sorted(df.n_train.unique())
    models = ["RF", "GP-only", "QK-only", "Hybrid"]
    colors = {"RF": "#2E7D32", "GP-only": "#1565C0", "QK-only": "#C62828", "Hybrid": "#6A1B9A"}
    markers = {"RF": "o", "GP-only": "s", "QK-only": "^", "Hybrid": "D"}
    linestyles = {"RF": "-", "GP-only": "--", "QK-only": ":", "Hybrid": "-."}

    for model in models:
        sub = df[df.model == model].dropna(subset=["acc"])
        if sub.empty:
            continue
        means = [sub[sub.n_train == s].acc.mean() for s in sizes]
        stds = [sub[sub.n_train == s].acc.std() for s in sizes]
        valid = [i for i, (m, s) in enumerate(zip(means, stds)) if not np.isnan(m)]
        s_valid = [sizes[i] for i in valid]
        m_valid = [means[i] for i in valid]
        st_valid = [stds[i] for i in valid]
        ax1.errorbar(s_valid, m_valid, yerr=st_valid, color=colors[model],
                     lw=2.5, marker=markers[model], markersize=7, capsize=3,
                     label=model, linestyle=linestyles[model], alpha=0.85)
        ax1.fill_between(s_valid, [m - s for m, s in zip(m_valid, st_valid)],
                         [m + s for m, s in zip(m_valid, st_valid)],
                         alpha=0.08, color=colors[model])

    ax1.axhline(y=0.700, color="gray", ls="--", lw=1.2, alpha=0.5, label="Majority baseline")
    ax1.set_xlabel("Training set size ($n_{\\mathrm{train}}$)", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Matched-$n$ Accuracy\n(28-dim features, 10 seeds)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(0.55, 1.05)

    for model in models:
        sub = df[df.model == model].dropna(subset=["f1"])
        if sub.empty:
            continue
        means = [sub[sub.n_train == s].f1.mean() for s in sizes]
        stds = [sub[sub.n_train == s].f1.std() for s in sizes]
        valid = [i for i, (m, s) in enumerate(zip(means, stds)) if not np.isnan(m)]
        s_valid = [sizes[i] for i in valid]
        m_valid = [means[i] for i in valid]
        st_valid = [stds[i] for i in valid]
        ax2.errorbar(s_valid, m_valid, yerr=st_valid, color=colors[model],
                     lw=2.5, marker=markers[model], markersize=7, capsize=3,
                     label=model, linestyle=linestyles[model], alpha=0.85)
        ax2.fill_between(s_valid, [m - s for m, s in zip(m_valid, st_valid)],
                         [m + s for m, s in zip(m_valid, st_valid)],
                         alpha=0.08, color=colors[model])

    ax2.axhline(y=0.0, color="gray", ls="--", lw=1.2, alpha=0.5, label="Zero F1 baseline")
    ax2.set_xlabel("Training set size ($n_{\\mathrm{train}}$)", fontsize=12)
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.set_title("Matched-$n$ F1 Score\n(28-dim features, 10 seeds)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle("Matched-$n$ Ablation: All Models at Identical Training Sizes",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig_matched_n_curves.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig_matched_n_curves.png")


# ================================================================
# FIGURE 14: Sensitivity Heatmap (NEW)
# ================================================================
def fig14_sensitivity_heatmap():
    import pandas as pd
    csv_path = "/home/arnav/Downloads/ccri/sensitivity_results.csv"
    csv_full_n = "/home/arnav/Downloads/ccri/sensitivity_full_n_results.csv"
    if not os.path.exists(csv_path):
        print("  SKIPPED fig14: sensitivity_results.csv not found.")
        return
    df = pd.read_csv(csv_path)

    has_full_n = os.path.exists(csv_full_n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = [("RF", "#2E7D32"), ("GP-only", "#1565C0")]

    for ax, (model, color) in zip(axes, models):
        sub = df[df.model == model]
        strengths = sorted(sub.strength.unique())
        seeds = sorted(sub.seed.unique())

        data_matrix = np.full((len(seeds), len(strengths)), np.nan)
        for i, seed in enumerate(seeds):
            for j, s in enumerate(strengths):
                row = sub[(sub.seed == seed) & (sub.strength == s)]
                if len(row) > 0 and row.acc.notna().any():
                    data_matrix[i, j] = row.acc.values[0]

        im = ax.imshow(data_matrix, aspect="auto", cmap="RdYlGn",
                       vmin=0.5, vmax=1.0, interpolation="nearest")
        ax.set_xticks(range(len(strengths)))
        ax.set_xticklabels([f"{s:.1f}" for s in strengths], fontsize=8, rotation=45)
        ax.set_yticks(range(len(seeds)))
        ax.set_yticklabels([f"Seed {s}" for s in seeds], fontsize=8)
        ax.set_xlabel("Attack Strength", fontsize=10)
        ax.set_title(f"{model} ($n=25$)", fontsize=12, fontweight="bold", color=color)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")

    fig.suptitle("Sensitivity Analysis: Matched-$n=25$ (QK-scale)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig_sensitivity_heatmap.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig_sensitivity_heatmap.png")


# ================================================================
# FIGURE 15: Adversarial Robustness (NEW)
# ================================================================
def fig15_adversarial_robustness():
    import pandas as pd
    csv_path = "/home/arnav/Downloads/ccri/adversarial_robustness_results.csv"
    if not os.path.exists(csv_path):
        print("  SKIPPED fig15: adversarial_robustness_results.csv not found. Run adversarial_robustness.py first.")
        return
    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    budgets = sorted(df.budget.unique())
    for model, color, marker in [("RF", "#2E7D32", "o"), ("GP", "#1565C0", "s")]:
        sub = df[df.model == model]
        det_rates = []
        det_stds = []
        for b in budgets:
            bsub = sub[sub.budget == b]
            if model == "RF":
                dr = bsub.detection_rate.values
            else:
                dr = bsub.detection_rate.values
            det_rates.append(np.mean(dr) if len(dr) > 0 else 1.0)
            det_stds.append(np.std(dr) if len(dr) > 0 else 0.0)

        ax1.errorbar(budgets, det_rates, yerr=det_stds, color=color, lw=2.5,
                     marker=marker, markersize=8, capsize=4, label=f"{model} detection rate")

    ax1.set_xlabel("Perturbation Budget (fraction of feature range)", fontsize=12)
    ax1.set_ylabel("Detection Rate", fontsize=12)
    ax1.set_title("Adversarial Robustness: Detection Rate vs. Evasion Budget",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11, loc="lower left")
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(True, alpha=0.2)

    # Right axis: GP uncertainty on evasive samples
    ax2 = ax1.twinx()
    gp_sub = df[(df.model == "GP") & (df.budget > 0)]
    if "mean_uncertainty" in gp_sub.columns and gp_sub["mean_uncertainty"].notna().any():
        uncert_means = [gp_sub[gp_sub.budget == b]["mean_uncertainty"].mean()
                        for b in budgets if b > 0]
        uncert_budgets = [b for b in budgets if b > 0]
        ax2.plot(uncert_budgets, uncert_means, color="#E65100", lw=2, ls="--",
                 marker="D", markersize=7, label="GP mean uncertainty")
        ax2.set_ylabel("Mean Predictive Uncertainty", fontsize=12, color="#E65100")
        ax2.tick_params(axis="y", labelcolor="#E65100")
        ax2.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig_adversarial_robustness.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig_adversarial_robustness.png")


# ================================================================
# FIGURE 16: QK Crossover (NEW)
# ================================================================
def fig16_qk_crossover():
    import pandas as pd
    csv_path = "/home/arnav/Downloads/ccri/qk_crossover_results.csv"
    if not os.path.exists(csv_path):
        print("  SKIPPED fig16: qk_crossover_results.csv not found. Run qk_crossover.py first.")
        return
    df = pd.read_csv(csv_path)

    datasets = df.dataset.unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5.5))
    if len(datasets) == 1:
        axes = [axes]

    model_colors = {
        "RBF-SVM": "#FF8F00", "GP-RBF": "#1565C0", "GP-Matern": "#2E7D32",
        "RF": "#6A1B9A", "QK-SVM": "#C62828",
    }

    for ax, ds in zip(axes, datasets):
        ds_sub = df[df.dataset == ds]
        models_present = ds_sub.model.unique()
        x_pos = np.arange(len(models_present))
        means = [ds_sub[ds_sub.model == m].acc.mean() for m in models_present]
        stds = [ds_sub[ds_sub.model == m].acc.std() for m in models_present]

        bars = ax.bar(x_pos, means, yerr=stds, color=[model_colors.get(m, "#999") for m in models_present],
                      edgecolor="white", linewidth=0.8, alpha=0.9, capsize=4)
        for bar, v, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models_present, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{ds.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.5, color="gray", ls="--", lw=1, alpha=0.4)
        ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle("Quantum Kernel Crossover: Performance on Classically-Hard Problems",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig_qk_crossover.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig_qk_crossover.png")


# ================================================================
# FIGURE 17: Sensitivity comparison n=25 vs n=700 (NEW)
# ================================================================
def fig17_sensitivity_full_n():
    import pandas as pd
    csv_n25 = "/home/arnav/Downloads/ccri/sensitivity_results.csv"
    csv_n700 = "/home/arnav/Downloads/ccri/sensitivity_full_n_results.csv"
    if not os.path.exists(csv_n25) or not os.path.exists(csv_n700):
        print("  SKIPPED fig17: sensitivity CSVs not found.")
        return

    df25 = pd.read_csv(csv_n25)
    df700 = pd.read_csv(csv_n700)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    models = [("RF", "#2E7D32"), ("GP-only", "#1565C0")]

    for ax, (model, color) in zip(axes, models):
        for df, ls, marker, label_suffix in [
            (df25, "--", "o", "$n=25$"),
            (df700, "-", "s", "$n=700$"),
        ]:
            sub = df[df.model == model]
            strengths = sorted(sub.strength.unique())
            means = [sub[sub.strength == s].acc.mean() for s in strengths]
            stds = [sub[sub.strength == s].acc.std() for s in strengths]
            ax.errorbar(strengths, means, yerr=stds, color=color, lw=2.5,
                        marker=marker, markersize=7, capsize=3, ls=ls,
                        label=f"{label_suffix} ({len(sub.seed.unique())} seeds)")

        ax.set_xlabel("Attack Perturbation Strength", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{model}", fontsize=13, fontweight="bold", color=color)
        ax.legend(fontsize=10, loc="lower right")
        ax.set_ylim(0.85, 1.02)
        ax.axvline(x=1.0, color="gray", ls=":", lw=1, alpha=0.5)
        ax.text(1.0, ax.get_ylim()[0] + 0.005, " default", fontsize=8, color="gray", va="bottom")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Sensitivity at Deployment Scale ($n=700$) vs.\ QK-Scale ($n=25$)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig_sensitivity_full_n.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("  Done: fig_sensitivity_full_n.png")


# ================================================================
# FIGURE 18: ECE Reliability Diagram (NEW)
# ================================================================
def fig18_ece_reliability():
    """Deprecated: this function previously corrupted the quantum kernel's
    predicted probabilities (np.clip to [0.3, 0.7] plus injected noise) before
    scoring its calibration, which manufactured its reported ECE rather than
    measuring it, and hardcoded the "coverage at u<0.3 = 89.7%" annotation
    instead of reading it off the actual computed curve. The honest,
    multi-seed, matched-training-budget replacement (including a fair
    Random-Forest calibration baseline) lives in
    scripts/calibration_comparison.py + scripts/gen_calibration_figure.py --
    run those two scripts instead of this function."""
    print("  Skipped: fig_ece_reliability.png is now generated by "
          "scripts/calibration_comparison.py + scripts/gen_calibration_figure.py")


import os

if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_framework()
    fig2_qber()
    fig3_radar()
    fig4_confusion_roc()
    fig5_benchmark()
    fig6_uncertainty()
    fig7_correlation()
    fig8_amplitude()
    fig9_transmittance()
    fig10_latency()
    fig11_importance()
    fig12_gp_scaling()
    fig13_matched_n_curves()
    fig14_sensitivity_heatmap()
    fig15_adversarial_robustness()
    fig16_qk_crossover()
    fig17_sensitivity_full_n()
    fig18_ece_reliability()
    print("All figures done.")
