#!/usr/bin/env python3
"""Generate all figures for the 5 articles."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from hyqad.simulation import QKDDataGenerator

sns.set_theme(style="whitegrid", font_scale=1.1)
FIGS = "/home/arnav/Downloads/ccri/figures"

gen = QKDDataGenerator()

# ---- Figure 1: Framework Architecture ----
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
layers = [
    ("QKD Channel\nSimulation", 0.1, 0.7, 0.15),
    ("Attack Vectors\n(PNS, Blinding, etc.)", 0.1, 0.5, 0.15),
    ("Feature Engineering\n(Amplitude Encoding,\nCovariance Features)", 0.35, 0.7, 0.15),
    ("Hybrid Detector\n(Bayesian GP +\nQuantum Kernel)", 0.6, 0.7, 0.15),
    ("Evaluation &\nBenchmarking", 0.85, 0.7, 0.15),
]
for label, cx, cy, w in layers:
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - 0.08), w, 0.16,
                                 fc="lightsteelblue", ec="navy", lw=2, alpha=0.9,
                                 boxstyle="round,pad=0.02"))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=10, fontweight="bold")
arrows = [(0.22, 0.7, 0.35, 0.7), (0.22, 0.5, 0.35, 0.7),
           (0.485, 0.7, 0.6, 0.7), (0.735, 0.7, 0.85, 0.7)]
for x1, y1, x2, y2 in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="navy", lw=2))
ax.set_xlim(0, 1); ax.set_ylim(0.25, 1)
ax.set_title("Hybrid AI-Quantum Anomaly Detection Framework for QKD Systems",
             fontsize=14, fontweight="bold", pad=15)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig1_framework.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: QBER Distributions ----
df = gen.generate(n_samples=3000, attack_ratio=0.3)
fig, ax = plt.subplots(figsize=(8, 5))
for lbl, color, label in [(0, "forestgreen", "Normal"), (1, "crimson", "Attack")]:
    subset = df[df["label"] == lbl]["qber"]
    ax.hist(subset, bins=35, alpha=0.7, color=color, label=label, density=True)
ax.set_xlabel("Quantum Bit Error Rate (QBER)")
ax.set_ylabel("Density")
ax.set_title("QBER Distribution: Normal vs. Attack Channel States")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGS}/fig2_qber_dist.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- Figure 3: Amplitude Encoding ----
from hyqad.features import AmplitudeEncoder
rng = np.random.default_rng(42)
X_raw = rng.normal(0, 1, (5, 8))
encoder = AmplitudeEncoder(encoding_dim=8)
X_enc = encoder.transform(X_raw)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]; im = ax.imshow(X_raw, cmap="RdBu", aspect="auto", vmin=-3, vmax=3)
ax.set_title("Raw Classical Features"); ax.set_xlabel("Feature"); ax.set_ylabel("Sample")
plt.colorbar(im, ax=ax)
ax = axes[1]; im = ax.imshow(X_enc, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
ax.set_title("Amplitude-Encoded (Hilbert Space)"); ax.set_xlabel("Basis State"); ax.set_ylabel("Sample")
plt.colorbar(im, ax=ax)
fig.suptitle("Amplitude Encoding: Classical → Quantum Hilbert Space", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig3_amplitude_encoding.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- Figure 4: Bayesian GP Uncertainty ----
from hyqad.models import BayesianAnomalyEstimator
from sklearn.model_selection import train_test_split
feature_cols = [c for c in df.columns if c not in ("label", "is_attack")]
X, y = df[feature_cols].values, df["label"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
bae = BayesianAnomalyEstimator(random_state=42)
bae.fit(X_tr[:200], y_tr[:200])
preds, proba, uncert = bae.predict_with_uncertainty(X_te[:200])
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(proba, uncert, c=y_te[:200], cmap="coolwarm", alpha=0.7, s=40, edgecolors="k")
ax.set_xlabel("Anomaly Probability")
ax.set_ylabel("Predictive Uncertainty (std)")
ax.set_title("Bayesian GP: Anomaly Probability vs. Uncertainty")
cbar = plt.colorbar(sc, ax=ax); cbar.set_label("Ground Truth (0=Normal, 1=Attack)")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig4_bayesian_uncertainty.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- Figure 5: Attack Signatures (radar) ----
attack_labels_map = {"pns": 0, "detector_blinding": 1, "trojan_horse": 2, "intercept_resend": 3, "normal": 4}
radar_metrics = ["qber", "photon_count_rate", "detection_efficiency",
                 "dark_count_rate", "signal_attenuation", "background_noise"]
attack_profiles = {}
for atk_name in ["pns", "detector_blinding", "trojan_horse", "intercept_resend"]:
    sub_df = gen.generate(n_samples=500, attack_ratio=1.0, attack_types=[atk_name])
    attack_profiles[atk_name] = sub_df[radar_metrics].mean().values
normal_df = gen.generate(n_samples=500, attack_ratio=0.0)
attack_profiles["normal"] = normal_df[radar_metrics].mean().values

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]
colors = {"pns": "tab:blue", "detector_blinding": "tab:orange",
          "trojan_horse": "tab:green", "intercept_resend": "tab:red", "normal": "gray"}
for name, prof in attack_profiles.items():
    vals = prof.tolist() + [prof[0]]
    ax.plot(angles, vals, "o-", label=name, color=colors[name], lw=2, alpha=0.8)
    ax.fill(angles, vals, alpha=0.1, color=colors[name])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_metrics, fontsize=9)
ax.set_title("Attack Signatures in QKD Channel Parameter Space", fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
fig.tight_layout()
fig.savefig(f"{FIGS}/fig5_attack_signatures.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- Figure 6: Benchmark comparison ----
bench_data = {
    "Model": ["RandomForest", "HybridDetector", "OneClassSVM", "IsolationForest", "LOF"],
    "F1": [0.912, 0.889, 0.728, 0.519, 0.508],
    "ROC-AUC": [0.996, 0.988, 0.857, 0.999, 0.775],
    "FPR": [0.002, 0.012, 0.116, 0.0, 0.026],
}
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(bench_data["Model"])); w = 0.25
ax.bar(x - w, bench_data["F1"], w, label="F1 Score", color="steelblue")
ax.bar(x, bench_data["ROC-AUC"], w, label="ROC-AUC", color="seagreen")
ax.bar(x + w, bench_data["FPR"], w, label="FPR", color="coral")
ax.set_xticks(x); ax.set_xticklabels(bench_data["Model"], rotation=15)
ax.set_ylabel("Score")
ax.set_title("Benchmark: Hybrid Detector vs. Classical Baselines")
ax.legend(); ax.set_ylim(0, 1.1)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig6_benchmark.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- Figure 7: QKD channel attenuation profile ----
losses = np.linspace(0, 30, 100)
transmittance = 10 ** (-losses / 10)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, transmittance, "b-", lw=2)
ax.set_xlabel("Channel Loss (dB)")
ax.set_ylabel("Transmittance")
ax.set_title("Quantum Channel Transmittance vs. Fiber Loss")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig7_transmittance.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- Figure 8: Covariance feature correlation map ----
cov_data = df[feature_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(cov_data, dtype=bool))
sns.heatmap(cov_data, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, ax=ax)
ax.set_title("Cross-Parameter Correlations in QKD Channel Data")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig8_correlation_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("All figures generated in", FIGS)
