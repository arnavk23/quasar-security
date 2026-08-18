# QUASAR

**QUantum-inspired Anomaly detection with Statistically-Audited Robustness**

**HYQAD** (QKD physical-layer telemetry) and **Q-ZTNA** (zero-trust network flows). Both studies use a matched-scale, matched-dimensionality protocol. All compared models are trained at identical sample sizes and feature dimensionalities, across multiple seeds with confidence intervals.

## Results summary

| Study | Metric | Result |
|---|---|---|
| HYQAD | Deployed Bayesian GP accuracy | 0.994 ± 0.002, sub-10 ms latency |
| HYQAD | Quantum kernel SVM vs. classical, matched-*n* | No benefit at any tested scale (*n* ≤ 25, *O(n²)* circuit cost) |
| HYQAD | Calibration, matched budget (*n*=700) | Uncalibrated RF (ECE 0.007) outperforms deployed GP (ECE 0.130) |
| Q-ZTNA | Amplitude-encoding kernel SVM vs. RBF-SVM | Statistically indistinguishable (0.947 vs. 0.948 acc., *p*=0.54) |
| Q-ZTNA | Zero-day (leave-one-attack-out) generalization | AE-SVM AUC as low as 0.45 vs. 0.81–0.89 for RBF-SVM |
| Q-ZTNA | Inference latency | AE-SVM ≈1.2× faster than RBF-SVM (Nyström) |

Across both domains, we find no robust accuracy advantage for the
amplitude-encoding/cosine-similarity kernel over classical kernels; this null
result is reported as a primary finding, not omitted.

## Papers

1. **Calibrated Uncertainty Quantification for QKD Anomaly Detection** (*HYQAD*)
2. **Amplitude-Encoding Kernels for Zero-Trust Network Anomaly Detection** (*Q-ZTNA*)

Progress slides: [papers/hyqad/presentation.pdf](papers/hyqad/presentation.pdf).`

## Installation

```bash
python -m pip install -e ".[dev]"
```

Requires Python ≥ 3.9. Core dependencies: `numpy`, `scipy`, `scikit-learn`, `xgboost`, `pandas`, `torch`, `pennylane`, `matplotlib`, `seaborn`.

## Quick start

```python
# HYQAD
from hyqad.config import Config
from hyqad.simulation import QKDDataGenerator
from hyqad.pipeline import InferencePipeline

config = Config(n_samples=500, attack_ratio=0.3)
train_df, val_df, test_df = QKDDataGenerator().generate_train_val_test(
    n_samples=config.n_samples, attack_ratio=config.attack_ratio)
cols = [c for c in train_df.columns if c != "label"]
pipeline = InferencePipeline(config).fit(train_df[cols].values, train_df["label"].values)
```

```python
# Q-ZTNA
from qztna.simulation import NetworkFlowGenerator
from qztna.models.kernel_svm import fit_kernel_svm

df = NetworkFlowGenerator(seed=42).generate_dataset(n_samples=8000, attack_ratio=0.2, random_state=0)
cols = [c for c in df.columns if c.startswith("f")]
model, _ = fit_kernel_svm("ae", df[cols].values, df["label"].values, random_state=0)
```

A full HYQAD demo (data generation → training → evaluation → visualization): `python scripts/hyqad/demo.py`.

## Reproducing the results

All scripts are run from the repository root (e.g. `python scripts/hyqad/benchmark_full.py`) and resolve their own paths regardless of working directory.

| Experiment | Script | Output |
|---|---|---|
| HYQAD full benchmark | `scripts/hyqad/benchmark_full.py` | `data/hyqad/benchmark_full.csv` |
| HYQAD matched-*n* kernel ablation | `scripts/hyqad/matched_n_ablation.py` | `data/hyqad/matched_n_results.csv` |
| HYQAD quantum-kernel crossover | `scripts/hyqad/qk_crossover.py` | `data/hyqad/qk_crossover_results.csv` |
| HYQAD sensitivity analysis | `scripts/hyqad/sensitivity_full_n.py` | `data/hyqad/sensitivity_full_n_results.csv` |
| HYQAD adversarial robustness | `scripts/hyqad/adversarial_robustness.py` | `data/hyqad/adversarial_robustness_results.csv` |
| HYQAD GP scaling | `scripts/hyqad/ablation_scaled.py` | `data/hyqad/gp_scaling.csv` |
| HYQAD calibration (GP vs. QK vs. RF) | `scripts/hyqad/calibration_comparison.py` | `data/hyqad/calibration_comparison.csv` |
| HYQAD figures | `scripts/hyqad/gen_paper_figures.py`, `gen_calibration_figure.py` | `figures/hyqad/*.png` |
| Q-ZTNA full experiment suite | `scripts/qztna/qztna_run_experiments.py` | `data/qztna/qztna_*.csv` |
| Q-ZTNA figures | `scripts/qztna/qztna_gen_figures.py` | `figures/qztna/*.png` |

Run tests with `python -m pytest`.

## Citing this work

See [`CITATION.cff`](CITATION.cff), or:

```bibtex
@unpublished{kapoor2026hyqad,
  title  = {Calibrated Uncertainty Quantification for QKD Anomaly Detection},
  author = {Kapoor, Arnav},
  year   = {2026},
  note   = {Manuscript. Code and data: https://github.com/arnavk23/quasar-security}
}

@unpublished{kapoor2026qztna,
  title  = {Amplitude-Encoding Kernels for Zero-Trust Network Anomaly Detection},
  author = {Kapoor, Arnav},
  year   = {2026},
  note   = {Manuscript. Code and data: https://github.com/arnavk23/quasar-security}
}
```