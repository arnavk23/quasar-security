# HYQAD & Q-ZTNA

**Calibrated uncertainty quantification and quantum-kernel methods for quantum-key-distribution and zero-trust network anomaly detection.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)
[![arXiv-style preprint](https://img.shields.io/badge/paper%20supplement-%E2%80%93-lightgrey)](papers/)

This repository contains the **complete code, data, figures, and paper sources** accompanying our research on hybrid quantum–classical anomaly detection for physical-layer QKD security and zero-trust networking. It is organized in the standard "paper supplement" layout so every table, figure, and result in the papers can be reproduced from source.

---

## Papers

Two complementary manuscripts are included, both built on the same `hyqad` codebase:

### 1. Calibrated Uncertainty Quantification for QKD Anomaly Detection

> **with a Scale-Limited Study of Amplitude-Encoding Kernels** — *HYQAD*

A calibrated uncertainty quantification framework for Quantum Key Distribution (QKD) anomaly detection, combining a **Bayesian Gaussian Process classifier** with quantum-inspired feature engineering. On synthetic QKD telemetry spanning four attack scenarios (Photon-Number-Splitting, Detector Blinding, Trojan Horse, Intercept-Resend), the deployed inference path achieves **0.994 ± 0.002 accuracy with sub-10 ms latency**, and calibrated predictive uncertainty enables selective prediction: at a modest uncertainty threshold the detector reaches **1.000 accuracy on 89.7 % of the test set**, deferring the remainder for human review.

- 📄 [Manuscript (PDF)](papers/paper.pdf) · [LaTeX source](papers/paper.tex)

### 2. Amplitude-Encoding Kernel Methods for Zero-Trust Network Anomaly Detection

> **A Cosine-Similarity Feature-Space Approach to Real-Time Threat Intelligence** — *Q-ZTNA*

An anomaly detection framework using an amplitude-encoding (cosine-similarity) kernel—a classically computable feature map K(𝐱ᵢ, 𝐱ⱼ) = |⟨𝐱ᵢ, 𝐱ⱼ⟩|²/‖𝐱ᵢ‖²‖𝐱ⱼ‖²—as the core of a kernel SVM with Nyström approximation. On synthetic enterprise-network telemetry spanning five attack classes, the amplitude-encoding SVM achieves **0.997 ± 0.001 detection accuracy with 0.42 ms mean inference latency on CPU**, and a matched-*n* ablation shows a statistically significant accuracy advantage over RBF and Laplacian kernels at training sizes *n* ≥ 500.

- 📄 [Manuscript (PDF)](papers/qcyber.pdf) · [LaTeX source](papers/qcyber.tex)

**Slides:** [Progress presentation (PDF)](papers/presentation.pdf)

---

## Repository layout

```
.
├── hyqad/               # Core Python package (source of truth for all experiments)
│   ├── simulation/      # QKD channel, attack, and data generators
│   ├── features/        # Amplitude encoding, covariance / MCD features
│   ├── models/          # Bayesian GP, quantum kernel, hybrid detector, optimizer
│   ├── pipeline/        # Inference pipeline and edge deployment
│   └── evaluation/      # Metrics and benchmark suites
├── scripts/             # Experiment and figure-generation scripts
├── tests/               # Unit tests (pytest)
├── data/                # Experiment results (CSV) — reproducibility artifacts
├── figures/             # Paper figures (PNG) referenced from the manuscripts
├── papers/              # LaTeX sources + compiled PDFs
├── articles/            # Article supplement series (.docx)
└── reports/             # Weekly progress reports (.docx)
```

## Installation

```bash
python -m pip install -e .          # install the hyqad package
python -m pip install -r scripts/requirements.txt
```

Requires Python ≥ 3.9. Main dependencies: `numpy`, `scipy`, `scikit-learn`, `pandas`, `torch`, `pennylane`, `matplotlib`, `seaborn`.

## Quick start

```python
from hyqad.config import Config
from hyqad.simulation import QKDDataGenerator
from hyqad.pipeline import InferencePipeline
from hyqad.evaluation import Evaluator

config = Config(n_samples=500, attack_ratio=0.3)
train_df, val_df, test_df = QKDDataGenerator().generate_train_val_test(
    n_samples=config.n_samples, attack_ratio=config.attack_ratio)

cols = [c for c in train_df.columns if c != "label"]
pipeline = InferencePipeline(config).fit(train_df[cols].values, train_df["label"].values)

y_pred = pipeline.predict(test_df[cols].values)
y_score = pipeline.predict_proba(test_df[cols].values)[:, 1]
metrics = Evaluator().evaluate(test_df["label"].values, y_pred, y_score=y_score)
```

A full end-to-end demo (data generation → training → evaluation → visualizations) is available as:

```bash
python scripts/demo.py
```

## Reproducing the results

Each experiment in the papers is backed by a script under `scripts/` and its outputs under `data/`:

| Paper experiment                 | Script                              | Output artifact                      |
| -------------------------------- | ----------------------------------- | ------------------------------------ |
| Full benchmark (5 seeds)         | `scripts/benchmark_full.py`         | `data/benchmark_full.csv`            |
| Matched-*n* kernel ablation      | `scripts/matched_n_ablation.py`     | `data/matched_n_results.csv`         |
| Quantum-kernel crossover study   | `scripts/qk_crossover.py`           | `data/qk_crossover_results.csv`      |
| Sensitivity analysis             | `scripts/sensitivity_full_n.py`     | `data/sensitivity_full_n_results.csv` |
| Adversarial robustness           | `scripts/adversarial_robustness.py` | `data/adversarial_robustness_results.csv` |
| GP scaling                       | `scripts/ablation_scaled.py`        | `data/gp_scaling.csv`                |
| Paper figures                    | `scripts/gen_paper_figures.py`      | `figures/*.png`                      |

Run the test suite with:

```bash
python -m pytest
```

## Citing this work

```bibtex
@article{kapoor_hyqad,
  title   = {Calibrated Uncertainty Quantification for QKD Anomaly Detection,
             with a Scale-Limited Study of Amplitude-Encoding Kernels},
  author  = {Kapoor, Arnav},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {Code and data: https://github.com/arnavk23/ccri}
}

@article{kapoor_qztna,
  title   = {Amplitude-Encoding Kernel Methods for Zero-Trust Network Anomaly Detection:
             A Cosine-Similarity Feature-Space Approach to Real-Time Threat Intelligence},
  author  = {Kapoor, Arnav},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {Code and data: https://github.com/arnavk23/ccri}
}
```

## License

This repository is released under the [MIT License](LICENSE).
