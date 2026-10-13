# QUASAR

**QUantum-inspired Anomaly detection with Statistically-Audited Robustness**

A matched-scale evaluation of the amplitude-encoding (quantum fidelity) kernel for security anomaly detection, in two domains:

- **HYQAD** — QKD physical-layer telemetry from a seeded decoy-state BB84 simulator with operating drift and an attack-strength axis *s*.
- **Q-ZTNA** — zero-trust network flows: a controlled copula-based flow simulator plus the public **UNSW-NB15** and **CIC-IDS2017** benchmarks.

Every compared model is trained on the same subsample at every training size, over multiple seeds with 95% confidence intervals.

## Results summary

| Setting | Finding |
|---|---|
| Kernel | The amplitude-encoding fidelity kernel equals squared cosine similarity and is computed exactly (verified against a PennyLane circuit to 7×10⁻¹⁵; ≈8×10⁴× faster). There is no small-*n* limit. |
| QKD telemetry, *s* = 0.3 | Supervised models within 0.80–0.84 MCC; RF best or tied-best. AE-SVM ≈ RBF-SVM (McNemar *p* > 0.05 on 10/10 seeds). The GP is the least calibrated model (ECE 0.047 vs 0.030 for isotonic RF). |
| Synthetic flows | AE-SVM ≈ RBF-SVM in accuracy (0.947 vs 0.948, *p* = 0.54); the norm-anchored AE-SVM reaches 0.960 / AUC 0.983. |
| UNSW-NB15 / CIC-IDS2017 | AE-SVM ≈ RBF-SVM (UNSW 0.862 vs 0.863; CIC 0.993 vs 0.994); XGBoost, RF and the DNN lead. |
| Zero-day (leave one attack type out) | Domain-dependent. In both controlled settings plain AE-SVM fails because its kernel discards input norm: synthetic flows 0.53 vs 0.84 AUC for RBF; QKD without its Mahalanobis feature 0.55 vs 0.94. A norm anchor, φ(x) = (x, c)/‖(x, c)‖, restores 0.79 and 0.94. On the public benchmarks it does not fail. Public data: UNSW-NB15 0.95 vs 0.93 (anchored 0.94); CIC-IDS2017 0.71 vs 0.66 (anchored 0.73). |

## Papers

| Paper | Source | Notes |
|---|---|---|
| *When Do Amplitude-Encoding Quantum Kernels Help Security Anomaly Detection?* (merged journal version) | [`papers/combined/quasar.tex`](papers/combined/quasar.tex) | Both domains, public datasets, mechanism and fix |
| *Calibrated and Kernel-Based Detection of QKD Implementation Attacks: A Controlled Simulation Study* | [`papers/hyqad/paper.tex`](papers/hyqad/paper.tex) | Conference version |
| *Amplitude-Encoding Kernels for Zero-Trust Network Anomaly Detection* | [`papers/qztna/qcyber.tex`](papers/qztna/qcyber.tex) | Conference version |

All three cite [`papers/references.bib`](papers/references.bib); every entry was checked, see [`papers/REFERENCE_AUDIT.md`](papers/REFERENCE_AUDIT.md).

## Installation

```bash
python -m pip install -e ".[dev]"
```

Requires Python ≥ 3.9. Dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `pennylane`, `matplotlib`, `seaborn`, `joblib`.

## Reproducing the results

Run from the repository root with `PYTHONPATH=.`. Set `OMP_NUM_THREADS=1`; parallelism is across seeds.

| Experiment | Script | Output |
|---|---|---|
| HYQAD (all experiments) | `scripts/hyqad/run_experiments.py [name ...]` | `data/hyqad/*.csv` |
| HYQAD figures | `scripts/hyqad/gen_figures.py` | `figures/hyqad/*.png` |
| Q-ZTNA synthetic flows | `scripts/qztna/qztna_run_experiments.py [name ...]` | `data/qztna/qztna_*.csv` |
| Public datasets (download) | `scripts/fetch_datasets.sh` | `data/external/` (git-ignored) |
| Q-ZTNA public datasets | `scripts/qztna/run_real_data.py [name ...]` | `data/qztna/real_*.csv` |
| Q-ZTNA figures | `scripts/qztna/qztna_gen_figures.py`, `scripts/qztna/gen_real_figures.py` | `figures/qztna/*.png` |
| Cross-domain figures | `scripts/combined_figures.py` | `figures/combined/*.png` |
| Norm-vs-direction diagnostic | `scripts/norm_diagnostic.py` | `data/norm_diagnostic.csv` |
| Fill paper numbers | `scripts/fill_papers.py` | `papers/*/…tex` from `*.template.tex` |
| Paper tables | `scripts/make_tables.py` | LaTeX rows printed from the CSVs |

`scripts/hyqad/demo.py` runs a short end-to-end HYQAD example. Run tests with `python -m pytest`.

## Changes in 0.3.0

Version 0.2.0 results are superseded. The fixes:

- **HYQAD simulator.** Every seed produced the same dataset, because the channel and attack simulators always used seed 42. Block timestamps leaked the label. The error-rate feature alone separated attacks perfectly. The gain used `(1-ηT)^μ` instead of the Poisson `1-e^{-ηTμ}`. The simulator now has operating drift, finite-size noise and an attack-strength parameter.
- **HYQAD features.** The old "28-dim" features were rolling averages of neighbouring rows of a shuffled table plus constant columns. The amplitude kernel saw only the first 8 of them. Features are now the 7 standardized measurements plus a Mahalanobis distance, and the kernel encodes all of them.
- **HYQAD quantum kernel.** It no longer silently falls back to an RBF kernel when a circuit call fails, and it is evaluated exactly at every training size. The old benchmark trained it on 25 samples only.
- **HYQAD baselines.** Outlier-score signs are fixed (OC-SVM AUC used to be reported as 0.000). IF and LOF are now fit on normal data, like OC-SVM.
- **HYQAD reporting.** Latency now excludes training time. Two figures used hard-coded numbers; every figure now reads the CSVs. The McNemar test the paper reported is now actually computed.
- **Q-ZTNA.** Added the public-dataset evaluation, deep baselines (DNN, autoencoder), the non-negative and norm-anchored kernel controls, and corrected text (Lipschitz bound, latency claims, a contradictory convergence section).
- **References.** Thirteen entries did not exist as cited: ten could not be found at all, and three had the wrong title, authors or venue. They were removed or replaced (see the audit).

## Citing this work

See [`CITATION.cff`](CITATION.cff).
