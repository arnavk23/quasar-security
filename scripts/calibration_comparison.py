"""
Honest calibration comparison for the HYQAD paper (papers/paper.tex), matching
the actual deployed-model configuration in scripts/benchmark_full.py exactly:
N=1000 samples, 70/30 split, 28-dim engineered features (covariance/MCD +
amplitude encoding), GP trained on the full ~700-sample training split, QK-SVM
on n<=25 samples.

This recomputes the Bayesian GP vs. quantum-kernel-SVM ECE comparison without
the artificial probability corruption present in the old fig18_ece_reliability()
in scripts/gen_paper_figures.py (which did
``np.clip(proba_qk, 0.3, 0.7) + rng.uniform(-0.05, 0.05, ...)`` before scoring
the quantum kernel's calibration -- that manufactures the QK-SVM's reported
ECE rather than measuring it), and adds the missing baseline a reviewer would
ask for: Platt- and isotonic-calibrated Random Forest, both matched to the
GP's ~700-sample training budget and at the same full budget (RF doesn't need
a smaller subset -- it already uses the full training split like the GP does).
It also recomputes the "accuracy at u<0.3" selective-prediction claim honestly
across seeds instead of trusting a single hardcoded annotation.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from hyqad.simulation import QKDDataGenerator
from hyqad.models import BayesianAnomalyEstimator, QuantumKernelEstimator
from hyqad.features import CovarianceFeatureExtractor, AmplitudeEncoder

DATA = Path("data")
N_SAMPLES = 1000
GP_SUBSET = 700
N_QK = 25


def build_28dim(X_tr_raw, X_te_raw):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_raw)
    X_te_sc = scaler.transform(X_te_raw)
    cov = CovarianceFeatureExtractor(smoothing=1e-6, window=20)
    X_tr_cov = cov.fit_transform(X_tr_sc)
    X_te_cov = cov.transform(X_te_sc)
    amp = AmplitudeEncoder(encoding_dim=8)
    X_tr_amp = amp.fit_transform(X_tr_cov)
    X_te_amp = amp.transform(X_te_cov)
    return (np.concatenate([X_tr_cov, X_tr_amp], axis=1),
            np.concatenate([X_te_cov, X_te_amp], axis=1))


def compute_ece(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs, counts = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for i in range(n_bins):
        idx = (probs >= bins[i]) & (probs < bins[i + 1] + (1e-9 if i == n_bins - 1 else 0))
        counts[i] = idx.sum()
        if counts[i] > 0:
            accs[i] = labels[idx].mean()
            confs[i] = probs[idx].mean()
    ece = float((counts / len(probs) * np.abs(accs - confs)).sum())
    return ece, accs, confs, counts, (bins[:-1] + bins[1:]) / 2


def run(n_seeds=10):
    rows = []
    reliability_rows = []
    for seed in range(n_seeds):
        gen = QKDDataGenerator(rng=np.random.default_rng(seed))
        df = gen.generate(n_samples=N_SAMPLES, attack_ratio=0.3)
        raw_cols = [c for c in df.columns if c not in ("label", "is_attack")]
        X_raw, y = df[raw_cols].values, df["label"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.3, stratify=y, random_state=seed)
        X_tr_28, X_te_28 = build_28dim(X_tr, X_te)

        # Bayesian GP (deployed model), full ~700-sample training budget.
        gp = BayesianAnomalyEstimator(kernel="rbf", n_restarts_optimizer=3, random_state=seed)
        gp.fit(X_tr_28[:GP_SUBSET], y_tr[:GP_SUBSET])
        proba_gp = gp.predict_proba(X_te_28)[:, 1]
        ece_gp, *_ = compute_ece(proba_gp, y_te)
        uncert_gp = np.sqrt(proba_gp * (1 - proba_gp))
        pred_gp = (proba_gp >= 0.5).astype(int)
        correct_gp = (pred_gp == y_te).astype(int)
        mask = uncert_gp < 0.3
        cov_u03 = float(mask.mean())
        acc_u03 = float(correct_gp[mask].mean()) if mask.sum() > 0 else np.nan

        # Quantum kernel SVM, n<=25, genuine predict_proba, NO post-hoc corruption.
        qk = QuantumKernelEstimator(n_qubits=3, kernel_type="amplitude", C=1.0, random_state=seed)
        qk.fit(X_tr_28[:N_QK], y_tr[:N_QK])
        proba_qk = qk.predict_proba(X_te_28)[:, 1]
        ece_qk, *_ = compute_ece(proba_qk, y_te)

        # Random Forest at the SAME 700-sample budget as the GP: raw, Platt, isotonic.
        rf = RandomForestClassifier(n_estimators=200, random_state=seed)
        rf.fit(X_tr_28[:GP_SUBSET], y_tr[:GP_SUBSET])
        proba_rf_raw = rf.predict_proba(X_te_28)[:, 1]
        ece_rf_raw, *_ = compute_ece(proba_rf_raw, y_te)

        rf_platt = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, random_state=seed), method="sigmoid", cv=5
        )
        rf_platt.fit(X_tr_28[:GP_SUBSET], y_tr[:GP_SUBSET])
        proba_rf_platt = rf_platt.predict_proba(X_te_28)[:, 1]
        ece_rf_platt, *_ = compute_ece(proba_rf_platt, y_te)

        rf_iso = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, random_state=seed), method="isotonic", cv=5
        )
        rf_iso.fit(X_tr_28[:GP_SUBSET], y_tr[:GP_SUBSET])
        proba_rf_iso = rf_iso.predict_proba(X_te_28)[:, 1]
        ece_rf_iso, *_ = compute_ece(proba_rf_iso, y_te)

        rows.append(dict(seed=seed, ece_gp=ece_gp, ece_qk=ece_qk, ece_rf_raw=ece_rf_raw,
                          ece_rf_platt=ece_rf_platt, ece_rf_iso=ece_rf_iso,
                          coverage_u03=cov_u03, accuracy_u03=acc_u03,
                          full_acc_gp=float(correct_gp.mean())))

        if seed == 0:
            named = {"GP": proba_gp, "QK-SVM": proba_qk, "rf_raw_700": proba_rf_raw,
                     "rf_iso_700": proba_rf_iso}
            for name, p in named.items():
                e, accs, confs, counts, centers = compute_ece(p, y_te)
                for a, c, n, ctr in zip(accs, confs, counts, centers):
                    if n > 0:
                        reliability_rows.append(dict(model=name, bin_center=ctr, acc=a, conf=c, count=n))
        print(f"seed {seed}: ece_gp={ece_gp:.4f} ece_qk={ece_qk:.4f} ece_rf_raw={ece_rf_raw:.4f} "
              f"ece_rf_platt={ece_rf_platt:.4f} ece_rf_iso={ece_rf_iso:.4f} "
              f"cov@u<0.3={cov_u03:.4f} acc@u<0.3={acc_u03:.4f} full_acc={correct_gp.mean():.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(DATA / "calibration_comparison.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(DATA / "calibration_reliability_seed0.csv", index=False)

    print("\n=== Mean +/- std over", n_seeds, "seeds ===")
    for col in [c for c in df_out.columns if c != "seed"]:
        print(f"{col}: {df_out[col].mean():.4f} +/- {df_out[col].std():.4f}")


if __name__ == "__main__":
    run()
