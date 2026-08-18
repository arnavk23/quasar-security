#!/usr/bin/env python3
"""Ablation at full N=5000 scale: GP uses ~3500 samples, QK capped at 25."""
import numpy as np, pandas as pd, warnings, time, os
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from hyqad.simulation import QKDDataGenerator
from hyqad.models import BayesianAnomalyEstimator, QuantumKernelEstimator
from hyqad.evaluation import Evaluator

N_SEEDS = 3
N_SAMPLES = 1000
ATTACK_RATIO = 0.3
N_QK = 25
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = str(ROOT / "data" / "hyqad" / "ablation_scaled.csv")
results = []

for seed in range(N_SEEDS):
    gen = QKDDataGenerator(rng=np.random.default_rng(seed))
    df = gen.generate(n_samples=N_SAMPLES, attack_ratio=ATTACK_RATIO)
    cols = [c for c in df.columns if c not in ("label", "is_attack")]
    X, y = df[cols].values, df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    n_tr = len(X_tr)

    # RF on full training set
    rf = RandomForestClassifier(n_estimators=200, random_state=seed)
    t0 = time.perf_counter(); rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te); ys = rf.predict_proba(X_te)[:, 1]
    lat = (time.perf_counter() - t0) / len(X_te) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append({"model":"RF","seed":seed,"n_train":n_tr,
        "acc":m.accuracy,"f1":m.f1_score,"auc":m.roc_auc,
        "ap":m.average_precision,"mcc":m.matthews_corrcoef,
        "bal_acc":m.balanced_accuracy,"spec":m.specificity,"lat":lat})
    print(f"  RF seed {seed} (n={n_tr})", flush=True)

    # GP-only on full training set
    gp = BayesianAnomalyEstimator(kernel="rbf", n_restarts_optimizer=3, random_state=seed)
    t0 = time.perf_counter(); gp.fit(X_tr, y_tr)
    yp = gp.predict(X_te); ys = gp.predict_proba(X_te)[:, 1]
    lat = (time.perf_counter() - t0) / len(X_te) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append({"model":"GP-only","seed":seed,"n_train":n_tr,
        "acc":m.accuracy,"f1":m.f1_score,"auc":m.roc_auc,
        "ap":m.average_precision,"mcc":m.matthews_corrcoef,
        "bal_acc":m.balanced_accuracy,"spec":m.specificity,"lat":lat})
    print(f"  GP-only seed {seed} (n={n_tr})", flush=True)

    # QK-only on 25 samples
    X_qk = X_tr[:N_QK]; y_qk = y_tr[:N_QK]
    qk = QuantumKernelEstimator(n_qubits=3, kernel_type="amplitude", random_state=seed)
    t0 = time.perf_counter(); qk.fit(X_qk, y_qk)
    yp = qk.predict(X_te); ys = qk.predict_proba(X_te)[:, 1]
    lat = (time.perf_counter() - t0) / len(X_te) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append({"model":"QK-only","seed":seed,"n_train":N_QK,
        "acc":m.accuracy,"f1":m.f1_score,"auc":m.roc_auc,
        "ap":m.average_precision,"mcc":m.matthews_corrcoef,
        "bal_acc":m.balanced_accuracy,"spec":m.specificity,"lat":lat})
    print(f"  QK-only seed {seed} (n={N_QK})", flush=True)

    # Hybrid: GP trained on full data, QK on 25, then meta-learner
    gp_full = BayesianAnomalyEstimator(kernel="rbf", n_restarts_optimizer=3, random_state=seed)
    gp_full.fit(X_tr, y_tr)
    qk_small = QuantumKernelEstimator(n_qubits=3, kernel_type="amplitude", random_state=seed)
    qk_small.fit(X_qk, y_qk)

    # Build meta-features from base estimators (on training data)
    # GP sees full set; QK sees only the 25 it was trained on
    Z_tr_gp = gp_full.predict_proba(X_tr)[:, 1].reshape(-1, 1)
    # QK only gets evaluated on X_qk subset for training meta-learner
    Z_qk_train = qk_small.predict_proba(X_qk)[:, 1].reshape(-1, 1)
    Z_tr = np.hstack([Z_tr_gp[:N_QK], Z_qk_train])  # meta-features: [GP_prob, QK_prob]

    meta = LogisticRegression(C=1.0, random_state=seed)
    t0 = time.perf_counter()
    meta.fit(Z_tr, y_qk)
    # At test time, GP scores all samples, QK scores all samples
    Z_te_gp = gp_full.predict_proba(X_te)[:, 1].reshape(-1, 1)
    Z_te_qk = qk_small.predict_proba(X_te)[:, 1].reshape(-1, 1)
    Z_te = np.hstack([Z_te_gp, Z_te_qk])
    ys = meta.predict_proba(Z_te)[:, 1]
    yp = meta.predict(Z_te)
    lat = (time.perf_counter() - t0) / len(X_te) * 1000

    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append({"model":"Hybrid","seed":seed,"n_train":f"{n_tr}(GP)+{N_QK}(QK)",
        "acc":m.accuracy,"f1":m.f1_score,"auc":m.roc_auc,
        "ap":m.average_precision,"mcc":m.matthews_corrcoef,
        "bal_acc":m.balanced_accuracy,"spec":m.specificity,"lat":lat})
    print(f"  Hybrid seed {seed} (GP={n_tr}, QK={N_QK})", flush=True)

    pd.DataFrame(results).to_csv(OUT, index=False)

df = pd.DataFrame(results)
print("\n=== Ablation at N=5000 scale ===")
print(f"{'Model':<12} {'n_train':<18} {'Acc':<10} {'F1':<10} {'AUC':<10} {'AP':<10} {'MCC':<10} {'Bal':<10} {'Spec':<10} {'Lat(ms)':<10}")
print("-"*110)
for m in ["RF","GP-only","QK-only","Hybrid"]:
    r = df[df.model == m]
    nt = r["n_train"].iloc[0]
    print(f"{m:<12} {str(nt):<18} " +
          "".join(f"{r[c].mean():.3f}±{r[c].std():.3f}  " for c in
                  ["acc","f1","auc","ap","mcc","bal_acc","spec"]) +
          f"{r['lat'].mean():.4f}±{r['lat'].std():.4f}")

df.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
