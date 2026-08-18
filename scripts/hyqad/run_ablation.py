#!/usr/bin/env python3
"""Ablation: GP-only, QK-only, Hybrid + baselines, multi-seed."""
import sys, numpy as np, pandas as pd, warnings, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore")
from hyqad.simulation import QKDDataGenerator
from hyqad.models import BayesianAnomalyEstimator, QuantumKernelEstimator, HybridAnomalyDetector
from hyqad.evaluation import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

N_SEEDS = 3
N_SAMPLES = 500
ATTACK_RATIO = 0.3
N_TRAIN_QK = 30       # small for QK speed
results = []

for seed in range(N_SEEDS):
    gen = QKDDataGenerator(rng=np.random.default_rng(seed))
    df = gen.generate(n_samples=N_SAMPLES, attack_ratio=ATTACK_RATIO)
    cols = [c for c in df.columns if c not in ("label", "is_attack")]
    X, y = df[cols].values, df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    # RF
    rf = RandomForestClassifier(n_estimators=200, random_state=seed)
    t0 = time.perf_counter(); rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te); ys = rf.predict_proba(X_te)[:, 1]
    lat = (time.perf_counter() - t0) / len(X_te) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append({"model":"RF","seed":seed,"acc":m.accuracy,"f1":m.f1_score,
        "auc":m.roc_auc,"ap":m.average_precision,"mcc":m.matthews_corrcoef,
        "bal_acc":m.balanced_accuracy,"spec":m.specificity,"lat":lat})
    print(f"  RF seed {seed} done", flush=True)

    # GP-only (full train set)
    gp = BayesianAnomalyEstimator(kernel="rbf", n_restarts_optimizer=3, random_state=seed)
    t0 = time.perf_counter(); gp.fit(X_tr, y_tr)
    yp = gp.predict(X_te); ys = gp.predict_proba(X_te)[:, 1]
    lat = (time.perf_counter() - t0) / len(X_te) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append({"model":"GP-only","seed":seed,"acc":m.accuracy,"f1":m.f1_score,
        "auc":m.roc_auc,"ap":m.average_precision,"mcc":m.matthews_corrcoef,
        "bal_acc":m.balanced_accuracy,"spec":m.specificity,"lat":lat})
    print(f"  GP seed {seed} done", flush=True)

    # QK-only (small train set)
    X_tr_qk = X_tr[:N_TRAIN_QK]; y_tr_qk = y_tr[:N_TRAIN_QK]
    qk = QuantumKernelEstimator(n_qubits=3, kernel_type="amplitude", random_state=seed)
    t0 = time.perf_counter(); qk.fit(X_tr_qk, y_tr_qk)
    yp = qk.predict(X_te); ys = qk.predict_proba(X_te)[:, 1]
    lat = (time.perf_counter() - t0) / len(X_te) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append({"model":"QK-only","seed":seed,"acc":m.accuracy,"f1":m.f1_score,
        "auc":m.roc_auc,"ap":m.average_precision,"mcc":m.matthews_corrcoef,
        "bal_acc":m.balanced_accuracy,"spec":m.specificity,"lat":lat})
    print(f"  QK seed {seed} done", flush=True)

    # Hybrid (small train set for QK, full for GP)
    hy = HybridAnomalyDetector(use_bayesian=True, use_quantum_kernel=True,
        bayesian_kernel="rbf", n_qubits=3, quantum_kernel_type="amplitude",
        meta_learner="logistic", n_restarts_optimizer=3, random_state=seed)
    # Fit on small set so QK doesn't blow up
    t0 = time.perf_counter(); hy.fit(X_tr_qk, y_tr_qk)
    yp = hy.predict(X_te); ys = hy.predict_proba(X_te)[:, 1]
    lat = (time.perf_counter() - t0) / len(X_te) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append({"model":"Hybrid","seed":seed,"acc":m.accuracy,"f1":m.f1_score,
        "auc":m.roc_auc,"ap":m.average_precision,"mcc":m.matthews_corrcoef,
        "bal_acc":m.balanced_accuracy,"spec":m.specificity,"lat":lat})
    print(f"  Hybrid seed {seed} done", flush=True)

df = pd.DataFrame(results)
print("\n=== Ablation (mean ± std over %d seeds) ===" % N_SEEDS)
print("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-12s" % (
    "Model","Acc","F1","AUC","AP","MCC","BalAcc","Spec","Lat(ms)"))
print("-"*100)
for m in ["RF","GP-only","QK-only","Hybrid"]:
    r = df[df.model == m]
    print("%-10s" % m +
          "".join("%-10s" % ("%.3f±%.3f" % (r[c].mean(), r[c].std())) for c in
                  ["acc","f1","auc","ap","mcc","bal_acc","spec"]) +
          "%-12s" % ("%.4f±%.4f" % (r["lat"].mean(), r["lat"].std())))

df.to_csv(str(ROOT / "data" / "hyqad" / "ablation_results.csv"), index=False)
print("\nSaved to ablation_results.csv")
