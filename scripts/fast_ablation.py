#!/usr/bin/env python3
"""Fast ablation: incremental saving, tiny QK sizes for speed."""
import sys, numpy as np, pandas as pd, warnings, time, os
warnings.filterwarnings("ignore")
from hyqad.simulation import QKDDataGenerator
from hyqad.models import BayesianAnomalyEstimator, QuantumKernelEstimator, HybridAnomalyDetector
from hyqad.evaluation import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

N_SEEDS = 3
N_SAMPLES = 500
ATTACK_RATIO = 0.3
N_TRAIN_QK = 25
OUT = "/home/arnav/Downloads/ccri/ablation_results.csv"
results = []

for seed in range(N_SEEDS):
    gen = QKDDataGenerator(rng=np.random.default_rng(seed))
    df = gen.generate(n_samples=N_SAMPLES, attack_ratio=ATTACK_RATIO)
    cols = [c for c in df.columns if c not in ("label", "is_attack")]
    X, y = df[cols].values, df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    for model_name, estimator_fn in [
        ("RF", lambda: RandomForestClassifier(n_estimators=200, random_state=seed)),
        ("GP-only", lambda: BayesianAnomalyEstimator(kernel="rbf", n_restarts_optimizer=3, random_state=seed)),
    ]:
        est = estimator_fn()
        t0 = time.perf_counter(); est.fit(X_tr, y_tr)
        yp = est.predict(X_te); ys = est.predict_proba(X_te)[:, 1]
        lat = (time.perf_counter() - t0) / len(X_te) * 1000
        m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
        results.append({"model":model_name,"seed":seed,"acc":m.accuracy,"f1":m.f1_score,
            "auc":m.roc_auc,"ap":m.average_precision,"mcc":m.matthews_corrcoef,
            "bal_acc":m.balanced_accuracy,"spec":m.specificity,"lat":lat})
        pd.DataFrame(results).to_csv(OUT, index=False)
        print(f"  {model_name} seed {seed} done", flush=True)

    # QK-only and Hybrid on tiny subset
    X_qk = X_tr[:N_TRAIN_QK]; y_qk = y_tr[:N_TRAIN_QK]
    for model_name, estimator_fn in [
        ("QK-only", lambda: QuantumKernelEstimator(n_qubits=3, kernel_type="amplitude", random_state=seed)),
        ("Hybrid", lambda: HybridAnomalyDetector(use_bayesian=True, use_quantum_kernel=True,
            bayesian_kernel="rbf", n_qubits=3, quantum_kernel_type="amplitude",
            meta_learner="logistic", n_restarts_optimizer=3, random_state=seed)),
    ]:
        est = estimator_fn()
        # Only fit on tiny subset
        t0 = time.perf_counter()
        if model_name == "Hybrid":
            # Hybrid: GP on full, QK on tiny, stacked on tiny
            # Temp workaround: fit Hybrid directly on tiny data
            est.fit(X_qk, y_qk)
        else:
            est.fit(X_qk, y_qk)
        yp = est.predict(X_te); ys = est.predict_proba(X_te)[:, 1]
        lat = (time.perf_counter() - t0) / len(X_te) * 1000
        m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
        results.append({"model":model_name,"seed":seed,"acc":m.accuracy,"f1":m.f1_score,
            "auc":m.roc_auc,"ap":m.average_precision,"mcc":m.matthews_corrcoef,
            "bal_acc":m.balanced_accuracy,"spec":m.specificity,"lat":lat})
        pd.DataFrame(results).to_csv(OUT, index=False)
        print(f"  {model_name} seed {seed} done", flush=True)

df = pd.DataFrame(results)
print("\n=== Ablation (mean ± std over %d seeds) ===" % N_SEEDS)
print("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-12s" % (
    "Model","Acc","F1","AUC","AP","MCC","BalAcc","Spec","Lat(ms)"))
print("-"*100)
for m in ["RF","GP-only","QK-only","Hybrid"]:
    r = df[df.model == m]
    if len(r) == 0:
        continue
    print("%-10s" % m +
          "".join("%-10s" % ("%.3f±%.3f" % (r[c].mean(), r[c].std())) for c in
                  ["acc","f1","auc","ap","mcc","bal_acc","spec"]) +
          "%-12s" % ("%.4f±%.4f" % (r["lat"].mean(), r["lat"].std())))

df.to_csv(OUT, index=False)
print("\nSaved:", OUT)
