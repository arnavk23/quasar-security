#!/usr/bin/env python3
"""
Matched-n ablation (practical version):
- GP/RF at all sizes (10 seeds) — fast
- QK-only at n=25 only (3 seeds) — documents the flat curve
- Shows GP/RF scale while QK remains at baseline
"""
import warnings, time, numpy as np, pandas as pd, sys
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from hyqad.simulation import QKDDataGenerator
from hyqad.models import BayesianAnomalyEstimator, QuantumKernelEstimator
from hyqad.features import CovarianceFeatureExtractor, AmplitudeEncoder
from hyqad.evaluation import Evaluator

N_SEEDS_RF_GP = 10
N_SEEDS_QK = 3
N_SAMPLES = 1000
OUT = "/home/arnav/Downloads/ccri/matched_n_results.csv"

GP_RF_SIZES = [25, 50, 100, 150, 200, 300, 500, 700]

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

results = []

# Phase 1: GP/RF at all sizes, 10 seeds
print("=" * 60)
print("Phase 1: GP/RF at all sizes (10 seeds)")
print("=" * 60)
for seed in range(N_SEEDS_RF_GP):
    gen = QKDDataGenerator(rng=np.random.default_rng(seed))
    df = gen.generate(n_samples=N_SAMPLES, attack_ratio=0.3)
    raw_cols = [c for c in df.columns if c not in ("label", "is_attack")]
    X_raw, y = df[raw_cols].values, df["label"].values
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=seed)
    X_tr_28_full, X_te_28 = build_28dim(X_tr_full, X_te)
    print(f"\nSeed {seed}:", flush=True)

    for n_train in GP_RF_SIZES:
        X_n = X_tr_28_full[:n_train]
        y_n = y_tr_full[:n_train]

        rf = RandomForestClassifier(n_estimators=200, random_state=seed)
        t0 = time.perf_counter(); rf.fit(X_n, y_n)
        yp = rf.predict(X_te_28); ys = rf.predict_proba(X_te_28)[:, 1]
        lat = (time.perf_counter() - t0) / len(X_te_28) * 1000
        m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
        results.append(dict(model="RF", seed=seed, n_train=n_train,
            acc=m.accuracy, f1=m.f1_score, auc=m.roc_auc, ap=m.average_precision,
            mcc=m.matthews_corrcoef, bal=m.balanced_accuracy, spec=m.specificity, lat=lat))

        gp = BayesianAnomalyEstimator(kernel="rbf", n_restarts_optimizer=3, random_state=seed)
        t0 = time.perf_counter(); gp.fit(X_n, y_n)
        yp = gp.predict(X_te_28); ys = gp.predict_proba(X_te_28)[:, 1]
        lat = (time.perf_counter() - t0) / len(X_te_28) * 1000
        m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
        results.append(dict(model="GP-only", seed=seed, n_train=n_train,
            acc=m.accuracy, f1=m.f1_score, auc=m.roc_auc, ap=m.average_precision,
            mcc=m.matthews_corrcoef, bal=m.balanced_accuracy, spec=m.specificity, lat=lat))

        if n_train in [25, 100, 300, 700]:
            print(f"  n={n_train}: RF={results[-2]['acc']:.3f} GP={results[-1]['acc']:.3f}", flush=True)

    pd.DataFrame(results).to_csv(OUT, index=False)

# Phase 2: QK-only at n=25, 3 seeds
print("\n" + "=" * 60)
print("Phase 2: QK-only at n=25 (3 seeds)")
print("=" * 60)
for seed in range(N_SEEDS_QK):
    gen = QKDDataGenerator(rng=np.random.default_rng(seed))
    df = gen.generate(n_samples=N_SAMPLES, attack_ratio=0.3)
    raw_cols = [c for c in df.columns if c not in ("label", "is_attack")]
    X_raw, y = df[raw_cols].values, df["label"].values
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=seed)
    X_tr_28_full, X_te_28 = build_28dim(X_tr_full, X_te)

    X_n = X_tr_28_full[:25]
    y_n = y_tr_full[:25]

    try:
        qk = QuantumKernelEstimator(n_qubits=3, kernel_type="amplitude", random_state=seed)
        t0 = time.perf_counter(); qk.fit(X_n, y_n)
        yp = qk.predict(X_te_28); ys = qk.predict_proba(X_te_28)[:, 1]
        lat = (time.perf_counter() - t0) / len(X_te_28) * 1000
        m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
        results.append(dict(model="QK-only", seed=seed, n_train=25,
            acc=m.accuracy, f1=m.f1_score, auc=m.roc_auc, ap=m.average_precision,
            mcc=m.matthews_corrcoef, bal=m.balanced_accuracy, spec=m.specificity, lat=lat))
        print(f"  QK seed {seed}: acc={m.accuracy:.3f} f1={m.f1_score:.3f}", flush=True)
    except Exception as e:
        print(f"  QK seed {seed} FAILED: {e}", flush=True)
        results.append(dict(model="QK-only", seed=seed, n_train=25,
            acc=np.nan,f1=np.nan,auc=np.nan,ap=np.nan,mcc=np.nan,bal=np.nan,spec=np.nan,lat=np.nan))

    pd.DataFrame(results).to_csv(OUT, index=False)

df = pd.read_csv(OUT)
print("\n=== Matched-n Summary ===")
for model in ["RF", "GP-only", "QK-only"]:
    sub = df[(df.model == model) & df.acc.notna()].sort_values("n_train")
    print(f"\n{model}:")
    for n in sub.n_train.unique():
        ns = sub[sub.n_train == n]
        print(f"  n={int(n):>4d}: acc={ns.acc.mean():.3f}±{ns.acc.std():.3f}  "
              f"f1={ns.f1.mean():.3f}±{ns.f1.std():.3f}  "
              f"(seeds={len(ns)})")
