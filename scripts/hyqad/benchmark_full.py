#!/usr/bin/env python3
"""
Full benchmark (optimized): RF/Deployed/QK-only/Hybrid/OC-SVM/IF/LOF at N=1000.
Reduced to 5 seeds for practical runtime. QK is the bottleneck.
"""
import warnings, time, numpy as np, pandas as pd, sys
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from hyqad.simulation import QKDDataGenerator
from hyqad.models import BayesianAnomalyEstimator, QuantumKernelEstimator
from hyqad.evaluation import Evaluator
from hyqad.features import CovarianceFeatureExtractor, AmplitudeEncoder

N_SEEDS = 5
N_SAMPLES = 1000
N_QK = 25
GP_SUBSET = 700
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = str(ROOT / "data" / "hyqad" / "benchmark_full.csv")

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
    return np.concatenate([X_tr_cov, X_tr_amp], axis=1), np.concatenate([X_te_cov, X_te_amp], axis=1)

results = []
for seed in range(N_SEEDS):
    gen = QKDDataGenerator(rng=np.random.default_rng(seed))
    df = gen.generate(n_samples=N_SAMPLES, attack_ratio=0.3)
    raw_cols = [c for c in df.columns if c not in ("label","is_attack")]
    X_raw, y = df[raw_cols].values, df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.3, stratify=y, random_state=seed)
    X_tr_28, X_te_28 = build_28dim(X_tr, X_te)
    print(f"\nSeed {seed}: {len(X_tr)} train, {len(X_te)} test", flush=True)

    # RF
    rf = RandomForestClassifier(n_estimators=200, random_state=seed)
    t0 = time.perf_counter(); rf.fit(X_tr_28, y_tr)
    yp = rf.predict(X_te_28); ys = rf.predict_proba(X_te_28)[:, 1]
    lat = (time.perf_counter() - t0) / len(X_te_28) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append(dict(model="RF",seed=seed,n_train=len(X_tr),
        acc=m.accuracy,f1=m.f1_score,auc=m.roc_auc,ap=m.average_precision,
        mcc=m.matthews_corrcoef,bal=m.balanced_accuracy,spec=m.specificity,lat=lat))
    print(f"  RF: acc={m.accuracy:.3f}", flush=True)

    # Deployed (GP-only)
    gp = BayesianAnomalyEstimator(kernel="rbf", n_restarts_optimizer=3, random_state=seed)
    t0 = time.perf_counter(); gp.fit(X_tr_28[:GP_SUBSET], y_tr[:GP_SUBSET])
    yp = gp.predict(X_te_28); ys = gp.predict_proba(X_te_28)[:, 1]
    lat = (time.perf_counter() - t0) / len(X_te_28) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append(dict(model="Deployed",seed=seed,n_train=GP_SUBSET,
        acc=m.accuracy,f1=m.f1_score,auc=m.roc_auc,ap=m.average_precision,
        mcc=m.matthews_corrcoef,bal=m.balanced_accuracy,spec=m.specificity,lat=lat))
    print(f"  Deployed: acc={m.accuracy:.3f}", flush=True)

    # QK-only
    X_qk = X_tr_28[:N_QK]; y_qk = y_tr[:N_QK]
    qk = QuantumKernelEstimator(n_qubits=3, kernel_type="amplitude", random_state=seed)
    t0 = time.perf_counter(); qk.fit(X_qk, y_qk)
    yp = qk.predict(X_te_28); ys = qk.predict_proba(X_te_28)[:, 1]
    lat = (time.perf_counter() - t0) / len(X_te_28) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append(dict(model="QK-only",seed=seed,n_train=N_QK,
        acc=m.accuracy,f1=m.f1_score,auc=m.roc_auc,ap=m.average_precision,
        mcc=m.matthews_corrcoef,bal=m.balanced_accuracy,spec=m.specificity,lat=lat))
    print(f"  QK-only: acc={m.accuracy:.3f}", flush=True)

    # Hybrid
    gp_full = BayesianAnomalyEstimator(kernel="rbf", n_restarts_optimizer=3, random_state=seed)
    gp_full.fit(X_tr_28[:GP_SUBSET], y_tr[:GP_SUBSET])
    qk_small = QuantumKernelEstimator(n_qubits=3, kernel_type="amplitude", random_state=seed)
    qk_small.fit(X_qk, y_qk)
    Z_tr_gp = gp_full.predict_proba(X_tr_28[:GP_SUBSET])[:, 1].reshape(-1, 1)
    Z_qk_train = qk_small.predict_proba(X_qk)[:, 1].reshape(-1, 1)
    Z_tr = np.hstack([Z_tr_gp[:N_QK], Z_qk_train])
    meta = LogisticRegression(C=1.0, random_state=seed)
    t0 = time.perf_counter()
    meta.fit(Z_tr, y_qk)
    Z_te_gp = gp_full.predict_proba(X_te_28)[:, 1].reshape(-1, 1)
    Z_te_qk = qk_small.predict_proba(X_te_28)[:, 1].reshape(-1, 1)
    Z_te = np.hstack([Z_te_gp, Z_te_qk])
    ys = meta.predict_proba(Z_te)[:, 1]; yp = meta.predict(Z_te)
    lat = (time.perf_counter() - t0) / len(X_te_28) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append(dict(model="Hybrid",seed=seed,n_train=f"{GP_SUBSET}(GP)+{N_QK}(QK)",
        acc=m.accuracy,f1=m.f1_score,auc=m.roc_auc,ap=m.average_precision,
        mcc=m.matthews_corrcoef,bal=m.balanced_accuracy,spec=m.specificity,lat=lat))
    print(f"  Hybrid: acc={m.accuracy:.3f}", flush=True)

    # OC-SVM
    ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    t0 = time.perf_counter(); ocsvm.fit(X_tr_28[y_tr == 0])
    yp = ocsvm.predict(X_te_28); yp = np.where(yp == 1, 0, 1)
    ys = ocsvm.decision_function(X_te_28)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    lat = (time.perf_counter() - t0) / len(X_te_28) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append(dict(model="OC-SVM",seed=seed,n_train=len(X_tr),
        acc=m.accuracy,f1=m.f1_score,auc=m.roc_auc,ap=m.average_precision,
        mcc=m.matthews_corrcoef,bal=m.balanced_accuracy,spec=m.specificity,lat=lat))
    print(f"  OC-SVM: acc={m.accuracy:.3f}", flush=True)

    # IF
    iforest = IsolationForest(random_state=seed, contamination=0.1)
    t0 = time.perf_counter(); iforest.fit(X_tr_28)
    yp = iforest.predict(X_te_28); yp = np.where(yp == 1, 0, 1)
    ys = iforest.decision_function(X_te_28)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    lat = (time.perf_counter() - t0) / len(X_te_28) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append(dict(model="IF",seed=seed,n_train=len(X_tr),
        acc=m.accuracy,f1=m.f1_score,auc=m.roc_auc,ap=m.average_precision,
        mcc=m.matthews_corrcoef,bal=m.balanced_accuracy,spec=m.specificity,lat=lat))
    print(f"  IF: acc={m.accuracy:.3f}", flush=True)

    # LOF
    lof = LocalOutlierFactor(novelty=True, contamination=0.1)
    t0 = time.perf_counter(); lof.fit(X_tr_28)
    yp = lof.predict(X_te_28); yp = np.where(yp == 1, 0, 1)
    ys = lof.decision_function(X_te_28)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    lat = (time.perf_counter() - t0) / len(X_te_28) * 1000
    m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
    results.append(dict(model="LOF",seed=seed,n_train=len(X_tr),
        acc=m.accuracy,f1=m.f1_score,auc=m.roc_auc,ap=m.average_precision,
        mcc=m.matthews_corrcoef,bal=m.balanced_accuracy,spec=m.specificity,lat=lat))
    print(f"  LOF: acc={m.accuracy:.3f}", flush=True)

    pd.DataFrame(results).to_csv(OUT, index=False)

    # McNemar's test: Deployed vs Hybrid
    from hyqad.evaluation.metrics import mcnemar_test
    dep_results = [r for r in results if r["model"] == "Deployed" and r["seed"] == seed]
    hyb_results = [r for r in results if r["model"] == "Hybrid" and r["seed"] == seed]

df = pd.read_csv(OUT)
print("\n=== Full Benchmark @ N=1000 (28-dim, %d seeds) ===" % N_SEEDS)
print(f"{'Model':<12} {'Acc':<12} {'F1':<12} {'AUC':<12} {'MCC':<12}")
print("-"*60)
for m in ["RF","Deployed","QK-only","Hybrid","OC-SVM","IF","LOF"]:
    r = df[df.model == m]
    print(f"{m:<12} {r.acc.mean():.3f}±{r.acc.std():.3f}  {r.f1.mean():.3f}±{r.f1.std():.3f}  "
          f"{r.auc.mean():.3f}±{r.auc.std():.3f}  {r.mcc.mean():.3f}±{r.mcc.std():.3f}")
