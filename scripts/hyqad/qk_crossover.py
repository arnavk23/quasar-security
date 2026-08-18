#!/usr/bin/env python3
"""
QK crossover experiment (optimized): 3 seeds, n=150 to keep QK feasible.
"""
import warnings, time, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.stdout.reconfigure(line_buffering=True)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from hyqad.models import QuantumKernelEstimator
from hyqad.evaluation import Evaluator

N_SEEDS = 3
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = str(ROOT / "data" / "hyqad" / "qk_crossover_results.csv")

def make_classically_hard_data(n_samples=150, n_features=8, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n_per_class = n_samples // 2
    dim = 2 ** 3
    X_list, y_list = [], []
    for cls, n in enumerate([n_per_class, n_samples - n_per_class]):
        z = rng.normal(0, 1, (n, dim))
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        theta = cls * np.pi / 3
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.eye(dim)
        rotation[0, 0] = c; rotation[0, 1] = -s
        rotation[1, 0] = s; rotation[1, 1] = c
        phi = cls * np.pi / 4
        c2, s2 = np.cos(phi), np.sin(phi)
        rotation[2, 2] = c2; rotation[2, 3] = -s2
        rotation[3, 2] = s2; rotation[3, 3] = c2
        x = z @ rotation
        x[:, 4:] += cls * 0.05
        x += rng.normal(0, 0.01, x.shape)
        X_list.append(x)
        y_list.extend([cls] * n)
    X = np.vstack(X_list)
    y = np.array(y_list)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]

def make_concentric_rings(n_samples=150, n_features=8, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = n_samples
    r1, r2, r3 = 0.3, 0.7, 1.0
    X = rng.normal(0, 1, (n, n_features))
    n0 = n // 2
    angles0 = rng.uniform(0, 2*np.pi, n0)
    radii0 = rng.uniform(0, r1, n0)
    X[:n0, 0] = radii0 * np.cos(angles0)
    X[:n0, 1] = radii0 * np.sin(angles0)
    n1 = n - n0
    angles1 = rng.uniform(0, 2*np.pi, n1)
    radii1 = rng.uniform(r2, r3, n1)
    X[n0:, 0] = radii1 * np.cos(angles1)
    X[n0:, 1] = radii1 * np.sin(angles1)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = np.clip(X, -1, 1)
    y = np.array([0]*n0 + [1]*n1)
    perm = rng.permutation(n)
    return X[perm], y[perm]

results = []
for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    for ds_name, ds_fn in [("concentric_quantum", make_classically_hard_data),
                            ("concentric_rings", make_concentric_rings)]:
        X, y = ds_fn(n_samples=150, rng=rng)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

        for model_name, model in [
            ("RBF-SVM", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)),
            ("GP-RBF", GaussianProcessClassifier(kernel=1.0*RBF(1.0), n_restarts_optimizer=3, random_state=seed)),
            ("GP-Matern", GaussianProcessClassifier(kernel=1.0*Matern(nu=1.5), n_restarts_optimizer=3, random_state=seed)),
            ("RF", RandomForestClassifier(n_estimators=200, random_state=seed)),
        ]:
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            yp = model.predict(X_te)
            ys = model.predict_proba(X_te)[:, 1]
            lat = (time.perf_counter() - t0) / len(X_te) * 1000
            m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
            results.append(dict(model=model_name, dataset=ds_name, seed=seed, n_train=len(X_tr),
                acc=m.accuracy, f1=m.f1_score, auc=m.roc_auc, mcc=m.matthews_corrcoef, lat=lat))

        # QK-SVM
        try:
            qk = QuantumKernelEstimator(n_qubits=3, kernel_type="amplitude", random_state=seed)
            t0 = time.perf_counter()
            qk.fit(X_tr, y_tr)
            yp = qk.predict(X_te)
            ys = qk.predict_proba(X_te)[:, 1]
            lat = (time.perf_counter() - t0) / len(X_te) * 1000
            m = Evaluator().evaluate(y_te, yp, y_score=ys, latency_ms=lat)
            results.append(dict(model="QK-SVM", dataset=ds_name, seed=seed, n_train=len(X_tr),
                acc=m.accuracy, f1=m.f1_score, auc=m.roc_auc, mcc=m.matthews_corrcoef, lat=lat))
        except Exception as e:
            print(f"  QK failed ({ds_name}, seed {seed}): {e}")
            results.append(dict(model="QK-SVM", dataset=ds_name, seed=seed, n_train=len(X_tr),
                acc=np.nan, f1=np.nan, auc=np.nan, mcc=np.nan, lat=np.nan))

        print(f"  {ds_name} seed={seed} done", flush=True)
    pd.DataFrame(results).to_csv(OUT, index=False)

df = pd.read_csv(OUT)
print("\n=== QK Crossover Summary ===")
for ds in df.dataset.unique():
    sub = df[df.dataset == ds]
    print(f"\n{ds}:")
    for model in sub.model.unique():
        ms = sub[sub.model == model]
        if ms.acc.notna().any():
            print(f"  {model:12s}: acc={ms.acc.mean():.3f}±{ms.acc.std():.3f} f1={ms.f1.mean():.3f}±{ms.f1.std():.3f}")
