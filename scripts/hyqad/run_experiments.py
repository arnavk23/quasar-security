#!/usr/bin/env python3
"""
Experiment driver for the QKD telemetry results in papers/qmi/qmi.tex.

Every table and figure in the paper is produced from the CSVs this script
writes to data/hyqad/; scripts/hyqad/gen_figures.py reads only those CSVs.

Feature pipeline (identical for every model): the 7 physical telemetry
features, standardized on the training split, plus a robust Mahalanobis
distance fit on training-split normal samples -> 8 dims. The amplitude
fidelity kernel therefore encodes all 8 features on exactly 3 qubits.

Usage:  python scripts/hyqad/run_experiments.py [experiment ...]
"""
import os
import sys
import time
import warnings
from pathlib import Path

# One BLAS thread per worker: parallelism comes from joblib across seeds.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM

from hyqad.features import CovarianceFeatureExtractor
from hyqad.models import BayesianAnomalyEstimator, HybridAnomalyDetector, QuantumKernelEstimator
from hyqad.models.quantum_kernel import amplitude_fidelity_kernel
from hyqad.simulation import QKDDataGenerator
from hyqad.simulation.data_generator import ATTACK_TYPES, FEATURE_COLS
from qztna.features.amplitude_kernel import NormAnchor

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "hyqad"
DATA.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
N_SAMPLES = 2000
STRENGTHS_MAIN = (1.0, 0.3)
N_JOBS = int(os.environ.get("HYQAD_JOBS", 6))
GP_HYPER_N = 300  # GP kernel hyperparameters are selected on a 300-sample stratified subset


# ---------------------------------------------------------------- helpers
def make_data(seed, n=N_SAMPLES, strength=1.0, attack_types=None):
    gen = QKDDataGenerator(rng=np.random.default_rng(seed), attack_strength=strength)
    return gen.generate(n_samples=n, attack_ratio=0.3, attack_types=attack_types)


class Features:
    """Standardize the 7 physical features, then append the Mahalanobis distance."""

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.cov = CovarianceFeatureExtractor().fit(self.sc.transform(X), y)
        return self

    def transform(self, X):
        return self.cov.transform(self.sc.transform(X))


def split(df, seed):
    X, y, atk = df[FEATURE_COLS].values, df["label"].values, df["attack_type"].values
    Xtr, Xte, ytr, yte, atr, ate = train_test_split(X, y, atk, test_size=0.3, stratify=y,
                                                    random_state=seed)
    f = Features().fit(Xtr, ytr)
    return f.transform(Xtr), f.transform(Xte), ytr, yte, atr, ate


def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
               for b in range(n_bins) if (idx == b).any())


def supervised_models(seed, anchored=False):
    models = _all_models(seed)
    if not anchored:
        models.pop("AE-SVM (anchored)")
    return models


def _all_models(seed):
    return {
        "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "RF-isotonic": lambda: CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1), method="isotonic", cv=3),
        "GP": lambda: BayesianAnomalyEstimator(kernel="rbf", n_restarts_optimizer=2, random_state=seed,
                                               hyperparam_subset=GP_HYPER_N),
        "AE-SVM": lambda: QuantumKernelEstimator(kernel_type="amplitude", backend="exact", random_state=seed),
        # appends c = median training norm before encoding (9 features -> 4 qubits)
        "AE-SVM (anchored)": lambda: make_pipeline(
            NormAnchor(), QuantumKernelEstimator(kernel_type="amplitude", backend="exact", random_state=seed)),
        "RBF-SVM": lambda: SVC(kernel="rbf", gamma="scale", C=1.0, probability=True,
                               class_weight="balanced", random_state=seed),
        "Hybrid": lambda: HybridAnomalyDetector(use_bayesian=True, use_quantum_kernel=True,
                                                n_restarts_optimizer=2, random_state=seed,
                                                gp_hyperparam_subset=GP_HYPER_N),
    }


def fit_unsupervised(name, Xtr, ytr, seed):
    """Novelty detectors, all fit on normal-class training samples only."""
    Xn = Xtr[ytr == 0]
    if name == "OC-SVM":
        return OneClassSVM(nu=0.1, kernel="rbf", gamma="scale").fit(Xn)
    if name == "IF":
        return IsolationForest(n_estimators=100, contamination=0.1, random_state=seed).fit(Xn)
    return LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True).fit(Xn)


def unsupervised_scores(m, X):
    """decision_function is higher for inliers, so its negation is the anomaly score."""
    return (m.predict(X) == -1).astype(int), -m.decision_function(X)


def metrics(y, yp, score, prob=None):
    row = dict(acc=accuracy_score(y, yp), f1=f1_score(y, yp, zero_division=0),
               auc=roc_auc_score(y, score), ap=average_precision_score(y, score),
               mcc=matthews_corrcoef(y, yp), bal=balanced_accuracy_score(y, yp),
               spec=((yp == 0) & (y == 0)).sum() / max((y == 0).sum(), 1),
               recall=((yp == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    if prob is not None:
        row.update(ece=ece(y, prob), brier=brier_score_loss(y, prob))
    return row


def mcnemar(y, a, b):
    """Exact McNemar test on discordant pairs."""
    n01 = int(((a == y) & (b != y)).sum())
    n10 = int(((a != y) & (b == y)).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def run_parallel(fn, jobs):
    out = Parallel(n_jobs=N_JOBS)(delayed(fn)(*j) for j in jobs)
    return pd.DataFrame([r for rows in out for r in rows])


# ---------------------------------------------------------------- experiments
def _benchmark_job(seed, strength):
    df = make_data(seed, strength=strength)
    Xtr, Xte, ytr, yte, _, _ = split(df, seed)
    rows, preds = [], {}
    for name, make in supervised_models(seed).items():
        m = make().fit(Xtr, ytr)
        prob = m.predict_proba(Xte)[:, 1]
        yp = m.predict(Xte)
        preds[name] = yp
        rows.append(dict(model=name, seed=seed, strength=strength, n_train=len(Xtr),
                         **metrics(yte, yp, prob, prob)))
    for name in ["OC-SVM", "IF", "LOF"]:
        m = fit_unsupervised(name, Xtr, ytr, seed)
        yp, score = unsupervised_scores(m, Xte)
        rows.append(dict(model=name, seed=seed, strength=strength, n_train=len(Xtr),
                         **metrics(yte, yp, score)))
    for a, b in [("GP", "Hybrid"), ("GP", "AE-SVM"), ("AE-SVM", "RBF-SVM"), ("GP", "RF")]:
        n01, n10, p = mcnemar(yte, preds[a], preds[b])
        rows.append(dict(model=f"mcnemar:{a}|{b}", seed=seed, strength=strength,
                         n01=n01, n10=n10, p=p))
    return rows


def exp_benchmark():
    print("[benchmark]")
    jobs = [(s, st) for st in STRENGTHS_MAIN for s in range(N_SEEDS)]
    run_parallel(_benchmark_job, jobs).to_csv(DATA / "benchmark.csv", index=False)


def _matched_n_job(seed, strength):
    df = make_data(seed, strength=strength)
    Xtr, Xte, ytr, yte, _, _ = split(df, seed)
    rows = []
    for n in (25, 50, 100, 200, 500, 1000, 1400):
        idx = np.arange(len(Xtr)) if n >= len(Xtr) else train_test_split(
            np.arange(len(Xtr)), train_size=n, stratify=ytr, random_state=seed)[0]
        for name, make in supervised_models(seed).items():
            if name == "RF-isotonic":
                continue
            m = make().fit(Xtr[idx], ytr[idx])
            prob = m.predict_proba(Xte)[:, 1]
            rows.append(dict(model=name, seed=seed, strength=strength, n_train=len(idx),
                             **metrics(yte, m.predict(Xte), prob, prob)))
    return rows


def exp_matched_n():
    print("[matched_n]")
    jobs = [(s, st) for st in STRENGTHS_MAIN for s in range(N_SEEDS)]
    run_parallel(_matched_n_job, jobs).to_csv(DATA / "matched_n.csv", index=False)


def _strength_job(seed, strength):
    df = make_data(seed, strength=strength)
    Xtr, Xte, ytr, yte, ate_tr, ate = split(df, seed)
    rows = []
    for name, make in supervised_models(seed).items():
        if name in ("RF-isotonic", "Hybrid"):
            continue
        m = make().fit(Xtr, ytr)
        yp = m.predict(Xte)
        prob = m.predict_proba(Xte)[:, 1]
        rows.append(dict(model=name, seed=seed, strength=strength, attack="all",
                         **metrics(yte, yp, prob, prob)))
        for atk in ATTACK_TYPES:
            mask = ate == atk
            rows.append(dict(model=name, seed=seed, strength=strength, attack=atk,
                             recall=float((yp[mask] == 1).mean())))
    return rows


def exp_strength():
    print("[strength]")
    jobs = [(s, st) for st in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0) for s in range(N_SEEDS)]
    run_parallel(_strength_job, jobs).to_csv(DATA / "strength_sweep.csv", index=False)


def _zeroday_job(seed, strength, held_out):
    train_types = [a for a in ATTACK_TYPES if a != held_out]
    df_tr = make_data(seed, strength=strength, attack_types=train_types)
    df_te = make_data(seed + 1000, n=1200, strength=strength, attack_types=[held_out])
    f = Features().fit(df_tr[FEATURE_COLS].values, df_tr["label"].values)
    Xtr, ytr = f.transform(df_tr[FEATURE_COLS].values), df_tr["label"].values
    Xte, yte = f.transform(df_te[FEATURE_COLS].values), df_te["label"].values
    rows = []
    for name, make in supervised_models(seed).items():
        if name in ("RF-isotonic", "Hybrid"):
            continue
        m = make().fit(Xtr, ytr)
        prob = m.predict_proba(Xte)[:, 1]
        rows.append(dict(model=name, seed=seed, strength=strength, held_out=held_out,
                         auc=roc_auc_score(yte, prob), recall=float((m.predict(Xte)[yte == 1] == 1).mean())))
    for name in ["OC-SVM", "IF"]:
        m = fit_unsupervised(name, Xtr, ytr, seed)
        _, score = unsupervised_scores(m, Xte)
        rows.append(dict(model=name, seed=seed, strength=strength, held_out=held_out,
                         auc=roc_auc_score(yte, score)))
    return rows


def exp_zeroday():
    print("[zeroday]")
    jobs = [(s, st, h) for st in STRENGTHS_MAIN for h in ATTACK_TYPES for s in range(N_SEEDS)]
    run_parallel(_zeroday_job, jobs).to_csv(DATA / "zeroday.csv", index=False)


def _alignment_job(seed, strength):
    from qztna.features.amplitude_kernel import kernel_target_alignment
    rows = []
    for atk in ATTACK_TYPES:
        df = make_data(seed, n=800, strength=strength, attack_types=[atk])
        Xtr, _, ytr, _, _, _ = split(df, seed)
        gamma = 1.0 / (Xtr.shape[1] * Xtr.var())
        sq = ((Xtr[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        for name, K in [("AE", amplitude_fidelity_kernel(Xtr, Xtr, 3)), ("RBF", np.exp(-gamma * sq))]:
            rows.append(dict(kernel=name, seed=seed, strength=strength, attack=atk,
                             alignment=kernel_target_alignment(K, ytr)))
    return rows


def exp_alignment():
    print("[alignment]")
    jobs = [(s, st) for st in STRENGTHS_MAIN for s in range(N_SEEDS)]
    run_parallel(_alignment_job, jobs).to_csv(DATA / "kernel_alignment.csv", index=False)


def _selective_job(seed, strength):
    df = make_data(seed, strength=strength)
    Xtr, Xte, ytr, yte, _, _ = split(df, seed)
    rows = []
    for name in ("GP", "RF-isotonic", "AE-SVM"):
        m = supervised_models(seed)[name]().fit(Xtr, ytr)
        prob = m.predict_proba(Xte)[:, 1]
        unc = np.sqrt(prob * (1 - prob))
        correct = ((prob >= 0.5).astype(int) == yte)
        for tau in np.round(np.arange(0.05, 0.501, 0.025), 3):
            keep = unc < tau
            rows.append(dict(model=name, seed=seed, strength=strength, tau=tau,
                             coverage=keep.mean(),
                             accuracy=correct[keep].mean() if keep.any() else np.nan))
        if name == "GP" and seed == 0:
            pd.DataFrame(dict(prob=prob, unc=unc, y=yte)).to_csv(
                DATA / f"gp_scores_s{strength}.csv", index=False)
    return rows


def exp_selective():
    print("[selective]")
    jobs = [(s, st) for st in STRENGTHS_MAIN for s in range(N_SEEDS)]
    run_parallel(_selective_job, jobs).to_csv(DATA / "selective_prediction.csv", index=False)


def _adversarial_job(seed, strength):
    """Gradient-free evasion: move each attack sample's top-k RF-important
    (standardized) features a fraction `budget` of the way to the normal-class median."""
    df = make_data(seed, strength=strength)
    Xtr, Xte, ytr, yte, _, _ = split(df, seed)
    rf = supervised_models(seed)["RF"]().fit(Xtr, ytr)
    gp = supervised_models(seed)["GP"]().fit(Xtr, ytr)
    top = np.argsort(rf.feature_importances_)[::-1][:5]
    target = np.median(Xtr[ytr == 0], axis=0)
    Xa = Xte[yte == 1]
    rows = []
    for budget in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        Xp = Xa.copy()
        Xp[:, top] += budget * (target[top] - Xp[:, top])
        p = gp.predict_proba(Xp)[:, 1]
        rows.append(dict(seed=seed, strength=strength, budget=budget,
                         rf_detect=float(rf.predict(Xp).mean()),
                         gp_detect=float((p >= 0.5).mean()),
                         gp_unc=float(np.sqrt(p * (1 - p)).mean())))
    return rows


def exp_adversarial():
    print("[adversarial]")
    jobs = [(s, st) for st in STRENGTHS_MAIN for s in range(N_SEEDS)]
    run_parallel(_adversarial_job, jobs).to_csv(DATA / "adversarial.csv", index=False)


def _crossover_job(seed):
    """Controls for the amplitude kernel's inductive bias.
    angular: label depends only on direction, norm is a heavy-tailed nuisance
             (AE kernel is norm-invariant -> should help).
    radial:  label depends only on norm (AE kernel discards norm -> chance)."""
    rng = np.random.default_rng(seed)
    d, n_pool = 8, 3000
    w = rng.normal(size=d); w /= np.linalg.norm(w)
    u = rng.normal(size=(n_pool, d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.lognormal(0, 1.0, n_pool)
    tasks = {
        "angular": (u * r[:, None], (np.abs(u @ w) > 0.35).astype(int)),
        "radial": (u * r[:, None], (r > np.median(r)).astype(int)),
    }
    rows = []
    for task, (X, y) in tasks.items():
        Xpool, Xte, ypool, yte = train_test_split(X, y, test_size=1000, stratify=y, random_state=seed)
        for n in (25, 50, 100, 200, 500, 1000):
            idx = train_test_split(np.arange(len(Xpool)), train_size=n, stratify=ypool,
                                   random_state=seed)[0]
            for name in ("AE-SVM", "RBF-SVM", "RF", "GP"):
                m = supervised_models(seed)[name]().fit(Xpool[idx], ypool[idx])
                rows.append(dict(task=task, model=name, seed=seed, n_train=n,
                                 acc=accuracy_score(yte, m.predict(Xte))))
    return rows


def exp_crossover():
    print("[crossover]")
    run_parallel(_crossover_job, [(s,) for s in range(N_SEEDS)]).to_csv(
        DATA / "kernel_crossover.csv", index=False)


def exp_latency():
    """Per-sample inference latency only (training excluded), single thread."""
    print("[latency]")
    df = make_data(0, strength=1.0)
    Xtr, Xte, ytr, yte, _, _ = split(df, 0)
    f = Features().fit(df[FEATURE_COLS].values[:100], df["label"].values[:100])
    raw = df[FEATURE_COLS].values[:len(Xte)]
    rows = []

    def bench(name, fn, X, reps=20):
        fn(X[:5])
        batch = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(X); batch.append((time.perf_counter() - t0) / len(X) * 1e3)
        single = []
        for x in X[:200]:
            t0 = time.perf_counter(); fn(x[None, :]); single.append((time.perf_counter() - t0) * 1e3)
        rows.append(dict(component=name, batch_ms_per_sample=float(np.median(batch)),
                         single_ms_median=float(np.median(single)),
                         single_ms_p95=float(np.percentile(single, 95))))

    bench("Feature extraction", f.transform, raw)
    for name in ("RF", "GP", "AE-SVM", "RBF-SVM", "Hybrid"):
        m = supervised_models(0)[name]().fit(Xtr, ytr)
        if name == "RF":
            m.set_params(n_jobs=1)
        bench(name, m.predict_proba, Xte)
    pd.DataFrame(rows).to_csv(DATA / "latency.csv", index=False)


def exp_circuit_check():
    """Record that the exact kernel equals the PennyLane overlap circuit."""
    print("[circuit_check]")
    df = make_data(0)
    Xtr, Xte, _, _, _, _ = split(df, 0)
    qk = QuantumKernelEstimator(backend="circuit")
    qk.n_qubits_ = qk._resolve_qubits(Xtr.shape[1])
    t0 = time.perf_counter()
    Kc = qk._quantum_kernel(Xte[:20], Xtr[:20])
    t_circ = (time.perf_counter() - t0) / Kc.size
    t0 = time.perf_counter()
    Ke = amplitude_fidelity_kernel(Xte[:20], Xtr[:20], qk.n_qubits_)
    t_exact = (time.perf_counter() - t0) / Ke.size
    pd.DataFrame([dict(n_qubits=qk.n_qubits_, n_features=Xtr.shape[1], max_abs_diff=float(np.abs(Kc - Ke).max()),
                       circuit_ms_per_entry=t_circ * 1e3, exact_ms_per_entry=t_exact * 1e3)]).to_csv(
        DATA / "circuit_check.csv", index=False)


def _anchor_job(seed, strength, held_out):
    """Anchored AE-SVM on the benchmark split (held_out=None) or a zero-day split."""
    make = supervised_models(seed, anchored=True)["AE-SVM (anchored)"]
    if held_out is None:
        df = make_data(seed, strength=strength)
        Xtr, Xte, ytr, yte, _, _ = split(df, seed)
        m = make().fit(Xtr, ytr)
        prob = m.predict_proba(Xte)[:, 1]
        return [dict(kind="benchmark", model="AE-SVM (anchored)", seed=seed, strength=strength,
                     n_train=len(Xtr), **metrics(yte, m.predict(Xte), prob, prob))]
    train_types = [a for a in ATTACK_TYPES if a != held_out]
    df_tr = make_data(seed, strength=strength, attack_types=train_types)
    df_te = make_data(seed + 1000, n=1200, strength=strength, attack_types=[held_out])
    f = Features().fit(df_tr[FEATURE_COLS].values, df_tr["label"].values)
    Xtr, ytr = f.transform(df_tr[FEATURE_COLS].values), df_tr["label"].values
    Xte, yte = f.transform(df_te[FEATURE_COLS].values), df_te["label"].values
    m = make().fit(Xtr, ytr)
    prob = m.predict_proba(Xte)[:, 1]
    return [dict(kind="zeroday", model="AE-SVM (anchored)", seed=seed, strength=strength, held_out=held_out,
                 auc=roc_auc_score(yte, prob), recall=float((m.predict(Xte)[yte == 1] == 1).mean()))]


def exp_anchor():
    """Adds the norm-anchored AE-SVM to benchmark.csv and zeroday.csv (same seeds and splits)."""
    print("[anchor]")
    jobs = [(s, st, h) for st in STRENGTHS_MAIN for h in [None] + ATTACK_TYPES for s in range(N_SEEDS)]
    new = run_parallel(_anchor_job, jobs)
    for kind, name in (("benchmark", "benchmark.csv"), ("zeroday", "zeroday.csv")):
        old = pd.read_csv(DATA / name)
        old = old[old.model != "AE-SVM (anchored)"]
        add = new[new.kind == kind].drop(columns="kind").dropna(axis=1, how="all")
        pd.concat([old, add], ignore_index=True).to_csv(DATA / name, index=False)


def _mahal_ablation_job(seed, strength, held_out):
    """Zero-day AUC with and without the Mahalanobis feature (column 7)."""
    train_types = [a for a in ATTACK_TYPES if a != held_out]
    df_tr = make_data(seed, strength=strength, attack_types=train_types)
    df_te = make_data(seed + 1000, n=1200, strength=strength, attack_types=[held_out])
    f = Features().fit(df_tr[FEATURE_COLS].values, df_tr["label"].values)
    Xtr, ytr = f.transform(df_tr[FEATURE_COLS].values), df_tr["label"].values
    Xte, yte = f.transform(df_te[FEATURE_COLS].values), df_te["label"].values
    rows = []
    models = supervised_models(seed, anchored=True)
    for feats, cols in (("with Mahalanobis", slice(None)), ("without Mahalanobis", slice(0, 7))):
        for name in ("AE-SVM", "AE-SVM (anchored)", "RBF-SVM"):
            m = models[name]().fit(Xtr[:, cols], ytr)
            rows.append(dict(seed=seed, strength=strength, held_out=held_out, features=feats, model=name,
                             auc=roc_auc_score(yte, m.predict_proba(Xte[:, cols])[:, 1])))
    return rows


def exp_mahal_ablation():
    print("[mahal_ablation]")
    jobs = [(s, st, h) for st in STRENGTHS_MAIN for h in ATTACK_TYPES for s in range(N_SEEDS)]
    run_parallel(_mahal_ablation_job, jobs).to_csv(DATA / "zeroday_mahal_ablation.csv", index=False)


EXPERIMENTS = dict(benchmark=exp_benchmark, matched_n=exp_matched_n, strength=exp_strength,
                   zeroday=exp_zeroday, alignment=exp_alignment, selective=exp_selective,
                   adversarial=exp_adversarial, crossover=exp_crossover, latency=exp_latency,
                   circuit_check=exp_circuit_check, anchor=exp_anchor,
                   mahal_ablation=exp_mahal_ablation)

if __name__ == "__main__":
    t0 = time.time()
    for name in sys.argv[1:] or EXPERIMENTS:
        t = time.time()
        EXPERIMENTS[name]()
        print(f"  {name} done in {time.time() - t:.0f}s")
    print(f"All done in {time.time() - t0:.0f}s")
