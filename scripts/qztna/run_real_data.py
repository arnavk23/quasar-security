#!/usr/bin/env python3
"""
Real-data evaluation for Q-ZTNA on UNSW-NB15 and CIC-IDS2017.

Writes data/qztna/real_*.csv. Run scripts/fetch_datasets.sh first.

Protocol (per dataset, per seed):
- training set: stratified-by-family subsample of N_TRAIN flows from the
  training partition; test set: stratified subsample of N_TEST flows from the
  test partition (UNSW-NB15 official split; CIC-IDS2017 fixed 70/30 split).
- inputs: StandardScaler fit on the training subsample ("std"). The
  "AE-SVM (nonneg)" variant instead uses MinMaxScaler clipped to [0, 1]
  ("mm"), so every input lies in the non-negative orthant and the squared
  cosine kernel's x -> -x ambiguity cannot arise.
- kernel SVMs use the Nystrom approximation with m = 1000 landmarks.

Usage:  python scripts/qztna/run_real_data.py [experiment ...]
"""
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier

from qztna.data.real_datasets import DATASETS
from qztna.features.amplitude_kernel import NormAnchor, kernel_target_alignment
from qztna.models.deep import AutoencoderDetector, DNNClassifier
from qztna.models.kernel_svm import KERNEL_FUNCS, NystromKernelSVM

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "qztna"
N_SEEDS = 5
N_SEEDS_ZERODAY = 3
N_TRAIN = 20_000
N_TEST = 50_000
N_LANDMARKS = 1000
N_JOBS = int(os.environ.get("QZTNA_JOBS", 5))

KERNEL_MODELS = {"AE-SVM": ("ae", "std"), "AE-SVM (nonneg)": ("ae", "mm"), "AE-SVM (anchored)": ("ae", "anchor"),
                 "RBF-SVM": ("rbf", "std"), "Laplacian-SVM": ("laplacian", "std"),
                 "Poly-SVM": ("poly", "std")}


def subsample(y_strat, n, seed):
    if n >= len(y_strat):
        return np.arange(len(y_strat))
    # families too small to stratify are merged into one stratum
    vals, counts = np.unique(y_strat, return_counts=True)
    strat = np.where(np.isin(y_strat, vals[counts < 10]), "__rare__", y_strat)
    return train_test_split(np.arange(len(y_strat)), train_size=n, stratify=strat, random_state=seed)[0]


class _Anchored:
    """Standardize, then append c = median training norm (see NormAnchor)."""

    def __init__(self, Xtr):
        self.sc = StandardScaler().fit(Xtr)
        self.na = NormAnchor().fit(self.sc.transform(Xtr))

    def transform(self, X):
        return self.na.transform(self.sc.transform(X))


def scalers(Xtr):
    return {"std": StandardScaler().fit(Xtr), "mm": MinMaxScaler(clip=True).fit(Xtr), "anchor": _Anchored(Xtr)}


def build(name, seed):
    if name in KERNEL_MODELS:
        return NystromKernelSVM(kernel=KERNEL_MODELS[name][0], landmark_frac=1.0, min_landmarks=N_LANDMARKS,
                                max_landmarks=N_LANDMARKS, random_state=seed)
    return {
        "RF": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1),
        "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss",
                                         random_state=seed, n_jobs=1, verbosity=0),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=seed),
        "DNN": lambda: DNNClassifier(random_state=seed),
        "Autoencoder": lambda: AutoencoderDetector(random_state=seed),
    }[name]()


def view(name, sc, X):
    space = KERNEL_MODELS.get(name, (None, "std"))[1]
    return sc[space].transform(X)


def fit_score(name, seed, sc, Xtr, ytr, Xte):
    """Returns (hard predictions, anomaly score) on Xte."""
    m = build(name, seed).fit(view(name, sc, Xtr), ytr)
    Xv = view(name, sc, Xte)
    if name == "Autoencoder":
        return m.predict(Xv), m.score(Xv)
    return m.predict(Xv), m.predict_proba(Xv)[:, 1]


def load(ds):
    """Loaded once in the parent; workers only ever receive subsampled arrays."""
    return DATASETS[ds][0]()


ALL_MODELS = list(KERNEL_MODELS) + ["RF", "XGBoost", "MLP", "DNN", "Autoencoder"]


# ---------------------------------------------------------------- detection benchmark
def _detection_job(ds, seed, Xtr, ytr, Xte, yte):
    sc = scalers(Xtr)
    rows = []
    for name in ALL_MODELS:
        t0 = time.perf_counter()
        yp, s = fit_score(name, seed, sc, Xtr, ytr, Xte)
        rows.append(dict(dataset=ds, model=name, seed=seed, fit_predict_s=time.perf_counter() - t0,
                         accuracy=accuracy_score(yte, yp), f1=f1_score(yte, yp),
                         auc=roc_auc_score(yte, s), ap=average_precision_score(yte, s),
                         precision=precision_score(yte, yp, zero_division=0),
                         recall=recall_score(yte, yp), fpr=float(yp[yte == 0].mean())))
        print(f"  {ds} seed {seed} {name}: acc={rows[-1]['accuracy']:.3f} auc={rows[-1]['auc']:.3f}")
    return rows


def exp_detection():
    print("[real detection]")
    jobs = []
    for ds in DATASETS:
        d = load(ds)
        for s in range(N_SEEDS):
            itr, ite = subsample(d["fam_train"], N_TRAIN, s), subsample(d["fam_test"], N_TEST, s)
            jobs.append((ds, s, d["X_train"][itr], d["y_train"][itr], d["X_test"][ite], d["y_test"][ite]))
        del d
    out = Parallel(n_jobs=N_JOBS)(delayed(_detection_job)(*j) for j in jobs)
    pd.DataFrame([r for rows in out for r in rows]).to_csv(OUT / "real_detection.csv", index=False)


# ---------------------------------------------------------------- matched n
def _matched_n_job(ds, seed, Xpool, ypool, fpool, Xte, yte):
    rows = []
    for n in (200, 1000, 5000, 20000):
        idx = subsample(fpool, n, seed)
        Xtr, ytr = Xpool[idx], ypool[idx]
        sc = scalers(Xtr)
        for name in ("AE-SVM", "AE-SVM (nonneg)", "RBF-SVM", "Laplacian-SVM"):
            yp, s = fit_score(name, seed, sc, Xtr, ytr, Xte)
            rows.append(dict(dataset=ds, model=name, seed=seed, n=n,
                             accuracy=accuracy_score(yte, yp), auc=roc_auc_score(yte, s)))
    return rows


def exp_matched_n():
    print("[real matched_n]")
    jobs = []
    for ds in DATASETS:
        d = load(ds)
        for s in range(N_SEEDS):
            itr, ite = subsample(d["fam_train"], N_TRAIN, s), subsample(d["fam_test"], N_TEST, s)
            jobs.append((ds, s, d["X_train"][itr], d["y_train"][itr], d["fam_train"][itr],
                         d["X_test"][ite], d["y_test"][ite]))
        del d
    out = Parallel(n_jobs=N_JOBS)(delayed(_matched_n_job)(*j) for j in jobs)
    pd.DataFrame([r for rows in out for r in rows]).to_csv(OUT / "real_matched_n.csv", index=False)


# ---------------------------------------------------------------- zero-day (leave one family out)
def _zeroday_indices(d, seed, held_out):
    pool = np.where(d["fam_train"] != held_out)[0]
    itr = pool[subsample(d["fam_train"][pool], N_TRAIN, seed)]
    tpool = np.where(np.isin(d["fam_test"], ["Normal", held_out]))[0]
    ite = tpool[subsample(d["fam_test"][tpool], min(N_TEST, len(tpool)), seed)]
    return itr, ite


def _zeroday_job(ds, seed, held_out, Xtr, ytr, Xte, yte):
    sc = scalers(Xtr)
    rows = []
    for name in ("AE-SVM", "AE-SVM (nonneg)", "RBF-SVM", "RF", "XGBoost", "DNN", "Autoencoder"):
        yp, s = fit_score(name, seed, sc, Xtr, ytr, Xte)
        rows.append(dict(dataset=ds, model=name, seed=seed, held_out=held_out,
                         n_heldout_test=int(yte.sum()), auc=roc_auc_score(yte, s),
                         recall=float(yp[yte == 1].mean()), fpr=float(yp[yte == 0].mean())))
    print(f"  {ds} seed {seed} held-out {held_out} done")
    return rows


def exp_zeroday():
    print("[real zeroday]")
    jobs = []
    for ds, (_, fams) in DATASETS.items():
        d = load(ds)
        for fam in fams:
            for s in range(N_SEEDS_ZERODAY):
                itr, ite = _zeroday_indices(d, s, fam)
                jobs.append((ds, s, fam, d["X_train"][itr], d["y_train"][itr], d["X_test"][ite], d["y_test"][ite]))
        del d
    out = Parallel(n_jobs=N_JOBS)(delayed(_zeroday_job)(*j) for j in jobs)
    pd.DataFrame([r for rows in out for r in rows]).to_csv(OUT / "real_zeroday.csv", index=False)


# ---------------------------------------------------------------- kernel-target alignment
def _alignment_job(ds, seed, fam, X, y):
    sc = scalers(X)
    rows = []
    for name, (k, space) in [("AE", ("ae", "std")), ("AE (nonneg)", ("ae", "mm")),
                             ("RBF", ("rbf", "std")), ("Laplacian", ("laplacian", "std"))]:
        Xs = sc[space].transform(X)
        rows.append(dict(dataset=ds, kernel=name, family=fam, seed=seed,
                         alignment=kernel_target_alignment(KERNEL_FUNCS[k](Xs, Xs), y)))
    return rows


def exp_alignment():
    print("[real alignment]")
    jobs = []
    for ds, (_, fams) in DATASETS.items():
        d = load(ds)
        for s in range(N_SEEDS):
            for fam in fams:
                pool = np.where(np.isin(d["fam_train"], ["Normal", fam]))[0]
                idx = pool[subsample(d["fam_train"][pool], 2000, s)]
                jobs.append((ds, s, fam, d["X_train"][idx], d["y_train"][idx]))
        del d
    out = Parallel(n_jobs=N_JOBS)(delayed(_alignment_job)(*j) for j in jobs)
    pd.DataFrame([r for rows in out for r in rows]).to_csv(OUT / "real_kernel_alignment.csv", index=False)


# ---------------------------------------------------------------- latency
def exp_latency():
    print("[real latency]")
    rows = []
    for ds in DATASETS:
        d = load(ds)
        itr = subsample(d["fam_train"], N_TRAIN, 0)
        ite = subsample(d["fam_test"], 5000, 0)
        Xtr, ytr, Xte = d["X_train"][itr], d["y_train"][itr], d["X_test"][ite]
        sc = scalers(Xtr)
        for name in ALL_MODELS:
            m = build(name, 0).fit(view(name, sc, Xtr), ytr)
            fn = m.score if name == "Autoencoder" else m.predict_proba
            Xv = view(name, sc, Xte)
            fn(Xv[:10])
            batch = []
            for _ in range(10):
                t0 = time.perf_counter(); fn(Xv); batch.append((time.perf_counter() - t0) / len(Xv) * 1e3)
            single = []
            for x in Xv[:300]:
                t0 = time.perf_counter(); fn(x[None, :]); single.append((time.perf_counter() - t0) * 1e3)
            rows.append(dict(dataset=ds, model=name, batch_ms_per_flow=float(np.median(batch)),
                             single_ms_median=float(np.median(single))))
    pd.DataFrame(rows).to_csv(OUT / "real_latency.csv", index=False)


def _anchor_detection_job(ds, seed, Xtr, ytr, Xte, yte):
    sc = scalers(Xtr)
    yp, s = fit_score("AE-SVM (anchored)", seed, sc, Xtr, ytr, Xte)
    return [dict(dataset=ds, model="AE-SVM (anchored)", seed=seed, accuracy=accuracy_score(yte, yp),
                 f1=f1_score(yte, yp), auc=roc_auc_score(yte, s), ap=average_precision_score(yte, s),
                 precision=precision_score(yte, yp, zero_division=0), recall=recall_score(yte, yp),
                 fpr=float(yp[yte == 0].mean()))]


def _anchor_zeroday_job(ds, seed, held_out, Xtr, ytr, Xte, yte):
    sc = scalers(Xtr)
    yp, s = fit_score("AE-SVM (anchored)", seed, sc, Xtr, ytr, Xte)
    return [dict(dataset=ds, model="AE-SVM (anchored)", seed=seed, held_out=held_out,
                 n_heldout_test=int(yte.sum()), auc=roc_auc_score(yte, s),
                 recall=float(yp[yte == 1].mean()), fpr=float(yp[yte == 0].mean()))]


def exp_anchor():
    """Adds the anchored AE-SVM to real_detection.csv / real_zeroday.csv (same splits and seeds)."""
    print("[real anchor]")
    det, zd = [], []
    for ds, (_, fams) in DATASETS.items():
        d = load(ds)
        for s in range(N_SEEDS):
            itr, ite = subsample(d["fam_train"], N_TRAIN, s), subsample(d["fam_test"], N_TEST, s)
            det.append((ds, s, d["X_train"][itr], d["y_train"][itr], d["X_test"][ite], d["y_test"][ite]))
        for fam in fams:
            for s in range(N_SEEDS_ZERODAY):
                itr, ite = _zeroday_indices(d, s, fam)
                zd.append((ds, s, fam, d["X_train"][itr], d["y_train"][itr], d["X_test"][ite], d["y_test"][ite]))
        del d
    for jobs, fn, name in ((det, _anchor_detection_job, "real_detection.csv"),
                           (zd, _anchor_zeroday_job, "real_zeroday.csv")):
        out = Parallel(n_jobs=N_JOBS)(delayed(fn)(*j) for j in jobs)
        new = pd.DataFrame([r for rows in out for r in rows])
        path = OUT / name
        old = pd.read_csv(path)
        old = old[old.model != "AE-SVM (anchored)"]
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


EXPERIMENTS = dict(detection=exp_detection, matched_n=exp_matched_n, zeroday=exp_zeroday,
                   alignment=exp_alignment, latency=exp_latency, anchor=exp_anchor)

if __name__ == "__main__":
    t0 = time.time()
    for name in sys.argv[1:] or EXPERIMENTS:
        t = time.time()
        EXPERIMENTS[name]()
        print(f"  {name} done in {time.time() - t:.0f}s")
    print(f"All done in {time.time() - t0:.0f}s")
