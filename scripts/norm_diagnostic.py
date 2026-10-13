#!/usr/bin/env python3
"""
Norm-vs-direction diagnostic for every leave-one-attack-out split.

For each held-out attack type we compute, on the same standardized features the
detectors see:
  norm_auc  - ROC-AUC of ||x|| (distance from the training mean) for
              held-out attack vs. normal: how much of the separation is magnitude;
  dir_auc   - ROC-AUC of a direction-only score: cosine distance of x/||x|| to the
              normalized mean of normal training traffic.
and join the AE-SVM minus RBF-SVM zero-day AUC gap from the experiment CSVs.
Writes data/norm_diagnostic.csv.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "qztna"))
sys.path.insert(0, str(ROOT / "scripts" / "hyqad"))


def scores(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    norm_auc = roc_auc_score(yte, np.linalg.norm(B, axis=1))
    mu = A[ytr == 0] / np.maximum(np.linalg.norm(A[ytr == 0], axis=1, keepdims=True), 1e-12)
    mu = mu.mean(0); mu /= np.linalg.norm(mu)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    dir_auc = roc_auc_score(yte, 1 - (Bn @ mu) ** 2)
    return norm_auc, dir_auc


def gap(df, held_col="held_out", **filt):
    for k, v in filt.items():
        df = df[df[k] == v]
    g = df.groupby(["model", held_col]).auc.mean().unstack(0)
    return (g["AE-SVM"] - g["RBF-SVM"]).to_dict()


rows = []

# synthetic flows
import qztna_run_experiments as q
gen = q.NetworkFlowGenerator(seed=42)
g_syn = gap(pd.read_csv(ROOT / "data/qztna/qztna_zeroday.csv"))
for h in q.ATTACK_TYPES:
    vals = []
    for seed in range(3):
        df_tr = gen.generate_dataset(6000, 0.2, random_state=seed, exclude_attack=h)
        df_te = gen.generate_dataset(3000, 0.25, attack_types=[h], random_state=seed + 1000)
        cols = [c for c in df_tr.columns if c.startswith("f")]
        vals.append(scores(df_tr[cols].values, df_tr.label.values, df_te[cols].values, df_te.label.values))
    n, d = np.mean(vals, 0)
    rows.append(dict(domain="Synthetic flows", held_out=h, norm_auc=n, dir_auc=d, ae_minus_rbf=g_syn[h]))

# QKD telemetry, with and without the Mahalanobis feature, s = 0.3
import run_experiments as hq
g_with = gap(pd.read_csv(ROOT / "data/hyqad/zeroday.csv"), strength=0.3)
ab = pd.read_csv(ROOT / "data/hyqad/zeroday_mahal_ablation.csv")
g_wo = gap(ab, strength=0.3, features="without Mahalanobis")
for h in hq.ATTACK_TYPES:
    v_with, v_wo = [], []
    for seed in range(3):
        tr = hq.make_data(seed, strength=0.3, attack_types=[a for a in hq.ATTACK_TYPES if a != h])
        te = hq.make_data(seed + 1000, n=1200, strength=0.3, attack_types=[h])
        f = hq.Features().fit(tr[hq.FEATURE_COLS].values, tr.label.values)
        A, B = f.transform(tr[hq.FEATURE_COLS].values), f.transform(te[hq.FEATURE_COLS].values)
        v_with.append(scores(A, tr.label.values, B, te.label.values))
        v_wo.append(scores(A[:, :7], tr.label.values, B[:, :7], te.label.values))
    for label, v, g in (("QKD", v_with, g_with), ("QKD, no Mahal.", v_wo, g_wo)):
        n, d = np.mean(v, 0)
        rows.append(dict(domain=label, held_out=h, norm_auc=n, dir_auc=d, ae_minus_rbf=g[h]))

# public datasets (same subsampling as run_real_data.py)
import run_real_data as rr
rz = pd.read_csv(ROOT / "data/qztna/real_zeroday.csv")
for ds, (loader, fams) in rr.DATASETS.items():
    d_all = loader()
    g_real = gap(rz, dataset=ds)
    for h in fams:
        vals = []
        for seed in range(rr.N_SEEDS_ZERODAY):
            itr, ite = rr._zeroday_indices(d_all, seed, h)
            vals.append(scores(d_all["X_train"][itr], d_all["y_train"][itr],
                               d_all["X_test"][ite], d_all["y_test"][ite]))
        n, d = np.mean(vals, 0)
        rows.append(dict(domain=ds, held_out=h, norm_auc=n, dir_auc=d, ae_minus_rbf=g_real[h]))
    del d_all

out = pd.DataFrame(rows)
out.to_csv(ROOT / "data" / "norm_diagnostic.csv", index=False)
print(out.round(3).to_string())
from scipy.stats import spearmanr, pearsonr
x = out.norm_auc - out.dir_auc
print("Spearman(norm_auc - dir_auc, AE-RBF gap):", spearmanr(x, out.ae_minus_rbf))
print("Pearson:", pearsonr(x, out.ae_minus_rbf))
print(out.groupby("domain")[["norm_auc", "dir_auc", "ae_minus_rbf"]].mean().round(3))
