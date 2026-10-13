#!/usr/bin/env python3
"""
Extra experiments for the Quantum Machine Intelligence version of the merged paper.

For every leave-one-attack-out split of the four domains (QKD with and without the
Mahalanobis feature, synthetic flows, UNSW-NB15, CIC-IDS2017) we reduce inputs to
q = 8 dimensions (PCA, fit on training data, when d > 8) and compare four exact
fidelity or classical kernels in the same precomputed-kernel SVM:

  AE           amplitude encoding (3 qubits for 8 features)
  AE-anchored  amplitude encoding of (z, c), c = median training norm
  RBF          Gaussian kernel, gamma chosen by 3-fold CV on known attacks
  Laplacian    exp(-gamma ||x - y||_1), gamma chosen the same way
  ZZ           ZZ feature map of Havlicek et al. (8 qubits, 2 layers), bandwidth
               beta chosen by 3-fold CV on known attacks

Each split yields an in-distribution AUC (known attack types, unseen samples) and
a held-out AUC (the left-out attack type). The two quantum kernels are then
re-run with every kernel entry estimated from a finite number of shots.
Writes data/qmi/qmi_kernels.csv.
"""
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "qztna"))
sys.path.insert(0, str(ROOT / "scripts" / "hyqad"))
from qztna.features.amplitude_kernel import amplitude_kernel  # noqa: E402
from qztna.features.zz_kernel import shot_noise, zz_kernel  # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

OUT = ROOT / "data" / "qmi"
Q = 8
N_TRAIN, N_INDIST, N_HELDOUT = 1000, 2000, 3000
N_SEEDS = 3
BETAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
GAMMAS = [g / Q for g in (0.1, 0.3, 1, 3, 10)]
SHOTS = [100, 1000, 10000]
N_JOBS = int(os.environ.get("QMI_JOBS", 3))


def rbf(A, B, g):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-g * np.maximum(d, 0))


def laplacian(A, B, g):
    return np.exp(-g * np.abs(A[:, None, :] - B[None, :, :]).sum(-1))


def svm_auc(Ktr, ytr, Kte, yte):
    m = SVC(C=1.0, kernel="precomputed").fit(Ktr, ytr)
    return roc_auc_score(yte, m.decision_function(Kte))


def cv_pick(kfun, params, Z, y, seed):
    """Parameter with the best 3-fold CV AUC on the training data (known attacks only)."""
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Z, y))
    scores = []
    for p in params:
        K = kfun(Z, Z, p)
        scores.append(np.mean([svm_auc(K[np.ix_(a, a)], y[a], K[np.ix_(b, a)], y[b]) for a, b in folds]))
    return params[int(np.argmax(scores))]


def offdiag(K):
    v = K[~np.eye(len(K), dtype=bool)]
    return float(v.mean()), float(v.std())


def job(domain, seed, held_out, Xpool, ypool, Xho, yho, scale=True):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ypool))
    itr, iin = idx[:N_TRAIN], idx[N_TRAIN:N_TRAIN + N_INDIST]
    iho = rng.permutation(len(yho))[:N_HELDOUT]
    Xtr, ytr, Xin, yin, Xh, yh = Xpool[itr], ypool[itr], Xpool[iin], ypool[iin], Xho[iho], yho[iho]

    # QKD features arrive prepared by the QKD pipeline (scale=False): seven standardized
    # measurements plus a non-negative Mahalanobis distance, exactly as in the main study.
    sc = StandardScaler().fit(Xtr)
    steps = [sc] if scale else []
    if Xtr.shape[1] > Q:
        steps += [PCA(Q, random_state=seed).fit(sc.transform(Xtr))]
        steps += [StandardScaler().fit(steps[1].transform(sc.transform(Xtr)))]

    def red(X):
        for s in steps:
            X = s.transform(X)
        return X

    Ztr, Zin, Zh = red(Xtr), red(Xin), red(Xh)
    c = float(np.median(np.linalg.norm(Ztr, axis=1)))
    anc = lambda Z: np.hstack([Z, np.full((len(Z), 1), c)])
    beta = cv_pick(lambda A, B, b: zz_kernel(b * A, b * B), BETAS, Ztr, ytr, seed)
    gamma = cv_pick(rbf, GAMMAS, Ztr, ytr, seed)
    gamma_l = cv_pick(laplacian, GAMMAS, Ztr, ytr, seed)
    kernels = {
        "AE": lambda A, B: amplitude_kernel(A, B),
        "AE-anchored": lambda A, B: amplitude_kernel(anc(A), anc(B)),
        "RBF": lambda A, B: rbf(A, B, gamma),
        "Laplacian": lambda A, B: laplacian(A, B, gamma_l),
        "ZZ": lambda A, B: zz_kernel(beta * A, beta * B),
    }
    rows = []
    base = dict(domain=domain, seed=seed, held_out=held_out, dim=Xtr.shape[1], beta=beta, gamma=gamma, gamma_laplacian=gamma_l, c=c)
    for name, k in kernels.items():
        Ktr, Kin, Kh = k(Ztr, Ztr), k(Zin, Ztr), k(Zh, Ztr)
        mu, sd = offdiag(Ktr)
        rows.append(dict(base, model=name, shots=0, auc_indist=svm_auc(Ktr, ytr, Kin, yin),
                         auc_heldout=svm_auc(Ktr, ytr, Kh, yh), k_mean=mu, k_std=sd))
        if name in ("AE-anchored", "ZZ"):
            for s in SHOTS:
                Ntr = shot_noise(Ktr, s, rng, symmetric=True)
                rows.append(dict(base, model=name, shots=s,
                                 auc_indist=svm_auc(Ntr, ytr, shot_noise(Kin, s, rng), yin),
                                 auc_heldout=svm_auc(Ntr, ytr, shot_noise(Kh, s, rng), yh), k_mean=mu, k_std=sd))
    # concentration of the ZZ kernel without bandwidth scaling
    mu, sd = offdiag(zz_kernel(Ztr[:300], Ztr[:300]))
    rows.append(dict(base, model="ZZ (beta=1)", shots=0, auc_indist=np.nan, auc_heldout=np.nan, k_mean=mu, k_std=sd))
    print(f"  {domain} seed {seed} held-out {held_out} beta {beta} done")
    return rows


def qkd_jobs():
    import run_experiments as hq
    jobs = []
    for h in hq.ATTACK_TYPES:
        for seed in range(N_SEEDS):
            tr = hq.make_data(seed, strength=0.3, attack_types=[a for a in hq.ATTACK_TYPES if a != h], n=3000)
            te = hq.make_data(seed + 1000, n=1200, strength=0.3, attack_types=[h])
            f = hq.Features().fit(tr[hq.FEATURE_COLS].values, tr.label.values)
            A, B = f.transform(tr[hq.FEATURE_COLS].values), f.transform(te[hq.FEATURE_COLS].values)
            jobs.append(("QKD", seed, h, A, tr.label.values, B, te.label.values, False))
            jobs.append(("QKD, no Mahal.", seed, h, A[:, :7], tr.label.values, B[:, :7], te.label.values, False))
            # control: re-standardizing makes the Mahalanobis distance signed
            jobs.append(("QKD, centred Mahal.", seed, h, A, tr.label.values, B, te.label.values, True))
    return jobs


def synthetic_jobs():
    import qztna_run_experiments as q
    gen = q.NetworkFlowGenerator(seed=42)
    jobs = []
    for h in q.ATTACK_TYPES:
        for seed in range(N_SEEDS):
            tr = gen.generate_dataset(6000, 0.2, random_state=seed, exclude_attack=h)
            te = gen.generate_dataset(3000, 0.25, attack_types=[h], random_state=seed + 1000)
            cols = [c for c in tr.columns if c.startswith("f")]
            jobs.append(("Synthetic flows", seed, h, tr[cols].values, tr.label.values, te[cols].values,
                         te.label.values))
    return jobs


def real_jobs():
    import run_real_data as rr
    jobs = []
    for ds, (loader, fams) in rr.DATASETS.items():
        d = loader()
        for h in fams:
            for seed in range(N_SEEDS):
                itr, ite = rr._zeroday_indices(d, seed, h)
                rng = np.random.default_rng(seed)
                itr = rng.permutation(itr)[:N_TRAIN + N_INDIST]
                ite = rng.permutation(ite)[:N_HELDOUT]
                jobs.append((ds, seed, h, d["X_train"][itr], d["y_train"][itr], d["X_test"][ite], d["y_test"][ite]))
        del d
    return jobs


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["qkd", "synthetic", "real"]
    jobs = []
    for w in which:
        jobs += {"qkd": qkd_jobs, "synthetic": synthetic_jobs, "real": real_jobs}[w]()
    print(f"{len(jobs)} jobs")
    out = Parallel(n_jobs=N_JOBS)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    old = OUT / "qmi_kernels.csv"
    if sys.argv[1:] and old.exists():
        prev = pd.read_csv(old)
        df = pd.concat([prev[~prev.domain.isin(df.domain.unique())], df], ignore_index=True)
    df.to_csv(OUT / "qmi_kernels.csv", index=False)
    s = df[df.shots == 0].groupby(["domain", "model"])[["auc_indist", "auc_heldout", "k_std"]].mean()
    print(s.round(3).to_string())
