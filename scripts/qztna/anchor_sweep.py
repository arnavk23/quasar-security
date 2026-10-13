#!/usr/bin/env python3
"""
Sweep of the norm-anchor constant c on the synthetic leave-one-attack-out
protocol of qztna_run_experiments.run_zeroday (same splits and seeds).
c = 0 is plain amplitude encoding; "median" is the median-training-norm rule.
Writes data/qztna/qztna_anchor_sweep.csv.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import qztna_run_experiments as q  # noqa: E402
from qztna.models.kernel_svm import fit_kernel_svm  # noqa: E402

C_VALUES = [0, 1, 2, 4, 8, 16, 32, "median"]


def job(seed, held_out):
    gen = q.NetworkFlowGenerator(seed=42)
    Xtr, _, ytr, _, _, _, sc = q.make_split(gen, n_samples=6000, attack_ratio=0.2, seed=seed,
                                             exclude_attack=held_out)
    df = gen.generate_dataset(n_samples=3000, attack_ratio=0.25, attack_types=[held_out],
                              random_state=seed + 1000)
    cols = [c for c in df.columns if c.startswith("f")]
    Xte, yte = sc.transform(df[cols].values), df["label"].values
    median = float(np.median(np.linalg.norm(Xtr, axis=1)))
    rows = []
    for c in C_VALUES:
        cv = median if c == "median" else float(c)
        add = (lambda X: X) if cv == 0 else (lambda X: np.hstack([X, np.full((len(X), 1), cv)]))
        model, _ = fit_kernel_svm("ae", add(Xtr), ytr, C=1.0, random_state=seed, nystrom_threshold=1500)
        rows.append(dict(seed=seed, held_out=held_out, c=str(c), c_value=cv,
                         auc=roc_auc_score(yte, model.predict_proba(add(Xte))[:, 1])))
    return rows


if __name__ == "__main__":
    jobs = [(s, h) for s in range(q.N_SEEDS_SMALL) for h in q.ATTACK_TYPES]
    out = Parallel(n_jobs=int(os.environ.get("QZTNA_JOBS", 4)))(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(ROOT / "data" / "qztna" / "qztna_anchor_sweep.csv", index=False)
    print(df.groupby("c").auc.mean())
