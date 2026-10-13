"""
Experiment driver for the synthetic-flow results in papers/qmi/qmi.tex.

Replaces the previous hardcoded-array figure script with genuine trained
models on genuinely simulated network flow data. Every CSV written here is
read directly by scripts/qztna/qztna_gen_figures.py to produce the paper
figures, and the same numbers are what papers/qmi/qmi.tex reports.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier

from qztna.simulation import NetworkFlowGenerator
from qztna.models.kernel_svm import fit_kernel_svm, NystromKernelSVM, KERNEL_FUNCS
from qztna.features.amplitude_kernel import NormAnchor, kernel_target_alignment
from qztna.certification.randomized_smoothing import certify_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "qztna"
DATA.mkdir(exist_ok=True, parents=True)
ATTACK_TYPES = ["port_scan", "ddos", "exfiltration", "dns_tunneling", "arp_spoofing"]
N_SEEDS_MAIN = 10
N_SEEDS_SMALL = 5


def make_split(gen, n_samples, attack_ratio, seed, attack_strength=1.0, noise_scale=0.5,
               exclude_attack=None):
    df = gen.generate_dataset(n_samples=n_samples, attack_ratio=attack_ratio,
                               attack_strength=attack_strength, noise_scale=noise_scale,
                               random_state=seed, exclude_attack=exclude_attack)
    cols = [c for c in df.columns if c.startswith("f")]
    X, y, atk = df[cols].values, df["label"].values, df["attack_type"].values
    Xtr, Xte, ytr, yte, atr, ate = train_test_split(
        X, y, atk, test_size=0.3, random_state=seed, stratify=y
    )
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte, atr, ate, sc


def fit_baselines(Xtr, ytr, seed):
    models = {
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, eval_metric="logloss", random_state=seed,
                                  scale_pos_weight=4.0, verbosity=0),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=seed),
    }
    fitted = {}
    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        fitted[name] = (m, time.perf_counter() - t0)
    return fitted


# =============================================================================
# 1. Detection performance benchmark (Table: detection)
# =============================================================================
def run_detection_benchmark():
    print("[1/9] Detection benchmark...")
    rows = []
    gen = NetworkFlowGenerator(seed=42)
    for seed in range(N_SEEDS_MAIN):
        Xtr, Xte, ytr, yte, atr, ate, _ = make_split(gen, n_samples=8000, attack_ratio=0.2, seed=seed)

        na = NormAnchor().fit(Xtr)
        for kname, label in [("ae", "AE-SVM"), ("rbf", "RBF-SVM"), ("laplacian", "Laplacian-SVM"),
                             ("poly", "Poly-SVM"), ("ae", "AE-SVM (anchored)")]:
            Xa, Xb = (na.transform(Xtr), na.transform(Xte)) if label.endswith("(anchored)") else (Xtr, Xte)
            model, fit_t = fit_kernel_svm(kname, Xa, ytr, C=1.0, random_state=seed, nystrom_threshold=1500)
            ypred = model.predict(Xb)
            yscore = model.predict_proba(Xb)[:, 1]
            rows.append(dict(model=label, seed=seed, accuracy=accuracy_score(yte, ypred),
                              f1=f1_score(yte, ypred), auc=roc_auc_score(yte, yscore),
                              precision=precision_score(yte, ypred, zero_division=0),
                              recall=recall_score(yte, ypred, zero_division=0)))

        for name, (m, fit_t) in fit_baselines(Xtr, ytr, seed).items():
            ypred = m.predict(Xte)
            yscore = m.predict_proba(Xte)[:, 1]
            rows.append(dict(model=name, seed=seed, accuracy=accuracy_score(yte, ypred),
                              f1=f1_score(yte, ypred), auc=roc_auc_score(yte, yscore),
                              precision=precision_score(yte, ypred, zero_division=0),
                              recall=recall_score(yte, ypred, zero_division=0)))
        print(f"  seed {seed} done")

    pd.DataFrame(rows).to_csv(DATA / "qztna_detection_benchmark.csv", index=False)


# =============================================================================
# 2. Matched-n ablation (Fig: matched_n)
# =============================================================================
def run_matched_n():
    print("[2/9] Matched-n ablation...")
    rows = []
    gen = NetworkFlowGenerator(seed=42)
    n_grid = [50, 100, 200, 500, 1000, 2000, 5000]
    for seed in range(N_SEEDS_SMALL):
        df = gen.generate_dataset(n_samples=12000, attack_ratio=0.2, attack_strength=1.0, random_state=seed)
        cols = [c for c in df.columns if c.startswith("f")]
        X, y = df[cols].values, df["label"].values
        X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=2000, random_state=seed, stratify=y)
        sc_full = StandardScaler().fit(X_pool)
        X_pool_s, X_test_s = sc_full.transform(X_pool), sc_full.transform(X_test)

        for n in n_grid:
            if n > len(X_pool_s):
                continue
            idx, _ = train_test_split(np.arange(len(X_pool_s)), train_size=n, random_state=seed,
                                       stratify=y_pool)
            Xn, yn = X_pool_s[idx], y_pool[idx]
            if len(np.unique(yn)) < 2:
                continue
            for kname in ["ae", "rbf", "laplacian"]:
                model, _ = fit_kernel_svm(kname, Xn, yn, C=1.0, random_state=seed, nystrom_threshold=1500)
                ypred = model.predict(X_test_s)
                rows.append(dict(kernel=kname, n=n, seed=seed, accuracy=accuracy_score(y_test, ypred)))
        print(f"  seed {seed} done")

    pd.DataFrame(rows).to_csv(DATA / "qztna_matched_n.csv", index=False)


# =============================================================================
# 3. Certified robustness (Table: robustness)
# =============================================================================
def run_certified_robustness():
    print("[3/9] Certified robustness...")
    rows = []
    gen = NetworkFlowGenerator(seed=42)
    sigma = 0.25
    for seed in range(3):
        Xtr, Xte, ytr, yte, atr, ate, _ = make_split(gen, n_samples=4000, attack_ratio=0.3, seed=seed)
        for kname, label in [("ae", "AE-SVM"), ("rbf", "RBF-SVM")]:
            model, _ = fit_kernel_svm(kname, Xtr, ytr, C=1.0, random_state=seed, nystrom_threshold=1500)
            for atk in ATTACK_TYPES:
                mask = (ate == atk) | (ate == "normal")
                mask_atk_only = ate == atk
                if mask_atk_only.sum() < 5:
                    continue
                idx = np.where(mask_atk_only)[0]
                recs = certify_dataset(model, Xte[idx], yte[idx], sigma=sigma, n0=50, n=300, alpha=0.05,
                                        max_points=40, random_state=seed)
                for r in recs:
                    rows.append(dict(model=label, attack=atk, seed=seed, radius=r["radius"],
                                      abstained=r["abstained"]))
        print(f"  seed {seed} done")

    pd.DataFrame(rows).to_csv(DATA / "qztna_certified_radii.csv", index=False)


# =============================================================================
# 4. Sensitivity to traffic noise (Fig: sensitivity)
# =============================================================================
def run_sensitivity():
    print("[4/9] Sensitivity to traffic noise...")
    rows = []
    gen = NetworkFlowGenerator(seed=42)
    noise_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for seed in range(N_SEEDS_SMALL):
        for noise in noise_grid:
            Xtr, Xte, ytr, yte, atr, ate, _ = make_split(gen, n_samples=4000, attack_ratio=0.2,
                                                            seed=seed, noise_scale=noise)
            for kname, label in [("ae", "AE-SVM"), ("rbf", "RBF-SVM")]:
                model, _ = fit_kernel_svm(kname, Xtr, ytr, C=1.0, random_state=seed, nystrom_threshold=1500)
                ypred = model.predict(Xte)
                rows.append(dict(model=label, noise=noise, seed=seed, accuracy=accuracy_score(yte, ypred)))
        print(f"  seed {seed} done")

    pd.DataFrame(rows).to_csv(DATA / "qztna_sensitivity.csv", index=False)


# =============================================================================
# 5. Zero-day generalization (Fig: zeroday)
# =============================================================================
def run_zeroday():
    print("[5/9] Zero-day generalization...")
    rows = []
    gen = NetworkFlowGenerator(seed=42)
    for seed in range(N_SEEDS_SMALL):
        for held_out in ATTACK_TYPES:
            # train on 4 attack types + normal
            Xtr, _, ytr, _, atr, _, sc = make_split(gen, n_samples=6000, attack_ratio=0.2, seed=seed,
                                                       exclude_attack=held_out)
            # test set: normal + held-out attack only
            df_test = gen.generate_dataset(n_samples=3000, attack_ratio=0.25,
                                            attack_types=[held_out], random_state=seed + 1000)
            cols = [c for c in df_test.columns if c.startswith("f")]
            Xte, yte = sc.transform(df_test[cols].values), df_test["label"].values
            if len(np.unique(yte)) < 2:
                continue
            # "nonneg": min-max scaled to [0, 1] so the squared-cosine kernel's
            # x -> -x ambiguity cannot arise (mechanism control for the AE-SVM gap)
            # "anchor": phi(x) = (x, c)/||(x, c)|| with c = median training norm keeps the
            # norm and sign information plain amplitude encoding discards
            mm = MinMaxScaler(clip=True).fit(Xtr)
            na = NormAnchor().fit(Xtr)
            views = {"std": (Xtr, Xte), "mm": (mm.transform(Xtr), mm.transform(Xte)),
                     "anchor": (na.transform(Xtr), na.transform(Xte))}
            for kname, label, space in [("ae", "AE-SVM", "std"), ("rbf", "RBF-SVM", "std"),
                                        ("ae", "AE-SVM (nonneg)", "mm"), ("ae", "AE-SVM (anchored)", "anchor")]:
                Xa, Xb = views[space]
                model, _ = fit_kernel_svm(kname, Xa, ytr, C=1.0, random_state=seed, nystrom_threshold=1500)
                yscore = model.predict_proba(Xb)[:, 1]
                rows.append(dict(model=label, held_out=held_out, seed=seed,
                                  auc=roc_auc_score(yte, yscore)))
        print(f"  seed {seed} done")

    pd.DataFrame(rows).to_csv(DATA / "qztna_zeroday.csv", index=False)


# =============================================================================
# 6. Kernel-target alignment (Fig: alignment)
# =============================================================================
def run_kernel_alignment():
    print("[6/9] Kernel-target alignment...")
    rows = []
    gen = NetworkFlowGenerator(seed=42)
    for seed in range(N_SEEDS_SMALL):
        for atk in ATTACK_TYPES:
            df = gen.generate_dataset(n_samples=800, attack_ratio=0.3, attack_types=[atk], random_state=seed)
            cols = [c for c in df.columns if c.startswith("f")]
            X = StandardScaler().fit_transform(df[cols].values)
            y = df["label"].values
            for kname, label in [("ae", "AE-SVM"), ("rbf", "RBF-SVM"), ("laplacian", "Laplacian-SVM")]:
                K = KERNEL_FUNCS[kname](X, X)
                align = kernel_target_alignment(K, y)
                rows.append(dict(kernel=label, attack=atk, seed=seed, alignment=align))
        print(f"  seed {seed} done")

    pd.DataFrame(rows).to_csv(DATA / "qztna_kernel_alignment.csv", index=False)


# =============================================================================
# 7. Dimensionality sweep (Fig: dimensionality)
# =============================================================================
def run_dimensionality():
    print("[7/9] Dimensionality sweep...")
    rows = []
    d_grid = [16, 32, 64, 128, 256]
    for seed in range(N_SEEDS_SMALL):
        for d in d_grid:
            n_header = max(int(round(d * 16 / 64)), 2)
            n_flow = max(int(round(d * 24 / 64)), 2)
            n_behavioral = d - n_header - n_flow
            gen = NetworkFlowGenerator(n_features=d, n_header=n_header, n_flow=n_flow,
                                        n_behavioral=n_behavioral, seed=42 + d)
            Xtr, Xte, ytr, yte, atr, ate, _ = make_split(gen, n_samples=4000, attack_ratio=0.2, seed=seed)
            for kname, label in [("ae", "AE-SVM"), ("rbf", "RBF-SVM")]:
                model, _ = fit_kernel_svm(kname, Xtr, ytr, C=1.0, random_state=seed, nystrom_threshold=1500)
                ypred = model.predict(Xte)
                rows.append(dict(model=label, d=d, seed=seed, accuracy=accuracy_score(yte, ypred)))
        print(f"  seed {seed} done")

    pd.DataFrame(rows).to_csv(DATA / "qztna_dimensionality.csv", index=False)


# =============================================================================
# 8. Latency benchmark (Table: latency)
# =============================================================================
def run_latency():
    print("[8/9] Latency benchmark...")
    gen = NetworkFlowGenerator(seed=42)
    Xtr, Xte, ytr, yte, atr, ate, sc = make_split(gen, n_samples=4000, attack_ratio=0.2, seed=0)

    rows = []

    def bench(name, model, X, n_runs=30):
        for _ in range(3):
            model.predict(X[:10])
        lats = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.predict(X)
            lats.append((time.perf_counter() - t0) * 1000 / len(X))
        rows.append(dict(component=name, latency_ms=float(np.mean(lats)), std_ms=float(np.std(lats))))

    ae_nystrom, _ = fit_kernel_svm("ae", Xtr, ytr, C=1.0, random_state=0, force_nystrom=True, landmark_frac=0.1, max_landmarks=50)
    ae_full, _ = fit_kernel_svm("ae", Xtr[:500], ytr[:500], C=1.0, random_state=0, force_exact=True)
    rbf_svm, _ = fit_kernel_svm("rbf", Xtr, ytr, C=1.0, random_state=0, force_nystrom=True, landmark_frac=0.1, max_landmarks=50)
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=0).fit(Xtr, ytr)

    bench("AE-SVM (Nystrom)", ae_nystrom, Xte)
    bench("AE-SVM (full)", ae_full, Xte[:500])
    bench("RBF-SVM (Nystrom)", rbf_svm, Xte)
    bench("Random Forest", rf, Xte)
    bench("MLP", mlp, Xte)
    rows.append(dict(component="Feature extraction (StandardScaler)",
                      latency_ms=float(np.mean([1000 * (time.perf_counter() - t0) / len(Xte)
                                                 for t0 in [time.perf_counter()] for _ in [sc.transform(Xte)]])),
                      std_ms=0.0))

    pd.DataFrame(rows).to_csv(DATA / "qztna_latency.csv", index=False)


# =============================================================================
# 9. Multiclass confusion / ROC / uncertainty (Figs: confusion, roc, uncertainty)
# =============================================================================
def run_multiclass_and_uncertainty():
    print("[9/9] Multiclass confusion, ROC, uncertainty...")
    gen = NetworkFlowGenerator(seed=42)
    df = gen.generate_dataset(n_samples=10000, attack_ratio=0.35, attack_strength=1.0, random_state=0)
    cols = [c for c in df.columns if c.startswith("f")]
    class_names = ["normal"] + ATTACK_TYPES
    class_map = {c: i for i, c in enumerate(class_names)}
    X = df[cols].values
    y_multi = df["attack_type"].map(class_map).values
    y_bin = df["label"].values

    Xtr, Xte, ytr_m, yte_m, ytr_b, yte_b = train_test_split(
        X, y_multi, y_bin, test_size=0.3, random_state=0, stratify=y_multi
    )
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    ovr = OneVsRestClassifier(NystromKernelSVM(kernel="ae", random_state=0, landmark_frac=0.1, max_landmarks=200))
    ovr.fit(Xtr_s, ytr_m)
    ypred_m = ovr.predict(Xte_s)

    cm = confusion_matrix(yte_m, ypred_m, labels=list(range(len(class_names))), normalize="true")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(DATA / "qztna_confusion.csv")

    roc_rows = []
    proba = ovr.predict_proba(Xte_s)
    for i, cname in enumerate(class_names):
        y_i = (yte_m == i).astype(int)
        if y_i.sum() == 0 or y_i.sum() == len(y_i):
            continue
        fpr, tpr, _ = roc_curve(y_i, proba[:, i])
        auc = roc_auc_score(y_i, proba[:, i])
        for f, t in zip(fpr, tpr):
            roc_rows.append(dict(cls=cname, fpr=f, tpr=t, auc=auc))
    pd.DataFrame(roc_rows).to_csv(DATA / "qztna_roc.csv", index=False)

    # Uncertainty / selective prediction on the binary AE-SVM
    ae_bin, _ = fit_kernel_svm("ae", Xtr_s, ytr_b, C=1.0, random_state=0, nystrom_threshold=1500)
    proba_b = ae_bin.predict_proba(Xte_s)[:, 1]
    unc = np.sqrt(proba_b * (1 - proba_b))
    pred_b = (proba_b >= 0.5).astype(int)
    correct = (pred_b == yte_b).astype(int)

    thresholds = np.linspace(0.05, 0.5, 20)
    unc_rows = []
    for tau in thresholds:
        keep = unc < tau
        coverage = keep.mean()
        acc = correct[keep].mean() if keep.sum() > 0 else np.nan
        unc_rows.append(dict(threshold=tau, coverage=coverage, accuracy=acc))
    pd.DataFrame(unc_rows).to_csv(DATA / "qztna_uncertainty.csv", index=False)


EXPERIMENTS = dict(detection=run_detection_benchmark, matched_n=run_matched_n,
                   certified=run_certified_robustness, sensitivity=run_sensitivity,
                   zeroday=run_zeroday, alignment=run_kernel_alignment,
                   dimensionality=run_dimensionality, latency=run_latency,
                   multiclass=run_multiclass_and_uncertainty)

if __name__ == "__main__":
    import sys
    t0 = time.time()
    for name in sys.argv[1:] or EXPERIMENTS:
        EXPERIMENTS[name]()
    print(f"All experiments done in {time.time() - t0:.1f}s")
