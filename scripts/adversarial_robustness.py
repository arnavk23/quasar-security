#!/usr/bin/env python3
"""
Adversarial robustness (optimized, 10 seeds): gradient-free evasion in 28-dim
feature space against RF and GP, with GP uncertainty tracking.
"""
import warnings, time, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.stdout.reconfigure(line_buffering=True)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from hyqad.simulation import QKDDataGenerator
from hyqad.models import BayesianAnomalyEstimator
from hyqad.features import CovarianceFeatureExtractor, AmplitudeEncoder
from hyqad.evaluation import Evaluator
from sklearn.preprocessing import StandardScaler

N_SEEDS = 10
N_SAMPLES = 1000
OUT = "/home/arnav/Downloads/ccri/adversarial_robustness_results.csv"

PERTURB_BUDGETS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

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
    X_tr_28 = np.concatenate([X_tr_cov, X_tr_amp], axis=1)
    X_te_28 = np.concatenate([X_te_cov, X_te_amp], axis=1)
    return X_tr_28, X_te_28

def transform_single(X_raw, scaler, cov, amp):
    X_sc = scaler.transform(X_raw)
    X_cov = cov.transform(X_sc)
    X_amp = amp.transform(X_cov)
    return np.concatenate([X_cov, X_amp], axis=1)

results = []

for seed in range(N_SEEDS):
    gen = QKDDataGenerator(rng=np.random.default_rng(seed))
    df = gen.generate(n_samples=N_SAMPLES, attack_ratio=0.3)
    raw_cols = [c for c in df.columns if c not in ("label", "is_attack")]
    X_raw, y = df[raw_cols].values, df["label"].values

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=seed)

    X_tr_28, X_te_28 = build_28dim(X_tr_raw, X_te_raw)

    # Train RF and fit pipeline
    scaler = StandardScaler().fit(X_tr_raw)
    cov = CovarianceFeatureExtractor(smoothing=1e-6, window=20).fit(scaler.transform(X_tr_raw))
    amp = AmplitudeEncoder(encoding_dim=8).fit(cov.transform(scaler.transform(X_tr_raw)))

    rf = RandomForestClassifier(n_estimators=200, random_state=seed)
    rf.fit(X_tr_28, y_tr)

    gp = BayesianAnomalyEstimator(kernel="rbf", n_restarts_optimizer=3, random_state=seed)
    gp.fit(X_tr_28, y_tr)

    # Feature ranges in 28-dim space for perturbation
    feat_ranges_28 = np.ptp(X_tr_28, axis=0)

    # Identify which 28-dim features are most important via RF
    importances = rf.feature_importances_
    top_features = np.argsort(importances)[-5:]  # top 5 most important

    attack_mask = y_te == 1
    n_attack = int(np.sum(attack_mask))

    # Baseline
    yp_rf = rf.predict(X_te_28)
    yp_gp = gp.predict(X_te_28)
    m_rf = Evaluator().evaluate(y_te, yp_rf, y_score=rf.predict_proba(X_te_28)[:, 1])
    m_gp = Evaluator().evaluate(y_te, yp_gp, y_score=gp.predict_proba(X_te_28)[:, 1])

    results.append(dict(model="RF", seed=seed, budget=0.0, n_evaded=0, n_attack=n_attack,
        detection_rate=m_rf.true_positive_rate, mean_uncertainty=0, high_uncert_frac=0))
    results.append(dict(model="GP", seed=seed, budget=0.0, n_evaded=0, n_attack=n_attack,
        detection_rate=m_gp.true_positive_rate, mean_uncertainty=0, high_uncert_frac=0))

    for budget in PERTURB_BUDGETS:
        if budget == 0:
            continue

        # Create evasive samples: perturb top-5 features of attack samples toward
        # the centroid of the normal class in 28-dim space
        normal_mask = y_tr == 0
        normal_centroid = np.mean(X_tr_28[normal_mask], axis=0)

        X_ev_28 = X_te_28.copy()
        atk_idx = np.where(attack_mask)[0]

        for i in atk_idx:
            x = X_ev_28[i].copy()
            for feat in top_features:
                feat_range = feat_ranges_28[feat]
                delta = feat_range * budget
                # Move toward normal centroid
                direction = np.sign(normal_centroid[feat] - x[feat])
                x[feat] += direction * delta
            X_ev_28[i] = x

        yp_rf_ev = rf.predict(X_ev_28)
        yp_gp_ev = gp.predict(X_ev_28)

        n_ev_rf = int(np.sum(yp_rf_ev[attack_mask] == 0))
        n_ev_gp = int(np.sum(yp_gp_ev[attack_mask] == 0))

        # GP uncertainty on attacked samples
        _, _, uncert = gp.predict_with_uncertainty(X_ev_28)
        atk_uncerts = uncert[attack_mask]
        mean_uncert = float(np.mean(atk_uncerts))
        high_uncert_frac = float(np.mean(atk_uncerts > 0.3))

        results.append(dict(model="RF", seed=seed, budget=budget,
            n_evaded=n_ev_rf, n_attack=n_attack,
            detection_rate=1.0 - n_ev_rf/max(n_attack,1),
            mean_uncertainty=0, high_uncert_frac=0))
        results.append(dict(model="GP", seed=seed, budget=budget,
            n_evaded=n_ev_gp, n_attack=n_attack,
            detection_rate=1.0 - n_ev_gp/max(n_attack,1),
            mean_uncertainty=mean_uncert, high_uncert_frac=high_uncert_frac))

        print(f"  seed={seed} budget={budget:.2f} RF_ev={n_ev_rf}/{n_attack} "
              f"GP_ev={n_ev_gp}/{n_attack} GP_unc={mean_uncert:.3f}", flush=True)

    pd.DataFrame(results).to_csv(OUT, index=False)

df = pd.read_csv(OUT)
print("\n=== Adversarial Robustness Summary ===")
print(f"{'Budget':<10} {'RF DetRate':<12} {'GP DetRate':<12} {'GP Uncert':<12} {'GP HiU%':<10}")
print("-" * 56)
for b in PERTURB_BUDGETS:
    sub = df[df.budget == b]
    rf_dr = sub[sub.model=="RF"].detection_rate.mean()
    gp_dr = sub[sub.model=="GP"].detection_rate.mean()
    gp_unc = sub[sub.model=="GP"].mean_uncertainty.mean()
    gp_hi = sub[sub.model=="GP"].high_uncert_frac.mean()
    print(f"{b:<10.2f} {rf_dr:<12.3f} {gp_dr:<12.3f} {gp_unc:<12.4f} {gp_hi:<10.3f}")
