#!/usr/bin/env python3
"""
Sensitivity analysis at FULL training size (n=700): sweep attack strength
for RF and GP trained on the full deployment-scale training set.
This complements sensitivity_analysis.py (n=25, matched with QK).
"""
import warnings, time, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.stdout.reconfigure(line_buffering=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from hyqad.simulation.qkd_channel import QKDChannelSimulator, ChannelState
from hyqad.simulation.attacks import AttackSimulator
from hyqad.simulation import QKDDataGenerator
from hyqad.models import BayesianAnomalyEstimator
from hyqad.features import CovarianceFeatureExtractor, AmplitudeEncoder
from hyqad.evaluation import Evaluator

N_SEEDS = 5
N_SAMPLES = 1000
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = str(ROOT / "data" / "hyqad" / "sensitivity_full_n_results.csv")

ATTACK_STRENGTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
ATTACK_TYPES = ["pns", "detector_blinding", "trojan_horse", "intercept_resend"]

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

def generate_scaled_data(n_samples, attack_ratio, strength, rng):
    gen = QKDDataGenerator(rng=rng)
    n_normal = int(n_samples * (1 - attack_ratio))
    n_attack = n_samples - n_normal

    normal_state = gen.channel.simulate(n_normal, noise_scale=0.05, include_anomalies=False)
    attacker = AttackSimulator(rng=rng)

    attack_states, attack_labels = [], []
    per_type = n_attack // len(ATTACK_TYPES)

    for i, atk in enumerate(ATTACK_TYPES):
        n_atk = per_type + (1 if i < (n_attack % len(ATTACK_TYPES)) else 0)
        clean = gen.channel.simulate(n_atk + 100, noise_scale=0.05, include_anomalies=False)

        if strength == 0:
            for j in range(n_atk):
                attack_states.append((clean.time[j], clean.qber[j], clean.photon_count_rate[j],
                    clean.detection_efficiency[j], clean.dark_count_rate[j], clean.error_rate[j],
                    clean.signal_attenuation[j], clean.background_noise[j]))
                attack_labels.append(0)
        else:
            if atk == "pns":
                s = attacker.apply_pns_attack(clean, split_fraction=0.3*strength)
            elif atk == "detector_blinding":
                s = attacker.apply_detector_blinding(clean, blinding_intensity=1.5*strength)
            elif atk == "trojan_horse":
                s = attacker.apply_trojan_horse(clean, power_ratio=0.05*strength)
            else:
                n_ir = len(clean.qber)
                induced = rng.uniform(0.1, 0.35, n_ir) * strength
                qber = np.clip(clean.qber + induced, 0, 0.5)
                low = max(0.05, 0.6 + 0.35*(1-min(strength,1.0)))
                high = min(1.0, 0.95 - 0.35*(1-min(strength,1.0)))
                if high <= low:
                    high = low + 0.01
                photon = np.clip(clean.photon_count_rate * rng.uniform(low, high, n_ir), 0, None)
                s = ChannelState(time=clean.time, qber=qber, photon_count_rate=photon,
                    detection_efficiency=clean.detection_efficiency, dark_count_rate=clean.dark_count_rate,
                    error_rate=clean.error_rate, signal_attenuation=clean.signal_attenuation,
                    background_noise=clean.background_noise)

            for j in range(n_atk):
                attack_states.append((s.time[j], s.qber[j], s.photon_count_rate[j],
                    s.detection_efficiency[j], s.dark_count_rate[j], s.error_rate[j],
                    s.signal_attenuation[j], s.background_noise[j]))
                attack_labels.append(1)

    normal_data = [(normal_state.time[i], normal_state.qber[i], normal_state.photon_count_rate[i],
        normal_state.detection_efficiency[i], normal_state.dark_count_rate[i], normal_state.error_rate[i],
        normal_state.signal_attenuation[i], normal_state.background_noise[i]) for i in range(n_normal)]

    cols = ["time","qber","photon_count_rate","detection_efficiency","dark_count_rate",
            "error_rate","signal_attenuation","background_noise"]
    df = pd.DataFrame(normal_data + attack_states, columns=cols)
    df["label"] = [0]*n_normal + attack_labels
    df = df.sample(frac=1, random_state=rng.integers(2**31)).reset_index(drop=True)
    return df

results = []

for seed in range(N_SEEDS):
    rng_base = np.random.default_rng(seed)

    for strength in ATTACK_STRENGTHS:
        df = generate_scaled_data(N_SAMPLES, 0.3, strength, rng_base)
        raw_cols = [c for c in df.columns if c not in ("label","is_attack")]
        X_raw, y = df[raw_cols].values, df["label"].values
        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X_raw, y, test_size=0.3, stratify=y, random_state=seed)
        X_tr_28, X_te_28 = build_28dim(X_tr_raw, X_te_raw)

        # Use FULL training set (no truncation)
        n_train = len(X_tr_28)

        # RF (full n)
        rf = RandomForestClassifier(n_estimators=200, random_state=seed)
        rf.fit(X_tr_28, y_tr)
        yp = rf.predict(X_te_28)
        m = Evaluator().evaluate(y_te, yp, y_score=rf.predict_proba(X_te_28)[:,1])
        results.append(dict(model="RF", seed=seed, strength=strength, n_train=n_train,
            acc=m.accuracy, f1=m.f1_score, auc=m.roc_auc, mcc=m.matthews_corrcoef))

        # GP (full n - up to 700 like benchmark)
        gp_n = min(n_train, 700)
        gp = BayesianAnomalyEstimator(kernel="rbf", n_restarts_optimizer=3, random_state=seed)
        gp.fit(X_tr_28[:gp_n], y_tr[:gp_n])
        yp = gp.predict(X_te_28)
        m = Evaluator().evaluate(y_te, yp, y_score=gp.predict_proba(X_te_28)[:,1])
        results.append(dict(model="GP-only", seed=seed, strength=strength, n_train=gp_n,
            acc=m.accuracy, f1=m.f1_score, auc=m.roc_auc, mcc=m.matthews_corrcoef))

        print(f"  s={seed} str={strength:.1f} RF(n={n_train})={results[-2]['acc']:.3f} GP(n={gp_n})={results[-1]['acc']:.3f}", flush=True)

    pd.DataFrame(results).to_csv(OUT, index=False)

df = pd.read_csv(OUT)
print("\n=== Sensitivity Summary (full n) ===")
print(f"{'Str':<6} {'RF Acc':<12} {'GP Acc':<12} {'GP F1':<12}")
print("-"*42)
for s in ATTACK_STRENGTHS:
    sub = df[df.strength == s]
    rf_a = sub[sub.model=="RF"].acc.mean()
    gp_a = sub[sub.model=="GP-only"].acc.mean()
    gp_f = sub[sub.model=="GP-only"].f1.mean()
    print(f"{s:<6.1f} {rf_a:<12.3f} {gp_a:<12.3f} {gp_f:<12.3f}")
