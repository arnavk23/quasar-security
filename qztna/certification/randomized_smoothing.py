"""Randomized smoothing certification (Cohen, Rosenfeld & Kolter, ICML 2019)."""

import numpy as np
from scipy import stats

ABSTAIN = -1


def _clopper_pearson_lower(count, n, alpha):
    if count == 0:
        return 0.0
    return float(stats.beta.ppf(alpha, count, n - count + 1))


def certify_point(model, x, sigma, n0=100, n=1000, alpha=0.05, rng=None):
    """Certify a single point. Returns (predicted_class, certified_radius) or (ABSTAIN, 0.0)."""
    rng = rng or np.random.default_rng()
    d = x.shape[0]

    noise0 = rng.normal(0, sigma, size=(n0, d))
    preds0 = model.predict(x[None, :] + noise0)
    counts0 = np.bincount(preds0.astype(int), minlength=2)
    c_a = int(np.argmax(counts0))

    noise = rng.normal(0, sigma, size=(n, d))
    preds = model.predict(x[None, :] + noise)
    n_a = int(np.sum(preds.astype(int) == c_a))

    p_a_lower = _clopper_pearson_lower(n_a, n, alpha)
    if p_a_lower <= 0.5:
        return ABSTAIN, 0.0
    radius = sigma * stats.norm.ppf(p_a_lower)
    return c_a, float(radius)


def certify_dataset(model, X, y_true, sigma, n0=100, n=1000, alpha=0.05,
                     max_points=200, random_state=0):
    """Certify up to max_points correctly-classified test points, returns per-point records."""
    rng = np.random.default_rng(random_state)
    y_pred_clean = model.predict(X)
    correct_idx = np.where(y_pred_clean == y_true)[0]
    if len(correct_idx) > max_points:
        correct_idx = rng.choice(correct_idx, size=max_points, replace=False)

    records = []
    for i in correct_idx:
        c_a, radius = certify_point(model, X[i], sigma, n0=n0, n=n, alpha=alpha, rng=rng)
        certified_correct = (c_a == y_true[i]) and (c_a != ABSTAIN)
        records.append({
            "idx": int(i),
            "predicted": int(c_a),
            "true": int(y_true[i]),
            "radius": radius if certified_correct else 0.0,
            "abstained": c_a == ABSTAIN,
        })
    return records
