import numpy as np
import pytest
from sklearn.datasets import make_classification
from hyqad.models import (
    BayesianAnomalyEstimator, BayesianLinearEstimator,
    HybridAnomalyDetector, JSOOptimizer,
)


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4,
        n_redundant=1, weights=[0.7, 0.3], random_state=42,
    )
    return X, y


def test_bayesian_estimator_fit_predict(classification_data):
    X, y = classification_data
    model = BayesianAnomalyEstimator(random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (200,)
    assert set(preds).issubset({0, 1})


def test_bayesian_estimator_uncertainty(classification_data):
    X, y = classification_data
    model = BayesianAnomalyEstimator(random_state=42)
    model.fit(X, y)
    preds, proba, uncert = model.predict_with_uncertainty(X)
    assert np.all(uncert >= 0)
    assert np.all(uncert <= 1)


def test_bayesian_linear_estimator(classification_data):
    X, y = classification_data
    model = BayesianLinearEstimator()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (200,)


def test_hybrid_detector_bayesian_only(classification_data):
    X, y = classification_data
    model = HybridAnomalyDetector(
        use_bayesian=True, use_quantum_kernel=False, random_state=42
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (200,)


def test_hybrid_detector_full(classification_data):
    X, y = classification_data
    model = HybridAnomalyDetector(
        use_bayesian=True, use_quantum_kernel=False,
        random_state=42,
    )
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (200, 2)


def test_hybrid_detector_uncertainty(classification_data):
    X, y = classification_data
    model = HybridAnomalyDetector(
        use_bayesian=True, use_quantum_kernel=False, random_state=42
    )
    model.fit(X, y)
    preds, scores, uncert = model.predict_with_uncertainty(X)
    assert np.all(uncert >= 0)
    assert np.all(uncert <= 1)


def test_jso_optimizer():
    def objective(x):
        return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

    opt = JSOOptimizer(method="L-BFGS-B", n_restarts=3, random_state=42)
    bounds = [(-10, 10), (-10, 10)]
    best_x, best_f = opt.optimize(objective, bounds)
    assert abs(best_f) < 1e-4
    assert abs(best_x[0] - 2) < 1e-2
    assert abs(best_x[1] - 3) < 1e-2


def test_jso_optimizer_differential_evolution():
    def objective(x):
        return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

    opt = JSOOptimizer(method="differential_evolution", max_iter=100, random_state=42)
    bounds = [(-10, 10), (-10, 10)]
    best_x, best_f = opt.optimize(objective, bounds)
    assert best_f < 1e-4
