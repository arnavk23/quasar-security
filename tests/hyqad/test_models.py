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


def test_quantum_kernel_exact_matches_circuit():
    pytest.importorskip("pennylane")
    from hyqad.models.quantum_kernel import QuantumKernelEstimator, amplitude_fidelity_kernel
    rng = np.random.default_rng(0)
    for d in (3, 8, 12):
        X, Y = rng.normal(size=(4, d)), rng.normal(size=(3, d))
        qk = QuantumKernelEstimator(backend="circuit")
        qk.n_qubits_ = qk._resolve_qubits(d)
        assert np.allclose(qk._quantum_kernel(X, Y), amplitude_fidelity_kernel(X, Y, qk.n_qubits_), atol=1e-10)


def test_quantum_kernel_uses_all_features():
    from hyqad.models.quantum_kernel import QuantumKernelEstimator
    X = np.zeros((2, 12)); X[0, 0] = 1; X[1, 0] = 1; X[1, 11] = 1
    qk = QuantumKernelEstimator().fit(np.vstack([X, -X]), [0, 0, 1, 1])
    assert qk.n_qubits_ == 4
    assert qk._quantum_kernel(X[:1], X[1:])[0, 0] < 0.99


def test_quantum_kernel_rejects_too_few_qubits():
    from hyqad.models.quantum_kernel import QuantumKernelEstimator
    with pytest.raises(ValueError):
        QuantumKernelEstimator(n_qubits=3).fit(np.random.randn(10, 12), [0, 1] * 5)


def test_outlier_scores_rank_anomalies_high():
    from sklearn.ensemble import IsolationForest
    from hyqad.evaluation.benchmarks import BenchmarkSuite
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(200, 3)), rng.normal(6, 1, size=(20, 3))])
    y = np.r_[np.zeros(200), np.ones(20)]
    raw = IsolationForest(random_state=0).fit(X).decision_function(X)
    score = BenchmarkSuite()._convert_outlier_score(raw, y)
    assert roc_auc_score(y, score) > 0.95
