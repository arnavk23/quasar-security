import numpy as np
import pytest
from qztna.features.zz_kernel import shot_noise, zz_kernel, zz_states


def test_zz_states_normalized():
    X = np.random.default_rng(0).normal(size=(6, 5))
    assert np.allclose(np.linalg.norm(zz_states(X), axis=1), 1.0)


def test_zz_kernel_is_fidelity_kernel():
    X = np.random.default_rng(1).normal(size=(8, 4))
    K = zz_kernel(X, X)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-10


def test_zz_kernel_matches_circuit():
    qml = pytest.importorskip("pennylane")
    q = 3
    dev = qml.device("default.qubit", wires=q)

    @qml.qnode(dev)
    def state(x):
        for _ in range(2):
            for w in range(q):
                qml.Hadamard(w)
            for w in range(q):
                qml.RZ(-2 * x[w], wires=w)
            for w in range(q - 1):
                qml.IsingZZ(-2 * (np.pi - x[w]) * (np.pi - x[w + 1]), wires=[w, w + 1])
        return qml.state()

    X = np.random.default_rng(2).normal(size=(4, q))
    S = np.array([state(x) for x in X])
    assert np.allclose(np.abs(S @ S.conj().T) ** 2, zz_kernel(X, X), atol=1e-12)


def test_shot_noise_is_unbiased_and_symmetric():
    rng = np.random.default_rng(3)
    K = zz_kernel(*(2 * [rng.normal(size=(30, 3))]))
    Kh = shot_noise(K, 100_000, rng, symmetric=True)
    assert np.allclose(Kh, Kh.T)
    assert np.abs(Kh - K).max() < 0.01


def test_amplitude_svm_is_rayleigh_quotient():
    """An SVM on K_A scores x by xhat^T W xhat + b, with W = sum_i a_i y_i xhat_i xhat_i^T."""
    from sklearn.svm import SVC
    from qztna.features.amplitude_kernel import amplitude_kernel
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    m = SVC(kernel="precomputed").fit(amplitude_kernel(X, X), y)
    T = rng.normal(size=(50, 5))
    unit = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
    S = unit(X[m.support_])
    W = (S.T * m.dual_coef_[0]) @ S
    rq = np.einsum("ij,jk,ik->i", unit(T), W, unit(T)) + m.intercept_[0]
    assert np.allclose(m.decision_function(amplitude_kernel(T, X)), rq, atol=1e-10)
    assert np.allclose(m.decision_function(amplitude_kernel(3 * T, X)), rq, atol=1e-10)
