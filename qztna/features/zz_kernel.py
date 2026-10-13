import numpy as np


def _signs(q):
    """(2^q, q) matrix of Z eigenvalues, +1 for bit 0 and -1 for bit 1."""
    z = np.arange(2 ** q)[:, None] >> np.arange(q)[None, :] & 1
    return 1.0 - 2.0 * z


def _hadamard(q):
    h = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    w = np.ones((1, 1))
    for _ in range(q):
        w = np.kron(w, h)
    return w


def zz_states(X, reps=2):
    """Statevectors of the ZZ feature map of Havlicek et al. (2019) with linear entanglement.

    U(x) = exp(i sum_i x_i Z_i + i sum_i (pi - x_i)(pi - x_{i+1}) Z_i Z_{i+1}), and the
    state is (U(x) H^q)^reps |0>. Rows of X are the already-scaled rotation angles.
    """
    X = np.asarray(X, dtype=np.float64)
    q = X.shape[1]
    S = _signs(q)
    i, j = np.arange(q - 1), np.arange(1, q)
    phase = X @ S.T + ((np.pi - X[:, i]) * (np.pi - X[:, j])) @ (S[:, i] * S[:, j]).T
    u = np.exp(1j * phase)
    W = _hadamard(q)
    psi = np.full((len(X), 2 ** q), 2.0 ** (-q / 2), dtype=np.complex128)
    for r in range(reps):
        if r > 0:
            psi = psi @ W
        psi = u * psi
    return psi


def zz_kernel(X, Y, reps=2):
    """Fidelity kernel |<psi(x)|psi(y)>|^2 of the ZZ feature map."""
    return np.abs(zz_states(X, reps) @ zz_states(Y, reps).conj().T) ** 2


def shot_noise(K, shots, rng, symmetric=False):
    """Compute-uncompute estimate: each entry is a Binomial(shots, K) frequency."""
    Kh = rng.binomial(shots, np.clip(K, 0.0, 1.0)) / shots
    if symmetric:
        Kh = np.triu(Kh, 1)
        Kh = Kh + Kh.T + np.eye(len(K))
    return Kh
