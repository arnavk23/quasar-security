import numpy as np


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def amplitude_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Squared cosine similarity kernel: K_A(x,y) = |<x/||x||, y/||y||>|^2."""
    Xn = _l2_normalize(np.asarray(X, dtype=np.float64))
    Yn = _l2_normalize(np.asarray(Y, dtype=np.float64))
    return (Xn @ Yn.T) ** 2


def weighted_amplitude_kernel(X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Trainable diagonal-weighted cosine kernel: K(x,y) = (sum_k theta_k x_k y_k)^2."""
    Xn = _l2_normalize(np.asarray(X, dtype=np.float64)) * theta
    Yn = _l2_normalize(np.asarray(Y, dtype=np.float64)) * theta
    return (Xn @ Yn.T) ** 2


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Cristianini et al. 2002 kernel-target alignment, y in {-1,+1}."""
    y = np.asarray(y, dtype=np.float64)
    y = np.where(y <= 0, -1.0, 1.0)
    yy = np.outer(y, y)
    num = np.sum(K * yy)
    den = np.sqrt(np.sum(K * K) * np.sum(yy * yy))
    return float(num / den) if den > 0 else 0.0
