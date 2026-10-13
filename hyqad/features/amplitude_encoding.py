import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin


class AmplitudeEncoder(BaseEstimator, TransformerMixin):
    """
    Amplitude-encoding map: zero-pads each d-dimensional vector to 2^q
    amplitudes (q = ceil(log2 d) unless ``encoding_dim`` is given) and
    L2-normalizes it. All input features are kept; an ``encoding_dim``
    smaller than d is an error rather than a silent truncation.
    """

    def __init__(self, encoding_dim: int | None = None, norm: bool = True):
        self.encoding_dim = encoding_dim
        self.norm = norm

    def fit(self, X: NDArray, y=None):
        return self

    def _dim(self, d: int) -> int:
        if self.encoding_dim is None:
            return 2 ** max(1, int(np.ceil(np.log2(max(d, 2)))))
        if self.encoding_dim < d:
            raise ValueError(f"encoding_dim={self.encoding_dim} cannot hold {d} features")
        return self.encoding_dim

    def transform(self, X: NDArray) -> NDArray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        dim = self._dim(X.shape[1])
        encoded = np.zeros((X.shape[0], dim))
        encoded[:, :X.shape[1]] = X
        norms = np.linalg.norm(encoded, axis=1)
        encoded[norms < 1e-12, 0] = 1.0
        if self.norm:
            encoded /= np.linalg.norm(encoded, axis=1, keepdims=True)
        return encoded

    def kernel(self, X1: NDArray, X2: NDArray) -> NDArray:
        """Fidelity kernel |<phi(x1), phi(x2)>|^2 (squared cosine similarity)."""
        return (self.transform(X1) @ self.transform(X2).T) ** 2
