import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin


class AmplitudeEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes classical QKD channel measurements into Hilbert space
    amplitude vectors. Transforms n-dimensional classical data into
    2^m-dimensional feature vectors via amplitude encoding,
    enabling quantum-inspired representation learning.
    """

    def __init__(self, encoding_dim: int = 8, norm: bool = True):
        self.encoding_dim = encoding_dim
        self.norm = norm

    def fit(self, X: NDArray, y=None):
        return self

    def _encode_single(self, x: NDArray) -> NDArray:
        n_features = len(x)
        dim = self.encoding_dim

        if n_features >= dim:
            projection = x[:dim]
        else:
            projection = np.zeros(dim)
            projection[:n_features] = x

        if np.linalg.norm(projection) < 1e-12:
            projection[0] = 1.0

        if self.norm:
            projection = projection / np.linalg.norm(projection)

        return projection

    def transform(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        encoded = np.array([self._encode_single(row) for row in X])
        return encoded

    def kernel(self, X1: NDArray, X2: NDArray, gamma: float = 1.0) -> NDArray:
        phi1 = self.transform(X1)
        phi2 = self.transform(X2)
        inner = phi1 @ phi2.T
        sq_dists = 2 - 2 * inner
        return np.exp(-gamma * sq_dists)
