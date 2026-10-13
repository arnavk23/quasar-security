import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, MinCovDet


class CovarianceFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Appends a robust Mahalanobis distance to each measurement vector.

    The covariance model is fit on normal-operation samples only (``y == 0``
    when labels are given), so the distance measures how far a block's joint
    telemetry is from the normal operating envelope. Output columns are the
    input features followed by the distance: shape (n, d + 1). Each row depends
    only on that row, so the transform is independent of sample order.
    """

    def __init__(self, smoothing: float = 1e-6, use_robust: bool = True):
        self.smoothing = smoothing
        self.use_robust = use_robust

    def fit(self, X: NDArray, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        X_ref = X if y is None else X[np.asarray(y) == 0]
        if self.use_robust and X_ref.shape[0] > 2 * self.n_features_in_:
            self.cov_estimator_ = MinCovDet(random_state=42).fit(X_ref)
        else:
            self.cov_estimator_ = EmpiricalCovariance().fit(X_ref)
        self.mean_ = self.cov_estimator_.location_
        self.covariance_ = self.cov_estimator_.covariance_
        self.precision_ = np.linalg.inv(
            self.covariance_ + np.eye(self.n_features_in_) * self.smoothing
        )
        return self

    def _mahalanobis_distance(self, X: NDArray) -> NDArray:
        centered = X - self.mean_
        return np.sqrt(np.maximum(np.sum(centered @ self.precision_ * centered, axis=1), 0.0))

    def transform(self, X: NDArray) -> NDArray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.concatenate([X, self._mahalanobis_distance(X)[:, None]], axis=1)

    def get_uncertainty_score(self, X: NDArray) -> NDArray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return self._mahalanobis_distance(X) / (1 + np.sqrt(np.trace(self.covariance_)))
