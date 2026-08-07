import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, MinCovDet


class CovarianceFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts covariance-aware uncertainty features from QKD signal
    measurements. Models cross-parameter correlations and their
    uncertainties using robust covariance estimation.

    Inspired by the NQComp 2026 materials discovery covariance-aware
    acquisition function approach.
    """

    def __init__(self, smoothing: float = 1e-6, use_robust: bool = True, window: int = 20):
        self.smoothing = smoothing
        self.use_robust = use_robust
        self.window = window

    def fit(self, X: NDArray, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        if self.use_robust and X.shape[0] > self.n_features_in_:
            self.cov_estimator_ = MinCovDet(random_state=42).fit(X)
        else:
            self.cov_estimator_ = EmpiricalCovariance().fit(X)
        self.mean_ = self.cov_estimator_.location_
        self.covariance_ = self.cov_estimator_.covariance_
        try:
            self.precision_ = self.cov_estimator_.precision_
        except Exception:
            self.precision_ = np.linalg.inv(
                self.covariance_ + np.eye(self.n_features_in_) * self.smoothing
            )
        return self

    def _mahalanobis_distance(self, X: NDArray) -> NDArray:
        centered = X - self.mean_
        try:
            return np.sqrt(np.sum(centered @ self.precision_ * centered, axis=1))
        except Exception:
            return np.zeros(X.shape[0])

    def transform(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = X.shape[0]
        window = min(self.window, n)

        local_means = np.array([
            X[max(0, i - window):i + 1].mean(axis=0)
            for i in range(n)
        ])
        local_stds = np.array([
            X[max(0, i - window):i + 1].std(axis=0)
            for i in range(n)
        ])
        local_stds = np.nan_to_num(local_stds, nan=1e-8)

        mahal = self._mahalanobis_distance(X).reshape(-1, 1)

        cov_norm = np.linalg.norm(self.covariance_, 'fro')
        cov_norm_col = np.full((n, 1), cov_norm)

        pairwise_corr = self.covariance_[np.triu_indices_from(self.covariance_, k=1)]
        mean_pairwise = np.full((n, 1), np.mean(np.abs(pairwise_corr)))
        max_pairwise = np.full((n, 1), np.max(np.abs(pairwise_corr)) if len(pairwise_corr) > 0 else 0)

        features = np.concatenate([
            local_means, local_stds, mahal, cov_norm_col, mean_pairwise, max_pairwise
        ], axis=1)

        return features

    def get_uncertainty_score(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=np.float64)
        mahal = self._mahalanobis_distance(X)
        cov_diag = np.diag(self.covariance_)
        trace_term = np.trace(self.covariance_)
        uncertainty = mahal / (1 + np.sqrt(trace_term))
        return uncertainty
