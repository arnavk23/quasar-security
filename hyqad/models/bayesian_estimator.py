import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, multivariate_normal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern


class BayesianAnomalyEstimator(BaseEstimator, ClassifierMixin):
    """
    Bayesian probabilistic anomaly detector using Gaussian Process
    classification with uncertainty quantification.

    Inspired by sktime/skpro probabilistic forecasting work,
    adapted for anomaly detection in QKD channels.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        length_scale: float = 1.0,
        length_scale_bounds: tuple[float, float] = (1e-3, 1e3),
        noise_level: float = 1e-3,
        noise_level_bounds: tuple[float, float] = (1e-5, 1e1),
        n_restarts_optimizer: int = 5,
        random_state: int = 42,
        max_iter_predict: int = 100,
        prior_strength: float = 1.0,
        hyperparam_subset: int | None = None,
    ):
        self.kernel = kernel
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.max_iter_predict = max_iter_predict
        self.prior_strength = prior_strength
        self.hyperparam_subset = hyperparam_subset

    def _build_kernel(self):
        if self.kernel == "rbf":
            k = RBF(
                length_scale=self.length_scale,
                length_scale_bounds=self.length_scale_bounds,
            )
        elif self.kernel == "matern":
            k = Matern(
                length_scale=self.length_scale,
                length_scale_bounds=self.length_scale_bounds,
                nu=1.5,
            )
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        return k + WhiteKernel(
            noise_level=self.noise_level,
            noise_level_bounds=self.noise_level_bounds,
        )

    def fit(self, X: NDArray, y: NDArray):
        kernel = self._build_kernel()
        n_restarts, optimizer = self.n_restarts_optimizer, "fmin_l_bfgs_b"
        if self.hyperparam_subset is not None and len(X) > self.hyperparam_subset:
            # Laplace marginal-likelihood optimization is O(n^3) per step; select
            # kernel hyperparameters on a stratified subset, then refit the
            # posterior on all of X with those hyperparameters held fixed.
            from sklearn.model_selection import train_test_split
            idx, _ = train_test_split(np.arange(len(X)), train_size=self.hyperparam_subset,
                                      stratify=y, random_state=self.random_state)
            kernel = GaussianProcessClassifier(
                kernel=kernel, n_restarts_optimizer=n_restarts, random_state=self.random_state,
                max_iter_predict=self.max_iter_predict,
            ).fit(X[idx], np.asarray(y)[idx]).kernel_
            n_restarts, optimizer = 0, None
        self.gpc_ = GaussianProcessClassifier(
            kernel=kernel,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts,
            random_state=self.random_state,
            max_iter_predict=self.max_iter_predict,
        )
        self.gpc_.fit(X, y)
        self.classes_ = self.gpc_.classes_
        return self

    def predict(self, X: NDArray) -> NDArray:
        return self.gpc_.predict(X)

    def predict_proba(self, X: NDArray) -> NDArray:
        return self.gpc_.predict_proba(X)

    def predict_with_uncertainty(self, X: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        proba = self.predict_proba(X)
        anomaly_proba = proba[:, 1] if self.classes_[1] == 1 else proba[:, 0]
        predictions = (anomaly_proba >= 0.5).astype(int)

        prediction_var = anomaly_proba * (1 - anomaly_proba)
        uncertainty = np.sqrt(prediction_var)

        return predictions, anomaly_proba, uncertainty

    def predict_anomaly_score(self, X: NDArray) -> NDArray:
        _, scores, _ = self.predict_with_uncertainty(X)
        return scores


class BayesianLinearEstimator(BaseEstimator, ClassifierMixin):
    """
    Lightweight Bayesian linear classifier for low-latency edge deployment.
    Uses variational inference approximation.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, max_iter: int = 500):
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter

    def fit(self, X: NDArray, y: NDArray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        XtX = X.T @ X
        Xty = X.T @ y

        self.w_mean_ = np.linalg.solve(
            XtX + (self.alpha / self.beta) * np.eye(d), Xty
        )
        self.w_cov_ = np.linalg.inv(self.beta * XtX + self.alpha * np.eye(d))
        self.intercept_ = np.mean(y - X @ self.w_mean_)
        return self

    def predict(self, X: NDArray) -> NDArray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=np.float64)
        mu = X @ self.w_mean_ + self.intercept_
        var = np.sum(X @ self.w_cov_ * X, axis=1) + 1.0 / self.beta
        sigma = np.sqrt(var + 1e-8)
        return norm.cdf(mu / sigma)

    def predict_with_uncertainty(self, X: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        proba = self.predict_proba(X)
        predictions = (proba >= 0.5).astype(int)
        uncertainty = np.sqrt(proba * (1 - proba))
        return predictions, proba, uncertainty
