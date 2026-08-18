"""
Kernel SVM classifiers with an exact-kernel path for small training sets and a
Nystrom-approximated path for large ones, matching the two regimes described
in the paper (exact precomputed-kernel SVM for controlled ablations, Nystrom
for deployment-scale training).
"""

import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import laplacian_kernel, polynomial_kernel, rbf_kernel
from sklearn.utils import check_random_state

from qztna.features.amplitude_kernel import amplitude_kernel

KERNEL_FUNCS = {
    "ae": amplitude_kernel,
    "rbf": lambda X, Y: rbf_kernel(X, Y, gamma=1.0 / X.shape[1]),
    "laplacian": lambda X, Y: laplacian_kernel(X, Y, gamma=1.0 / X.shape[1]),
    "poly": lambda X, Y: polynomial_kernel(X, Y, degree=3, gamma=1.0 / X.shape[1], coef0=1.0),
}


class ExactKernelSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel="ae", C=1.0, probability=True):
        self.kernel = kernel
        self.C = C
        self.probability = probability

    def fit(self, X, y):
        kf = KERNEL_FUNCS[self.kernel]
        self._X_train = np.asarray(X, dtype=np.float64)
        self.svc_ = SVC(kernel=kf, C=self.C, probability=self.probability, class_weight="balanced")
        self.svc_.fit(self._X_train, y)
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(X)

    def predict(self, X):
        return self.svc_.predict(X)

    def predict_proba(self, X):
        return self.svc_.predict_proba(X)


class NystromKernelSVM(BaseEstimator, ClassifierMixin):
    """Nystrom-approximated kernel SVM: explicit low-rank feature map + linear SVM."""

    def __init__(self, kernel="ae", landmark_frac=0.1, max_landmarks=500, min_landmarks=25,
                 C=1.0, random_state=0, calibrate=True):
        self.kernel = kernel
        self.landmark_frac = landmark_frac
        self.max_landmarks = max_landmarks
        self.min_landmarks = min_landmarks
        self.C = C
        self.random_state = random_state
        self.calibrate = calibrate

    def _phi(self, X):
        K_nm = self._kernel_func(np.asarray(X, dtype=np.float64), self.landmarks_)
        return K_nm @ self._W

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        rng = check_random_state(self.random_state)
        m = int(np.clip(np.ceil(self.landmark_frac * n), self.min_landmarks, self.max_landmarks))
        m = min(m, n)
        idx = rng.choice(n, size=m, replace=False)
        self.landmarks_ = X[idx]
        self.n_landmarks_ = m
        self._kernel_func = KERNEL_FUNCS[self.kernel]

        K_mm = self._kernel_func(self.landmarks_, self.landmarks_)
        eigval, eigvec = np.linalg.eigh(K_mm)
        eigval = np.clip(eigval, 1e-10, None)
        self._W = eigvec @ np.diag(1.0 / np.sqrt(eigval))

        Phi = self._phi(X)
        base = LinearSVC(C=self.C, max_iter=5000, dual="auto", class_weight="balanced")
        if self.calibrate:
            self.clf_ = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        else:
            self.clf_ = base
        self.clf_.fit(Phi, y)
        return self

    def decision_function(self, X):
        Phi = self._phi(X)
        if hasattr(self.clf_, "decision_function"):
            return self.clf_.decision_function(Phi)
        return self.clf_.predict_proba(Phi)[:, 1] - 0.5

    def predict(self, X):
        return self.clf_.predict(self._phi(X))

    def predict_proba(self, X):
        return self.clf_.predict_proba(self._phi(X))


def fit_kernel_svm(kernel, X, y, C=1.0, random_state=0, nystrom_threshold=1500,
                    landmark_frac=0.1, max_landmarks=500, force_nystrom=False,
                    force_exact=False):
    n = len(X)
    use_nystrom = force_nystrom or (not force_exact and n > nystrom_threshold)
    t0 = time.perf_counter()
    if use_nystrom:
        model = NystromKernelSVM(kernel=kernel, C=C, random_state=random_state,
                                  landmark_frac=landmark_frac, max_landmarks=max_landmarks)
    else:
        model = ExactKernelSVM(kernel=kernel, C=C)
    model.fit(X, y)
    fit_time = time.perf_counter() - t0
    return model, fit_time
