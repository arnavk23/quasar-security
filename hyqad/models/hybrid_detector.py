import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hyqad.models.bayesian_estimator import BayesianAnomalyEstimator
from hyqad.models.quantum_kernel import QuantumKernelEstimator


class HybridAnomalyDetector(BaseEstimator, ClassifierMixin):
    """
    Hybrid quantum-classical anomaly detector that combines:
    1. Bayesian probabilistic estimator (uncertainty-aware)
    2. Quantum kernel estimator (quantum feature space)
    3. Meta-learner combining both with learned weights

    Provides calibrated anomaly scores with uncertainty estimates.
    """

    def __init__(
        self,
        use_bayesian: bool = True,
        use_quantum_kernel: bool = True,
        bayesian_kernel: str = "rbf",
        n_qubits: int = 3,
        quantum_kernel_type: str = "amplitude",
        meta_learner: str = "logistic",
        n_restarts_optimizer: int = 5,
        random_state: int = 42,
    ):
        self.use_bayesian = use_bayesian
        self.use_quantum_kernel = use_quantum_kernel
        self.bayesian_kernel = bayesian_kernel
        self.n_qubits = n_qubits
        self.quantum_kernel_type = quantum_kernel_type
        self.meta_learner = meta_learner
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    def _build_estimators(self):
        estimators = []

        if self.use_bayesian:
            bayesian = BayesianAnomalyEstimator(
                kernel=self.bayesian_kernel,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.random_state,
            )
            estimators.append(("bayesian", bayesian))

        if self.use_quantum_kernel:
            quantum = QuantumKernelEstimator(
                n_qubits=self.n_qubits,
                kernel_type=self.quantum_kernel_type,
                random_state=self.random_state,
            )
            estimators.append(("quantum", quantum))

        if not estimators:
            estimators.append((
                "fallback",
                RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            ))

        return estimators

    def _build_meta_learner(self):
        if self.meta_learner == "logistic":
            return LogisticRegression(C=1.0, random_state=self.random_state)
        elif self.meta_learner == "rf":
            return RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        else:
            return LogisticRegression(C=1.0, random_state=self.random_state)

    def fit(self, X: NDArray, y: NDArray):
        estimators = self._build_estimators()
        meta = self._build_meta_learner()

        if len(estimators) == 1:
            self.estimator_ = estimators[0][1]
            self.estimator_.fit(X, y)
            self.is_single_ = True
        else:
            self.is_single_ = False
            self.stack_ = StackingClassifier(
                estimators=estimators,
                final_estimator=meta,
                cv=3,
                stack_method="predict_proba",
            )
            self.stack_.fit(X, y)

        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: NDArray) -> NDArray:
        if self.is_single_:
            return self.estimator_.predict(X)
        return self.stack_.predict(X)

    def predict_proba(self, X: NDArray) -> NDArray:
        if self.is_single_:
            return self.estimator_.predict_proba(X)
        return self.stack_.predict_proba(X)

    def predict_with_uncertainty(self, X: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        proba = self.predict_proba(X)
        anomaly_proba = proba[:, 1]
        predictions = (anomaly_proba >= 0.5).astype(int)
        aleatoric = np.sqrt(anomaly_proba * (1 - anomaly_proba))

        if hasattr(self, 'estimator_') and hasattr(self.estimator_, 'predict_with_uncertainty'):
            _, _, epi = self.estimator_.predict_with_uncertainty(X)
            aleatoric = np.maximum(aleatoric, epi)

        return predictions, anomaly_proba, aleatoric

    def predict_anomaly_score(self, X: NDArray) -> NDArray:
        _, scores, _ = self.predict_with_uncertainty(X)
        return scores
