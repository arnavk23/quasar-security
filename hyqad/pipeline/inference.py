import time
import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hyqad.models.hybrid_detector import HybridAnomalyDetector
from hyqad.features.covariance_features import CovarianceFeatureExtractor
from hyqad.features.amplitude_encoding import AmplitudeEncoder
from hyqad.config import Config


class InferencePipeline:
    """
    Low-latency inference pipeline optimized for edge deployment.
    Combines feature engineering and hybrid detection in a
    streamlined sklearn-compatible pipeline.

    Supports GPU/parallel acceleration and configurable
    latency vs. accuracy trade-offs.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._pipeline = None
        self._latency_history: list[float] = []

    def _build_pipeline(self) -> Pipeline:
        steps = []

        scaler = StandardScaler()
        steps.append(("scaler", scaler))

        if self.config.features.use_covariance_features:
            cov = CovarianceFeatureExtractor(
                smoothing=self.config.features.covariance_smoothing,
                window=20,
            )
            steps.append(("cov_features", cov))

        if self.config.features.use_amplitude_encoding:
            amp = AmplitudeEncoder(encoding_dim=self.config.features.encoding_dim)
            steps.append(("amplitude_encoding", amp))

        detector = HybridAnomalyDetector(
            use_bayesian=self.config.pipeline.use_bayesian_estimator,
            use_quantum_kernel=self.config.pipeline.use_quantum_kernel,
            n_qubits=self.config.model.n_quantum_qubits,
            quantum_kernel_type=self.config.model.quantum_kernel_type,
            n_restarts_optimizer=self.config.model.n_restarts_optimizer,
            random_state=self.config.model.random_state,
        )
        steps.append(("detector", detector))

        return Pipeline(steps)

    def fit(self, X: NDArray, y: NDArray):
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: NDArray) -> NDArray:
        start = time.perf_counter()
        result = self._pipeline.predict(X)
        self._latency_history.append(time.perf_counter() - start)
        return result

    def predict_proba(self, X: NDArray) -> NDArray:
        start = time.perf_counter()
        result = self._pipeline.predict_proba(X)
        self._latency_history.append(time.perf_counter() - start)
        return result

    def predict_with_uncertainty(self, X: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        start = time.perf_counter()
        proba = self.predict_proba(X)
        anomaly_proba = proba[:, 1]
        predictions = (anomaly_proba >= 0.5).astype(int)
        uncertainty = np.sqrt(anomaly_proba * (1 - anomaly_proba))
        self._latency_history.append(time.perf_counter() - start)
        return predictions, anomaly_proba, uncertainty

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_history:
            return 0.0
        return float(np.mean(self._latency_history) * 1000)

    @property
    def p95_latency_ms(self) -> float:
        if not self._latency_history:
            return 0.0
        return float(np.percentile(self._latency_history, 95) * 1000)

    def reset_latency(self):
        self._latency_history.clear()

    def benchmark_throughput(self, X: NDArray, n_runs: int = 100) -> dict:
        samples_per_sec = []
        for _ in range(n_runs):
            start = time.perf_counter()
            self.predict(X)
            elapsed = time.perf_counter() - start
            samples_per_sec.append(len(X) / elapsed if elapsed > 0 else 0)
        return {
            "throughput_mean": float(np.mean(samples_per_sec)),
            "throughput_std": float(np.std(samples_per_sec)),
            "latency_ms_mean": self.avg_latency_ms,
            "latency_ms_p95": self.p95_latency_ms,
        }

    def warmup(self, X: NDArray, n_warmup: int = 10):
        for _ in range(n_warmup):
            _ = self._pipeline.predict(X[:min(10, len(X))])
        self.reset_latency()
