import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from hyqad.evaluation.metrics import Evaluator, DetectionMetrics


class BenchmarkSuite:
    """
    Benchmarks the hybrid detector against classical ML baselines:
    - Random Forest
    - Isolation Forest
    - One-Class SVM
    - Local Outlier Factor

    Reports accuracy, F1, ROC-AUC, FPR, and latency for each.
    """

    BASELINES = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "IsolationForest": IsolationForest(random_state=42, contamination=0.1),
        "OneClassSVM": OneClassSVM(nu=0.1, kernel="rbf", gamma="scale"),
        "LocalOutlierFactor": LocalOutlierFactor(novelty=True, contamination=0.1),
    }

    def __init__(self, evaluator: Evaluator | None = None):
        self.evaluator = evaluator or Evaluator()
        self.results: dict[str, DetectionMetrics] = {}

    def _convert_outlier_score(self, scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Classifier probabilities: column 1 is P(attack).
        if scores.ndim == 2 and scores.shape[1] == 2:
            return scores[:, 1]
        # Outlier detectors' decision_function is higher for *inliers*, so it is
        # always negated to get an anomaly score (never sign-flipped on the data).
        scores = -scores
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    def run_baseline(
        self, name: str, model, X_train, y_train, X_test, y_test
    ) -> DetectionMetrics:
        start = time.perf_counter()

        # Novelty detectors are fit on normal training samples only.
        if name == "LocalOutlierFactor":
            model.fit(X_train[y_train == 0])
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 0, 1)
            y_score = model.decision_function(X_test)
            y_score = self._convert_outlier_score(y_score, y_test)
        elif name == "OneClassSVM":
            model.fit(X_train[y_train == 0])
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 0, 1)
            y_score = model.decision_function(X_test)
            y_score = self._convert_outlier_score(y_score, y_test)
        elif name == "IsolationForest":
            model.fit(X_train[y_train == 0])
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 0, 1)
            y_score = model.decision_function(X_test)
            y_score = self._convert_outlier_score(y_score, y_test)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
                y_score = self._convert_outlier_score(y_score, y_test)
            else:
                y_score = None

        elapsed_ms = (time.perf_counter() - start) * 1000

        metrics = self.evaluator.evaluate(
            y_test, y_pred, y_score=y_score, latency_ms=elapsed_ms
        )
        self.results[name] = metrics
        return metrics

    def benchmark_all(
        self, X_train, y_train, X_test, y_test, include_hybrid: bool = True,
        hybrid_detector=None,
    ) -> pd.DataFrame:
        for name, model in self.BASELINES.items():
            self.run_baseline(name, model, X_train, y_train, X_test, y_test)

        if include_hybrid and hybrid_detector is not None:
            start = time.perf_counter()
            hybrid_detector.fit(X_train, y_train)
            y_pred = hybrid_detector.predict(X_test)
            y_score = hybrid_detector.predict_proba(X_test)[:, 1]
            elapsed_ms = (time.perf_counter() - start) * 1000

            metrics = self.evaluator.evaluate(
                y_test, y_pred, y_score=y_score, latency_ms=elapsed_ms
            )
            self.results["HybridDetector"] = metrics

        rows = []
        for name, m in self.results.items():
            rows.append({
                "Model": name,
                "Accuracy": m.accuracy,
                "F1": m.f1_score,
                "ROC-AUC": m.roc_auc,
                "Precision": m.precision,
                "Recall": m.recall,
                "FPR": m.false_positive_rate,
                "MCC": m.matthews_corrcoef,
                "Latency(ms)": m.latency_ms,
            })

        return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
