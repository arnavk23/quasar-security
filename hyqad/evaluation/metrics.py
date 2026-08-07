import time
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score,
)


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test for paired binary classification comparison.

    Tests whether two classifiers disagree significantly on the same test set.
    H0: the two classifiers have the same error rate.

    Returns:
        statistic: chi-squared statistic
        p_value: two-sided p-value
        contingency: 2x2 table [[both_correct, a_wrong_b_correct],
                                 [a_correct_b_wrong, both_wrong]]
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    b01 = int(np.sum(~correct_a & correct_b))
    b10 = int(np.sum(correct_a & ~correct_b))

    contingency = np.array([
        [int(np.sum(correct_a & correct_b)), b01],
        [b10, int(np.sum(~correct_a & ~correct_b))],
    ])

    if b01 + b10 == 0:
        return 0.0, 1.0, contingency

    if b01 + b10 < 25:
        statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    else:
        statistic = (b01 - b10) ** 2 / (b01 + b10)

    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return float(statistic), float(p_value), contingency


def paired_bootstrap_test(y_true, scores_a, scores_b, y_pred_a, y_pred_b,
                           n_bootstrap=10000, metric="accuracy", rng_seed=42):
    """
    Paired bootstrap resampling test for comparing two classifiers.

    Resamples the test set with replacement and computes the difference
    in the chosen metric at each resample. Returns the fraction of
    resamples where model B outperforms model A.

    Returns:
        diff_mean: mean accuracy difference (B - A) across bootstrap samples
        diff_ci_95: 95% CI of the difference [lower, upper]
        p_value_b_better: fraction of resamples where B > A
    """
    rng = np.random.default_rng(rng_seed)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)

    metric_fn = {
        "accuracy": accuracy_score,
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
    }[metric]

    diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ya = y_pred_a[idx]
        yb = y_pred_b[idx]
        ma = metric_fn(yt, ya)
        mb = metric_fn(yt, yb)
        diffs[i] = mb - ma

    diff_mean = float(np.mean(diffs))
    diff_ci = [float(np.percentile(diffs, 2.5)),
               float(np.percentile(diffs, 97.5))]
    p_b_better = float(np.mean(diffs > 0))

    return diff_mean, diff_ci, p_b_better


@dataclass
class DetectionMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    average_precision: float = 0.0
    matthews_corrcoef: float = 0.0
    balanced_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    true_positive_rate: float = 0.0
    specificity: float = 0.0
    latency_ms: float = 0.0
    n_samples: int = 0
    n_attacks: int = 0
    per_attack_metrics: dict = field(default_factory=dict)


class Evaluator:
    """
    Comprehensive evaluation suite for QKD anomaly detection.
    Computes detection accuracy, false positive rate, ROC AUC,
    latency, and attack-type breakdown.
    """

    def __init__(self):
        self.metrics_history: list[dict] = []

    def evaluate(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        y_score: NDArray | None = None,
        latency_ms: float = 0.0,
        attack_types: list[str] | None = None,
    ) -> DetectionMetrics:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        metrics = DetectionMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            matthews_corrcoef=matthews_corrcoef(y_true, y_pred),
            balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
            latency_ms=latency_ms,
            n_samples=len(y_true),
            n_attacks=int(y_true.sum()),
        )

        if y_score is not None:
            try:
                metrics.roc_auc = roc_auc_score(y_true, y_score)
            except Exception:
                metrics.roc_auc = 0.5
            try:
                metrics.average_precision = average_precision_score(y_true, y_score)
            except Exception:
                metrics.average_precision = 0.0

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics.true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return metrics

    def evaluate_with_attack_breakdown(
        self,
        df,
        y_pred_col: str = "prediction",
        y_true_col: str = "label",
        attack_type_col: str | None = None,
    ) -> DetectionMetrics:
        y_true = df[y_true_col].values
        y_pred = df[y_pred_col].values

        metrics = self.evaluate(y_true, y_pred)

        if attack_type_col and attack_type_col in df.columns:
            per_attack = {}
            for atk_type in df[attack_type_col].unique():
                if atk_type == "normal":
                    continue
                mask = df[attack_type_col] == atk_type
                sub_true = y_true[mask]
                sub_pred = y_pred[mask]
                if len(sub_true) > 0:
                    per_attack[atk_type] = {
                        "accuracy": accuracy_score(sub_true, sub_pred),
                        "recall": recall_score(sub_true, sub_pred, zero_division=0),
                        "precision": precision_score(sub_true, sub_pred, zero_division=0),
                        "n_samples": int(mask.sum()),
                    }
            metrics.per_attack_metrics = per_attack

        return metrics

    def record(self, name: str, metrics: DetectionMetrics):
        self.metrics_history.append({"name": name, **metrics.__dict__})

    def summary(self) -> str:
        if not self.metrics_history:
            return "No evaluations recorded."
        lines = []
        for entry in self.metrics_history:
            name = entry["name"]
            acc = entry.get("accuracy", 0)
            f1 = entry.get("f1_score", 0)
            auc = entry.get("roc_auc", 0)
            fpr = entry.get("false_positive_rate", 0)
            lat = entry.get("latency_ms", 0)
            lines.append(
                f"  {name:30s}  Acc={acc:.4f}  F1={f1:.4f}  "
                f"AUC={auc:.4f}  FPR={fpr:.4f}  Lat={lat:.2f}ms"
            )
        return "\n".join(lines)

    def compare_classifiers(self, y_true, y_pred_a, y_pred_b,
                             model_a_name="Model A", model_b_name="Model B",
                             y_score_a=None, y_score_b=None):
        """
        Comprehensive paired comparison between two classifiers on the same test set.
        Returns a dict with McNemar's test and paired bootstrap results.
        """
        y_true = np.asarray(y_true)

        # McNemar's test
        chi2, p_mcnemar, contingency = mcnemar_test(y_true, y_pred_a, y_pred_b)

        # Paired bootstrap on accuracy
        diff_acc, ci_acc, p_acc = paired_bootstrap_test(
            y_true, y_score_a, y_score_b, y_pred_a, y_pred_b,
            n_bootstrap=10000, metric="accuracy"
        )

        # Paired bootstrap on F1
        diff_f1, ci_f1, p_f1 = paired_bootstrap_test(
            y_true, y_score_a, y_score_b, y_pred_a, y_pred_b,
            n_bootstrap=10000, metric="f1"
        )

        return {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "mcnemar_chi2": chi2,
            "mcnemar_p": p_mcnemar,
            "mcnemar_contingency": contingency,
            "acc_diff_mean": diff_acc,
            "acc_diff_ci_95": ci_acc,
            "acc_p_b_better": p_acc,
            "f1_diff_mean": diff_f1,
            "f1_diff_ci_95": ci_f1,
            "f1_p_b_better": p_f1,
            "n_test": len(y_true),
        }

    def benchmark_latency(self, model, X: NDArray, n_runs: int = 50) -> dict:
        latencies = []
        for _ in range(5):
            model.predict(X[:10])
        for _ in range(n_runs):
            start = time.perf_counter()
            model.predict(X)
            latencies.append((time.perf_counter() - start) * 1000 / len(X))
        return {
            "latency_per_sample_ms": float(np.mean(latencies)),
            "latency_std_ms": float(np.std(latencies)),
            "throughput_samples_per_sec": float(1000 / np.mean(latencies)),
        }
