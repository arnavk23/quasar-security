#!/usr/bin/env python3
"""
Hybrid AI-Quantum Anomaly Detection Framework for QKD Systems
================================================================
End-to-end demonstration: generate data, train detector, evaluate.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter

from hyqad.simulation import QKDDataGenerator
from hyqad.features import AmplitudeEncoder, CovarianceFeatureExtractor
from hyqad.models import HybridAnomalyDetector, BayesianAnomalyEstimator, JSOOptimizer
from hyqad.pipeline import InferencePipeline
from hyqad.evaluation import Evaluator, BenchmarkSuite
from hyqad.config import Config
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "figures" / "hyqad"

sns.set_theme(style="whitegrid")


def main():
    print("=" * 72)
    print("Hybrid AI-Quantum Anomaly Detection Framework for QKD Systems")
    print("=" * 72)

    config = Config(
        n_samples=500,
        attack_ratio=0.3,
    )
    config.pipeline.use_quantum_kernel = False
    config.pipeline.use_bayesian_estimator = True
    config.features.use_covariance_features = False
    config.features.use_amplitude_encoding = False
    config.model.max_iter = 100
    config.model.n_restarts_optimizer = 1

    print(f"\n[1/6] Generating {config.n_samples} QKD channel samples "
          f"(attack ratio={config.attack_ratio})...")
    generator = QKDDataGenerator()
    train_df, val_df, test_df = generator.generate_train_val_test(
        n_samples=config.n_samples,
        attack_ratio=config.attack_ratio,
        train_split=config.model.train_split,
        val_split=config.model.val_split,
        random_state=config.model.random_state,
    )

    print(f"  Train: {len(train_df)} ({train_df['label'].sum()} attacks)")
    print(f"  Val:   {len(val_df)} ({val_df['label'].sum()} attacks)")
    print(f"  Test:  {len(test_df)} ({test_df['label'].sum()} attacks)")

    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print(f"\n[2/6] Quantum-inspired feature engineering...")
    amp_encoder = AmplitudeEncoder(encoding_dim=config.features.encoding_dim)
    X_train_amp = amp_encoder.transform(X_train)
    print(f"  Amplitude encoding: {X_train.shape[1]} -> {X_train_amp.shape[1]} dims")

    cov_extractor = CovarianceFeatureExtractor(smoothing=config.features.covariance_smoothing)
    cov_extractor.fit(X_train)
    unc_scores = cov_extractor.get_uncertainty_score(X_train)
    print(f"  Covariance features: {cov_extractor.transform(X_train).shape[1]} derived features")
    print(f"  Mean uncertainty score (train): {unc_scores.mean():.6f}")

    print(f"\n[3/6] Training hybrid quantum-classical detector...")
    pipeline = InferencePipeline(config)
    t0 = perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = perf_counter() - t0
    print(f"  Model fit complete in {fit_time:.2f}s")

    print(f"\n[4/6] Evaluating on test set...")
    evaluator = Evaluator()
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    latency = pipeline.avg_latency_ms

    metrics = evaluator.evaluate(y_test, y_pred, y_score=y_score, latency_ms=latency)
    print(f"  Accuracy:           {metrics.accuracy:.4f}")
    print(f"  Precision:          {metrics.precision:.4f}")
    print(f"  Recall:             {metrics.recall:.4f}")
    print(f"  F1 Score:           {metrics.f1_score:.4f}")
    print(f"  ROC-AUC:            {metrics.roc_auc:.4f}")
    print(f"  Avg Precision:      {metrics.average_precision:.4f}")
    print(f"  MCC:                {metrics.matthews_corrcoef:.4f}")
    print(f"  False Positive Rate: {metrics.false_positive_rate:.4f}")
    print(f"  True Positive Rate:  {metrics.true_positive_rate:.4f}")
    print(f"  Specificity:        {metrics.specificity:.4f}")
    print(f"  Avg Latency:        {latency:.3f} ms/sample")

    print(f"\n[5/6] Benchmarking against classical baselines...")
    benchmark = BenchmarkSuite(evaluator)
    results_df = benchmark.benchmark_all(
        X_train, y_train, X_test, y_test,
        include_hybrid=True, hybrid_detector=pipeline._pipeline.named_steps["detector"],
    )
    print(f"\n{results_df.to_string(index=False)}")

    print(f"\n[6/6] Generating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(y_score[y_test == 0], bins=30, alpha=0.6, label="Normal", density=True)
    ax.hist(y_score[y_test == 1], bins=30, alpha=0.6, label="Attack", density=True)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Class")
    ax.legend()

    ax = axes[0, 1]
    perf_df = results_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "F1", "ROC-AUC", "FPR"],
        var_name="Metric", value_name="Score"
    )
    sns.barplot(data=perf_df, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_title("Performance Comparison Across Models")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 0]
    train_qber = train_df[train_df["label"] == 0]["qber"].values
    test_qber_normal = test_df[test_df["label"] == 0]["qber"].values
    test_qber_attack = test_df[test_df["label"] == 1]["qber"].values
    ax.hist(train_qber, bins=30, alpha=0.5, label="Train (normal)", density=True)
    ax.hist(test_qber_normal, bins=30, alpha=0.5, label="Test (normal)", density=True)
    ax.hist(test_qber_attack, bins=30, alpha=0.5, label="Test (attack)", density=True)
    ax.set_xlabel("QBER")
    ax.set_ylabel("Density")
    ax.set_title("QBER Distribution")
    ax.legend()

    ax = axes[1, 1]
    _, _, uncertainties = pipeline.predict_with_uncertainty(X_test)
    ax.scatter(y_score, uncertainties, c=y_test, cmap="coolwarm", alpha=0.5, s=10)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Predictive Uncertainty")
    ax.set_title("Uncertainty vs. Score (color=ground truth)")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Attack")

    plt.tight_layout()
    plt.savefig(str(FIGS / "hyqad_results.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {FIGS / 'hyqad_results.png'}")

    best_f1 = results_df.iloc[0]
    print(f"\n{'=' * 72}")
    print(f"Summary: Best model = {best_f1['Model']} "
          f"(F1={best_f1['F1']:.4f}, AUC={best_f1['ROC-AUC']:.4f})")
    print(f"         Hybrid detector F1={metrics.f1_score:.4f}, "
          f"AUC={metrics.roc_auc:.4f}, FPR={metrics.false_positive_rate:.4f}")
    print(f"         Latency={latency:.3f} ms/sample")
    print(f"{'=' * 72}")
    print("Done.")


if __name__ == "__main__":
    main()
