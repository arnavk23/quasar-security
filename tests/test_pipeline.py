import numpy as np
import pytest
from sklearn.datasets import make_classification
from hyqad.pipeline import InferencePipeline
from hyqad.config import Config


@pytest.fixture
def pipeline_data():
    X, y = make_classification(
        n_samples=200, n_features=8, n_informative=5,
        weights=[0.7, 0.3], random_state=42,
    )
    return X, y


def test_pipeline_fit_predict(pipeline_data):
    X, y = pipeline_data
    config = Config()
    config.features.use_covariance_features = True
    config.features.use_amplitude_encoding = False
    config.pipeline.use_quantum_kernel = False
    config.pipeline.use_bayesian_estimator = True

    pipe = InferencePipeline(config)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (200,)


def test_pipeline_probabilities(pipeline_data):
    X, y = pipeline_data
    config = Config()
    config.features.use_covariance_features = False
    config.features.use_amplitude_encoding = False
    config.pipeline.use_quantum_kernel = False
    config.pipeline.use_bayesian_estimator = True

    pipe = InferencePipeline(config)
    pipe.fit(X[:100], y[:100])
    proba = pipe.predict_proba(X[100:])
    assert proba.shape[1] == 2
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_uncertainty(pipeline_data):
    X, y = pipeline_data
    config = Config()
    config.features.use_covariance_features = False
    config.features.use_amplitude_encoding = False
    config.pipeline.use_quantum_kernel = False
    config.pipeline.use_bayesian_estimator = True

    pipe = InferencePipeline(config)
    pipe.fit(X[:100], y[:100])
    preds, scores, uncert = pipe.predict_with_uncertainty(X[100:])
    assert np.all(uncert >= 0)
    assert np.all(uncert <= 1)


def test_pipeline_latency_tracking(pipeline_data):
    X, y = pipeline_data
    config = Config()
    config.features.use_covariance_features = False
    config.features.use_amplitude_encoding = False
    config.pipeline.use_quantum_kernel = False
    config.pipeline.use_bayesian_estimator = True

    pipe = InferencePipeline(config)
    pipe.fit(X[:50], y[:50])
    pipe.predict(X[50:])
    pipe.predict(X[50:])
    assert pipe.avg_latency_ms > 0
    assert pipe.p95_latency_ms > 0


def test_pipeline_benchmark(pipeline_data):
    X, y = pipeline_data
    config = Config()
    config.features.use_covariance_features = False
    config.features.use_amplitude_encoding = False
    config.pipeline.use_quantum_kernel = False
    config.pipeline.use_bayesian_estimator = True

    pipe = InferencePipeline(config)
    pipe.fit(X[:50], y[:50])
    results = pipe.benchmark_throughput(X[50:], n_runs=5)
    assert results["throughput_mean"] > 0
    assert results["latency_ms_mean"] > 0
