import numpy as np
import pytest
from hyqad.features import AmplitudeEncoder, CovarianceFeatureExtractor


def test_amplitude_encoding_shape():
    encoder = AmplitudeEncoder(encoding_dim=8)
    X = np.random.randn(10, 6)
    encoded = encoder.transform(X)
    assert encoded.shape == (10, 8)


def test_amplitude_encoding_normalization():
    encoder = AmplitudeEncoder(encoding_dim=8, norm=True)
    X = np.random.randn(5, 4)
    encoded = encoder.transform(X)
    norms = np.linalg.norm(encoded, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_amplitude_encoding_padding():
    encoder = AmplitudeEncoder(encoding_dim=8)
    X = np.random.randn(3, 10)
    encoded = encoder.transform(X)
    assert encoded.shape == (3, 8)


def test_amplitude_encoding_single_row():
    encoder = AmplitudeEncoder(encoding_dim=4)
    x = np.array([1.0, 2.0, 3.0])
    encoded = encoder.transform(x)
    assert encoded.shape == (1, 4)


def test_amplitude_kernel():
    encoder = AmplitudeEncoder(encoding_dim=4)
    X1 = np.random.randn(5, 3)
    X2 = np.random.randn(3, 3)
    K = encoder.kernel(X1, X2)
    assert K.shape == (5, 3)
    assert np.all(K >= 0)


def test_covariance_features_shape():
    extractor = CovarianceFeatureExtractor(smoothing=1e-6)
    X = np.random.randn(50, 4)
    extractor.fit(X)
    features = extractor.transform(X)
    n_expected = 4 + 4 + 1 + 1 + 1 + 1
    assert features.shape == (50, n_expected)


def test_covariance_uncertainty_score():
    extractor = CovarianceFeatureExtractor(smoothing=1e-6)
    X = np.random.randn(50, 4)
    extractor.fit(X)
    scores = extractor.get_uncertainty_score(X)
    assert scores.shape == (50,)
    assert np.all(scores >= 0)


def test_covariance_fit_transform_consistency():
    extractor = CovarianceFeatureExtractor(smoothing=1e-6)
    X_train = np.random.randn(100, 3)
    X_test = np.random.randn(10, 3)
    extractor.fit(X_train)
    features = extractor.transform(X_test)
    assert features.shape[0] == 10
