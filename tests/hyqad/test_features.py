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


def test_amplitude_encoding_never_truncates():
    X = np.random.randn(3, 10)
    assert AmplitudeEncoder().transform(X).shape == (3, 16)
    with pytest.raises(ValueError):
        AmplitudeEncoder(encoding_dim=8).transform(X)


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
    assert np.all(K >= 0) and np.all(K <= 1 + 1e-12)
    assert np.allclose(np.diag(encoder.kernel(X1, X1)), 1.0)


def test_covariance_features_shape():
    extractor = CovarianceFeatureExtractor(smoothing=1e-6)
    X = np.random.randn(50, 4)
    extractor.fit(X)
    features = extractor.transform(X)
    assert features.shape == (50, 5)
    assert np.allclose(features[:, :4], X)


def test_covariance_features_are_order_independent():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    extractor = CovarianceFeatureExtractor().fit(X)
    perm = rng.permutation(60)
    assert np.allclose(extractor.transform(X)[perm], extractor.transform(X[perm]))


def test_covariance_fit_uses_normal_class_only():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(100, 3)), rng.normal(5, 1, size=(30, 3))])
    y = np.r_[np.zeros(100), np.ones(30)]
    extractor = CovarianceFeatureExtractor().fit(X, y)
    d = extractor.transform(X)[:, -1]
    assert d[y == 1].mean() > 3 * d[y == 0].mean()


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
