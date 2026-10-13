import numpy as np
import pytest
from qztna.simulation import NetworkFlowGenerator
from qztna.features import amplitude_kernel, kernel_target_alignment
from qztna.models.kernel_svm import fit_kernel_svm, KERNEL_FUNCS
from qztna.certification import certify_point


def test_flow_generator_shape_and_labels():
    gen = NetworkFlowGenerator(seed=0)
    df = gen.generate_dataset(n_samples=200, attack_ratio=0.2, random_state=0)
    assert df.shape[0] == 200
    assert set(df["label"].unique()) <= {0, 1}
    assert df["attack_type"].nunique() >= 2


def test_flow_generator_attack_dims_within_bounds():
    gen = NetworkFlowGenerator(n_features=64, n_header=16, n_flow=24, n_behavioral=24, seed=0)
    for dims in gen.attack_dims.values():
        assert all(0 <= d < 64 for d in dims)
        assert len(dims) > 0


def test_amplitude_kernel_properties():
    X = np.random.randn(5, 8)
    K = amplitude_kernel(X, X)
    assert K.shape == (5, 5)
    assert np.all(K >= -1e-9) and np.all(K <= 1 + 1e-9)
    assert np.allclose(np.diag(K), 1.0, atol=1e-9)
    assert np.allclose(K, K.T, atol=1e-9)


def test_kernel_target_alignment_perfect_separation():
    rng = np.random.default_rng(0)
    X_pos = rng.normal(loc=5, scale=0.1, size=(20, 4))
    X_neg = rng.normal(loc=-5, scale=0.1, size=(20, 4))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 20 + [0] * 20)
    K = KERNEL_FUNCS["rbf"](X, X)
    align = kernel_target_alignment(K, y)
    assert align > 0.3


def test_fit_kernel_svm_exact_and_nystrom():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 6))
    y = (X[:, 0] > 0).astype(int)
    model_exact, _ = fit_kernel_svm("ae", X, y, random_state=0, force_exact=True)
    assert model_exact.predict(X).shape == (100,)
    model_nystrom, _ = fit_kernel_svm("rbf", X, y, random_state=0, force_nystrom=True, max_landmarks=20)
    assert model_nystrom.predict(X).shape == (100,)


def test_certify_point_returns_valid_radius():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    model, _ = fit_kernel_svm("ae", X, y, random_state=0, force_exact=True)
    c_a, radius = certify_point(model, X[0], sigma=0.25, n0=20, n=50, rng=rng)
    assert radius >= 0.0


def test_amplitude_kernel_is_norm_and_sign_invariant():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(5, 7)), rng.normal(size=(4, 7))
    K = amplitude_kernel(X, Y)
    assert np.allclose(amplitude_kernel(3.0 * X, Y), K)
    assert np.allclose(amplitude_kernel(-X, Y), K)


def test_norm_anchor_breaks_norm_and_sign_invariance():
    from qztna.features.amplitude_kernel import NormAnchor
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    na = NormAnchor().fit(X)
    assert np.isclose(na.c_, np.median(np.linalg.norm(X, axis=1)))
    x = X[:1]
    k_scaled = amplitude_kernel(na.transform(x), na.transform(3.0 * x))[0, 0]
    k_flipped = amplitude_kernel(na.transform(x), na.transform(-x))[0, 0]
    assert k_scaled < 0.999 and k_flipped < 0.999


def test_deep_baselines_fit_and_score():
    from qztna.models.deep import AutoencoderDetector, DNNClassifier
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(300, 5)), rng.normal(3, 1, size=(100, 5))]).astype(np.float32)
    y = np.r_[np.zeros(300, int), np.ones(100, int)]
    dnn = DNNClassifier(epochs=5, random_state=0).fit(X, y)
    assert dnn.predict_proba(X).shape == (400, 2)
    assert (dnn.predict(X) == y).mean() > 0.9
    ae = AutoencoderDetector(epochs=10, random_state=0).fit(X, y)
    s = ae.score(X)
    assert s[y == 1].mean() > s[y == 0].mean()


def test_real_dataset_loaders_if_present():
    from qztna.data.real_datasets import EXTERNAL, load_unsw
    if not (EXTERNAL / "unsw" / "UNSW_NB15_training-set.csv").exists():
        pytest.skip("UNSW-NB15 not downloaded (scripts/fetch_datasets.sh)")
    d = load_unsw()
    assert d["X_train"].shape == (175341, 68) and d["X_test"].shape == (82332, 68)
    assert np.isfinite(d["X_train"]).all()
    assert set(np.unique(d["fam_train"])) >= {"Normal", "Generic", "Exploits", "Worms"}
