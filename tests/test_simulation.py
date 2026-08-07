import numpy as np
import pytest
from hyqad.simulation import QKDChannelSimulator, AttackSimulator, QKDDataGenerator


def test_channel_simulation_basic():
    sim = QKDChannelSimulator(length_km=50, rng=np.random.default_rng(42))
    state = sim.simulate(100, noise_scale=0.05)
    assert len(state.qber) == 100
    assert np.all(state.qber >= 0)
    assert np.all(state.qber <= 0.5)
    assert np.all(state.photon_count_rate >= 0)


def test_channel_transmittance():
    sim = QKDChannelSimulator(length_km=10, fiber_loss_db_per_km=0.2)
    expected = 10 ** (-2.0 / 10)
    assert abs(sim.transmittance - expected) < 1e-10


def test_pns_attack():
    sim = QKDChannelSimulator(rng=np.random.default_rng(42))
    attacker = AttackSimulator(rng=np.random.default_rng(42))
    state = sim.simulate(50)
    attacked = attacker.apply_pns_attack(state, split_fraction=0.3)
    assert np.any(attacked.qber != state.qber)
    assert np.all(attacked.photon_count_rate <= state.photon_count_rate + 1e-10)


def test_detector_blinding():
    sim = QKDChannelSimulator(rng=np.random.default_rng(42))
    attacker = AttackSimulator(rng=np.random.default_rng(42))
    state = sim.simulate(50)
    attacked = attacker.apply_detector_blinding(state, blinding_intensity=1.5)
    assert np.any(attacked.detection_efficiency < state.detection_efficiency + 1e-10)


def test_trojan_horse():
    sim = QKDChannelSimulator(rng=np.random.default_rng(42))
    attacker = AttackSimulator(rng=np.random.default_rng(42))
    state = sim.simulate(50)
    attacked = attacker.apply_trojan_horse(state, power_ratio=0.05)
    assert np.any(attacked.background_noise > state.background_noise)


def test_intercept_resend():
    sim = QKDChannelSimulator(rng=np.random.default_rng(42))
    attacker = AttackSimulator(rng=np.random.default_rng(42))
    state = sim.simulate(50)
    attacked = attacker.apply_intercept_resend(state)
    assert np.any(attacked.qber > state.qber)


def test_all_attack_types():
    attacker = AttackSimulator(rng=np.random.default_rng(42))
    sim = QKDChannelSimulator(rng=np.random.default_rng(42))
    state = sim.simulate(20)
    for atk in ["pns", "detector_blinding", "trojan_horse", "intercept_resend"]:
        attacked = attacker.apply_attack(state, atk)
        assert attacked is not None


def test_data_generator():
    gen = QKDDataGenerator()
    df = gen.generate(n_samples=200, attack_ratio=0.3)
    assert len(df) == 200
    assert "label" in df.columns
    assert "is_attack" in df.columns
    assert 0.25 <= df["label"].mean() <= 0.35


def test_train_val_test_split():
    gen = QKDDataGenerator()
    train, val, test = gen.generate_train_val_test(
        n_samples=1000, attack_ratio=0.3, random_state=42
    )
    assert len(train) + len(val) + len(test) == 1000
    assert abs(train["label"].mean() - 0.3) <= 0.05
