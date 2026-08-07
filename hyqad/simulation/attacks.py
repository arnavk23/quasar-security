import numpy as np
from numpy.typing import NDArray
from hyqad.simulation.qkd_channel import ChannelState


class AttackSimulator:
    """
    Simulates various attack vectors on QKD systems:
    - Photon-Number-Splitting (PNS) attack
    - Detector Blinding attack
    - Trojan Horse attack
    - Intercept-Resend attack
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng(42)

    def apply_pns_attack(
        self, state: ChannelState, split_fraction: float = 0.3
    ) -> ChannelState:
        n = len(state.qber)
        qber = state.qber.copy()
        photon_counts = state.photon_count_rate.copy()

        stolen_photons = photon_counts * split_fraction
        photon_counts -= stolen_photons
        photon_counts = np.clip(photon_counts, 0, None)

        qber_increase = self.rng.uniform(0.02, 0.08, n)
        qber = np.clip(qber + qber_increase, 0, 0.5)

        return ChannelState(
            time=state.time, qber=qber, photon_count_rate=photon_counts,
            detection_efficiency=state.detection_efficiency,
            dark_count_rate=state.dark_count_rate,
            error_rate=np.clip(state.error_rate + self.rng.uniform(0.01, 0.04, n), 0, 0.5),
            signal_attenuation=state.signal_attenuation + self.rng.uniform(0.5, 2.0, n),
            background_noise=state.background_noise,
        )

    def apply_detector_blinding(
        self, state: ChannelState, blinding_intensity: float = 1.5
    ) -> ChannelState:
        n = len(state.qber)
        qber = state.qber.copy()
        det_eff = state.detection_efficiency.copy()

        blinding_factor = 1.0 / (1.0 + blinding_intensity * self.rng.uniform(0.5, 1.5, n))
        det_eff = det_eff * blinding_factor
        det_eff = np.clip(det_eff, 0, 1)

        qber_boost = self.rng.uniform(0.05, 0.25, n)
        qber = np.clip(qber + qber_boost, 0, 0.5)

        dark_rate_boost = state.dark_count_rate * self.rng.uniform(1, 5, n)

        return ChannelState(
            time=state.time, qber=qber, photon_count_rate=state.photon_count_rate,
            detection_efficiency=det_eff, dark_count_rate=dark_rate_boost,
            error_rate=np.clip(state.error_rate + self.rng.uniform(0.03, 0.08, n), 0, 0.5),
            signal_attenuation=state.signal_attenuation,
            background_noise=state.background_noise * self.rng.uniform(1.0, 3.0, n),
        )

    def apply_trojan_horse(
        self, state: ChannelState, power_ratio: float = 0.05
    ) -> ChannelState:
        n = len(state.qber)
        bg_noise = state.background_noise.copy()
        qber = state.qber.copy()

        injected_power = power_ratio * self.rng.uniform(0.5, 2.0, n)
        bg_noise = bg_noise * (1 + injected_power * 10)

        qber_increase = self.rng.uniform(0.01, 0.06, n)
        qber = np.clip(qber + qber_increase, 0, 0.5)

        return ChannelState(
            time=state.time, qber=qber, photon_count_rate=state.photon_count_rate,
            detection_efficiency=state.detection_efficiency,
            dark_count_rate=state.dark_count_rate,
            error_rate=np.clip(state.error_rate + qber_increase * 0.5, 0, 0.5),
            signal_attenuation=state.signal_attenuation,
            background_noise=bg_noise,
        )

    def apply_intercept_resend(
        self, state: ChannelState
    ) -> ChannelState:
        n = len(state.qber)
        qber = state.qber.copy()
        error_rate = state.error_rate.copy()

        induced_qber = self.rng.uniform(0.1, 0.35, n)
        qber = np.clip(qber + induced_qber, 0, 0.5)
        error_rate = np.clip(error_rate + induced_qber * 0.3, 0, 0.5)

        return ChannelState(
            time=state.time, qber=qber,
            photon_count_rate=state.photon_count_rate * self.rng.uniform(0.6, 0.95, n),
            detection_efficiency=state.detection_efficiency,
            dark_count_rate=state.dark_count_rate * self.rng.uniform(1.0, 2.0, n),
            error_rate=error_rate,
            signal_attenuation=state.signal_attenuation + self.rng.uniform(0.2, 1.0, n),
            background_noise=state.background_noise * self.rng.uniform(1.0, 1.5, n),
        )

    def apply_attack(self, state: ChannelState, attack_type: str, **kwargs) -> ChannelState:
        methods = {
            "pns": self.apply_pns_attack,
            "detector_blinding": self.apply_detector_blinding,
            "trojan_horse": self.apply_trojan_horse,
            "intercept_resend": self.apply_intercept_resend,
        }
        if attack_type not in methods:
            raise ValueError(f"Unknown attack type: {attack_type}. Choose from {list(methods.keys())}")
        return methods[attack_type](state, **kwargs)
