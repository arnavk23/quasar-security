import numpy as np
from hyqad.simulation.qkd_channel import ChannelState


class AttackSimulator:
    """
    Simulates telemetry signatures of four QKD implementation attacks:
    - Photon-Number-Splitting (PNS)
    - Detector Blinding
    - Trojan Horse
    - Intercept-Resend

    Perturbation magnitudes are modeling assumptions, not experimentally
    calibrated values. Every deviation from the clean state is multiplied by
    ``strength`` (1.0 = default magnitudes, 0 = no attack signature), so the
    same models can be swept from blatant to subtle.
    """

    def __init__(self, rng: np.random.Generator | None = None, strength: float = 1.0):
        self.rng = rng or np.random.default_rng(42)
        self.strength = strength

    def apply_pns_attack(
        self, state: ChannelState, split_fraction: float = 0.3
    ) -> ChannelState:
        n = len(state.qber)
        s = self.strength
        photon_counts = np.clip(state.photon_count_rate * (1 - split_fraction * s), 0, None)
        qber = np.clip(state.qber + s * self.rng.uniform(0.02, 0.08, n), 0, 0.5)
        return ChannelState(
            time=state.time, qber=qber, photon_count_rate=photon_counts,
            detection_efficiency=state.detection_efficiency,
            dark_count_rate=state.dark_count_rate,
            error_rate=np.clip(state.error_rate + s * self.rng.uniform(0.01, 0.04, n), 0, 0.5),
            signal_attenuation=state.signal_attenuation + s * self.rng.uniform(0.5, 2.0, n),
            background_noise=state.background_noise,
        )

    def apply_detector_blinding(
        self, state: ChannelState, blinding_intensity: float = 1.5
    ) -> ChannelState:
        n = len(state.qber)
        s = self.strength
        blinding_factor = 1.0 / (1.0 + s * blinding_intensity * self.rng.uniform(0.5, 1.5, n))
        det_eff = np.clip(state.detection_efficiency * blinding_factor, 0, 1)
        qber = np.clip(state.qber + s * self.rng.uniform(0.05, 0.25, n), 0, 0.5)
        return ChannelState(
            time=state.time, qber=qber, photon_count_rate=state.photon_count_rate,
            detection_efficiency=det_eff,
            dark_count_rate=state.dark_count_rate * (1 + s * self.rng.uniform(0, 4, n)),
            error_rate=np.clip(state.error_rate + s * self.rng.uniform(0.03, 0.08, n), 0, 0.5),
            signal_attenuation=state.signal_attenuation,
            background_noise=state.background_noise * (1 + s * self.rng.uniform(0, 2, n)),
        )

    def apply_trojan_horse(
        self, state: ChannelState, power_ratio: float = 0.05
    ) -> ChannelState:
        n = len(state.qber)
        s = self.strength
        injected_power = power_ratio * self.rng.uniform(0.5, 2.0, n)
        bg_noise = state.background_noise * (1 + s * injected_power * 10)
        qber_increase = s * self.rng.uniform(0.01, 0.06, n)
        return ChannelState(
            time=state.time, qber=np.clip(state.qber + qber_increase, 0, 0.5),
            photon_count_rate=state.photon_count_rate,
            detection_efficiency=state.detection_efficiency,
            dark_count_rate=state.dark_count_rate,
            error_rate=np.clip(state.error_rate + qber_increase * 0.5, 0, 0.5),
            signal_attenuation=state.signal_attenuation,
            background_noise=bg_noise,
        )

    def apply_intercept_resend(self, state: ChannelState) -> ChannelState:
        n = len(state.qber)
        s = self.strength
        induced_qber = s * self.rng.uniform(0.1, 0.35, n)
        return ChannelState(
            time=state.time,
            qber=np.clip(state.qber + induced_qber, 0, 0.5),
            photon_count_rate=state.photon_count_rate * (1 - s * self.rng.uniform(0.05, 0.4, n)),
            detection_efficiency=state.detection_efficiency,
            dark_count_rate=state.dark_count_rate * (1 + s * self.rng.uniform(0, 1, n)),
            error_rate=np.clip(state.error_rate + induced_qber * 0.3, 0, 0.5),
            signal_attenuation=state.signal_attenuation + s * self.rng.uniform(0.2, 1.0, n),
            background_noise=state.background_noise * (1 + s * self.rng.uniform(0, 0.5, n)),
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
