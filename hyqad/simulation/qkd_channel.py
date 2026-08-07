import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class ChannelState:
    """Represents a snapshot of QKD channel measurements."""
    time: NDArray[np.float64]
    qber: NDArray[np.float64]
    photon_count_rate: NDArray[np.float64]
    detection_efficiency: NDArray[np.float64]
    dark_count_rate: NDArray[np.float64]
    error_rate: NDArray[np.float64]
    signal_attenuation: NDArray[np.float64]
    background_noise: NDArray[np.float64]


class QKDChannelSimulator:
    """
    Simulates a realistic Quantum Key Distribution channel
    incorporating quantum channel noise, photon loss, and
    environmental disturbances.
    """

    def __init__(
        self,
        length_km: float = 50.0,
        fiber_loss_db_per_km: float = 0.2,
        dark_count_prob: float = 1e-5,
        detector_efficiency: float = 0.9,
        misalignment_error: float = 0.01,
        afterpulse_prob: float = 0.005,
        background_noise_rate: float = 100.0,
        rng: np.random.Generator | None = None,
    ):
        self.length_km = length_km
        self.fiber_loss_db_per_km = fiber_loss_db_per_km
        self.dark_count_prob = dark_count_prob
        self.detector_efficiency = detector_efficiency
        self.misalignment_error = misalignment_error
        self.afterpulse_prob = afterpulse_prob
        self.background_noise_rate = background_noise_rate
        self.rng = rng or np.random.default_rng(42)

    @property
    def total_loss_db(self) -> float:
        return self.fiber_loss_db_per_km * self.length_km

    @property
    def transmittance(self) -> float:
        return 10 ** (-self.total_loss_db / 10)

    def _compute_qber(self, signal_intensity: float) -> float:
        tau = self.transmittance
        Y0 = self.dark_count_prob
        ed = self.misalignment_error
        mu = signal_intensity

        Y0_term = 0.5 * Y0
        signal_term = 1 - (1 - self.detector_efficiency * tau) ** mu
        gain = Y0_term + signal_term * (1 + self.afterpulse_prob)

        eq = 0.5
        qber_num = Y0_term * eq + signal_term * ed * (1 + self.afterpulse_prob)
        qber_den = gain + 1e-15
        return qber_num / qber_den

    def _compute_photon_count_rate(self, signal_intensity: float) -> float:
        tau = self.transmittance
        mu = signal_intensity
        return self.detector_efficiency * tau * mu

    def simulate(
        self,
        n_samples: int,
        signal_intensity_range: tuple[float, float] = (0.1, 1.0),
        noise_scale: float = 0.05,
        include_anomalies: bool = False,
        anomaly_fraction: float = 0.0,
    ) -> ChannelState:
        times = np.sort(self.rng.uniform(0, 1, n_samples))
        signal_intensities = self.rng.uniform(*signal_intensity_range, n_samples)

        qber = np.array([self._compute_qber(mu) for mu in signal_intensities])
        photon_counts = np.array([self._compute_photon_count_rate(mu) for mu in signal_intensities])

        qber += self.rng.normal(0, noise_scale * 0.1, n_samples)
        qber = np.clip(qber, 0, 0.5)
        photon_counts += self.rng.normal(0, noise_scale * photon_counts, n_samples)
        photon_counts = np.clip(photon_counts, 0, None)

        det_eff = self.detector_efficiency + self.rng.normal(0, noise_scale * 0.02, n_samples)
        det_eff = np.clip(det_eff, 0, 1)

        dark_counts = self.dark_count_prob + self.rng.exponential(
            self.dark_count_prob * noise_scale, n_samples
        )
        error_rates = self.misalignment_error + self.rng.normal(0, noise_scale * 0.005, n_samples)
        error_rates = np.clip(error_rates, 0, 0.5)

        attenuation = self.total_loss_db + self.rng.normal(0, noise_scale * 0.1, n_samples)
        bg_noise = self.background_noise_rate + self.rng.normal(
            0, noise_scale * self.background_noise_rate, n_samples
        )
        bg_noise = np.clip(bg_noise, 0, None)

        if include_anomalies and anomaly_fraction > 0:
            n_anom = int(n_samples * anomaly_fraction)
            anom_idx = self.rng.choice(n_samples, n_anom, replace=False)
            qber[anom_idx] = self.rng.uniform(0.15, 0.5, n_anom)
            photon_counts[anom_idx] *= self.rng.uniform(0.1, 0.5, n_anom)

        return ChannelState(
            time=times,
            qber=qber,
            photon_count_rate=photon_counts,
            detection_efficiency=det_eff,
            dark_count_rate=dark_counts,
            error_rate=error_rates,
            signal_attenuation=attenuation,
            background_noise=bg_noise,
        )
