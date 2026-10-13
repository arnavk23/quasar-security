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
    Simulates per-block telemetry of a decoy-state BB84 link over fiber.

    Each sample is one key-generation block. Normal operation is not a single
    fixed operating point: the simulator applies slow operating-condition drift
    (polarization misalignment, fiber/connector loss, detector efficiency, dark
    counts, ambient background) whose magnitude is set by ``drift_scale``, and
    finite-size statistical noise on the measured error rates whose magnitude is
    set by ``block_size``. Without this drift every normal sample sits at
    essentially the same point and any attack perturbation becomes trivially
    separable.
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
        drift_scale: float = 1.0,
        block_size: int = 5000,
        rng: np.random.Generator | None = None,
    ):
        self.length_km = length_km
        self.fiber_loss_db_per_km = fiber_loss_db_per_km
        self.dark_count_prob = dark_count_prob
        self.detector_efficiency = detector_efficiency
        self.misalignment_error = misalignment_error
        self.afterpulse_prob = afterpulse_prob
        self.background_noise_rate = background_noise_rate
        self.drift_scale = drift_scale
        self.block_size = block_size
        self.rng = rng or np.random.default_rng(42)

    @property
    def total_loss_db(self) -> float:
        return self.fiber_loss_db_per_km * self.length_km

    @property
    def transmittance(self) -> float:
        return 10 ** (-self.total_loss_db / 10)

    def _gain_and_qber(self, mu, eta, tau, y0, e_d):
        """Decoy-state BB84 gain and QBER for a Poisson source (Lo, Ma & Chen 2005)."""
        signal = (1.0 - np.exp(-eta * tau * mu)) * (1.0 + self.afterpulse_prob)
        gain = y0 + signal
        qber = (0.5 * y0 + e_d * signal) / (gain + 1e-15)
        return gain, qber

    def simulate(
        self,
        n_samples: int,
        signal_intensity_range: tuple[float, float] = (0.1, 1.0),
        noise_scale: float = 0.05,
        include_anomalies: bool = False,
        anomaly_fraction: float = 0.0,
    ) -> ChannelState:
        rng = self.rng
        s = self.drift_scale
        times = rng.uniform(0, 1, n_samples)
        mu = rng.uniform(*signal_intensity_range, n_samples)

        # Slow operating-condition drift (per block).
        attenuation = self.total_loss_db + rng.normal(0, 0.3 * s, n_samples)
        tau = 10 ** (-attenuation / 10)
        eta = np.clip(self.detector_efficiency * (1 + rng.normal(0, 0.03 * s, n_samples)), 0.05, 1.0)
        e_d = self.misalignment_error * rng.lognormal(0, 0.4 * s, n_samples)
        y0 = self.dark_count_prob * rng.lognormal(0, 0.3 * s, n_samples)
        bg = self.background_noise_rate * rng.lognormal(0, 0.3 * s, n_samples)

        _, qber_true = self._gain_and_qber(mu, eta, tau, y0, e_d)

        # Finite-size estimation noise on the two error-rate estimates.
        se = np.sqrt(qber_true * (1 - qber_true) / self.block_size)
        qber = np.clip(qber_true + rng.normal(0, 1, n_samples) * se, 0, 0.5)
        error_rate = np.clip(qber_true + rng.normal(0, 1, n_samples) * se, 0, 0.5)

        photon_counts = eta * tau * mu
        photon_counts = np.clip(photon_counts * (1 + rng.normal(0, noise_scale, n_samples)), 0, None)
        det_eff = np.clip(eta + rng.normal(0, noise_scale * 0.02, n_samples), 0, 1)
        dark_counts = y0 * (1 + rng.normal(0, noise_scale, n_samples)).clip(0.1)
        att_meas = attenuation + rng.normal(0, noise_scale, n_samples)
        bg_noise = np.clip(bg * (1 + rng.normal(0, noise_scale, n_samples)), 0, None)

        if include_anomalies and anomaly_fraction > 0:
            n_anom = int(n_samples * anomaly_fraction)
            anom_idx = rng.choice(n_samples, n_anom, replace=False)
            qber[anom_idx] = rng.uniform(0.15, 0.5, n_anom)
            photon_counts[anom_idx] *= rng.uniform(0.1, 0.5, n_anom)

        return ChannelState(
            time=times,
            qber=qber,
            photon_count_rate=photon_counts,
            detection_efficiency=det_eff,
            dark_count_rate=dark_counts,
            error_rate=error_rate,
            signal_attenuation=att_meas,
            background_noise=bg_noise,
        )
