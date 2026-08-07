from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ChannelConfig:
    length_km: float = 50.0
    fiber_loss_db_per_km: float = 0.2
    dark_count_prob: float = 1e-5
    detector_efficiency: float = 0.9
    misalignment_error: float = 0.01
    afterpulse_prob: float = 0.005
    background_noise_rate: float = 100.0


@dataclass
class AttackConfig:
    attack_types: list[str] = field(
        default_factory=lambda: ["pns", "detector_blinding", "trojan_horse", "intercept_resend"]
    )
    pns_photon_split_fraction: float = 0.3
    blinding_intensity: float = 1.5
    trojan_horse_power_ratio: float = 0.05


@dataclass
class FeatureConfig:
    encoding_dim: int = 8
    covariance_smoothing: float = 1e-6
    use_amplitude_encoding: bool = True
    use_covariance_features: bool = True


@dataclass
class ModelConfig:
    bayesian_prior_strength: float = 1.0
    quantum_kernel_type: Literal["gaussian", "angle", "amplitude"] = "amplitude"
    n_quantum_qubits: int = 3
    quantum_kernel_gamma: float = 1.0
    lr: float = 0.01
    max_iter: int = 500
    batch_size: int = 64
    n_mcmc_samples: int = 200
    n_restarts_optimizer: int = 5
    train_split: float = 0.7
    val_split: float = 0.15
    random_state: int = 42


@dataclass
class PipelineConfig:
    device: Literal["cpu", "cuda", "auto"] = "auto"
    n_jobs: int = -1
    low_latency_mode: bool = True
    use_quantum_kernel: bool = True
    use_bayesian_estimator: bool = True


@dataclass
class Config:
    n_samples: int = 5000
    attack_ratio: float = 0.3
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
