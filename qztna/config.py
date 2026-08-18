from dataclasses import dataclass, field


@dataclass
class NetworkConfig:
    n_features: int = 64
    n_header: int = 16
    n_flow: int = 24
    n_behavioral: int = 24
    within_group_corr: float = 0.35
    cross_group_corr: float = 0.10


@dataclass
class AttackConfig:
    attack_types: list[str] = field(
        default_factory=lambda: ["port_scan", "ddos", "exfiltration", "dns_tunneling", "arp_spoofing"]
    )
    attack_strength: float = 1.0
    noise_scale: float = 0.5


@dataclass
class DataConfig:
    n_samples: int = 20000
    attack_ratio: float = 0.2
    train_split: float = 0.7
    val_split: float = 0.15
    random_state: int = 42
