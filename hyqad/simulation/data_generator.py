import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from hyqad.simulation.qkd_channel import QKDChannelSimulator
from hyqad.simulation.attacks import AttackSimulator

FEATURE_COLS = ["qber", "photon_count_rate", "detection_efficiency", "dark_count_rate",
                "error_rate", "signal_attenuation", "background_noise"]
ATTACK_TYPES = ["pns", "detector_blinding", "trojan_horse", "intercept_resend"]


class QKDDataGenerator:
    """
    End-to-end data generator that creates labeled QKD channel
    measurements for anomaly detection.

    The ``rng`` drives every random draw (channel drift, measurement noise,
    attack perturbations, shuffling), so different seeds give genuinely
    different datasets. Custom simulators passed in keep their own RNGs.
    """

    def __init__(
        self,
        channel_simulator: QKDChannelSimulator | None = None,
        attack_simulator: AttackSimulator | None = None,
        rng: np.random.Generator | None = None,
        attack_strength: float = 1.0,
        drift_scale: float = 1.0,
    ):
        self.rng = rng or np.random.default_rng(42)
        self.channel = channel_simulator or QKDChannelSimulator(rng=self.rng, drift_scale=drift_scale)
        self.attacker = attack_simulator or AttackSimulator(rng=self.rng, strength=attack_strength)

    def generate(
        self,
        n_samples: int = 5000,
        attack_ratio: float = 0.3,
        attack_types: list[str] | None = None,
        noise_scale: float = 0.05,
        return_channel_states: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, list]:
        attack_types = attack_types or ATTACK_TYPES

        n_normal = int(n_samples * (1 - attack_ratio))
        n_attack = n_samples - n_normal

        normal_state = self.channel.simulate(n_normal, noise_scale=noise_scale)
        cols = ["time"] + FEATURE_COLS

        def to_frame(state, label, atk):
            df = pd.DataFrame({c: getattr(state, c) for c in cols})
            df["label"] = label
            df["attack_type"] = atk
            return df

        frames = [to_frame(normal_state, 0, "normal")]
        per_type = n_attack // len(attack_types)
        remainder = n_attack % len(attack_types)
        for i, atk in enumerate(attack_types):
            n_atk = per_type + (1 if i < remainder else 0)
            if n_atk == 0:
                continue
            clean = self.channel.simulate(n_atk, noise_scale=noise_scale)
            frames.append(to_frame(self.attacker.apply_attack(clean, atk), 1, atk))

        df = pd.concat(frames, ignore_index=True)
        df["is_attack"] = df["label"].astype(bool)
        df = df.sample(frac=1, random_state=int(self.rng.integers(2**31))).reset_index(drop=True)

        if return_channel_states:
            return df, [normal_state]
        return df

    def generate_train_val_test(
        self, n_samples: int = 5000, attack_ratio: float = 0.3,
        train_split: float = 0.7, val_split: float = 0.15,
        random_state: int = 42, **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.generate(n_samples=n_samples, attack_ratio=attack_ratio, **kwargs)
        X = df.drop(columns=["label", "is_attack", "attack_type"])
        y = df["label"]

        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=1 - train_split, random_state=random_state, stratify=y
        )
        val_frac = val_split / (1 - train_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=val_frac, random_state=random_state, stratify=y_tv
        )

        train_df = X_train.copy()
        train_df["label"] = y_train.values
        val_df = X_val.copy()
        val_df["label"] = y_val.values
        test_df = X_test.copy()
        test_df["label"] = y_test.values

        return train_df, val_df, test_df
