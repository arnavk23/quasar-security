import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from hyqad.simulation.qkd_channel import QKDChannelSimulator
from hyqad.simulation.attacks import AttackSimulator


class QKDDataGenerator:
    """
    End-to-end data generator that creates labeled QKD channel
    measurements for anomaly detection.
    """

    def __init__(
        self,
        channel_simulator: QKDChannelSimulator | None = None,
        attack_simulator: AttackSimulator | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.channel = channel_simulator or QKDChannelSimulator()
        self.attacker = attack_simulator or AttackSimulator()
        self.rng = rng or np.random.default_rng(42)

    def generate(
        self,
        n_samples: int = 5000,
        attack_ratio: float = 0.3,
        attack_types: list[str] | None = None,
        noise_scale: float = 0.05,
        return_channel_states: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, list]:
        if attack_types is None:
            attack_types = ["pns", "detector_blinding", "trojan_horse", "intercept_resend"]

        n_normal = int(n_samples * (1 - attack_ratio))
        n_attack = n_samples - n_normal

        normal_state = self.channel.simulate(
            n_normal, noise_scale=noise_scale, include_anomalies=False
        )

        attack_states = []
        attack_labels = []
        per_type = n_attack // len(attack_types)
        remainder = n_attack % len(attack_types)

        for i, atk in enumerate(attack_types):
            n_atk = per_type + (1 if i < remainder else 0)
            clean = self.channel.simulate(
                n_atk + 100, noise_scale=noise_scale, include_anomalies=False
            )
            attacked = self.attacker.apply_attack(clean, atk)
            for j in range(n_atk):
                attack_states.append((
                    attacked.time[j], attacked.qber[j],
                    attacked.photon_count_rate[j], attacked.detection_efficiency[j],
                    attacked.dark_count_rate[j], attacked.error_rate[j],
                    attacked.signal_attenuation[j], attacked.background_noise[j],
                ))
                attack_labels.append(1)

        normal_data = [(normal_state.time[i], normal_state.qber[i],
                        normal_state.photon_count_rate[i], normal_state.detection_efficiency[i],
                        normal_state.dark_count_rate[i], normal_state.error_rate[i],
                        normal_state.signal_attenuation[i], normal_state.background_noise[i])
                       for i in range(n_normal)]
        normal_labels = [0] * n_normal

        all_data = normal_data + attack_states
        all_labels = normal_labels + attack_labels

        cols = ["time", "qber", "photon_count_rate", "detection_efficiency",
                "dark_count_rate", "error_rate", "signal_attenuation", "background_noise"]
        df = pd.DataFrame(all_data, columns=cols)
        df["label"] = all_labels
        df["is_attack"] = df["label"].astype(bool)

        df = df.sample(frac=1, random_state=self.rng.integers(2**31)).reset_index(drop=True)

        if return_channel_states:
            return df, [normal_state]
        return df

    def generate_train_val_test(
        self, n_samples: int = 5000, attack_ratio: float = 0.3,
        train_split: float = 0.7, val_split: float = 0.15,
        random_state: int = 42, **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.generate(n_samples=n_samples, attack_ratio=attack_ratio, **kwargs)
        X = df.drop(columns=["label", "is_attack"])
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
