"""
Loaders for the public network-intrusion benchmarks used in the Q-ZTNA
real-data evaluation. Raw files are fetched by scripts/qztna/fetch_datasets.sh
into data/external/ (git-ignored); processed arrays are cached as .npz.

UNSW-NB15 (Moustafa & Slay, 2015): the official training/testing partition
(175,341 / 82,332 records). ``id`` is dropped; ``proto``/``service``/``state``
are one-hot encoded (top categories in the training partition, rest -> other).

CIC-IDS2017 (Sharafaldin et al., 2018): the eight MachineLearningCVE day files
(2,830,743 flows). Rows with NaN/inf are dropped, exact duplicate rows are
removed, constant and duplicated columns are dropped, and classes with fewer
than 100 flows after de-duplication (Infiltration, Heartbleed, Web Attack SQL
Injection) are removed. There is no official split, so a stratified 70/30
train/test split with a fixed seed is used.

Both: every numeric feature x is mapped to sign(x) * log1p(|x|) to tame the
heavy tails of byte/packet counts. Standardization is left to the experiment
code so that it is always fit on the training split only.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "data" / "external"

UNSW_FAMILIES = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
                 "Analysis", "Backdoor", "Shellcode", "Worms"]

CICIDS_FAMILY = {
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS", "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS", "PortScan": "PortScan", "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Bot": "Bot", "Web Attack Brute Force": "WebAttack", "Web Attack XSS": "WebAttack",
}
CICIDS_FAMILIES = ["DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot"]


def _signed_log(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


def _one_hot(train: pd.Series, test: pd.Series, top: int):
    keep = train.value_counts().index[:top]
    cats = list(keep) + ["__other__"]
    tr = pd.Categorical(train.where(train.isin(keep), "__other__"), categories=cats)
    te = pd.Categorical(test.where(test.isin(keep), "__other__"), categories=cats)
    return (pd.get_dummies(tr, prefix=train.name).values.astype(float),
            pd.get_dummies(te, prefix=train.name).values.astype(float),
            [f"{train.name}={c}" for c in cats])


def load_unsw(cache: bool = True):
    """Returns dict with X_train, y_train, fam_train, X_test, y_test, fam_test, features."""
    path = EXTERNAL / "unsw_processed.npz"
    if cache and path.exists():
        return dict(np.load(path, allow_pickle=True))
    tr = pd.read_csv(EXTERNAL / "unsw" / "UNSW_NB15_training-set.csv")
    te = pd.read_csv(EXTERNAL / "unsw" / "UNSW_NB15_testing-set.csv")
    tr.columns = [c.strip().lstrip("﻿") for c in tr.columns]
    te.columns = tr.columns
    cat_cols = ["proto", "service", "state"]
    num_cols = [c for c in tr.columns if c not in cat_cols + ["id", "attack_cat", "label"]]
    Xtr, Xte = [_signed_log(tr[num_cols].values.astype(float))], [_signed_log(te[num_cols].values.astype(float))]
    feats = list(num_cols)
    for c, top in zip(cat_cols, (10, 10, 6)):
        a, b, names = _one_hot(tr[c].astype(str), te[c].astype(str), top)
        Xtr.append(a); Xte.append(b); feats += names
    out = dict(
        X_train=np.hstack(Xtr), y_train=tr["label"].values.astype(int),
        fam_train=np.where(tr["label"] == 1, tr["attack_cat"].str.strip(), "Normal").astype(str),
        X_test=np.hstack(Xte), y_test=te["label"].values.astype(int),
        fam_test=np.where(te["label"] == 1, te["attack_cat"].str.strip(), "Normal").astype(str),
        features=np.array(feats),
    )
    if cache:
        np.savez_compressed(path, **out)
    return out


def load_cicids2017(cache: bool = True, split_seed: int = 0):
    """Returns dict with X_train, y_train, fam_train, X_test, y_test, fam_test, features."""
    path = EXTERNAL / f"cicids2017_processed_s{split_seed}.npz"
    if cache and path.exists():
        return dict(np.load(path, allow_pickle=True))
    files = sorted(glob.glob(str(EXTERNAL / "cicids2017" / "*.csv")))
    df = pd.concat([pd.read_csv(f, skipinitialspace=True, encoding="latin-1") for f in files],
                   ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df["Label"] = df["Label"].str.replace("�", "", regex=False).str.replace("ï¿½", "", regex=False)
    df["Label"] = df["Label"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop(columns=[c for c in df.columns if c.endswith(".1")])
    num = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(num.values).all(axis=1)
    df, num = df[ok], num[ok]
    keep_rows = ~pd.concat([num, df["Label"]], axis=1).duplicated()
    df, num = df[keep_rows], num[keep_rows]
    num = num.loc[:, num.nunique() > 1]
    counts = df["Label"].value_counts()
    ok = df["Label"].isin(counts[counts >= 100].index).values
    df, num = df[ok], num[ok]

    y = (df["Label"] != "BENIGN").values.astype(int)
    fam = np.where(y == 1, df["Label"].map(CICIDS_FAMILY).fillna("Other").values, "Normal").astype(str)
    X = _signed_log(num.values.astype(float))
    Xtr, Xte, ytr, yte, ftr, fte = train_test_split(X, y, fam, test_size=0.3, stratify=fam,
                                                    random_state=split_seed)
    out = dict(X_train=Xtr, y_train=ytr, fam_train=ftr, X_test=Xte, y_test=yte, fam_test=fte,
               features=np.array(num.columns))
    if cache:
        np.savez_compressed(path, **out)
    return out


DATASETS = {"UNSW-NB15": (load_unsw, UNSW_FAMILIES), "CIC-IDS2017": (load_cicids2017, CICIDS_FAMILIES)}
