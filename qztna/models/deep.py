"""
Deep-learning IDS baselines (PyTorch, CPU).

DNNClassifier: supervised 3x256 MLP with batch norm and dropout, trained with
  class-balanced cross-entropy and early stopping on a 10% validation split.
AutoencoderDetector: unsupervised reconstruction-error detector in the spirit
  of Kitsune (Mirsky et al., NDSS 2018), trained on benign traffic only; the
  alarm threshold is the 99th percentile of reconstruction error on held-out
  benign training flows. It never sees an attack label, so it is the natural
  zero-day reference point.
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn


def _batches(n, batch_size, rng):
    idx = rng.permutation(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(256, 256, 256), dropout=0.2, lr=1e-3, epochs=40, batch_size=256,
                 patience=5, random_state=0):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def _net(self, d):
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        return nn.Sequential(*layers, nn.Linear(prev, 2))

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, stratify=y,
                                              random_state=self.random_state)
        Xtr, Xva = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
        ytr, yva = torch.tensor(ytr, dtype=torch.long), torch.tensor(yva, dtype=torch.long)
        counts = torch.bincount(ytr, minlength=2).float()
        loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts.clamp(min=1)))
        self.net_ = self._net(X.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net_.train()
            for b in _batches(len(Xtr), self.batch_size, rng):
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss_fn(self.net_(Xtr[b]), ytr[b]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                val = loss_fn(self.net_(Xva), yva).item()
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            logits = self.net_(torch.tensor(np.asarray(X), dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


class AutoencoderDetector(BaseEstimator):
    def __init__(self, bottleneck=8, hidden=64, lr=1e-3, epochs=40, batch_size=256, patience=5,
                 quantile=0.99, random_state=0):
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.quantile = quantile
        self.random_state = random_state

    def fit(self, X, y=None):
        """Trains on benign rows only (y == 0) when labels are given."""
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        Xb = X if y is None else X[np.asarray(y) == 0]
        Xtr, Xva = train_test_split(Xb, test_size=0.1, random_state=self.random_state)
        Xtr, Xva = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
        d = X.shape[1]
        self.net_ = nn.Sequential(nn.Linear(d, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.bottleneck),
                                  nn.ReLU(), nn.Linear(self.bottleneck, self.hidden), nn.ReLU(),
                                  nn.Linear(self.hidden, d))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net_.train()
            for b in _batches(len(Xtr), self.batch_size, rng):
                opt.zero_grad()
                ((self.net_(Xtr[b]) - Xtr[b]) ** 2).mean().backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                val = ((self.net_(Xva) - Xva) ** 2).mean().item()
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.threshold_ = float(np.quantile(self.score(Xva.numpy()), self.quantile))
        return self

    def score(self, X):
        """Reconstruction MSE per flow: higher = more anomalous."""
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
            return ((self.net_(Xt) - Xt) ** 2).mean(1).numpy()

    def predict(self, X):
        return (self.score(X) > self.threshold_).astype(int)
