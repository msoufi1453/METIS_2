"""Shared test helpers: a fast stand-in model for campaigns, so tests do not need TabPFN."""
import numpy as np

from metis.campaign import MODELS, register_model
from metis.surrogates import Prediction, Surrogate

FAST = 'fast-test-model'


class BaggedRidge(Surrogate):
    """Ridge regressions on bootstrap samples (features and their squares): the mean and
    spread of their predictions. Deterministic for a seed, and quick."""

    name = 'bagged_ridge'

    def __init__(self, members=12, alpha=1e-2, seed=0):
        self.members, self.alpha, self.seed = members, alpha, seed

    def _design(self, X):
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return np.hstack([np.ones((len(Z), 1)), Z, Z ** 2])

    def fit(self, X, y, noise_var=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        self.mean_, self.scale_ = X.mean(axis=0), np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        rng = np.random.default_rng(self.seed)
        A = self._design(X)
        self.coef_ = []
        for _ in range(self.members):
            rows = rng.integers(0, len(X), size=len(X))
            a, b = A[rows], y[rows]
            self.coef_.append(np.linalg.solve(a.T @ a + self.alpha * np.eye(a.shape[1]), a.T @ b))
        return self

    def predict(self, X):
        predictions = np.stack([self._design(X) @ c for c in self.coef_])
        return Prediction(predictions.mean(axis=0), predictions.std(axis=0))

    def describe(self):
        return {'name': self.name, 'members': self.members}


def register_fast_model():
    """Register the stand-in under ``FAST`` (idempotent) and return its key."""
    if FAST not in MODELS:
        register_model(FAST, 'Fast test model', lambda seed=0, **options: BaggedRidge(seed=seed, **options),
                       max_variables=6000, pool_size=(400, 400))
    return FAST
