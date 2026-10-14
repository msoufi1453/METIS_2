"""The surrogate interface every model plugs into."""
import copy
from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    """Surrogate output for a batch of inputs.

    ``std`` is the model's own uncertainty: GP posterior std of the latent yield, ensemble
    member spread, or the spread of TabPFN's predictive distribution. Acquisitions use it
    as-is; metis_bench.metrics.calibration measures how well it matches real errors.
    """
    mean: np.ndarray
    std: np.ndarray

    def __len__(self):
        return len(self.mean)

    def take(self, index):
        return Prediction(self.mean[index], self.std[index])


class Surrogate:
    """Base class for models that predict yield, with uncertainty, from model inputs.

    Inputs are the METIS feature matrix: one column per factor concentration plus one-hot
    columns for alternatives, in ``feature_columns(concentrations_limits)`` order.

    Subclasses implement ``fit`` and ``predict``. A surrogate whose ``accepts_noise`` is
    True also uses ``noise_var``, the per-sample measurement variance of the training
    labels (e.g. ``yield_std ** 2 / n_replicates`` from triplicates). Others ignore it.
    """

    name = 'surrogate'
    accepts_noise = False

    def fit(self, X, y, noise_var=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def refit(self, X, y, noise_var=None):
        """Refit on new data keeping hyperparameters fixed. Defaults to ``fit``."""
        return self.fit(X, y, noise_var=noise_var)

    def clone(self):
        """A copy that can be refitted without touching this one (used for fantasies)."""
        return copy.deepcopy(self)

    def describe(self):
        """JSON-serialisable settings, logged with every benchmark run."""
        return {'name': self.name}


def as_2d_float(X):
    X = np.asarray(X, dtype=float)
    return X.reshape(len(X), -1)


def as_1d_float(y):
    return np.asarray(y, dtype=float).reshape(-1)


def inverse_variance_weights(noise_var, floor=None):
    """Normalised weights 1 / (noise_var + floor) with mean 1.

    ``floor`` keeps a perfect replicate (std 0) from dominating; it defaults to the median
    noise variance, so weights stay within a sensible range.
    """
    v = np.asarray(noise_var, dtype=float).reshape(-1)
    if floor is None:
        positive = v[v > 0]
        floor = float(np.median(positive)) if len(positive) else 1.0
    w = 1.0 / (v + floor)
    return w / w.mean()
