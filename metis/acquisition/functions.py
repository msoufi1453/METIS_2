"""Acquisition functions: score candidates from a surrogate prediction (higher is better)."""
import numpy as np
from scipy.stats import norm


class Acquisition:
    name = 'acquisition'

    def __call__(self, prediction, best_y=None):
        raise NotImplementedError

    def describe(self):
        return {'name': self.name}


class UCB(Acquisition):
    """METIS's score: ``exploitation * mean + exploration * std``.

    ``exploration`` is the per-round coefficient (1.41 / 1.0 / 0.5 in the notebooks); a
    schedule from ``metis.schedules`` updates it between rounds.
    """

    name = 'ucb'

    def __init__(self, exploitation=1.0, exploration=1.0):
        self.exploitation = exploitation
        self.exploration = exploration

    def __call__(self, prediction, best_y=None):
        return self.exploitation * prediction.mean + self.exploration * prediction.std

    def describe(self):
        return {'name': self.name, 'exploitation': self.exploitation, 'exploration': self.exploration}


class ExpectedImprovement(Acquisition):
    name = 'ei'

    def __init__(self, xi=0.0):
        self.xi = xi

    def __call__(self, prediction, best_y=None):
        mean, std = np.asarray(prediction.mean, float), np.maximum(np.asarray(prediction.std, float), 1e-12)
        best = np.max(mean) if best_y is None else best_y
        z = (mean - best - self.xi) / std
        return (mean - best - self.xi) * norm.cdf(z) + std * norm.pdf(z)

    def describe(self):
        return {'name': self.name, 'xi': self.xi}
