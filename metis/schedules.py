"""Exploration schedules: the UCB exploration coefficient for each round.

``round_index`` is the round being proposed: 2 is the first model-guided round after
the day-1 random design, as ``exploration[day + 1]`` in the notebooks.
``history`` holds the best measured yield after each completed round, which adaptive
schedules may use.
"""
import numpy as np


class Schedule:
    name = 'schedule'

    def __call__(self, round_index, n_rounds, history=None):
        raise NotImplementedError

    def describe(self):
        return {'name': self.name, **{k: v for k, v in vars(self).items() if not k.startswith('_')}}


class LegacySchedule(Schedule):
    """The notebooks' hand-set list: 1.41 for days 1 to 3, 1.0 for 4 to 7, 0.5 for 8 to 10."""

    name = 'legacy'
    TABLE = {1: 1.41, 2: 1.41, 3: 1.41, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 0.5, 9: 0.5, 10: 0.5}

    def __call__(self, round_index, n_rounds, history=None):
        return self.TABLE.get(round_index, 0.5)


class ConstantSchedule(Schedule):
    name = 'constant'

    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, round_index, n_rounds, history=None):
        return self.beta


def _progress(round_index, n_rounds, first=2):
    return float(np.clip((round_index - first) / max(n_rounds - first, 1), 0.0, 1.0))


class LinearDecaySchedule(Schedule):
    """``start`` in the first guided round down to ``end`` in the last."""

    name = 'linear'

    def __init__(self, start=2.0, end=0.25):
        self.start = start
        self.end = end

    def __call__(self, round_index, n_rounds, history=None):
        return self.start + (self.end - self.start) * _progress(round_index, n_rounds)


class ExponentialDecaySchedule(Schedule):
    """``start * (end / start) ** progress``: a fast early drop, then a long tail."""

    name = 'exponential'

    def __init__(self, start=2.0, end=0.25):
        self.start = start
        self.end = end

    def __call__(self, round_index, n_rounds, history=None):
        return self.start * (self.end / self.start) ** _progress(round_index, n_rounds)


class GPUCBSchedule(Schedule):
    """Srinivas et al. (2010) finite-domain rate, scaled: ``scale * sqrt(2 log(|D| t^2 pi^2 / 6 delta))``.

    ``t`` counts guided rounds. The theory grows the bonus slowly over time; ``scale``
    absorbs the gap between theory and practice (the theoretical constants are known to
    over-explore).
    """

    name = 'gp_ucb'

    def __init__(self, scale=0.2, domain_size=10000, delta=0.1):
        self.scale = scale
        self.domain_size = domain_size
        self.delta = delta

    def __call__(self, round_index, n_rounds, history=None):
        t = max(round_index - 1, 1)
        return self.scale * float(np.sqrt(2 * np.log(self.domain_size * t ** 2 * np.pi ** 2 / (6 * self.delta))))


class AdaptiveSchedule(Schedule):
    """Explore more after stagnation, less after progress.

    After each round, if the best measured yield rose by more than ``tolerance`` (relative)
    the coefficient is multiplied by ``down``; otherwise by ``up``. The result is clipped to
    [``low``, ``high``]. Starts at ``start``.
    """

    name = 'adaptive'

    def __init__(self, start=1.41, down=0.7, up=1.4, tolerance=0.01, low=0.1, high=4.0):
        self.start = start
        self.down = down
        self.up = up
        self.tolerance = tolerance
        self.low = low
        self.high = high

    def __call__(self, round_index, n_rounds, history=None):
        beta = self.start
        history = list(history or [])
        for prev, cur in zip(history[:-1], history[1:]):
            improved = cur > prev + self.tolerance * max(abs(prev), 1e-12)
            beta = float(np.clip(beta * (self.down if improved else self.up), self.low, self.high))
        return beta


REGISTRY = {'legacy': LegacySchedule, 'constant': ConstantSchedule, 'linear': LinearDecaySchedule,
            'exponential': ExponentialDecaySchedule, 'gp_ucb': GPUCBSchedule, 'adaptive': AdaptiveSchedule}


def make(cfg):
    cfg = dict(cfg or {'type': 'legacy'})
    return REGISTRY[cfg.pop('type')](**cfg)
