"""Per-factor response shapes: parameterised families on the factor's unit scale.

Every shape maps ``u`` in [0, 1] (the level's position on the factor's own scale, log
for log-spaced factors) to [0, 1], normalised so its maximum over [0, 1] is 1.
Continuous parameters cover the forms between the named families: a Hill curve with
``n`` near 1 is almost linear, with ``n`` around 20 it is a switch.
"""
from dataclasses import dataclass, field

import numpy as np

FAMILIES = ('linear', 'logarithmic', 'exponential', 'hill', 'bell', 'biphasic', 'crash')
_U = np.linspace(0.0, 1.0, 2001)


def _hill(u, K, n):
    un = np.power(np.clip(u, 0, None), n)
    return un / (K ** n + un)


def _raw(family, p, u):
    u = np.asarray(u, dtype=float)
    if family == 'linear':
        return u if p.get('increasing', True) else 1.0 - u
    if family == 'logarithmic':
        return np.log1p(p['k'] * u) / np.log1p(p['k'])
    if family == 'exponential':
        return np.expm1(p['k'] * u) / np.expm1(p['k'])
    if family == 'hill':
        h = _hill(u, p['K'], p['n'])
        return h if p.get('increasing', True) else 1.0 - h
    if family == 'bell':
        return np.exp(-0.5 * ((u - p['mu']) / p['width']) ** 2)
    if family == 'biphasic':
        return _hill(u, p['K1'], p['n1']) * (1.0 - _hill(u, p['K2'], p['n2']))
    if family == 'crash':
        rise = np.power(np.clip(u / p['theta'], 0, 1), p['power'])
        return np.where(u <= p['theta'], rise, p['after'])
    raise ValueError(family)


@dataclass
class Shape:
    family: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self._norm = float(_raw(self.family, self.params, _U).max()) or 1.0

    def __call__(self, u):
        return np.clip(_raw(self.family, self.params, u) / self._norm, 0.0, 1.0)

    def to_dict(self):
        return {'family': self.family, 'params': self.params}


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_shape(family, rng, sharp=None, increasing=None):
    """Draw a shape from ``family`` with random parameters.

    ``sharp`` (Hill only): True gives switch-like (n 8 to 25), False gradual (n 1 to 3),
    None anywhere in 1 to 20. ``increasing`` fixes the direction where it applies.
    """
    p = {}
    if family == 'linear':
        p['increasing'] = bool(rng.random() < 0.8) if increasing is None else increasing
    elif family == 'logarithmic':
        p['k'] = _logu(rng, 2, 200)
    elif family == 'exponential':
        p['k'] = float(rng.uniform(1, 6))
    elif family == 'hill':
        p['n'] = {True: _logu(rng, 8, 25), False: _logu(rng, 1, 3), None: _logu(rng, 1, 20)}[sharp]
        p['K'] = float(rng.uniform(0.2, 0.8))
        p['increasing'] = True if increasing is None else increasing
    elif family == 'bell':
        p['mu'] = float(rng.uniform(0.15, 0.85))
        p['width'] = _logu(rng, 0.1, 0.45)
    elif family == 'biphasic':
        p['K1'] = float(rng.uniform(0.1, 0.45))
        p['K2'] = float(rng.uniform(p['K1'] + 0.2, 0.95))
        p['n1'] = _logu(rng, 1.5, 6)
        p['n2'] = _logu(rng, 1.5, 6)
    elif family == 'crash':
        p['theta'] = float(rng.uniform(0.45, 0.85))
        p['power'] = float(rng.uniform(0.6, 2.0))
        p['after'] = float(rng.uniform(0.0, 0.08))
    else:
        raise ValueError(family)
    return Shape(family, p)
