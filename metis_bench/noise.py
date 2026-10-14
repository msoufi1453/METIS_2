"""Measurement layer: replicate wells with noise, outliers and per-round batch effects.

Each design point is measured in ``n_replicates`` wells (triplicates, as in METIS). A
round's wells share one plate effect: yields are multiplied by
``exp(N(0, batch_sd))`` and shifted by ``N(0, offset_sd * f_max)``. Each well adds noise
with standard deviation ``sd(f) = sigma_abs * f_max + sigma_rel * f``, which is constant
when ``sigma_rel = 0`` and grows with the signal otherwise. With probability
``outlier_prob`` a well is an outlier: half are failed wells (about zero), half are
spikes (x1.5 to x3). Yields are clipped at zero. The mean and sample std of the wells
are what METIS stores as ``yield`` and ``yield_std``.

Presets (relative to the oracle's ``f_max``):

* ``realistic``: replicate noise fitted to Borkowski's triplicates (yield_std = 0.0545 +
  0.0274 * yield, max yield 3.32), 1% outlier wells, 5% plate-to-plate CV. The data
  can't separate batch effects from the design, so the batch values are assumptions.
* ``homoscedastic``: constant 5% of f_max, no outliers or batch effects.
* ``high``: about 2.5x the realistic replicate noise, 3% outliers, 15% plate CV and
  2% offsets.
* ``none``: exact yields (std 0).
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    sigma_abs: float = 0.0
    sigma_rel: float = 0.0
    outlier_prob: float = 0.0
    batch_sd: float = 0.0
    offset_sd: float = 0.0
    n_replicates: int = 3
    name: str = 'custom'

    def sd(self, f, f_max):
        return self.sigma_abs * f_max + self.sigma_rel * np.abs(f)

    def measure(self, f, f_max, rng):
        """Measure true yields ``f`` (one round, one plate). Returns (mean, std, wells)."""
        f = np.asarray(f, dtype=float)
        n, k = len(f), self.n_replicates
        multiplier = np.exp(rng.normal(0.0, self.batch_sd)) if self.batch_sd else 1.0
        offset = rng.normal(0.0, self.offset_sd * f_max) if self.offset_sd else 0.0
        wells = multiplier * f[:, None] + offset + self.sd(f, f_max)[:, None] * rng.standard_normal((n, k))
        if self.outlier_prob:
            hit = rng.random((n, k)) < self.outlier_prob
            failed = rng.random((n, k)) < 0.5
            dropout = np.abs(rng.normal(0.0, 0.02 * f_max, (n, k)))
            spike = wells * rng.uniform(1.5, 3.0, (n, k))
            wells = np.where(hit & failed, dropout, np.where(hit, spike, wells))
        wells = np.clip(wells, 0.0, None)
        std = wells.std(axis=1, ddof=1) if k > 1 else np.zeros(n)
        return wells.mean(axis=1), std, wells

    def to_dict(self):
        return asdict(self)


PRESETS = {
    'none': NoiseModel(n_replicates=1, name='none'),
    'homoscedastic': NoiseModel(sigma_abs=0.05, name='homoscedastic'),
    'realistic': NoiseModel(sigma_abs=0.0164, sigma_rel=0.0274, outlier_prob=0.01, batch_sd=0.05, name='realistic'),
    'high': NoiseModel(sigma_abs=0.04, sigma_rel=0.07, outlier_prob=0.03, batch_sd=0.15, offset_sd=0.02, name='high'),
}


def get_noise(name_or_dict):
    if isinstance(name_or_dict, NoiseModel):
        return name_or_dict
    if isinstance(name_or_dict, dict):
        return NoiseModel(**name_or_dict)
    return PRESETS[name_or_dict]
