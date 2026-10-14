"""Synthetic droplet grids shaped like METIS experiments."""
import numpy as np

from metis.space import DesignSpace, Factor

DROP_NL = 100.0
REACTION_NL = 20000.0


def make_space(n_factors, rng, n_categorical=0, log_fraction=0.3, feasible_fraction=None, name='synthetic'):
    """A grid of ``n_factors`` factors with a volume budget.

    * Linear factors: 5 to 12 levels, 1 or 2 drops per step, optionally no zero level.
    * Log-scale factors (about ``log_fraction``): a zero level plus doses spanning 2 to 3.5
      decades. As with METIS's multiple stocks, each dose costs 1 to 3 drops.
    * ``n_categorical`` factors carry 3 or 4 alternatives (e.g. promoters) on top of 3 to
      5 concentration levels; alternatives cost no volume.
    * The budget is set so ``feasible_fraction`` (default 0.3 to 0.9) of uniformly random
      grid points are executable. Drop counts are integers, which exact maxima rely on.
    """
    if feasible_fraction is None:
        feasible_fraction = float(rng.uniform(0.3, 0.9))
    categorical = set(rng.choice(n_factors, size=n_categorical, replace=False).tolist()) if n_categorical else set()
    factors = []
    for i in range(n_factors):
        fname = 'X{:02d}'.format(i + 1)
        stock = float(10 ** rng.uniform(0, 3))
        if i in categorical:
            n_levels = int(rng.integers(3, 6))
            drops = np.arange(1, n_levels + 1) * int(rng.choice([1, 2]))
            levels = drops * stock * DROP_NL / REACTION_NL
            alts = tuple('v{}'.format(k + 1) for k in range(int(rng.integers(3, 5))))
            factors.append(Factor(fname, levels, drops, alternatives=alts))
        elif rng.random() < log_fraction:
            n_levels = int(rng.integers(6, 11))
            decades = float(rng.uniform(2.0, 3.5))
            positive = stock * 1e-3 * 10 ** np.linspace(0, decades, n_levels - 1)
            levels = np.concatenate([[0.0], positive])
            drops = np.concatenate([[0], 1 + np.arange(n_levels - 1) % 3])
            factors.append(Factor(fname, levels, drops, log_scale=True))
        else:
            n_levels = int(rng.integers(5, 13))
            step = int(rng.choice([1, 2]))
            start = int(rng.choice([0, 1]))
            drops = (np.arange(n_levels) + start) * step
            levels = drops * stock * DROP_NL / REACTION_NL
            factors.append(Factor(fname, levels, drops))
    space = DesignSpace(factors, drop_nl=DROP_NL, max_nl=None, name=name)
    probe = rng.integers(0, space.cardinalities, size=(20000, space.n_dims))
    totals = space.drops(probe)
    max_drops = max(int(np.floor(np.quantile(totals, feasible_fraction))), int(totals.min()))
    space.max_nl = max_drops * DROP_NL
    space.feasible_fraction = feasible_fraction
    return space
