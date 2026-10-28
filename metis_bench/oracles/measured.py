"""Measured-data oracles: replaying published campaigns, as the METIS paper did for PURE.

An experiment returns the yield that was actually measured for that recipe, so only
measured recipes can be chosen. Each dataset's rows are concatenated from its per-day
files (matching columns by name), identical recipes are merged (mean yield), and a
fixed 20% of recipes is held out to score prediction accuracy (split seed 41, as in the
PURE notebook). The remaining 80% is the candidate pool. The optimum is the best
measured yield in the pool, so regret is exact with respect to the data.

Datasets:

* ``pure``: Matsuura et al. 2009 via Examples/PURE (1,094 recipes, 69 factors).
* ``borkowski``: Borkowski et al. 2020 via Examples/Cell_Free_System_Simulation
  (1,017 rows, 1,007 distinct recipes, 11 factors).
* ``cetch_yield`` / ``cetch_efficiency``: the METIS CETCH campaigns (Examples/CETCH;
  5 days x 125 and 8 days x 125 recipes, 32 columns including a one-hot HEPES pH).
  Note: the repository's ``Yield_all_data.csv`` stacks the day files by position, but
  days 2 to 5 order their HEPES pH columns differently from day 1, so 500 of its 625
  rows carry the wrong pH. The per-day files are concatenated by column name here
  instead.
"""
import functools
import glob
import pathlib

import numpy as np
import pandas as pd

from metis.space import DesignSpace

from .base import Oracle, PooledSpace
from .real import BORKOWSKI_FACTORS, ROOT, _levels_space, load_borkowski, load_pure

DATA_DIR = ROOT / 'data' / 'replay'
CETCH = ROOT / 'Examples' / 'CETCH' / 'Data'


def _cetch(folder):
    files = sorted(glob.glob(str(CETCH / folder / 'Results_*.csv')), key=lambda p: int(pathlib.Path(p).stem.split('_')[1]))
    frames = [pd.read_csv(f).assign(day=int(pathlib.Path(f).stem.split('_')[1])) for f in files]
    # pd.concat aligns by column name, which is the point: the day files order columns differently.
    data = pd.concat(frames, ignore_index=True)
    factors = list(frames[0].columns[:frames[0].columns.get_loc('Well_Name_1')])
    return data[factors + ['yield', 'yield_std', 'day']], factors


def load_dataset(name):
    """``(rows, factor_columns)``: every measured row, with ``yield`` (and ``yield_std``/``day`` if known)."""
    if name == 'pure':
        data = load_pure()
        return data, [c for c in data.columns if c != 'yield']
    if name == 'borkowski':
        data = load_borkowski()
        return data[BORKOWSKI_FACTORS + ['yield', 'yield_std', 'plate']].rename(columns={'plate': 'day'}), BORKOWSKI_FACTORS
    if name == 'cetch_yield':
        return _cetch('Yield_optimization')
    if name == 'cetch_efficiency':
        return _cetch('Efficiency_optimization')
    raise KeyError(name)


def write_concatenated(out_dir=DATA_DIR):
    """Write each dataset's concatenated rows to ``data/replay/<name>.csv`` (the oracle tables)."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in MEASURED:
        data, _ = load_dataset(name)
        path = out_dir / '{}.csv'.format(name)
        data.to_csv(path, index=False)
        paths.append(path)
    return paths


class MeasuredOracle(Oracle):
    """Returns the measured (mean) yield of a recorded recipe. ``test_idx`` is held out."""

    def __init__(self, name, test_fraction=0.2, split_seed=41):
        rows, factors = load_dataset(name)
        merged = rows.groupby(factors, sort=False)['yield'].agg(['mean', 'size']).reset_index()
        space_all = DesignSpace(_levels_space(merged[factors], name))
        idx = space_all.from_features(merged[factors].to_numpy(float))
        rng = np.random.default_rng(split_seed)
        order = rng.permutation(len(merged))
        n_test = int(round(test_fraction * len(merged)))
        test, pool = order[:n_test], order[n_test:]
        self.id = 'measured_' + name
        self.space = PooledSpace(space_all.factors, idx[pool], name=self.id)
        self.test_idx = idx[test]
        self._lookup = dict(zip(space_all.keys(idx), merged['mean'].to_numpy(float)))
        pool_y = merged['mean'].to_numpy(float)[pool]
        best = int(np.argmax(pool_y))
        self.f_max, self.x_max, self.max_is_exact = float(pool_y[best]), idx[pool][best], True
        self._f_ref = float(np.median(pool_y))       # median measured recipe in the pool
        self.meta = {'source': 'measured data: ' + name, 'n_rows': len(rows), 'n_recipes': len(merged),
                     'n_merged_duplicates': int((merged['size'] > 1).sum()), 'n_factors': len(factors),
                     'n_pool': len(pool), 'n_test': n_test, 'split_seed': split_seed}

    def __call__(self, idx):
        return np.array([self._lookup[k] for k in self.space.keys(np.atleast_2d(idx))], dtype=float)


MEASURED = ('pure', 'borkowski', 'cetch_yield', 'cetch_efficiency')


@functools.lru_cache(maxsize=8)
def build_measured(name):
    return MeasuredOracle(name)
