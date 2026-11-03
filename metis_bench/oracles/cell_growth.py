"""Bacterial growth across strains and media (Aida & Ying 2024, Commun. Biol. 7:1691).

Saturated population density K (maximum OD600, mean of four biological replicates) of
every combination of 115 E. coli strains (114 single-gene knockouts and their wild type
BW25113) and 135 synthetic media made from 45 chemicals (46 log10 concentration columns,
H+ included). The grid is exhaustive: any strain-medium neighbour of a measured recipe was
measured too, so hill-climbing can move freely and every point it reaches has real data.

Data: the paper's Supplementary Data (CC BY-NC-ND 4.0), not redistributed with this
repository; see data/README.md. Put the spreadsheet at
``data/external/cell_growth_Data_Data.xlsx`` or point ``METIS_CELL_GROWTH`` at it.

Cleaning, done here at load time so the file stays as published:

* ``Unique ID``, ``Combination ID`` and ``Condition ID_unique`` are dropped. They only
  re-encode ``Medium ID`` and ``Strain (gene)``.
* As in the paper, profiles without K or with a replicate SD of K of 0.1 or more are left
  out (13,944 of 15,525 remain).
* The model sees the 46 concentrations and the strain (one-hot). ``Medium ID`` only names
  media; the composition carries the same information.

Design space: each concentration column is a level factor (its measured values, already
log10, so no further log scale) and the strain is a categorical factor. A recipe's
neighbours, i.e. one hill-climbing move, are the same medium with any other strain, or the
same strain in one of the ``k_media`` media closest in composition.
"""
import os
import pathlib
import pickle
import re

import numpy as np
import pandas as pd

from metis.space import Factor

from .base import Oracle, PooledSpace
from .real import ROOT

DEFAULT_FILE = ROOT / 'data' / 'external' / 'cell_growth_Data_Data.xlsx'
CACHE = ROOT / 'artifacts' / 'cache' / 'cell_growth.pkl'
ID_COLUMNS = ['Unique ID', 'Combination ID', 'Condition ID_unique']
SD_MAX = 0.1
SOURCE = ('Aida H, Ying B-W (2024) Data-driven discovery of the interplay between genetic and environmental '
          'factors in bacterial growth. Commun Biol 7:1691. doi:10.1038/s42003-024-07347-3 (CC BY-NC-ND 4.0)')


def data_file():
    return pathlib.Path(os.environ.get('METIS_CELL_GROWTH', DEFAULT_FILE))


def _chemical(column):
    """'log10[Glucose]\\n(mM)' -> 'Glucose'."""
    match = re.match(r'log10\[(.+?)\]', column)
    return match.group(1) if match else column


def load_cell_growth():
    """``(rows, concentration_columns)``. Rows: ``strain``, ``medium``, the 46 log10 mM
    concentrations (named by chemical), ``yield`` (K) and ``yield_std`` (SD of K)."""
    path = data_file()
    if not path.exists():
        raise FileNotFoundError('{} not found. The cell growth data is not redistributed with this repository: '
                                'see data/README.md, or set METIS_CELL_GROWTH to the spreadsheet.'.format(path))
    stamp = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
    if CACHE.exists():
        cached = pickle.loads(CACHE.read_bytes())
        if cached.get('stamp') == stamp:
            return cached['rows'], cached['columns']
    raw = pd.read_excel(path)
    data = raw.drop(columns=ID_COLUMNS)
    concentrations = [c for c in data.columns if str(c).startswith('log10[')]
    data = data.rename(columns={**{c: _chemical(c) for c in concentrations}, 'Strain (gene)': 'strain',
                                'Medium ID': 'medium', 'K_ave': 'yield', 'K_ave_sd': 'yield_std'})
    columns = [_chemical(c) for c in concentrations]
    keep = data['yield'].notna() & (data['yield_std'] < SD_MAX)
    rows = data.loc[keep, ['strain', 'medium'] + columns + ['yield', 'yield_std']].reset_index(drop=True)
    rows['strain'] = rows['strain'].astype(str)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps({'stamp': stamp, 'rows': rows, 'columns': columns}))
    return rows, columns


class StrainMediumSpace(PooledSpace):
    """Measured strain-medium recipes; a move changes the strain or steps to a nearby medium."""

    def __init__(self, factors, pool, n_conc, k_media=8, name='cell_growth'):
        super().__init__(factors, pool, name=name)
        self.n_conc = n_conc
        self.k_media = k_media
        self.strain_dim = self.n_dims - 1
        medium_keys = [r.tobytes() for r in np.ascontiguousarray(self.pool[:, :n_conc].astype(np.uint8))]
        media = sorted(set(medium_keys))
        self._medium_of = {k: i for i, k in enumerate(media)}
        medium_id = np.array([self._medium_of[k] for k in medium_keys])
        self._rows = {(m, s): i for i, (m, s) in enumerate(zip(medium_id, self.pool[:, self.strain_dim]))}
        self._by_medium = {}
        for i, m in enumerate(medium_id):
            self._by_medium.setdefault(m, []).append(i)
        first = {m: rows[0] for m, rows in self._by_medium.items()}
        E = self.unit(self.pool[[first[m] for m in range(len(media))]])[:, :n_conc]
        d = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        self._near_media = np.argsort(d, axis=1, kind='stable')[:, :k_media]
        self.n_media = len(media)

    def neighbors(self, row, k=None, feasible_only=True):
        row = np.asarray(row, dtype=np.int64)
        m = self._medium_of[np.ascontiguousarray(row[:self.n_conc].astype(np.uint8)).tobytes()]
        s = row[self.strain_dim]
        same_medium = [i for i in self._by_medium[m] if self.pool[i, self.strain_dim] != s]
        near_media = [self._rows[(n, s)] for n in self._near_media[m] if (n, s) in self._rows]
        return self.pool[same_medium + near_media]


class CellGrowthOracle(Oracle):
    """Measured K of a strain-medium recipe; 20% of recipes held out (split seed 41)."""

    def __init__(self, test_fraction=0.2, split_seed=41, k_media=8):
        rows, columns = load_cell_growth()
        merged = rows.groupby(['strain'] + columns, sort=False).agg(
            mean=('yield', 'mean'), size=('yield', 'size'), medium=('medium', 'first')).reset_index()
        strains = tuple(sorted(merged['strain'].unique()))
        factors = [Factor(c, np.unique(merged[c].to_numpy(float)), log_scale=False) for c in columns]
        factors.append(Factor('strain', [1.0], alternatives=strains))
        idx = np.empty((len(merged), len(columns) + 2), dtype=np.int64)
        for d, (factor, c) in enumerate(zip(factors, columns)):
            values = merged[c].to_numpy(float)
            idx[:, d] = np.searchsorted(factor.levels, values)
            assert np.array_equal(factor.levels[idx[:, d]], values)
        idx[:, -2] = 0
        idx[:, -1] = merged['strain'].map({s: i for i, s in enumerate(strains)}).to_numpy()
        rng = np.random.default_rng(split_seed)
        order = rng.permutation(len(merged))
        n_test = int(round(test_fraction * len(merged)))
        test, pool = order[:n_test], order[n_test:]
        self.id = 'measured_cell_growth'
        self.space = StrainMediumSpace(factors, idx[pool], n_conc=len(columns), k_media=k_media, name=self.id)
        self.test_idx = idx[test]
        self._lookup = dict(zip(self.space.keys(idx), merged['mean'].to_numpy(float)))
        pool_y = merged['mean'].to_numpy(float)[pool]
        best = int(np.argmax(pool_y))
        self.f_max, self.x_max, self.max_is_exact = float(pool_y[best]), idx[pool][best], True
        self._f_ref = float(np.median(pool_y))
        self.meta = {'source': SOURCE, 'n_rows': len(rows), 'n_recipes': len(merged),
                     'n_merged_duplicates': int((merged['size'] > 1).sum()), 'n_factors': len(columns) + 1,
                     'n_strains': len(strains), 'n_media': int(merged['medium'].nunique()), 'k_media': k_media,
                     'n_pool': len(pool), 'n_test': n_test, 'split_seed': split_seed}

    def __call__(self, idx):
        return np.array([self._lookup[k] for k in self.space.keys(np.atleast_2d(idx))], dtype=float)
