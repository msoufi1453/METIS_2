"""Measured-data replay: oracle tables, held-out split, lookup and the TabPFN version guard."""
import numpy as np
import pandas as pd
import pytest

from metis_bench.oracles.measured import CETCH, MEASURED, build_measured, load_dataset


@pytest.mark.parametrize('name', MEASURED)
def test_pool_and_test_split_disjoint_and_cover_all_recipes(name):
    o = build_measured(name)
    pool_keys = set(o.space.keys(o.space.pool))
    test_keys = set(o.space.keys(o.test_idx))
    assert not pool_keys & test_keys
    assert len(pool_keys) + len(test_keys) == o.meta['n_recipes']
    assert o.meta['n_test'] == round(0.2 * o.meta['n_recipes'])


@pytest.mark.parametrize('name', MEASURED)
def test_lookup_returns_measured_mean_and_max_is_best_pool_recipe(name):
    o = build_measured(name)
    rows, factors = load_dataset(name)
    y = o(o.space.pool)
    assert o.f_max == pytest.approx(y.max())
    assert o(o.x_max[None])[0] == pytest.approx(o.f_max)
    # Every value the oracle returns is a measured (mean) yield from the data.
    means = rows.groupby(factors)['yield'].mean().to_numpy()
    assert np.isin(np.round(y, 9), np.round(means, 9)).all()


def test_cetch_days_concatenated_by_column_name():
    """Days 2-5 order the one-hot HEPES pH columns differently from day 1; values must follow names."""
    rows, factors = load_dataset('cetch_yield')
    day2 = pd.read_csv(CETCH / 'Yield_optimization' / 'Results_2.csv')
    ph = [c for c in factors if 'pH' in c or 'HEPES' in c]
    assert ph
    mine = rows[rows.day == 2][ph].reset_index(drop=True)
    pd.testing.assert_frame_equal(mine, day2[ph], check_dtype=False)
    assert len(rows) == 625 and rows.day.nunique() == 5
    assert len(load_dataset('cetch_efficiency')[0]) == 1000


def test_tabpfn_version_guard():
    tabpfn = pytest.importorskip('tabpfn')
    from metis.surrogates import TabPFNSurrogate

    major = int(tabpfn.__version__.split('.')[0])
    wrong = 'v2' if major >= 9 else 'v3.5'
    X, y = np.random.default_rng(0).random((10, 3)), np.arange(10.0)
    with pytest.raises(RuntimeError, match='needs the tabpfn'):
        TabPFNSurrogate(version=wrong).fit(X, y)
    with pytest.raises(ValueError):
        TabPFNSurrogate(version='v9').fit(X, y)
