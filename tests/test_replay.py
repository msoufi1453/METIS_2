"""Measured-data replay: oracle tables, held-out split, lookup, the TabPFN version guard and the figures."""
import json

import numpy as np
import pandas as pd
import pytest

from metis_bench import replay
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


def test_ranges_and_swarm_offsets():
    assert replay._ranges([2, 3, 4, 7, 9, 10]) == '2–4, 7, 9–10'
    y = np.array([1.0, 1.01, 1.02, 1.03, 5.0])
    x = replay._swarm_offsets(y, bin_height=0.5)
    assert x[4] == 0.0                          # alone in its band: centred
    assert len(set(np.round(x[:4], 9))) == 4    # same band: fanned out, no two on top of each other
    assert np.abs(x).max() <= 0.36 + 1e-12
    assert np.array_equal(x, replay._swarm_offsets(y, bin_height=0.5))


def _fake_run(arm, seed, rounds_found):
    """A minimal campaign result: 2 rounds of 3 recipes on one dataset."""
    true = [1.0, 2.0, 3.0, 2.5, 4.0 if rounds_found else 3.5, 1.0]
    guided = [{'round': 2, 'test_r2': 0.5, 'test_spearman': 0.6, 'test_cov80': 0.0 if arm == 'linear' else 0.7}]
    return {'problem': {'subtype': 'pure', 'f_max': 4.0}, 'arm_name': arm, 'seed': seed,
            'evaluated': {'round': [1, 1, 1, 2, 2, 2], 'true': true},
            'rounds': guided if arm != 'random' else [{'round': 2}],
            'endpoints': {'auc_norm_regret': 0.1}, 'elapsed_s': 1.0}


def test_write_replay_figures_and_tables(tmp_path):
    for arm in ('incumbent', 'linear', 'random'):
        for seed in (0, 1):
            path = tmp_path / 'measured_pure_none' / arm / 'seed_{:03d}.json'.format(seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_fake_run(arm, seed, rounds_found=arm != 'random')))
    cfg = {'name': 'fake', 'arms': [{'name': 'incumbent'}, {'name': 'linear'}, {'name': 'random'}],
           'families': {'R1': {'reference': 'incumbent', 'arms': ['linear']}}}
    out = replay.write_replay(cfg, tmp_path)
    for name in ('yields_by_round.png', 'best_so_far.png', 'model_accuracy.png', 'model_comparison.png',
                 'summary.csv', 'summary.md'):
        assert (out / name).stat().st_size > 0, name
    summary = (out / 'summary.md').read_text(encoding='utf-8')
    assert '| PURE | XGBoost (METIS) | 2/2 |' in summary
    assert '| PURE | Random | 0/2 |' in summary
    assert '| Linear | XGBoost (METIS) | PURE | 2 | 0 / 2 / 0 |' in summary
