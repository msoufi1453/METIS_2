"""Campaign runner: pairing by seed, logging, and the experiment task runner."""
import json

import numpy as np
import pytest

from metis_bench.campaign import Settings, run_campaign, run_tasks
from metis_bench.synthetic import FunctionSpec

PROBLEM = {'synthetic': FunctionSpec('switch_on', active='subset', seed=2).to_dict(), 'n_dims': 12,
           'noise': 'realistic'}
SETTINGS = Settings(batch_size=8, n_rounds=3, pool_size=300, test_points=100)
CHEAP_XGB = {'type': 'xgb_ensemble', 'n_iter': 3, 'ensemble_len': 3}


@pytest.fixture(scope='module')
def pair():
    a = run_campaign(PROBLEM, {'name': 'a', 'surrogate': CHEAP_XGB}, seed=5, settings=SETTINGS)
    b = run_campaign(PROBLEM, {'name': 'b', 'surrogate': {'type': 'gp_matern', 'n_restarts': 0}}, seed=5,
                     settings=SETTINGS)
    return a, b


def test_same_seed_shares_initial_design_and_noise(pair):
    a, b = pair
    first = lambda r: [i for i, rd in zip(r['evaluated']['idx'], r['evaluated']['round']) if rd == 1]  # noqa: E731
    assert first(a) == first(b)
    # Same wells, same draws: the round-1 measured means match exactly.
    m = lambda r: [x for x, rd in zip(r['evaluated']['mean'], r['evaluated']['round']) if rd == 1]  # noqa: E731
    assert m(a) == m(b)


def test_campaign_log_is_complete(pair):
    a, _ = pair
    assert len(a['evaluated']['idx']) == SETTINGS.batch_size * SETTINGS.n_rounds
    assert len(set(map(tuple, a['evaluated']['idx']))) == len(a['evaluated']['idx'])   # never re-measures
    assert [r['round'] for r in a['rounds']] == [2, 3]
    assert {'test_r2', 'test_cov80', 'batch_spearman', 'norm_regret', 'exploration'} <= set(a['rounds'][0])
    assert a['rounds'][0]['exploration'] == 1.41
    regrets = [t['norm_regret'] for t in a['trajectory']]
    assert all(x >= y for x, y in zip(regrets, regrets[1:]))          # best-so-far never worsens
    assert a['endpoints']['final_norm_regret'] == regrets[-1]


def test_random_search_and_imported_start():
    r = run_campaign(PROBLEM, {'name': 'rs', 'selector': {'type': 'random'}}, seed=1, settings=SETTINGS)
    assert len(r['evaluated']['idx']) == 24 and 'test_r2' not in r['rounds'][0]
    imp = dict(PROBLEM, imported={'n': 8, 'kind': 'ofat'})
    fill = run_campaign(imp, {'name': 'fill', 'init': {'type': 'fill'}, 'surrogate': CHEAP_XGB}, seed=1,
                        settings=SETTINGS)
    rounds = fill['evaluated']['round']
    assert rounds.count(0) == 8 and rounds.count(1) == 8


def test_run_tasks_is_resumable(tmp_path):
    path = tmp_path / 'r' / 'seed_000.json'
    task = (PROBLEM, {'name': 'rs', 'selector': {'type': 'random'}}, 0, vars(SETTINGS), str(path))
    assert run_tasks([task], progress=lambda *_: None) == []
    stamp = path.stat().st_mtime_ns
    assert json.loads(path.read_text())['seed'] == 0
    run_tasks([task], progress=lambda *_: None)
    assert path.stat().st_mtime_ns == stamp
