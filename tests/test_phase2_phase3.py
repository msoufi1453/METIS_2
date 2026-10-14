"""Initial designs, schedules, ablation analysis and experiment-file resolution."""
import json

import numpy as np
import pandas as pd
import pytest

from metis import initialization, schedules
from metis.initialization import min_pairwise_distance
from metis_bench import ablation, experiments, tuning
from metis_bench.synthetic import FunctionSpec, build_function


@pytest.fixture(scope='module')
def space():
    return build_function(FunctionSpec('broad', n_categorical=2, seed=5), 12).space


@pytest.mark.parametrize('kind', ['random', 'sobol', 'lhs_maximin'])
def test_initial_designs_are_feasible_unique(space, kind):
    init = initialization.make({'type': kind})
    idx = init(space, 32, np.random.default_rng(0))
    assert idx.shape == (32, space.n_dims)
    assert space.feasible(idx).all() and len(set(space.keys(idx))) == 32


def test_space_filling_designs_spread_more_than_random(space):
    def spread(kind):
        return np.mean([min_pairwise_distance(space.embedding(initialization.make({'type': kind})(
            space, 32, np.random.default_rng(s)))) for s in range(5)])
    assert spread('lhs_maximin') > spread('random')
    assert spread('sobol') > 0.8 * spread('random')


def test_fill_avoids_imported_region(space):
    from metis_bench.problems import imported_design
    rng = np.random.default_rng(1)
    existing = imported_design(space, 32, rng)
    fill = initialization.make({'type': 'fill'})(space, 32, rng, existing=existing)
    rand = initialization.make({'type': 'random'})(space, 32, rng, existing=existing)
    E_old = space.embedding(existing)

    def nearest_old(points):
        return np.min(np.linalg.norm(space.embedding(points)[:, None] - E_old[None], axis=2), axis=1).mean()
    assert nearest_old(fill) > nearest_old(rand)
    assert not set(space.keys(fill)) & set(space.keys(existing))


def test_schedules():
    assert schedules.make({'type': 'legacy'})(2, 10) == 1.41
    assert schedules.make({'type': 'legacy'})(10, 10) == 0.5
    lin = schedules.make({'type': 'linear', 'start': 2.0, 'end': 0.5})
    assert lin(2, 10) == 2.0 and lin(10, 10) == 0.5
    exp = schedules.make({'type': 'exponential', 'start': 2.0, 'end': 0.5})
    assert exp(6, 10) == pytest.approx(1.0)
    ada = schedules.make({'type': 'adaptive', 'start': 1.0, 'down': 0.5, 'up': 2.0})
    assert ada(4, 10, history=[1.0, 2.0, 2.0]) == pytest.approx(1.0)   # progress, then stagnation
    gp = schedules.make({'type': 'gp_ucb', 'scale': 0.2})
    assert gp(3, 10) > gp(2, 10) > 0


def test_ablation_attribution_recovers_planted_effects():
    names = ['a', 'b', 'c']
    arms = ablation.ablation_arms({n: {'x': n} for n in names})
    assert len(arms) == 8 and arms[0]['name'] == 'incumbent'
    rng = np.random.default_rng(0)
    rows = []
    for p in range(15):
        base = rng.uniform(0.3, 0.7)
        for s in range(6):
            shared = rng.normal(scale=0.02)
            for arm in arms:
                S = set(arm['ablation_subset'])
                effect = -0.10 * ('a' in S) - 0.02 * ('b' in S) - 0.05 * ({'a', 'c'} <= S)
                rows.append({'problem_id': p, 'seed': s, 'arm': arm['name'],
                             'final_norm_regret': base + shared + effect + rng.normal(scale=0.005)})
    coef, shap, per_subset = ablation.analyze(pd.DataFrame(rows), names, n_boot=200)
    coef = coef.set_index('term').effect_on_regret
    assert coef['a'] == pytest.approx(-0.10, abs=0.01) and coef['axc'] == pytest.approx(-0.05, abs=0.01)
    shap = shap.set_index('intervention').shapley_improvement
    assert shap.sum() == pytest.approx(0.17, abs=0.01)
    assert shap['a'] > shap['c'] > shap['b']


def test_tuning_candidates_and_winner(tmp_path):
    cfg = {'tuning': {'part': 'schedule'}, 'base_arm': {'surrogate': {'type': 'gp_matern'}},
           'candidates': [{'type': 'constant', 'beta': [0.5, 1.0]}, {'type': 'linear', 'start': [2.0], 'end': [0.1, 0.5]}]}
    arms = tuning.candidate_arms(cfg)
    assert len(arms) == 4 and all(a['surrogate']['type'] == 'gp_matern' for a in arms)
    runs = pd.DataFrame([{'arm': a['name'], 'problem_id': p, 'final_norm_regret': 0.1 * i + 0.01 * p}
                         for i, a in enumerate(arms) for p in range(3)])
    winner, table = tuning.select_winner(runs)
    assert winner == arms[0]['name']
    cfg['arms'] = arms
    path = tuning.write_winner(tmp_path / 'w.json', cfg, winner, table)
    assert json.loads(path.read_text())['fragment'] == {'type': 'constant', 'beta': 0.5}


@pytest.mark.parametrize('name', ['phase2_init', 'phase2_import', 'phase2_surrogates', 'phase2_batch'])
def test_experiment_files_load(name):
    for pilot in (False, True):
        cfg = experiments.load('experiments/{}.yaml'.format(name), pilot=pilot)
        assert cfg['arms'] and all('name' in a for a in cfg['arms'])
        assert cfg['name'].endswith('-pilot') == pilot


def test_local_pool_is_valid_and_concentrates_near_the_best(space):
    from metis.pools import LocalPool, RandomPool
    rng = np.random.default_rng(3)
    observed = space.sample(40, rng)
    y = -np.linalg.norm(space.unit(observed) - 0.3, axis=1)
    local = LocalPool(local_share=0.5).generate(space, 400, rng, observed, y)
    rand = RandomPool().generate(space, 400, rng, observed, y)
    for pool in (local, rand):
        assert space.feasible(pool).all() and len(set(space.keys(pool))) == len(pool)
        assert not set(space.keys(pool)) & set(space.keys(observed))
    best = space.embedding(observed[np.argsort(y)[::-1][:8]])

    def near(pool):
        return np.mean(np.min(np.linalg.norm(space.embedding(pool)[:, None] - best[None], axis=2), axis=1) < 0.5)
    assert near(local) > near(rand) + 0.3
