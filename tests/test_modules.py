"""TabPFN module tests: soft distance penalty, fantasy clone, hill-climbing, per-round switches, snapshots."""
import numpy as np
import pytest

from metis import hillclimb
from metis.acquisition import UCB, SelectionContext, SoftPenalty, TopK
from metis.surrogates import Prediction, Surrogate
from metis_bench import arms as arm_lib
from metis_bench import experiments
from metis_bench.campaign import Settings, run_campaign, task_key
from metis_bench.oracles.cell_growth import data_file
from metis_bench.problems import get_problem, typical_spacing
from metis_bench.synthetic import FunctionSpec, build_function

SPEC = FunctionSpec('broad', active='all', zero_fraction=0.0, n_categorical=0, contrast=129.0, seed=8)


@pytest.fixture(scope='module')
def fn():
    return build_function(SPEC, 12)


class Known(Surrogate):
    """Predicts the true function exactly, with no uncertainty: a perfect, smooth model."""
    name = 'known'
    accepts_noise = False

    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y, noise_var=None):
        return self

    def predict(self, X):
        idx = self.fn.space.from_features(X)
        return Prediction(self.fn(idx), np.zeros(len(idx)))


def _ctx(scores, E):
    return SelectionContext(candidates=E, prediction=Prediction(np.asarray(scores, float), np.zeros(len(scores))),
                            surrogate=None, acquisition=UCB(1.0, 0.0), embedding=E)


def test_soft_penalty_spreads_the_batch_and_tiny_radius_is_topk():
    # Two clusters: five near-identical high scorers at x=0 and lower ones at x=10.
    E = np.array([[0.0], [0.01], [0.02], [0.03], [0.04], [10.0], [10.01], [10.02]])
    scores = [10, 9.9, 9.8, 9.7, 9.6, 5, 4.9, 4.8]
    top = TopK().select(_ctx(scores, E), 3).indices
    assert top.tolist() == [0, 1, 2]
    assert SoftPenalty(radius=1e-6).select(_ctx(scores, E), 3).indices.tolist() == [0, 1, 2]
    spread = SoftPenalty(radius=1.0).select(_ctx(scores, E), 2).indices
    assert spread.tolist() == [0, 5]                     # the second pick leaves the first cluster
    assert len(set(SoftPenalty(radius=1.0).select(_ctx(scores, E), 8).indices.tolist())) == 8


def test_soft_penalty_radius_scales_with_dataset_spacing():
    sel = arm_lib.make_selector({'type': 'soft_penalty', 'radius_scale': 3.0, 'rounds': [4, 10]}, spacing=0.2)
    assert sel.radius == pytest.approx(0.6)
    assert arm_lib.active({'type': 'soft_penalty', 'rounds': [4, 10]}, 3) is False
    assert arm_lib.active({'type': 'soft_penalty', 'rounds': [4, 10]}, 4) is True
    assert arm_lib.active({'type': 'topk'}, 2) is True
    assert arm_lib.active(None, 2) is False


def test_hill_climb_reaches_local_tops_and_respects_the_cap(fn):
    space, model, acq = fn.space, Known(fn), UCB(1.0, 0.0)
    pool = space.sample(300, np.random.default_rng(1))
    pred = model.predict(space.features(pool))
    c = hillclimb.climb(space, model, acq, pool, pred, n_starts=5, max_steps=20)
    assert (c.steps > 0).all() and (c.end_scores >= c.start_scores).all()
    assert np.allclose(c.end_scores, fn(c.ends))
    for end, steps in zip(c.ends, c.steps):
        if steps < 20:                                   # stopped at a local top: no neighbour is better
            assert fn(space.neighbors(end)).max() <= fn(end[None])[0] + 1e-12
    assert space.feasible(c.ends).all()
    capped = hillclimb.climb(space, model, acq, pool, pred, n_starts=5, max_steps=1)
    assert (capped.steps <= 1).all() and capped.capped.sum() == (capped.steps == 1).sum()
    assert len(c.new) == len(c.new_prediction.mean) and np.allclose(c.new_prediction.mean, fn(c.new))


def test_hill_climb_never_proposes_measured_recipes(fn):
    space, model, acq = fn.space, Known(fn), UCB(1.0, 0.0)
    pool = space.sample(200, np.random.default_rng(2))
    first = hillclimb.climb(space, model, acq, pool, model.predict(space.features(pool)), n_starts=3)
    measured = space.keys(first.ends)
    again = hillclimb.climb(space, model, acq, pool, model.predict(space.features(pool)), measured_keys=measured,
                            n_starts=3)
    assert not set(space.keys(again.new)) & set(measured)


def test_tabpfn_clone_is_an_unfitted_twin_with_the_same_seed():
    from metis.surrogates import TabPFNSurrogate
    s = TabPFNSurrogate(n_estimators=8, device='auto', version='v3.5', random_state=7)
    s.seed_ = 7
    twin = s.clone()
    assert isinstance(twin, TabPFNSurrogate) and not hasattr(twin, 'model_')
    assert (twin.n_estimators, twin.device, twin.version, twin.random_state) == (8, 'auto', 'v3.5', 7)


@pytest.fixture(scope='module')
def tiny_problem():
    return {'synthetic': SPEC.to_dict(), 'n_dims': 12, 'noise': 'none',
            'labels': {'kind': 'synthetic', 'subtype': 'synthetic'}}


def test_campaign_switches_modules_by_round_and_logs_diagnostics(tiny_problem):
    arm = {'name': 'm', 'surrogate': {'type': 'gp_matern'}, 'schedule': {'type': 'legacy'},
           'selector': {'type': 'soft_penalty', 'radius_scale': 1.0, 'rounds': [3, 4]},
           'hillclimb': {'n_starts': 4, 'max_steps': 5, 'rounds': [2, 2]}}
    settings = Settings(batch_size=6, n_rounds=4, pool_size=300, test_points=50, snapshot=True)
    r = run_campaign(tiny_problem, arm, 0, settings)
    by_round = {x['round']: x for x in r['rounds']}
    assert [by_round[k]['selector'] for k in (2, 3, 4)] == ['topk', 'soft_penalty', 'soft_penalty']
    assert 'hc_steps' in by_round[2] and 'hc_steps' not in by_round[3]
    assert all(0 <= s <= 5 for s in by_round[2]['hc_steps'])
    assert all(by_round[k]['batch_min_gap'] > 0 for k in (2, 3, 4))
    assert r['problem']['spacing'] == pytest.approx(typical_spacing(get_problem(tiny_problem).oracle))
    snaps = r['_snapshots']
    assert {'r2_cand', 'r2_score', 'r2_chosen', 'r2_n_pool', 'r4_cand'} <= set(snaps)
    assert len(snaps['r3_cand']) == len(snaps['r3_score'])


def test_plain_arm_unchanged_by_the_new_code_path(tiny_problem):
    """Fit and predict now happen in the campaign; a plain arm must pick what suggest() would."""
    arm = {'name': 'p', 'surrogate': {'type': 'gp_matern'}, 'schedule': {'type': 'legacy'}}
    settings = Settings(batch_size=5, n_rounds=3, pool_size=200, test_points=40)
    a = run_campaign(tiny_problem, arm, 3, settings)
    b = run_campaign(tiny_problem, arm, 3, settings)
    assert a['evaluated']['idx'] == b['evaluated']['idx']
    assert '_snapshots' not in a


def test_module_experiment_tasks():
    cfg = experiments.load('experiments/tabpfn_modules.yaml')
    names = [a['name'] for a in cfg['arms']]
    assert names == ['control', 'penalty_low', 'penalty_high', 'fantasy', 'hillclimb']
    for a in cfg['arms']:
        assert a['surrogate'] == {'type': 'tabpfn', 'version': 'v3.5', 'n_estimators': 8, 'device': 'auto'}
        assert arm_lib.resolve(a)['schedule']['type'] == 'exploit_ramp'      # TabPFN's tuned schedule
    problems = [p['labels']['subtype'] for p in experiments.problem_configs(cfg)]
    assert problems == ['synthetic', 'borkowski', 'cell_growth']
    if not data_file().exists():
        pytest.skip('cell growth data not present')
    tasks = experiments.tasks(cfg, 'results')
    per = {}
    for problem_cfg, arm, seed, settings, _ in tasks:
        per.setdefault(problem_cfg['labels']['subtype'], set()).add(arm['name'])
        assert settings['snapshot'] == (seed == 0) and settings['batch_size'] == 10
    assert 'hillclimb' not in per['borkowski'] and 'hillclimb' in per['cell_growth'] and 'hillclimb' in per['synthetic']
    assert len(tasks) == 3 * (5 + 4 + 5)


def test_paired_effects_and_category_groups():
    import pandas as pd
    from metis.space import DesignSpace, Factor
    from metis_bench.modules_report import _same_levels, paired_effects
    runs = pd.DataFrame({'dataset': ['d'] * 4, 'arm': ['control', 'control', 'm', 'm'], 'seed': [0, 1, 0, 1],
                         'auc_norm_regret': [0.3, 0.4, 0.2, 0.5], 'best_fraction': [0.8, 0.9, 1.0, 0.7]})
    e = paired_effects(runs).sort_values('seed')
    assert np.allclose(e.d_auc, [-0.1, 0.1]) and np.allclose(e.d_best, [0.2, -0.2])
    space = DesignSpace([Factor('a', [0.0, 1.0, 2.0]), Factor('strain', [1.0], alternatives=('x', 'y', 'z'))])
    picks = np.array([[1, 0, 0], [1, 0, 2], [2, 0, 1], [1, 0, 1]])     # three picks share a=1, one has a=2
    groups = _same_levels(space, picks, np.arange(8.0).reshape(4, 2))
    assert [(list(p), n) for p, n in groups] == [([0.0, 1.0], 3)]


def test_task_key_ignores_snapshot_off():
    task = ({'measured': 'borkowski', 'noise': 'none'}, {'name': 'x'}, 0, {'batch_size': 10}, 'results/a/b/c/s.json')
    off = (task[0], task[1], 0, {'batch_size': 10, 'snapshot': False}, task[4])
    on = (task[0], task[1], 0, {'batch_size': 10, 'snapshot': True}, task[4])
    assert task_key(task) == task_key(off) != task_key(on)


@pytest.mark.skipif(not data_file().exists(), reason='cell growth data not present (data/README.md)')
def test_cell_growth_grid_and_neighbours():
    from metis_bench.oracles.measured import build_measured
    o = build_measured('cell_growth')
    assert o.meta['n_recipes'] == 13944 and o.meta['n_strains'] == 115 and o.meta['n_media'] == 135
    sp = o.space
    row = sp.pool[0]
    nb = sp.neighbors(row)
    assert sp.feasible(nb).all() and len(nb)
    conc = slice(0, sp.n_conc)
    same_medium = (nb[:, conc] == row[conc]).all(axis=1)
    assert (nb[same_medium, sp.strain_dim] != row[sp.strain_dim]).all()               # strain changed
    assert (nb[~same_medium, sp.strain_dim] == row[sp.strain_dim]).all()              # or medium changed
    assert (~same_medium).sum() <= sp.k_media
    assert o.f_max == pytest.approx(o(o.x_max[None])[0])
