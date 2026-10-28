"""Amendment 3: shared identical campaigns, the stratified ablation sample, arm filters, configs."""
import json
import pathlib

import pytest

from metis_bench import experiments
from metis_bench.__main__ import main
from metis_bench.arms import INCUMBENT, resolve
from metis_bench.campaign import Settings, run_tasks, shared_path, task_key
from metis_bench.synthetic import FunctionSpec

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBLEM = {'synthetic': FunctionSpec('switch_on', active='subset', seed=2).to_dict(), 'n_dims': 12,
           'noise': 'realistic', 'labels': {'landscape': 'threshold'}}
SETTINGS = vars(Settings(batch_size=8, n_rounds=3, pool_size=300, test_points=100))
RANDOM_ARM = {'selector': {'type': 'random'}}


def _task(out, experiment, name, arm=None, seed=0, problem=PROBLEM):
    path = out / experiment / 'p' / name / 'seed_{:03d}.json'.format(seed)
    return (problem, {**(arm or RANDOM_ARM), 'name': name}, seed, SETTINGS, str(path))


def test_identical_campaigns_run_once_and_are_shared(tmp_path):
    a = _task(tmp_path, 'exp1', 'arm_a')
    b = _task(tmp_path, 'exp1', 'arm_b')                     # same campaign, different name
    other = _task(tmp_path, 'exp1', 'arm_c', seed=1)         # different seed: its own campaign
    assert task_key(a) == task_key(b) != task_key(other)
    messages = []
    assert run_tasks([a, b, other], progress=messages.append) == []
    assert '2 distinct campaigns' in messages[0]
    ra, rb = (json.loads(pathlib.Path(t[4]).read_text()) for t in (a, b))
    assert ra['arm_name'] == 'arm_a' and rb['arm_name'] == 'arm_b'
    assert ra['evaluated'] == rb['evaluated'] and ra['shared_key'] == task_key(a)
    # A later experiment asking for the same campaign reuses it without computing.
    c = _task(tmp_path, 'exp2', 'incumbent_check')
    messages = []
    run_tasks([c], progress=messages.append)
    assert '1 reused' in messages[0] and '0 distinct' in messages[0]
    assert json.loads(pathlib.Path(c[4]).read_text())['evaluated'] == ra['evaluated']
    assert shared_path(c, task_key(c)).exists()


def test_labels_and_names_do_not_change_the_key():
    relabelled = dict(PROBLEM, labels={'landscape': 'something else'})
    assert task_key(_task(pathlib.Path('x'), 'e', 'a')) == task_key(_task(pathlib.Path('y'), 'f', 'b', problem=relabelled))
    noisy = dict(PROBLEM, noise='high')
    assert task_key(_task(pathlib.Path('x'), 'e', 'a')) != task_key(_task(pathlib.Path('x'), 'e', 'a', problem=noisy))


def test_stratified_sample_spreads_over_strata():
    functions = [{'function_id': '{}.{}.{}'.format(s, d, k), 'spec': {'subtype': s}, 'n_dims': d}
                 for s in ('a', 'b', 'c') for d in (12, 27) for k in range(3)]
    chosen = experiments.stratified_sample(functions, 8)
    strata = [(f['spec']['subtype'], f['n_dims']) for f in chosen]
    assert len(chosen) == 8 and len(set(strata)) == 6          # every stratum before any second pick
    assert chosen == experiments.stratified_sample(list(reversed(functions)), 8)


def test_ablation_problems_are_thirty_realistic():
    cfg = {'problems': {'suite': 'suite/suite.json', 'split': 'test', 'dims': [12, 27], 'stratified_sample': 28,
                        'real': ['borkowski', 'pure_pool'], 'noise': ['realistic']}}
    problems = experiments.problem_configs(cfg)
    assert len(problems) == 30 and {p['noise'] for p in problems} == {'realistic'}
    synthetic = [p for p in problems if 'synthetic' in p]
    assert len({(p['synthetic']['subtype'], p['n_dims']) for p in synthetic}) == 18


def test_seeds_and_tabpfn_coverage():
    # Amendment 6: the surrogate comparison keeps 20 seeds, the side questions use 10.
    assert experiments.load(ROOT / 'experiments/phase2_surrogates.yaml')['seeds'] == [0, 20]
    for name in ('phase2_init', 'phase2_batch'):
        assert experiments.load(ROOT / 'experiments/{}.yaml'.format(name))['seeds'] == [0, 10]
    cfg = experiments.load(ROOT / 'experiments/phase2_surrogates.yaml')
    tabpfn = next(a for a in cfg['arms'] if a['name'] == 'surr_tabpfn')
    assert tabpfn['surrogate']['device'] == 'auto'
    assert 'stratified_sample' not in cfg['problems']           # full coverage for the surrogate comparison


def test_incumbent_check_is_the_true_incumbent(tmp_path, monkeypatch):
    winners = {'interventions': {'F3c:pool': {'pool': {'type': 'local'}},
                                 'F2:surrogate': {'surrogate': {'type': 'gp_matern'}}}}
    (tmp_path / 'artifacts' / 'tuning').mkdir(parents=True)
    (tmp_path / 'artifacts' / 'winners.json').write_text(json.dumps(winners))
    (tmp_path / 'artifacts' / 'tuning' / 'schedule.json').write_text(
        json.dumps({'part': 'schedule', 'fragment': {'type': 'constant', 'beta': 1.0}}))
    monkeypatch.setattr(experiments, 'ROOT', tmp_path)
    cfg = experiments.load(ROOT / 'experiments/phase3_schedule.yaml')
    check = next(a for a in cfg['arms'] if a['name'] == 'incumbent_check')
    same = {k: v for k, v in resolve(check).items() if k != 'name'}
    assert same == {k: v for k, v in INCUMBENT.items() if k != 'name'}


def test_cli_arm_filters(capsys):
    main(['tasks', 'experiments/phase2_surrogates.yaml', '--pilot', '--arms', 'surr_tabpfn'])
    only = capsys.readouterr().out.split()
    assert only and all('surr_tabpfn' in p for p in only)
    main(['tasks', 'experiments/phase2_surrogates.yaml', '--pilot', '--skip-arms', 'surr_tabpfn'])
    rest = capsys.readouterr().out.split()
    assert rest and not any('surr_tabpfn' in p for p in rest)
