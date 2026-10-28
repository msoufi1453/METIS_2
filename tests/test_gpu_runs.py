"""Running TabPFN on a GPU next to CPU arms: the GPU check, the separate worker pool, provenance."""
import json

import pytest
import yaml

from metis_bench import campaign
from metis_bench.__main__ import main
from metis_bench.campaign import Settings, run_campaign, run_tasks, uses_gpu


def test_uses_gpu():
    task = lambda surrogate: ({}, {'name': 'a', 'surrogate': surrogate}, 0, {}, 'x')   # noqa: E731
    assert uses_gpu(task({'type': 'tabpfn', 'device': 'auto'}))
    assert uses_gpu(task({'type': 'tabpfn', 'device': 'cuda'}))
    assert not uses_gpu(task({'type': 'tabpfn', 'device': 'cpu'}))
    assert not uses_gpu(task({'type': 'tabpfn'}))           # TabPFNSurrogate's own default is the CPU
    assert not uses_gpu(({}, {'name': 'incumbent'}, 0, {}, 'x'))


def test_tabpfn_runs_stop_without_a_gpu(tmp_path, capsys):
    torch = pytest.importorskip('torch')
    if torch.cuda.is_available():
        pytest.skip('this machine has a GPU')
    spec = {'name': 'gpu_check', 'arms': [{'name': 't', 'surrogate': {'type': 'tabpfn', 'version': 'v3.5',
                                                                       'device': 'auto'},
                                           'schedule': {'type': 'legacy'}}],
            'problems': {'measured': [{'name': 'borkowski', 'settings': {'batch_size': 10}}]},
            'settings': {'n_rounds': 2}, 'seeds': [0, 1]}
    path = tmp_path / 'gpu_check.yaml'
    path.write_text(yaml.safe_dump(spec))
    assert main(['run', str(path), '--out', str(tmp_path / 'results')]) == 2
    err = capsys.readouterr().err
    assert 'no CUDA GPU' in err and '--allow-cpu-tabpfn' in err
    assert not (tmp_path / 'results').exists()       # stopped before any campaign


def test_gpu_campaigns_get_their_own_pool(tmp_path, monkeypatch):
    """Route one cheap campaign to the GPU pool (the routing happens in this process)."""
    monkeypatch.setattr(campaign, 'uses_gpu', lambda task: task[1]['name'] == 'gpu_side')
    problem = {'measured': 'borkowski', 'noise': 'none'}
    settings = {**vars(Settings()), 'batch_size': 8, 'n_rounds': 2, 'pool_size': 100000}
    tasks = [(problem, {'name': name, 'surrogate': {'type': 'linear_ensemble', 'members': 2}}, 0, settings,
              str(tmp_path / 'exp' / 'p' / name / 'seed_000.json')) for name in ('gpu_side', 'cpu_side')]
    tasks[1][1]['init'] = {'type': 'sobol'}          # a different campaign, so both are computed
    assert run_tasks(tasks, jobs=2, gpu_jobs=1, progress=lambda *_: None) == []
    for _, arm, _, _, path in tasks:
        assert json.loads(open(path).read())['arm_name'] == arm['name']


def test_results_record_code_and_surrogate():
    result = run_campaign({'measured': 'borkowski', 'noise': 'none'},
                          {'name': 'lin', 'surrogate': {'type': 'linear_ensemble', 'members': 2}}, 0,
                          Settings(batch_size=8, n_rounds=2, pool_size=100000))
    env = result['environment']
    assert len(env['code_hash']) == 16 and env['surrogate']['name'] == 'regressor_ensemble'
    json.dumps(env)
