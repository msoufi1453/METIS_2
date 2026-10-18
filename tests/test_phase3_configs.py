"""Phase 3 experiment files resolve against winners and tuning outputs (mocked here)."""
import json
import pathlib

import pytest

from metis_bench import experiments

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture()
def mock_outputs(tmp_path, monkeypatch):
    """Winners (GP + local penalisation + noise feeding) and tuning files, full and pilot."""
    winners = {'pilot': False, 'interventions': {
        'F2:surrogate': {'surrogate': {'type': 'gp_matern', 'nu': 2.5}},
        'F3:selector': {'selector': {'type': 'local_penalization', 'min_radius': 0.15}},
        'F3b:use_noise': {'use_noise': True}}}
    schedule = {'part': 'schedule', 'fragment': {'type': 'linear', 'start': 2.0, 'end': 0.25}}
    surrogate = {'part': 'surrogate', 'fragment': {'type': 'gp_matern', 'nu': 1.5}}
    files = {'artifacts/winners.json': winners, 'artifacts/tuning/schedule.json': schedule,
             'artifacts/tuning/surrogate.json': surrogate}
    for rel, content in files.items():
        for suffix in ('', '-pilot'):
            path = tmp_path / rel.replace('.json', suffix + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content))
    (tmp_path / 'experiments').symlink_to(ROOT / 'experiments', target_is_directory=True) \
        if hasattr(pathlib.Path, 'symlink_to') and False else None
    monkeypatch.setattr(experiments, 'ROOT', tmp_path)
    return tmp_path


@pytest.mark.parametrize('pilot', [False, True])
def test_schedule_tuning_builds_on_winners(mock_outputs, pilot):
    cfg = experiments.load(ROOT / 'experiments/phase3_schedule_tune.yaml', pilot=pilot)
    assert all(a['surrogate']['type'] == 'gp_matern' and a['use_noise'] for a in cfg['arms'])
    assert len({a['name'] for a in cfg['arms']}) == len(cfg["arms"]) == (6 if pilot else 15)


@pytest.mark.parametrize('pilot', [False, True])
def test_hparam_grid_follows_winning_surrogate(mock_outputs, pilot):
    cfg = experiments.load(ROOT / 'experiments/phase3_hparams_tune.yaml', pilot=pilot)
    assert all(a['surrogate']['type'] == 'gp_matern' for a in cfg['arms'])
    assert all(a['schedule'] == {'type': 'linear', 'start': 2.0, 'end': 0.25} for a in cfg['arms'])


def test_confirmatory_files(mock_outputs):
    sched = {a['name']: a for a in experiments.load(ROOT / 'experiments/phase3_schedule.yaml')['arms']}
    assert sched['tuned_schedule']['schedule']['type'] == 'linear'
    assert sched['winners_legacy_schedule']['schedule'] == {'type': 'legacy'}
    assert sched['incumbent_check']['surrogate'] == {'type': 'xgb_ensemble'}
    hp = {a['name']: a for a in experiments.load(ROOT / 'experiments/phase3_hparams.yaml')['arms']}
    assert hp['tuned']['surrogate']['nu'] == 1.5 and hp['untuned']['surrogate']['nu'] == 2.5


def test_ablation_is_full_factorial(mock_outputs):
    cfg = experiments.load(ROOT / 'experiments/phase3_ablation.yaml')
    assert cfg['interventions'] == ['batch', 'noise', 'schedule', 'surrogate']
    assert len(cfg['arms']) == 16 and cfg['arms'][0]['name'] == 'incumbent'
    full = next(a for a in cfg['arms'] if len(a['ablation_subset']) == 4)
    assert full['surrogate'] == {'type': 'gp_matern', 'nu': 1.5}   # tuned hyperparameters
    assert full['schedule']['type'] == 'linear' and full['use_noise'] is True


def test_pilot_ablation_can_restrict_parts(mock_outputs):
    cfg = experiments.load(ROOT / 'experiments/phase3_ablation.yaml', pilot=True)
    assert set(cfg['interventions']) <= {'surrogate', 'batch', 'pool', 'schedule'}
    assert len(cfg['arms']) == 2 ** len(cfg['interventions'])
