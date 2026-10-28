"""Phase 3 experiment files resolve against winners and tuning outputs (mocked here)."""
import json
import pathlib

import pytest

from metis_bench import experiments

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture()
def mock_outputs(tmp_path, monkeypatch):
    """Winners (GP + local penalisation + local pool) and tuning files, full and pilot."""
    winners = {'pilot': False, 'interventions': {
        'F2:surrogate': {'surrogate': {'type': 'gp_matern', 'nu': 2.5}},
        'F3:selector': {'selector': {'type': 'local_penalization', 'min_radius': 0.15}},
        'F3c:pool': {'pool': {'type': 'local', 'local_share': 0.5}}}}
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
def test_hparam_grid_follows_winning_surrogate(mock_outputs, pilot):
    cfg = experiments.load(ROOT / 'experiments/phase3_hparams_tune.yaml', pilot=pilot)
    assert all(a['surrogate']['type'] == 'gp_matern' for a in cfg['arms'])
    assert all('schedule' not in a for a in cfg['arms'])       # amendment 6: the model's own schedule


def test_confirmatory_files(mock_outputs):
    hp = {a['name']: a for a in experiments.load(ROOT / 'experiments/phase3_hparams.yaml')['arms']}
    assert hp['tuned']['surrogate']['nu'] == 1.5 and hp['untuned']['surrogate']['nu'] == 2.5


def test_ablation_is_full_factorial(mock_outputs):
    cfg = experiments.load(ROOT / 'experiments/phase3_ablation.yaml')
    assert cfg['interventions'] == ['batch', 'pool', 'surrogate']   # no schedule part (amendment 6)
    assert len(cfg['arms']) == 8 and cfg['arms'][0]['name'] == 'incumbent'
    full = next(a for a in cfg['arms'] if len(a['ablation_subset']) == 3)
    assert full['surrogate'] == {'type': 'gp_matern', 'nu': 1.5}   # tuned hyperparameters
    assert full['pool']['type'] == 'local' and 'schedule' not in full


def test_pilot_ablation_can_restrict_parts(mock_outputs):
    cfg = experiments.load(ROOT / 'experiments/phase3_ablation.yaml', pilot=True)
    assert set(cfg['interventions']) <= {'surrogate', 'batch', 'pool'}
    assert len(cfg['arms']) == 2 ** len(cfg['interventions'])
