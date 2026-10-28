"""TabPFN's own exploration schedule (amendment 4): the ramp family, the default, the tuning run."""
import json

import numpy as np
import pytest
import yaml

from metis.schedules import ExploitRampSchedule, LegacySchedule, make
from metis_bench import arms, experiments, tuning
from metis_bench.__main__ import main
from metis_bench.campaign import Settings, run_campaign

RAMP = {'type': 'exploit_ramp', 'slope': 1.5}


@pytest.fixture
def tuned_file(tmp_path, monkeypatch):
    path = tmp_path / 'tabpfn_schedule.json'
    path.write_text(json.dumps({'part': 'schedule', 'fragment': RAMP}))
    monkeypatch.setattr(arms, 'TABPFN_SCHEDULE', path)
    return path


def test_ramp_weights():
    w = [ExploitRampSchedule(1.0).weights(r, 10) for r in range(2, 11)]
    assert w[0] == pytest.approx((1 / 9, 8 / 9)) and w[-1] == (1.0, 0.0)
    assert all(e + x == pytest.approx(1.0) for e, x in w)
    assert np.all(np.diff([e for e, _ in w]) > 0)                     # exploitation only rises
    fast = ExploitRampSchedule(4.5)
    assert fast.weights(2, 10) == (0.5, 0.5) and fast.weights(3, 10) == (1.0, 0.0)
    assert ExploitRampSchedule(0.8).weights(10, 10)[0] == pytest.approx(0.8)   # never fully greedy
    assert make({'type': 'exploit_ramp', 'slope': 2.25}).weights(5, 10) == (1.0, 0.0)
    assert LegacySchedule().weights(2, 10) == (None, 1.41)            # other schedules leave exploitation alone


def test_tuning_file_candidates_follow_the_spec():
    cfg = experiments.load('experiments/tabpfn_schedule_tune.yaml')
    ramps = [a['schedule'] for a in cfg['arms'] if a['schedule']['type'] == 'exploit_ramp']
    assert len(ramps) == 10 and len({r['slope'] for r in ramps}) == 10
    for r in ramps:
        e_first, _ = make(r).weights(2, 10)
        e_last, _ = make(r).weights(10, 10)
        assert e_first <= 0.5 and e_last >= 0.8      # start mostly exploring, end mostly exploiting
    assert all(a['surrogate']['type'] == 'tabpfn' and a['surrogate']['version'] == 'v3.5' for a in cfg['arms'])
    assert [a['name'] for a in cfg['arms']][-1] == 'tabpfn_v1_schedule'
    assert cfg['tuning']['output'] == 'experiments/tuned/tabpfn_schedule.json'


def test_campaign_applies_both_weights():
    problem = {'measured': 'borkowski', 'noise': 'none'}
    arm = {'name': 'gp_ramp', 'surrogate': {'type': 'gp_matern', 'n_restarts': 0}, 'schedule': RAMP}
    rounds = run_campaign(problem, arm, 0, Settings(batch_size=10, n_rounds=4, pool_size=100000))['rounds']
    expected = [ExploitRampSchedule(1.5).weights(r, 4) for r in (2, 3, 4)]
    assert [(x['exploitation'], x['exploration']) for x in rounds] == pytest.approx(expected)
    legacy = run_campaign(problem, {**arm, 'schedule': {'type': 'legacy'}}, 0,
                          Settings(batch_size=10, n_rounds=3, pool_size=100000))['rounds']
    assert [(x['exploitation'], x['exploration']) for x in legacy] == [(1.0, 1.41), (1.0, 1.41)]


def test_tabpfn_arms_take_the_tuned_schedule(tuned_file):
    tabpfn = {'name': 't', 'surrogate': {'type': 'tabpfn'}}
    assert arms.resolve(tabpfn)['schedule'] == RAMP
    assert arms.resolve({**tabpfn, 'schedule': {'type': 'legacy'}})['schedule'] == {'type': 'legacy'}
    assert arms.resolve({'name': 'x'})['schedule'] == {'type': 'legacy'}            # XGBoost keeps v1
    # The decision rule sees "TabPFN with its own schedule" as one surrogate intervention ...
    assert tuning._differing_parts(tabpfn, {'name': 'incumbent'}) == ['surrogate']
    # ... but a schedule set on purpose is a second part.
    assert tuning._differing_parts({**tabpfn, 'schedule': {'type': 'constant'}}, {'name': 'incumbent'}) == [
        'surrogate', 'schedule']


def test_untuned_tabpfn_refuses_to_run_but_pilots_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(arms, 'TABPFN_SCHEDULE', tmp_path / 'missing.json')
    with pytest.raises(FileNotFoundError, match='tabpfn_schedule_tune'):
        arms.resolve({'name': 't', 'surrogate': {'type': 'tabpfn'}})
    cfg = experiments.load('experiments/phase2_surrogates.yaml', pilot=True)
    tabpfn = next(a for a in cfg['arms'] if a['name'] == 'surr_tabpfn')
    assert tabpfn['schedule'] == {'type': 'legacy'}
    full = experiments.load('experiments/phase2_surrogates.yaml')
    assert 'schedule' not in next(a for a in full['arms'] if a['name'] == 'surr_tabpfn')


def test_reference_arm_cannot_win():
    import pandas as pd
    runs = pd.DataFrame({'arm': ['ref', 'a', 'b'] * 2, 'problem_id': ['p'] * 3 + ['q'] * 3,
                         'auc_norm_regret': [0.0, 0.2, 0.1, 0.0, 0.2, 0.1]})
    winner, table = tuning.select_winner(runs, metric='auc_norm_regret', exclude=['ref'])
    assert winner == 'b' and table.index[0] == 'ref'


def test_tune_command_end_to_end(tmp_path):
    """The whole tune path on one dataset, with a fast GP standing in for TabPFN."""
    out = tmp_path / 'tuned.json'
    spec = {'name': 'ramp_tune', 'tuning': {'part': 'schedule', 'output': str(out), 'metric': 'auc_norm_regret'},
            'base_arm': {'surrogate': {'type': 'gp_matern', 'n_restarts': 0}},
            'candidates': [{'type': 'exploit_ramp', 'slope': [1.0, 4.5]}],
            'reference_arms': [{'name': 'v1_schedule', 'schedule': {'type': 'legacy'}}],
            'problems': {'measured': [{'name': 'borkowski', 'settings': {'batch_size': 10}},
                                     {'name': 'cetch_yield', 'settings': {'batch_size': 10}}]},  # a name with a space
            'settings': {'n_rounds': 3, 'pool_size': 100000}, 'seeds': [0, 2],
            'pilot': {'seeds': [0, 1]}}   # a pilot block must not mark the real run as a pilot
    path = tmp_path / 'ramp_tune.yaml'
    path.write_text(yaml.safe_dump(spec))
    assert main(['tune', str(path), '--out', str(tmp_path / 'results')]) == 0
    tuned = json.loads(out.read_text())
    assert tuned['part'] == 'schedule' and tuned['fragment']['type'] == 'exploit_ramp'
    assert tuned['metric'] == 'auc_norm_regret' and tuned['winner'] != 'v1_schedule' and tuned['pilot'] is False
    report = tmp_path / 'results' / 'ramp_tune' / 'tuning'
    text = (report / 'tuning_report.md').read_text(encoding='utf-8')
    assert '(winner)' in text and 'reference' in text and 'Borkowski' in text and 'CETCH yield' in text
    assert (report / 'tuning_curve.png').stat().st_size > 0
