"""Amendment 6: fewer XGBoost runs (trimmed side questions at 10 seeds, no schedule re-tuning)."""
import json
import pathlib

import pandas as pd

from metis_bench import experiments, report

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _arms(name):
    return [a['name'] for a in experiments.load(ROOT / 'experiments/{}.yaml'.format(name))['arms']]


def test_phase2_arms():
    assert _arms('phase2_surrogates') == ['incumbent', 'surr_gp', 'surr_tabpfn']
    assert _arms('phase2_init') == ['incumbent', 'init_sobol', 'random_search']
    assert _arms('phase2_batch') == ['incumbent', 'batch_fantasy', 'batch_lp', 'pool_local']
    batch = experiments.load(ROOT / 'experiments/phase2_batch.yaml')
    assert set(batch['families']) == {'F3', 'F3c'}


def test_pipelines_skip_dropped_steps_and_start_with_the_surrogates():
    for script in ('run_pipeline.sh', 'run_pipeline.ps1'):
        text = (ROOT / 'scripts' / script).read_text()
        code = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
        for dropped in ('phase2_import', 'phase3_schedule'):
            assert dropped not in code, (script, dropped)
        assert code.index('phase2_surrogates') < code.index('phase2_init') < code.index('phase2_batch')


def test_report_ignores_results_outside_the_plan():
    """Files from a dropped arm or extra seeds stay on disk but are not analysed."""
    frame = pd.DataFrame([{'problem_id': 'p', 'arm': arm, 'seed': seed, 'final_norm_regret': 0.1}
                          for arm in ('incumbent', 'init_sobol', 'init_lhs') for seed in range(20)])
    kept = report.in_plan(experiments.load(ROOT / 'experiments/phase2_init.yaml'), frame)
    assert set(kept.arm) == {'incumbent', 'init_sobol'} and set(kept.seed) == set(range(10))
    assert report.in_plan({'arms': [], 'seeds': [0, 1]}, frame.iloc[:0]).empty


def test_preregistration_records_amendment_6():
    text = (ROOT / 'docs' / 'PREREGISTRATION.md').read_text(encoding='utf-8')
    assert 'Amendment 6' in text
    assert json.loads((ROOT / 'experiments/tuned/tabpfn_schedule.json').read_text())['pilot'] is False
