import numpy as np
import pandas as pd
import pytest

from metis_bench import metrics, stats
from metis_bench.noise import PRESETS


def test_trajectory_and_endpoints():
    # f_max = 10, f_ref = 0. Round 1 best 4, round 2 best 9.6 (measured badly), round 3 hits 10.
    rounds = [1, 1, 2, 2, 3]
    true = [4.0, 2.0, 9.6, 1.0, 10.0]
    measured = [4.1, 2.2, 3.0, 1.1, 9.9]
    rows = metrics.trajectory(rounds, true, measured, f_max=10.0, f_ref=0.0, n_rounds=3)
    assert [r['norm_regret'] for r in rows] == pytest.approx([0.6, 0.04, 0.0])
    # After round 2 the best *measured* recipe is still the round-1 one (4.1 > 3.0).
    assert rows[1]['norm_rec_regret'] == pytest.approx(0.6)
    assert rows[2]['norm_cum_regret'] == pytest.approx(np.mean([0.6, 0.8, 0.04, 0.9, 0.0]))
    ep = metrics.endpoints(rows, f_max=10.0)
    assert ep['final_norm_regret'] == 0.0 and ep['found_optimum']
    assert ep['rounds_to_target'] == 2
    assert ep['auc_norm_regret'] == pytest.approx(np.mean([0.6, 0.04, 0.0]))


def test_calibration_of_a_calibrated_model():
    rng = np.random.default_rng(0)
    mean = rng.normal(size=20000)
    std = np.full_like(mean, 0.5)
    truth = mean + rng.normal(scale=0.5, size=mean.size)
    cal = metrics.calibration(mean, std, truth)
    assert cal['cov80'] == pytest.approx(0.8, abs=0.01)
    assert cal['mace'] < 0.01 and cal['z_sd'] == pytest.approx(1.0, abs=0.02)
    overconfident = metrics.calibration(mean, std / 4, truth)
    assert overconfident['cov80'] < 0.4 and overconfident['nlpd'] > cal['nlpd']


def _runs(effect, n_problems=12, n_seeds=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_problems):
        base = rng.uniform(0.2, 0.8)
        for s in range(n_seeds):
            shared = rng.normal(scale=0.05)
            rows.append({'problem_id': 'p{}'.format(p), 'seed': s, 'arm': 'ref', 'm': base + shared})
            rows.append({'problem_id': 'p{}'.format(p), 'seed': s, 'arm': 'better', 'm': base + shared - effect
                         + rng.normal(scale=0.02)})
            rows.append({'problem_id': 'p{}'.format(p), 'seed': s, 'arm': 'same', 'm': base + shared
                         + rng.normal(scale=0.02)})
    return pd.DataFrame(rows)


def test_compare_detects_effect_and_controls_null():
    out = stats.compare(_runs(0.1), ['better', 'same'], 'ref', 'm', n_boot=500).set_index('arm')
    assert out.loc['better', 'significant'] and out.loc['better', 'mean_diff'] == pytest.approx(-0.1, abs=0.01)
    assert out.loc['better', 'ci_low'] < -0.09 < out.loc['better', 'ci_high'] or out.loc['better', 'ci_high'] < 0
    assert not out.loc['same', 'significant']
    assert out.loc['better', 'share_better'] == 1.0


def test_holm():
    adjusted = stats.holm([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_noise_presets():
    rng = np.random.default_rng(0)
    f = np.linspace(0, 1, 4000)
    mean, std, wells = PRESETS['realistic'].measure(f, 1.0, rng)
    assert wells.shape == (4000, 3) and (wells >= 0).all()
    # Replicate spread grows with signal (heteroscedastic) as in the Borkowski triplicates.
    assert np.median(std[f > 0.8]) > 1.5 * np.median(std[f < 0.2])
    mean0, std0, _ = PRESETS['none'].measure(f, 1.0, rng)
    assert np.array_equal(mean0, f) and not std0.any()
