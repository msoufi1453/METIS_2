import numpy as np
import pandas as pd
import pytest

from metis_bench import calibration as C
from metis_bench import descriptors as D
from metis_bench.synthetic import SUBTYPES, FunctionSpec, build_function


def test_threshold_statistic_is_bounded_and_detects_steps():
    u = np.linspace(0, 1, 8)
    counts = np.ones(8)
    step, _ = D._profile_threshold(u, (u > 0.5).astype(float), counts)
    smooth, _ = D._profile_threshold(u, u ** 2, counts)
    flat, _ = D._profile_threshold(u, np.full(8, 3.0) + 1e-15 * np.arange(8), counts)
    assert step > 0.2 > smooth and smooth < 0 and flat == 0.0
    for _ in range(50):
        v, _ = D._profile_threshold(u, np.random.default_rng(_).random(8), counts)
        assert -1 <= v <= 1


def _refs():
    rows = []
    for dataset, base in (('borkowski', 0.2), ('pure', 0.8)):
        for model in ('xgb', 'gp'):
            rows.append({'dataset': dataset, 'source': 'ref_' + model,
                         **{c: base for c in D.REALISM_DESCRIPTORS}})
    return pd.DataFrame(rows)


def test_realism_is_distance_to_nearest_reference():
    refs = _refs()
    synth = pd.DataFrame([{c: v for c in D.REALISM_DESCRIPTORS} for v in (0.2, 0.75, 3.0)])
    d, nearest = C.realism_scores(synth, refs, D.REALISM_DESCRIPTORS)
    assert d[0] == pytest.approx(0) and nearest[0].startswith('borkowski')
    assert nearest[1].startswith('pure') and d[2] > d[1] > d[0]


def test_skew_outliers_do_not_swamp_the_scale():
    refs = _refs()
    synth = pd.DataFrame([{**{c: 0.2 for c in D.REALISM_DESCRIPTORS}, 'skew': s} for s in (0.2, 1.0, 60.0)])
    Z = C.standardize(synth, ['skew'])[0]['skew'].to_numpy()
    raw = synth['skew'].to_numpy()
    # After the signed log, skew 1 vs 0.2 keeps about ten times more of the scale than raw.
    share = (Z[1] - Z[0]) / (Z[2] - Z[0])
    assert share > 0.1 and share > 8 * (raw[1] - raw[0]) / (raw[2] - raw[0])


def test_selection_mixes_realistic_and_coverage_picks():
    rng = np.random.default_rng(0)
    rows = []
    for subtype in SUBTYPES:
        for k in range(8):
            spec = FunctionSpec(subtype, seed=k)
            level = rng.uniform(0, 1.5)
            for n_dims in (12, 27):
                rows.append({'spec_id': spec.id, 'subtype': subtype, 'n_dims': n_dims,
                             **{c: level + rng.normal(scale=0.05) for c in D.SCREENING_DESCRIPTORS}})
    selected = C.select_suite(pd.DataFrame(rows), _refs(), per_subtype=5)
    assert len(selected) == 45 and selected.groupby('subtype').size().eq(5).all()
    assert (selected.selection == 'realistic').sum() == 27 and (selected.selection == 'coverage').sum() == 18


def test_split_is_stratified_and_deterministic():
    ids = ['{}.{}'.format(s, k) for s in SUBTYPES for k in range(5)]
    subtypes = [s for s in SUBTYPES for _ in range(5)]
    split = C.assign_split(ids, subtypes)
    assert split == C.assign_split(list(reversed(ids)), list(reversed(subtypes)))
    per_subtype = pd.Series(split).groupby(lambda i: i.split('.')[0]).apply(lambda s: (s == 'dev').sum())
    assert per_subtype.eq(2).all()


def test_landscape_descriptors_on_a_suite_function():
    fn = build_function(FunctionSpec('broad', active='subset', seed=1), 12)
    d = D.landscape_descriptors(fn, fn.space, np.random.default_rng(0), n_base=128, n_starts=10, full=False)
    assert set(D.SCREENING_DESCRIPTORS) <= set(d)
    assert 0 <= d['s_first'] <= 1 and 0 <= d['frac_factors_80'] <= 1 and 0 < d['peak_frac'] <= 1
