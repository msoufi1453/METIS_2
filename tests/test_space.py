import numpy as np
import pandas as pd
import pytest

from metis import DesignSpace, Factor, concentration_to_volume, feature_columns, percentage_possible
from metis.design import allowed_output
from tests.regression.scenario import LIMITS, MAX_NL, REACTION_NL, DROP_NL


@pytest.fixture(scope='module')
def space():
    return DesignSpace.from_limits(LIMITS, reaction_vol_nl=REACTION_NL, drop_size_nl=DROP_NL, max_nl=MAX_NL)


def test_columns_and_levels_follow_metis(space):
    assert space.columns == feature_columns(LIMITS)
    for factor in space.factors:
        concs, _ = allowed_output(LIMITS[factor.name], reaction_vol_nl=REACTION_NL, drop_size_nl=DROP_NL)
        assert np.allclose(factor.levels, np.sort(concs))


def test_sample_is_feasible_unique_and_respects_exclusions(space):
    rng = np.random.default_rng(0)
    first = space.sample(300, rng)
    assert len(first) == 300 and space.feasible(first).all()
    assert len(set(space.keys(first))) == 300
    second = space.sample(300, rng, exclude=space.keys(first))
    assert not set(space.keys(first)) & set(space.keys(second))


def test_volume_matches_concentration_to_volume(space):
    idx = space.sample(50, np.random.default_rng(1))
    volumes = concentration_to_volume(space.frame(idx), LIMITS, reaction_mixture_vol_nl=REACTION_NL, fixed_parts={})
    variable = volumes[list(LIMITS)].sum(axis=1).to_numpy()
    assert np.allclose(variable, space.volume_nl(idx))


def test_round_trips(space):
    idx = space.sample(200, np.random.default_rng(2))
    assert (space.from_features(space.features(idx)) == idx).all()
    assert (space.from_unit(space.unit(idx)) == idx).all()
    assert (space.from_frame(space.frame(idx)) == idx).all()


def test_enumeration_matches_percentage_possible():
    limits = {k: LIMITS[k] for k in ['PEG', 'DTT', 'Spermidine', 'TF-plasmid']}
    space = DesignSpace.from_limits(limits, reaction_vol_nl=REACTION_NL, drop_size_nl=DROP_NL, max_nl=2500)
    feasible = sum(len(chunk) for chunk in space.enumerate())
    lists = {i: allowed_output(v, reaction_vol_nl=REACTION_NL, drop_size_nl=DROP_NL)[1] for i, v in enumerate(limits.values())}
    percent, total = percentage_possible(lists, threshold=2500)
    assert total == space.size
    assert feasible == round(percent * total / 100)


def test_log_scale_factor_and_neighbors():
    f = Factor('enzyme', [0, 1, 10, 100, 1000], drops=[0, 1, 2, 3, 4], log_scale=True)
    space = DesignSpace([f, Factor('salt', [1, 2, 3], drops=[1, 2, 3])], max_nl=500)
    units = f.level_units
    assert units[0] == 0 and units[-1] == 1 and np.allclose(np.diff(units[1:]), np.diff(units[1:])[0])
    nb = space.neighbors(np.array([2, 1]))
    assert {tuple(r) for r in nb} == {(1, 1), (3, 1), (2, 0), (2, 2)}
    assert space.feasible(np.array([[4, 2]]))[0] is np.False_
