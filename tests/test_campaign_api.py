"""metis.campaign: variables, liquid handling, the recipe space, campaigns, the workbook,
pipetting and the iDOT file, the analysis functions, day 0 data and the tutorial.

A fast stand-in model (tests/helpers.py) replaces TabPFN, except in the one test marked
``tabpfn``.
"""
import io
import math
import re

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from metis.campaign import (MODELS, Campaign, CampaignError, Categorical, Continuous, Discrete, LiquidHandling,
                            RecipeSpace, VariableError, WorkbookError, analysis, day0, make_model, tutorial, workbook)
from metis.campaign import columns as col
from metis.campaign.batch import PenalizedFantasy, make_selector, penalty_radius, spacing
from metis.campaign.pipetting import WELLS, assign_wells, idot_protocol
from metis.campaign.variables import check_variables, from_dict
from metis.idot import IdotError, idot_builder
from metis.surrogates import Prediction

from .helpers import register_fast_model

FAST = register_fast_model()


def design():
    return [Continuous('Mg', 0, 10, unit='mM', stock=500), Discrete('K', [40, 80, 120, 160], unit='mM', stock=4000),
            Categorical('Promoter', ['T7', 'T3', 'SP6'])]


def lab(table, noise=0.0, seed=0):
    """A made-up experiment: best at Mg = 6, more K is better, T7 adds 0.5."""
    rng = np.random.default_rng(seed)
    return (2.0 - 0.05 * (table['Mg'].astype(float) - 6) ** 2 + table['K'].astype(float) / 100
            + 0.5 * (table['Promoter'] == 'T7') + rng.normal(0, noise, len(table)))


def measured_campaign(n=12, replicates=1, liquid=True, **options):
    """A campaign with one Latin-hypercube round measured."""
    campaign = Campaign(design(), target='sfGFP', model=FAST, name='test',
                        liquid_handling=LiquidHandling(sample_volume_ul=10) if liquid else None, **options)
    campaign.add_round(campaign.propose(n), replicates=replicates)
    campaign.data[campaign.target] = lab(campaign.data, noise=0.02)
    return campaign


# --------------------------------------------------------------------------- variables
def test_variables_normalise_and_round_trip():
    v = Discrete(' K   glutamate ', [80, 40, 160.0], unit=' mM ', stock=4000)
    assert (v.name, v.levels, v.unit, v.label) == ('K glutamate', (40.0, 80.0, 160.0), 'mM', 'K glutamate (mM)')
    for variable in design() + [Categorical('Strain', ['a', 'b'], volume_ul=0.5)]:
        assert from_dict(variable.to_dict()) == variable
    assert Categorical('Strain', [' wild  type ', 'ko']).categories == ('wild type', 'ko')
    assert not Discrete('Extract', [33]).varies and Continuous('x', 0, 1).varies


@pytest.mark.parametrize('build', [
    lambda: Continuous('', 0, 1), lambda: Continuous('x', 1, 1), lambda: Continuous('x', 2, 1),
    lambda: Continuous('x', 0, float('nan')), lambda: Continuous('x', 0, 1, stock=0), lambda: Continuous('x', 'a', 1),
    lambda: Discrete('x', []), lambda: Discrete('x', [1, 1]), lambda: Discrete('x', range(101)),
    lambda: Discrete('x', [1, float('inf')]), lambda: Categorical('x', []), lambda: Categorical('x', ['a', 'a']),
    lambda: Categorical('x', ['a', ' ']), lambda: Categorical('x', 'abc'), lambda: Categorical('x', ['a'], volume_ul=-1),
    lambda: from_dict({'type': 'other', 'name': 'x'})])
def test_bad_variables_are_refused(build):
    with pytest.raises(VariableError):
        build()


def test_check_variables():
    assert len(check_variables(design(), 'sfGFP')) == 3
    for variables, target in [([], 'y'), (design(), ''), (design() + [Continuous('mg', 0, 1)], 'y'),
                              ([Continuous('Round', 0, 1)], 'y'), ([Continuous('y', 0, 1)], 'Y'),
                              ([Continuous('Predicted y', 0, 1)], 'y'), ([Discrete('x', [1])], 'y'),
                              ([Continuous('a', 0, 1, unit='mM'), Continuous('a (mM)', 0, 1)], 'y')]:
        with pytest.raises(VariableError):
            check_variables(variables, target)
    with pytest.raises(VariableError, match='at most 2'):
        check_variables(design(), 'y', max_variables=2)


# --------------------------------------------------------------------------- liquid handling
def test_droplets_and_snapping():
    liquid = LiquidHandling(sample_volume_ul=10, droplet_nl=4)
    mg = Continuous('Mg', 0, 10, unit='mM', stock=500)
    assert liquid.budget == 2500
    assert liquid.step(mg) == pytest.approx(0.2)                      # 500 mM * 4 nL / 10 uL
    assert liquid.droplets(mg, [0, 0.29, 0.31, 10]).tolist() == [0, 1, 2, 50]
    assert liquid.snap(mg, [0.29, 3.33]).tolist() == [0.2, 3.4]
    assert liquid.droplet_range(Continuous('x', 0.3, 0.75, stock=500)) == (2, 3)
    assert float(liquid.volume_ul(50)) == pytest.approx(0.2)
    assert LiquidHandling.from_dict(liquid.to_dict()) == liquid and LiquidHandling.from_dict(None) is None
    for bad in ({'sample_volume_ul': 0}, {'sample_volume_ul': 10, 'droplet_nl': -1},
                {'sample_volume_ul': 0.001, 'droplet_nl': 4}, {'sample_volume_ul': 'x'}):
        with pytest.raises(VariableError):
            LiquidHandling(**bad)


def _messages(liquid, variables, level):
    return [issue.message for issue in liquid.check(variables) if issue.level == level]


def test_liquid_checks_catch_unreachable_ranges():
    liquid = LiquidHandling(sample_volume_ul=10, droplet_nl=4)
    assert liquid.check(design()) == []
    assert 'Enter the stock' in _messages(liquid, [Continuous('a', 0, 1)], 'error')[0]
    assert 'negative' in _messages(liquid, [Continuous('a', -1, 1, stock=10)], 'error')[0]
    assert 'above the stock' in _messages(liquid, [Continuous('a', 0, 20, stock=10)], 'error')[0]
    # stock too concentrated: one droplet adds 4, but the range is 1 to 3
    assert 'more dilute stock' in _messages(liquid, [Continuous('a', 1, 3, stock=10000)], 'error')[0]
    assert 'only 2 pipettable' in _messages(liquid, [Continuous('a', 0, 5, stock=10000)], 'warning')[0]
    assert 'less than half a droplet' in _messages(liquid, [Discrete('a', [1, 8], stock=10000)], 'error')[0]
    assert 'same number of droplets' in _messages(liquid, [Discrete('a', [4, 5], stock=10000)], 'error')[0]
    assert 'rounded to whole droplets (3 -> 4)' in _messages(liquid, [Discrete('a', [3, 8], stock=10000)], 'warning')[0]
    assert 'less than half a droplet' in _messages(liquid, [Categorical('c', ['a', 'b'], volume_ul=0.001)], 'error')[0]
    assert 'rounded' in _messages(liquid, [Categorical('c', ['a', 'b'], volume_ul=0.005)], 'warning')[0]


def test_liquid_checks_compare_volumes_with_the_sample():
    liquid = LiquidHandling(sample_volume_ul=10, droplet_nl=4)
    big = [Continuous('a', 0, 80, unit='mM', stock=100), Continuous('b', 0, 50, unit='mM', stock=100),
           Discrete('c', [1], unit='mM', stock=100)]
    warnings = _messages(liquid, big, 'warning')
    assert len(warnings) == 1 and '13.1 µL together, more than the 10 µL sample' in warnings[0]
    assert warnings[0].index('a (8 µL)') < warnings[0].index('b (5 µL)') < warnings[0].index('c (0.1 µL)')
    assert not _messages(liquid, big, 'error')
    too_much = [Discrete('a', [60], stock=100), Discrete('b', [50], stock=100), Continuous('c', 0, 1, stock=100)]
    assert 'Even at their lowest values' in _messages(liquid, too_much, 'error')[0]
    table = liquid.summary(design() + [Categorical('d', ['x', 'y'], volume_ul=0.4)])
    assert list(table['Variable']) == ['Mg (mM)', 'K (mM)', 'd']
    assert table['Pipettable values'].tolist() == [51, 4, 2]
    assert table['Volume at maximum (µL)'].tolist() == pytest.approx([0.2, 0.4, 0.4])


# --------------------------------------------------------------------------- the pipettable volume
def pipettable(volume=8.25):
    """10 µL samples of which the dispenser fills ``volume`` µL (4 nL droplets)."""
    return LiquidHandling(sample_volume_ul=10, droplet_nl=4, pipettable_volume_ul=volume)


def test_pipettable_volume_limits_the_droplets_not_the_concentrations():
    whole, part = LiquidHandling(sample_volume_ul=10, droplet_nl=4), pipettable()
    mg = Continuous('Mg', 0, 10, unit='mM', stock=500)
    assert part.step(mg) == whole.step(mg) == pytest.approx(0.2)        # per the whole 10 uL sample
    assert part.snap(mg, [3.33]).tolist() == whole.snap(mg, [3.33]).tolist() == [3.4]
    assert part.droplet_range(mg) == whole.droplet_range(mg) == (0, 50)
    assert (whole.budget, part.budget) == (2500, 2062)                  # 8.25 uL / 4 nL = 2062.5 droplets
    assert (part.pipettable_ul, part.reserved_ul, whole.pipettable_ul, whole.reserved_ul) == (8.25, 1.75, 10.0, 0.0)
    assert whole.pipettable_volume_ul is None and whole.limit == 'the 10 µL sample'
    assert part.limit == 'the 8.25 µL that can be pipetted into each 10 µL sample'
    assert LiquidHandling(10, pipettable_volume_ul=10) == whole               # the whole sample: stored as None
    assert LiquidHandling(10, 4, 'Water', '8.25') == part
    assert part.to_dict() == {'sample_volume_ul': 10.0, 'droplet_nl': 4.0, 'diluent': 'Water',
                              'pipettable_volume_ul': 8.25}
    assert 'pipettable_volume_ul' not in whole.to_dict()                    # older versions can still read it
    assert LiquidHandling.from_dict(part.to_dict()) == part and LiquidHandling.from_dict(whole.to_dict()) == whole
    for bad, message in [(0, 'must be above 0'), (-1, 'must be above 0'), (float('nan'), 'must be above 0'),
                         ('x', 'must be a number'), (10.5, r'The pipettable volume \(10\.5 µL\) cannot be more than '
                                                         r'the sample volume \(10 µL\)')]:
        with pytest.raises(VariableError, match=message):
            pipettable(bad)
    with pytest.raises(VariableError, match='larger than the pipettable volume'):
        pipettable(0.003)                                                   # 3 nL, less than one droplet


def test_liquid_checks_compare_volumes_with_the_pipettable_volume():
    whole, part = LiquidHandling(sample_volume_ul=10, droplet_nl=4), pipettable()
    roomy = [Continuous('a', 0, 50, unit='mM', stock=100), Continuous('b', 0, 40, unit='mM', stock=100)]  # 5 + 4 uL
    assert whole.check(roomy) == []
    warnings = _messages(part, roomy, 'warning')
    assert len(warnings) == 1 and not _messages(part, roomy, 'error')
    assert '9 µL together, more than the 8.25 µL that can be pipetted into each 10 µL sample' in warnings[0]
    assert warnings[0].index('a (5 µL)') < warnings[0].index('b (4 µL)')
    fixed = [Discrete('a', [50], stock=100), Discrete('b', [40], stock=100)]                         # always 9 uL
    assert whole.check(fixed) == []
    errors = _messages(part, fixed, 'error')
    assert len(errors) == 1 and errors[0] == (
        'Even at their lowest values the components need 9 µL together, more than the 8.25 µL that can be '
        'pipetted into each 10 µL sample. Use more concentrated stocks, make more of each sample pipettable or '
        'lower the minimums.')
    too_much = [Discrete('a', [60], stock=100), Discrete('b', [50], stock=100)]
    assert _messages(whole, too_much, 'error') == [
        'Even at their lowest values the components need 11 µL together, more than the 10 µL sample. Use more '
        'concentrated stocks or lower the minimums.']
    assert part.summary(roomy).equals(whole.summary(roomy))           # volumes per variable do not change


def test_recipes_fit_the_pipettable_volume():
    variables = [Continuous(name, 0, 60, stock=100) for name in 'abc']                         # up to 6 uL each
    whole, part = RecipeSpace(variables, LiquidHandling(sample_volume_ul=10)), RecipeSpace(variables, pipettable())
    assert part.budget == 2062 and part.feasible_fraction() < whole.feasible_fraction()
    rng = np.random.default_rng(0)
    for R in (part.sample(200, rng), part.lhs(15, rng)):
        assert part.feasible(R).all() and (part.droplets(R) <= 2062).all()
    assert not part.feasible(whole.sample(400, rng)).all()             # some recipes fit 10 uL but not 8.25 uL
    row = part.encode(pd.DataFrame({'a': [40.0], 'b': [40.0], 'c': [2.4]}))[0]     # 4 + 4 + 0.24 = 8.24 uL
    assert part.droplets(row).tolist() == [2060] and part.feasible(row).all()
    near = part.neighbors(row)
    assert len(near) and part.feasible(near).all() and (part.droplets(near) < 2060).all()   # only moves down fit
    assert len(whole.neighbors(row)) > len(near)


def test_campaign_and_pipetting_with_a_pipettable_volume():
    campaign = Campaign(design() + [Categorical('Template', ['A', 'B'], volume_ul=0.5)], target='sfGFP', model=FAST,
                        liquid_handling=pipettable())
    assert campaign.check() == []
    campaign.add_round(campaign.propose(6))
    assert (campaign.volume_ul(campaign.data[campaign.names]) <= 8.25).all()
    plan = campaign.pipetting()
    table = plan.table(1).fillna(0)
    assert table.index[-1] == 'Water' and np.allclose(table.sum(), 8.248)       # 2062 whole droplets of 4 nL
    rows = campaign.data.set_index(col.WELL)
    assert np.allclose(table.loc['Mg (500 mM stock)'], rows['Mg'] * 10 / 500)  # volumes for the whole 10 uL
    assert np.allclose(table.loc['K (4000 mM stock)'], rows['K'] * 10 / 4000)
    assert 'Each sample is 10 µL: 8.25 µL from this table and 1.75 µL that the dispenser does not add.' in plan.notes
    assert 'Whole 4 nL droplets cannot make exactly 8.25 µL: each well gets 8.248 µL from this table.' in plan.notes
    idot = plan.idot('MoS', 'pipettable')
    assert idot.source['Dispensed (µL)'].sum() == pytest.approx(6 * 8.248) and not idot.warnings
    exact = Campaign(design(), model=FAST, liquid_handling=pipettable(8.24))               # 2060 whole droplets
    exact.add_round(exact.propose(3))
    notes = exact.pipetting().notes
    assert np.allclose(exact.pipetting().table(1).fillna(0).sum(), 8.24)
    assert not any('cannot make exactly' in note for note in notes) and any('1.76 µL' in note for note in notes)
    whole = measured_campaign(4)                                                 # no pipettable volume: as before
    assert not any('dispenser does not add' in note or 'cannot make exactly' in note
                   for note in whole.pipetting().notes)


def test_rounds_that_do_not_fit_the_pipettable_volume():
    campaign = Campaign([Continuous('a', 0, 60, stock=100), Continuous('b', 0, 60, stock=100)], model=FAST,
                        liquid_handling=pipettable())
    recipes = pd.DataFrame({'a': [40.0, 50.0], 'b': [40.0, 40.0]})                        # 8 uL and 9 uL
    assert campaign.volume_ul(recipes).tolist() == pytest.approx([8.0, 9.0])
    with pytest.raises(CampaignError, match='Data point 2 needs more than the 8.25 µL that can be pipetted into '
                                            'each 10 µL sample'):
        campaign.add_round(recipes)
    assert campaign.add_round(recipes.iloc[:1]) == 1
    rare = Campaign([Continuous(name, 0, 100, stock=100) for name in 'abcdefgh'], model=FAST,
                    liquid_handling=pipettable())
    message = next(issue.message for issue in rare.check() if 'Fewer than 1 in 100' in issue.message)
    assert message == ('Fewer than 1 in 100 random recipes fit in the 8.25 µL that can be pipetted into each 10 µL '
                       'sample, so a round cannot be drawn. Lower the maximums, make more of each sample pipettable '
                       'or use more concentrated stocks.')


def test_workbook_keeps_the_pipettable_volume():
    campaign = Campaign(design(), target='sfGFP', model=FAST, name='test', liquid_handling=pipettable())
    campaign.add_round(campaign.propose(4))
    data = campaign.to_workbook()
    loaded = Campaign.from_workbook(data, require_results=False)
    assert loaded.liquid == campaign.liquid and loaded.liquid.pipettable_ul == 8.25
    assert loaded.history[-1]['liquid_handling']['pipettable_volume_ul'] == 8.25
    meta = {row[0]: row[1] for row in load_workbook(io.BytesIO(data))['Metadata'].iter_rows(values_only=True)}
    assert (meta['Sample volume (µL)'], meta['Pipettable volume (µL)']) == (10, 8.25)
    old = Campaign.from_workbook(measured_campaign(4).to_workbook())             # no pipettable volume
    assert old.liquid.pipettable_volume_ul is None and old.liquid.budget == 2500


# --------------------------------------------------------------------------- the recipe space
def _space(liquid=True):
    return RecipeSpace(design(), LiquidHandling(sample_volume_ul=10) if liquid else None)


@pytest.mark.parametrize('liquid', [True, False])
def test_sampling_stays_inside_the_design(liquid):
    space = _space(liquid)
    R = space.sample(300, np.random.default_rng(0))
    frame = space.decode(R)
    assert len(R) == 300 and len(set(space.keys(R))) == 300
    assert frame['Mg'].between(0, 10).all() and set(frame['K']) <= {40, 80, 120, 160}
    assert set(frame['Promoter']) == {'T7', 'T3', 'SP6'}
    assert np.array_equal(space.snap(R), R) and space.feasible(R).all()
    if liquid:
        assert np.allclose(frame['Mg'] / 0.2, np.round(frame['Mg'] / 0.2))          # whole droplets
    assert np.array_equal(space.encode(frame), R)
    assert space.features(R).shape == (300, 5) and space.embedding(R).shape == (300, 5)
    assert space.feature_groups == [('Mg', [0]), ('K', [1]), ('Promoter', [2, 3, 4])]
    assert space.choices() == {'Mg': None, 'K': [40.0, 80.0, 120.0, 160.0], 'Promoter': ['T7', 'T3', 'SP6']}


def test_sampling_excludes_and_exhausts():
    space = RecipeSpace([Discrete('a', [1, 2, 3]), Categorical('b', ['x', 'y'])])
    everything = space.sample(100, np.random.default_rng(0))
    assert len(everything) == 6 == space.size and len(set(space.keys(everything))) == 6
    rest = space.sample(100, np.random.default_rng(1), exclude=space.keys(everything[:4]))
    assert set(space.keys(rest)) == set(space.keys(everything[4:]))
    assert len(space.sample(5, np.random.default_rng(0), exclude=space.keys(everything))) == 0
    assert math.isinf(_space(False).size) and _space(True).size == 51 * 4 * 3


def test_latin_hypercube_uses_every_slice_once():
    space = RecipeSpace([Continuous('a', 0, 1), Continuous('b', 10, 20), Discrete('c', [1, 2]),
                         Categorical('d', ['x', 'y', 'z', 'w', 'v'])])
    n = 20
    frame = space.decode(space.lhs(n, np.random.default_rng(3)))
    assert len(frame) == n
    assert sorted(np.floor(frame['a'] * n).astype(int)) == list(range(n))
    assert sorted(np.floor((frame['b'] - 10) / 10 * n).astype(int)) == list(range(n))
    assert frame['c'].value_counts().tolist() == [10, 10] and frame['d'].value_counts().tolist() == [4] * 5
    one_design = space.embedding(space.lhs(n, np.random.default_rng(3), n_designs=1))
    best_of_20 = space.embedding(space.lhs(n, np.random.default_rng(3)))
    from metis.campaign.space import min_distance
    assert min_distance(best_of_20) >= min_distance(one_design)


def test_latin_hypercube_respects_the_volume_and_small_spaces():
    liquid = LiquidHandling(sample_volume_ul=10)
    tight = RecipeSpace([Continuous(name, 0, 60, stock=100) for name in 'abc'], liquid)       # up to 18 of 10 uL
    R = tight.lhs(15, np.random.default_rng(0))
    assert len(R) == 15 and tight.feasible(R).all() and len(set(tight.keys(R))) == 15
    assert 0.5 < tight.feasible_fraction() < 0.75                  # three volumes of 0 to 6 uL in a 10 uL sample
    small = RecipeSpace([Discrete('a', [1, 2]), Categorical('b', ['x', 'y'])])
    assert len(small.lhs(10, np.random.default_rng(0))) == 4                                   # all there is
    taken = small.keys(small.sample(4, np.random.default_rng(0))[:3])
    assert len(small.lhs(10, np.random.default_rng(0), exclude=taken)) == 1


def test_neighbours_are_one_move_away():
    space = _space(True)
    row = space.encode(pd.DataFrame({'Mg': [5.0], 'K': [80.0], 'Promoter': ['T3']}))[0]
    frame = space.decode(space.neighbors(row))
    moves = {(m, k, p) for m, k, p in frame.itertuples(index=False)}
    assert moves == {(4.4, 80, 'T3'), (5.6, 80, 'T3'), (5, 40, 'T3'), (5, 120, 'T3'), (5, 80, 'T7'), (5, 80, 'SP6')}
    corner = space.encode(pd.DataFrame({'Mg': [0.0], 'K': [40.0], 'Promoter': ['T7']}))[0]
    assert len(space.neighbors(corner)) == 4                    # no step below the range
    free = _space(False)
    steps = free.decode(free.neighbors(row))['Mg']
    assert sorted(set(steps)) == [4.5, 5.0, 5.5]                # 1/20 of the range
    full = RecipeSpace([Discrete('a', [50], stock=100), Continuous('b', 0, 50, stock=100)],
                       LiquidHandling(sample_volume_ul=10))
    at_limit = full.encode(pd.DataFrame({'a': [50.0], 'b': [50.0]}))[0]
    assert full.feasible(at_limit).all() and len(full.neighbors(at_limit)) == 1     # only down fits the sample


def test_encoding_measured_values_off_the_design():
    space = RecipeSpace(design(), LiquidHandling(sample_volume_ul=10), {'Promoter': ['old', 'T7', 'older']})
    frame = pd.DataFrame({'Mg': [12.34, '1,5'], 'K': [55, 80], 'Promoter': [' old ', 'older']})
    with pytest.raises(VariableError, match='not a number'):
        space.encode(frame)
    frame['Mg'] = [12.34, 1.5]
    R = space.encode(frame)
    assert space.decode(R)['Promoter'].tolist() == ['old', 'older']
    assert space.features(R).shape == (2, 2 + 5) and space.choices()['Promoter'] == ['T7', 'T3', 'SP6']
    assert space.decode(space.snap(R))['Mg'].tolist() == [10.0, 1.6]         # into the range, onto a droplet
    assert space.keys(np.array([[0.1 + 0.2, 80, 0]])) == space.keys(np.array([[0.3, 80, 0]]))
    assert space.keys(np.array([[-0.0, 80, 0]])) == space.keys(np.array([[0.0, 80, 0]]))
    with pytest.raises(VariableError, match='not one of its categories'):
        space.encode(pd.DataFrame({'Mg': [1], 'K': [40], 'Promoter': ['T9']}))
    with pytest.raises(VariableError, match='No column for K'):
        space.encode(pd.DataFrame({'Mg': [1], 'Promoter': ['T7']}))


def _random_design(rng):
    variables = []
    for i in range(int(rng.integers(1, 7))):
        kind = rng.choice(['continuous', 'discrete', 'categorical'])
        if kind == 'continuous':
            low = float(rng.uniform(0, 5))
            variables.append(Continuous('v{}'.format(i), low, low + float(rng.uniform(1, 50)), stock=1000.0))
        elif kind == 'discrete':
            levels = np.unique(np.round(rng.uniform(0.5, 60, size=int(rng.integers(1, 8))), 1))
            variables.append(Discrete('v{}'.format(i), levels, stock=1000.0))
        else:
            variables.append(Categorical('v{}'.format(i), ['c{}'.format(k) for k in range(int(rng.integers(1, 6)))],
                                         volume_ul=float(rng.choice([0, 0.2]))))
    liquid = LiquidHandling(sample_volume_ul=float(rng.choice([5, 10, 20])), droplet_nl=float(rng.choice([2.5, 4, 8])))
    return variables, (liquid if rng.random() < 0.6 else None)


@pytest.mark.parametrize('seed', range(40))
def test_random_designs_keep_their_promises(seed):
    """Whatever the design: draws are distinct, allowed, inside the sample, and survive a round trip."""
    rng = np.random.default_rng(seed)
    variables, liquid = _random_design(rng)
    if liquid is not None and any(issue.blocking for issue in liquid.check(variables)):
        pytest.skip('this random design cannot be pipetted')
    space = RecipeSpace(variables, liquid)
    for R in (space.sample(60, rng), space.lhs(12, rng)):
        assert len(set(space.keys(R))) == len(R) and space.feasible(R).all()
        assert np.array_equal(space.snap(R), R)
        assert np.array_equal(space.encode(space.decode(R)), R)
        assert np.isfinite(space.features(R)).all() and np.isfinite(space.embedding(R)).all()
        for v, values in zip(variables, R.T):
            if v.kind == 'categorical':
                assert set(values) <= set(range(len(v.categories)))
            elif v.kind == 'discrete':          # its steps, rounded to whole droplets when pipetted
                steps = v.levels if liquid is None else liquid.snap(v, v.levels)
                assert set(values) <= set(float(s) for s in steps)
            else:
                assert (values >= v.low - 1e-9).all() and (values <= v.high + 1e-9).all()
        if liquid is not None:
            assert (space.droplets(R) <= liquid.budget).all()
        for row in R[:3]:
            near = space.neighbors(row)
            assert space.feasible(near).all() and all((n != row).sum() == 1 for n in near)
    if space.size <= 60:
        assert len(space.sample(60, rng)) == sum(1 for _ in space.keys(space.sample(10 ** 4, rng)))


# --------------------------------------------------------------------------- campaigns
def test_first_round_is_a_latin_hypercube_and_gets_ids_and_wells():
    campaign = Campaign(design(), target='sfGFP', model=FAST, liquid_handling=LiquidHandling(sample_volume_ul=10))
    assert campaign.next_round == 1 and campaign.rounds == [] and campaign.check() == []
    proposal = campaign.propose(10)
    assert not proposal.guided and proposal.prediction is None and proposal.fitted is None
    assert proposal.settings['method'] == 'latin hypercube' and list(proposal.table.columns)[0] == col.POINT
    with pytest.raises(CampaignError):
        proposal.predict(proposal.recipes)
    with pytest.raises(CampaignError):
        proposal.climb()
    assert campaign.propose(10).recipes.equals(proposal.recipes)                       # reproducible
    assert not campaign.propose(10, seed=5).recipes.equals(proposal.recipes)
    assert campaign.add_round(proposal, replicates=3) == 1
    data = campaign.data
    assert len(data) == 30 and data[col.ID].tolist() == sorted(list(range(1, 11)) * 3)
    assert data[col.REPLICATE].tolist() == [1, 2, 3] * 10 and data[col.POINT].tolist() == data[col.ID].tolist()
    assert data[col.WELL].tolist() == WELLS[:30] and set(data[col.PLATE]) == {1}
    assert data.groupby(col.ID)['Mg'].nunique().eq(1).all() and campaign.pending.equals(data)
    assert campaign.history[0]['n_samples'] == 10 and campaign.history[0]['replicates'] == 3
    assert campaign.next_round == 2 and len(campaign.samples()) == 10 and campaign.best() is None


def test_model_guided_rounds_and_their_modules():
    campaign = measured_campaign(14)
    assert campaign.default_exploitation() == 0.5
    measured = set(campaign.space().keys(campaign.space().encode(campaign.data[campaign.names])))
    plain = campaign.propose(8)
    assert plain.guided and plain.settings['exploitation'] == 0.5 and plain.settings['training_rows'] == 14
    assert list(plain.prediction.columns) == ['mean', 'std', 'low', 'high']
    assert (plain.prediction['low'] <= plain.prediction['mean']).all()
    assert (plain.prediction['mean'] <= plain.prediction['high']).all()
    assert list(plain.table.columns) == [col.POINT, 'Mg', 'K', 'Promoter'] + col.prediction_columns('sfGFP')
    seen = {}
    for name, options in {'plain': {}, 'penalty': {'penalty': 1.0}, 'fantasy': {'fantasy': True},
                          'both': {'penalty': 0.5, 'fantasy': True}, 'climb': {'hill_climb': True},
                          'explore': {'exploitation': 0.0}, 'exploit': {'exploitation': 1.0}}.items():
        proposal = campaign.propose(8, **options)
        space = proposal.fitted.space
        keys = space.keys(space.encode(proposal.recipes))
        assert len(set(keys)) == 8 and not measured & set(keys), name
        assert space.feasible(space.encode(proposal.recipes)).all()
        seen[name] = proposal
    assert seen['plain'].recipes.equals(plain.recipes)
    spread = {name: spacing(p.fitted.space.embedding(p.fitted.space.encode(p.recipes))) for name, p in seen.items()}
    assert spread['penalty'] > spread['plain']                          # the penalty pushes the batch apart
    assert seen['penalty'].settings['penalty_radius'] == pytest.approx(3 * seen['penalty'].settings['candidate_spacing'])
    assert seen['exploit'].prediction['mean'].mean() >= seen['explore'].prediction['mean'].mean()
    assert seen['explore'].prediction['std'].mean() >= seen['exploit'].prediction['std'].mean()
    for bad in ({'exploitation': 1.5}, {'n_samples': 0}):
        with pytest.raises(CampaignError):
            campaign.propose(**{'n_samples': 8, **bad})
    with pytest.raises(CampaignError, match='two measured results'):
        Campaign(design(), model=FAST).propose(5, guided=True)
    assert not campaign.propose(6, guided=False).guided                 # a Latin hypercube on request


def test_hill_climbing_keeps_the_start_and_improves_the_score():
    campaign = measured_campaign(14, liquid=False)
    plain = campaign.propose(6, pool_size=30, exploitation=1.0)
    climbed = plain.climb(max_steps=30)
    assert climbed.before is plain and climbed.settings['hill_climb'] and not plain.settings['hill_climb']
    assert climbed.prediction['mean'].max() >= plain.prediction['mean'].max() - 1e-9
    assert climbed.settings['climbed_in_batch'] == len(climbed.moves) > 0
    assert max(climbed.settings['climb_moves']) <= 30
    starts = climbed.moves[['start ' + name for name in campaign.names]]
    assert set(climbed.moves[col.POINT]) <= set(range(1, 7)) and starts.notna().all().all()
    assert campaign.propose(6, pool_size=30, exploitation=1.0, hill_climb=True).recipes.equals(climbed.recipes)
    predicted = climbed.predict(climbed.recipes)
    assert np.allclose(predicted['mean'], climbed.prediction['mean'])


def test_schedule_and_round_numbers_over_three_rounds():
    campaign = measured_campaign(10)
    assert campaign.rounds == [1] and campaign.next_round == 2
    for expected in (0.5, 1.0, 1.0):
        assert campaign.default_exploitation() == expected
        proposal = campaign.propose(5)
        assert proposal.settings['exploitation'] == expected and proposal.round == campaign.next_round
        campaign.add_round(proposal)
        campaign.data.loc[campaign.data['sfGFP'].isna(), 'sfGFP'] = lab(campaign.pending)
        campaign = Campaign.from_workbook(campaign.to_workbook())
    assert campaign.rounds == [1, 2, 3, 4] and len(campaign.history) == 4
    progress = campaign.progress()
    assert progress['Recipes'].tolist() == [10, 5, 5, 5] and progress['Best so far'].is_monotonic_increasing
    assert campaign.best()['sfGFP'] == progress['Best so far'].iloc[-1]
    assert campaign.data[col.ID].tolist() == list(range(1, 26))
    stored = campaign.data.loc[campaign.data[col.ROUND] == 2, col.prediction_columns('sfGFP')]
    assert stored.notna().all().all() and campaign.data.loc[campaign.data[col.ROUND] == 1, col.LOW].isna().all()


def test_day0_data_groups_replicates_and_reads_headers_with_units():
    campaign = Campaign(design(), target='sfGFP', model=FAST)
    table = pd.DataFrame({'Mg (mM)': ['1,5', 1.5, 3, 3, 9], ' K (mM) ': [40, 40, 80, 80, 160],
                          'Promoter': ['T7', 'T7', ' T3', 'T3', 'old'], 'sfGFP': [1.0, '1,2', 2.0, None, 3.0],
                          'note': list('abcde')})
    assert campaign.add_data(table) == 4                              # the row without a result is skipped
    data = campaign.data
    assert data[col.ROUND].eq(0).all() and data[col.ID].tolist() == [1, 1, 2, 3]
    assert data[col.REPLICATE].tolist() == [1, 2, 1, 1] and data['sfGFP'].tolist() == [1.0, 1.2, 2.0, 3.0]
    assert data['Mg'].tolist() == [1.5, 1.5, 3.0, 9.0] and data['Promoter'].tolist() == ['T7', 'T7', 'T3', 'old']
    assert campaign.next_round == 1 and campaign.samples()['sfGFP'].tolist() == pytest.approx([1.1, 2.0, 3.0])
    proposal = campaign.propose(4)                                    # day 0 data is enough for the model
    assert proposal.guided and proposal.round == 1 and 'old' not in set(proposal.recipes['Promoter'])
    assert campaign.add_data(table, round_number=0) == 4 and campaign.data[col.ID].max() == 6
    assert campaign.data[col.POINT].tolist() == [1, 1, 2, 3, 4, 4, 5, 6]
    for frame, message in [(table.drop(columns='sfGFP'), 'no "sfGFP" column'),
                           (table.drop(columns='Promoter'), 'no column for Promoter'),
                           (table.assign(sfGFP='x'), 'is not a number'), (table.assign(sfGFP=None), 'No row'),
                           (table.assign(**{'Mg (mM)': ['a', 1, 1, 1, 1]}), 'Line 2: "a" in the Mg column')]:
        with pytest.raises(CampaignError, match=message):
            Campaign(design(), target='sfGFP', model=FAST).add_data(frame)
    filled = Campaign(design(), target='sfGFP', model=FAST)
    filled.add_data(table.drop(columns='Promoter'), fill={'Promoter': 'T3'})
    assert set(filled.data['Promoter']) == {'T3'}


def test_redesign_changes_ranges_adds_and_drops_variables():
    campaign = measured_campaign(12)
    new = [Continuous('Mg', 2, 20, unit='mM', stock=500), Categorical('Promoter', ['T7', 'T3']),
           Discrete('DTT', [0, 1, 2], unit='mM', stock=100)]
    with pytest.raises(CampaignError, match='DTT is new'):
        campaign.redesign(new)
    other = campaign.redesign(new, fill={'DTT': 0})
    assert other.names == ['Mg', 'Promoter', 'DTT'] and other.extra == ['K'] and other.liquid == campaign.liquid
    assert other.data['DTT'].eq(0).all() and other.data['K'].equals(campaign.data['K'])
    assert campaign.names == ['Mg', 'K', 'Promoter'] and 'DTT' not in campaign.data          # the original is untouched
    proposal = other.propose(6)
    assert proposal.recipes['Mg'].between(2, 20).all() and set(proposal.recipes['Promoter']) <= {'T7', 'T3'}
    other.add_round(proposal)
    loaded = Campaign.from_workbook(other.to_workbook(), require_results=False)
    assert loaded.extra == ['K'] and loaded.data['K'].iloc[:12].tolist() == campaign.data['K'].tolist()
    assert loaded.data['K'].iloc[12:].isna().all() and loaded.names == other.names
    assert campaign.redesign(design(), liquid_handling=None).liquid is None
    with pytest.raises(CampaignError, match='K has no value in 6 of the 18 rows'):
        other.redesign(design())
    back = other.redesign(design(), fill={'K': 40})                   # K comes back: old values kept, gaps filled
    assert back.extra == ['DTT'] and back.data['K'].iloc[:12].tolist() == campaign.data['K'].tolist()
    assert back.data['K'].iloc[12:].eq(40).all() and back.propose(3).guided
    with pytest.raises(CampaignError, match='is not a number'):
        campaign.redesign([Continuous('Mg', 0, 10), Continuous('Promoter', 0, 1)])


def test_snap_volume_and_rounds_that_do_not_fit():
    liquid = LiquidHandling(sample_volume_ul=10)
    campaign = Campaign([Continuous('a', 0, 60, stock=100), Continuous('b', 0, 60, stock=100)], model=FAST,
                        liquid_handling=liquid)
    recipes = pd.DataFrame({'a': [10.03, 70], 'b': [20, 55.55]})
    snapped = campaign.snap(recipes)
    assert snapped['a'].tolist() == [10.04, 60.0] and snapped['b'].tolist() == [20.0, 55.56]
    assert campaign.volume_ul(snapped).tolist() == pytest.approx([3.004, 11.556])
    with pytest.raises(CampaignError, match='Data point 2 needs more than the 10 µL'):
        campaign.add_round(snapped)
    assert campaign.add_round(snapped.iloc[:1]) == 1
    with pytest.raises(CampaignError):
        Campaign(design(), model=FAST).volume_ul(recipes)
    tight = Campaign([Continuous(name, 59, 60, stock=100) for name in 'abc'], model=FAST, liquid_handling=liquid)
    assert any('Even at their lowest' in issue.message for issue in tight.check())
    with pytest.raises(CampaignError, match='Even at their lowest'):
        tight.propose(3)
    rare = Campaign([Continuous(name, 0, 100, stock=100) for name in 'abcdefgh'], model=FAST, liquid_handling=liquid)
    assert any('Fewer than 1 in 100' in issue.message for issue in rare.check())


def test_a_design_with_nothing_left_to_propose():
    campaign = Campaign([Discrete('a', [1, 2]), Categorical('b', ['x', 'y'])], model=FAST)
    first = campaign.propose(10)
    assert len(first.recipes) == 4 and 'Only 4 new recipe' in first.warnings[0]
    campaign.add_round(first)
    campaign.data['Result'] = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(CampaignError, match='Every recipe of this design has been measured'):
        campaign.propose(2)
    with pytest.raises(CampaignError, match='No new recipe'):
        campaign.propose(2, guided=False)


# --------------------------------------------------------------------------- the workbook
def test_workbook_round_trip_and_its_sheets():
    campaign = measured_campaign(6, replicates=2)
    campaign.add_round(campaign.propose(4, penalty=0.0))
    data = campaign.to_workbook()
    book = load_workbook(io.BytesIO(data))
    assert book.sheetnames == ['Experiments', 'Instructions', 'Metadata', 'State']
    sheet = book['Experiments']
    header = [cell.value for cell in sheet[1]]
    assert header == ['Sample ID', 'Round', 'Data point', 'Replicate', 'Plate', 'Well', 'Mg (mM)', 'K (mM)', 'Promoter',
                      'Predicted sfGFP', '80% interval low', '80% interval high', 'sfGFP']
    results = [row[-1] for row in sheet.iter_rows(min_row=2)]
    assert [cell.value is None for cell in results] == [False] * 12 + [True] * 4
    assert all(cell.fill.start_color.rgb.endswith('FFF2CC') for cell in results[12:])        # highlighted to fill in
    meta = {row[0]: row[1] for row in book['Metadata'].iter_rows(values_only=True)}
    for key in ('Random seed', 'Created', 'Sample volume (µL)', 'Mg', 'Exploitation', 'Environment', 'Model',
                'PCA map: variation on axis 1, axis 2', 'PCA axis 1 weights', 'Penalty radius'):
        assert key in meta, key
    assert 'continuous, 0 to 10 mM; stock 500 mM' == meta['Mg'] and 'Python' in meta['Environment']
    assert 'Type each measured value into the "sfGFP" column' in book['Instructions']['A3'].value

    with pytest.raises(CampaignError, match='4 of the 16 rows have no value in the "sfGFP" column: row 14 '
                                            r'\(sample 7, replicate 1\)'):
        Campaign.from_workbook(data)
    loaded = Campaign.from_workbook(data, require_results=False)
    pd.testing.assert_frame_equal(loaded.data, campaign.data, check_dtype=False)
    assert loaded.variables == campaign.variables and loaded.liquid == campaign.liquid
    assert (loaded.name, loaded.target, loaded.seed, loaded.model) == ('test', 'sfGFP', 0, FAST)
    assert loaded.history == campaign.history and loaded.history[1]['settings']['penalty'] == 0.0
    assert loaded.history[1]['pca']['columns'] == ['Mg', 'K', 'Promoter = T7', 'Promoter = T3', 'Promoter = SP6']
    assert loaded.fingerprint() == campaign.fingerprint() != measured_campaign(6).fingerprint()
    assert Campaign.from_workbook(loaded.to_workbook(), require_results=False).to_workbook()[:4] == b'PK\x03\x04'


def _edit(data, change):
    """The workbook after ``change(experiments_sheet, book)``, as bytes."""
    book = load_workbook(io.BytesIO(data))
    change(book['Experiments'], book)
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def test_a_filled_in_workbook_is_read_as_the_user_left_it():
    campaign = measured_campaign(4)
    campaign.add_round(campaign.propose(3))
    data = campaign.to_workbook()

    def fill(sheet, book):
        for row, value in zip(range(6, 9), (1.5, '2,25', ' 3 ')):          # numbers, a decimal comma, spaces
            sheet.cell(row=row, column=13, value=value)
        sheet.cell(row=1, column=14, value='Notes')
        sheet.cell(row=6, column=14, value='=cloudy').data_type = 's'       # text that starts with "="
        sheet.cell(row=7, column=2, value='Round 2')
        sheet.cell(row=8, column=7, value=7.77)                            # what was actually pipetted
        sheet.insert_rows(5)                                               # a blank line in the middle

    loaded = Campaign.from_workbook(_edit(data, fill))
    assert loaded.data['sfGFP'].iloc[4:].tolist() == [1.5, 2.25, 3.0] and loaded.rounds == [1, 2]
    assert loaded.extra == ['Notes'] and loaded.data['Notes'].iloc[4] == '=cloudy'
    assert loaded.data['Mg'].iloc[6] == 7.77 and loaded.next_round == 3
    again = Campaign.from_workbook(loaded.to_workbook())                  # the notes column and its "=" survive
    assert again.data['Notes'].iloc[4] == '=cloudy' and again.extra == ['Notes']
    sheet = load_workbook(io.BytesIO(loaded.to_workbook()))['Experiments']
    notes = [cell.value for cell in sheet[1]].index('Notes') + 1
    cell = sheet.cell(row=6, column=notes)
    assert cell.data_type == 's' and cell.value == '=cloudy'              # text, not a formula

    def drop_failed(sheet, book):
        for row, value in zip(range(6, 8), (1.5, 2.0)):
            sheet.cell(row=row, column=13, value=value)
        sheet.delete_rows(8)

    assert len(Campaign.from_workbook(_edit(data, drop_failed)).data) == 6


@pytest.mark.parametrize('change, message', [
    (lambda sheet, book: book.remove(book['State']), 'no "State" sheet'),
    (lambda sheet, book: book['State'].cell(row=2, column=1, value='json:{broken'), '"State" sheet was changed'),
    (lambda sheet, book: sheet.delete_cols(7), 'no "Mg" column'),
    (lambda sheet, book: sheet.delete_rows(2, 100), 'has no rows'),
    (lambda sheet, book: sheet.cell(row=3, column=13, value='n/a'), 'Row 3: "n/a" in the sfGFP column'),
    (lambda sheet, book: setattr(sheet.cell(row=3, column=7), 'value', None), 'Row 3: Mg has no value'),
    (lambda sheet, book: sheet.cell(row=3, column=1, value='x'), 'Row 3: the Sample ID column needs a whole number'),
    (lambda sheet, book: sheet.cell(row=3, column=1, value=1), 'sample 1 already has a replicate 1'),
    (lambda sheet, book: sheet.cell(row=1, column=8, value='Mg (mM)'), 'two columns called "Mg \\(mM\\)"'),
    (lambda sheet, book: setattr(sheet.cell(row=1, column=9), 'value', None), 'column without a header')])
def test_damaged_workbooks_give_a_clear_message(change, message):
    data = measured_campaign(4).to_workbook()
    with pytest.raises((CampaignError, WorkbookError), match=message):
        Campaign.from_workbook(_edit(data, change))


def test_other_workbook_problems():
    with pytest.raises(WorkbookError, match='cannot be opened'):
        Campaign.from_workbook(b'not a workbook')
    campaign = measured_campaign(4, replicates=2)

    def change_one_replicate(sheet, book):
        sheet.cell(row=3, column=7, value=9.9)

    with pytest.raises(CampaignError, match='rows of sample 1 have different recipes'):
        Campaign.from_workbook(_edit(campaign.to_workbook(), change_one_replicate))
    state = campaign._state()
    table = campaign.data.rename(columns={'Mg': 'Mg (mM)', 'K': 'K (mM)'})
    headers = {name: name for name in table.columns}
    for bad in ({**state, 'format': 99}, {**state, 'model': 'gone'}, {**state, 'variables': [{'type': 'x'}]}):
        data = workbook.write(table, headers, 'sfGFP', bad, ['x'], [('a', None)])
        with pytest.raises(CampaignError):
            Campaign.from_workbook(data)
    path_free = Campaign(design(), model=FAST)                              # no liquid handling: no well columns
    path_free.add_round(path_free.propose(3))
    sheet = load_workbook(io.BytesIO(path_free.to_workbook()))['Experiments']
    assert 'Well' not in [cell.value for cell in sheet[1]]


def test_save_and_load(tmp_path):
    campaign = measured_campaign(5)
    path = tmp_path / 'round1.xlsx'
    campaign.save(path)
    assert Campaign.load(path).fingerprint() == campaign.fingerprint()


# --------------------------------------------------------------------------- pipetting and the iDOT file
def test_pipetting_plan_matches_the_recipes():
    campaign = Campaign(design() + [Categorical('Template', ['A', 'B'], volume_ul=0.5)], target='sfGFP', model=FAST,
                        liquid_handling=LiquidHandling(sample_volume_ul=10))
    campaign.add_round(campaign.propose(5), replicates=2)
    plan = campaign.pipetting()
    table = plan.table(1)
    assert list(table.columns) == WELLS[:10] and len(plan.plates) == 1
    assert list(table.index[:2]) == ['Mg (500 mM stock)', 'K (4000 mM stock)'] and table.index[-1] == 'Water'
    assert set(table.index[2:-1]) == {'Template: ' + name for name in set(campaign.data['Template'])}
    assert np.allclose(table.fillna(0).sum(), 10.0)                           # every well is topped up to the sample
    rows = campaign.data.set_index(col.WELL)
    assert np.allclose(table.loc['Mg (500 mM stock)'].fillna(0), rows['Mg'] * 10 / 500)
    assert np.allclose(table.loc['K (4000 mM stock)'], rows['K'] * 10 / 4000)
    for template in set(rows['Template']):
        assert table.loc['Template: ' + template].fillna(0).tolist() == (rows['Template'] == template).mul(0.5).tolist()
    assert any('Promoter is not pipetted' in note for note in plan.notes)
    assert plan.layout[col.ID].tolist() == campaign.data[col.ID].tolist()
    sheet = idot_builder.parse_sheet(plan.csv(1))                              # the builder reads what we write
    assert sheet.wells == ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'A2', 'B2']
    assert not sheet.warnings and plan.csv(1).splitlines()[0] == ',' + ','.join(WELLS[:10])
    cells = [cell for line in plan.csv(1).splitlines()[1:] for cell in line.split(',')[1:] if cell]
    assert cells and all(re.fullmatch(r'\d+(\.\d+)?', cell) for cell in cells)   # plain decimals, no exponents


def test_idot_protocol_holds_the_same_volumes():
    campaign = measured_campaign(6)
    campaign.add_round(campaign.propose(7))
    plan = campaign.pipetting()
    idot = plan.idot('MoS', 'round 2 / test')
    assert idot.filename == 'round_2_test.idotx' and idot.data.startswith(b'\xef\xbb\xbf<?xml')
    assert b'\r\n' in idot.data and b'Title="round_2_test"' in idot.data
    project = idot_builder.read_project(idot.data)
    table = plan.table(1)
    names = dict(zip(idot.liquids['Liquid in the sheet'], idot.liquids['Name in the iDOT software']))
    assert names['Mg (500 mM stock)'] == 'MoS_Mg_500_mM_stock'
    for well in table.columns:
        steps = {step['SourceLiquid']: float(step['Volume']) * 1e6 for step in project.wells[idot_builder.parse_well(well)]}
        expected = {names[liquid]: volume for liquid, volume in table[well].dropna().items()}
        assert steps == pytest.approx(expected)
    assert sum(1 for steps in project.wells.values() if steps) == 7
    assert idot.source['Dispensed (µL)'].sum() == pytest.approx(70.0)          # 7 wells of 10 uL
    assert (idot.source['Load at least (µL, with 5% extra)'] <= 80).all()
    assert len(idot.source) == 3 and not idot.warnings                         # one source well per liquid
    campaign.add_round(campaign.propose(20, seed=3))
    large = campaign.pipetting().idot('MoS', 'round 3')
    water = large.source[large.source['Liquid'] == 'MoS_Water']
    assert len(water) == 3 and any('split over 3 source wells' in warning for warning in large.warnings)
    for initials in ('', 'two words', 'x' * 11):
        with pytest.raises(IdotError):
            plan.idot(initials, 'name')
    with pytest.raises(IdotError):
        idot_protocol(',1A\n', 'MoS', 'empty')


def test_pipetting_over_several_plates_and_its_errors():
    assert assign_wells(98)[95:] == [(1, '12H'), (2, '1A'), (2, '1B')]
    campaign = Campaign([Continuous('a', 0, 10, stock=100)], model=FAST, liquid_handling=LiquidHandling(10))
    campaign.add_round(campaign.propose(50), replicates=2)
    plan = campaign.pipetting()
    assert [sheet.shape[1] for sheet in plan.plates] == [96, 4] and plan.layout[col.PLATE].tolist() == [1] * 96 + [2] * 4
    assert plan.csv(2).splitlines()[0] == ',1A,1B,1C,1D' and plan.idot('AB', 'p', plate=2).data
    assert campaign.pipetting(1).layout.equals(plan.layout)
    with pytest.raises(CampaignError):
        campaign.pipetting(3)
    with pytest.raises(CampaignError, match='needs liquid handling'):
        Campaign(design(), model=FAST).pipetting()
    clash = Campaign([Continuous('Water', 0, 10, stock=100)], model=FAST, liquid_handling=LiquidHandling(10, 4, 'Water (stock 100)'))
    clash.add_round(clash.propose(2))
    with pytest.raises(VariableError, match='Two liquids would be called'):
        clash.pipetting()


# --------------------------------------------------------------------------- analysis
class Formula:
    """A stand-in model that answers with a formula."""

    def __init__(self, formula):
        self.formula = formula

    def predict(self, X):
        mean = self.formula(np.asarray(X, dtype=float))
        return Prediction(mean, np.zeros(len(mean)))


def test_importance_ranks_what_the_model_uses():
    rng = np.random.default_rng(0)
    X = rng.random((120, 5))
    y = 3 * X[:, 0] + X[:, 1] + 2 * X[:, 3]
    groups = [('a', [0]), ('b', [1]), ('c', [2]), ('de', [3, 4])]
    ranking = analysis.importance(Formula(lambda Z: 3 * Z[:, 0] + Z[:, 1] + 2 * Z[:, 3]), X, y, groups, max_rows=80)
    assert list(ranking.index) == ['a', 'de', 'b', 'c'] and ranking['c'] == pytest.approx(0.0)
    assert ranking.attrs['r2'] == pytest.approx(1.0) and ranking.name == 'R2 drop when shuffled'
    assert analysis.importance(Formula(lambda Z: Z[:, 0]), X[:1], y[:1], groups).eq(0).all()
    assert analysis.importance(Formula(lambda Z: Z[:, 0]), X, np.ones(120), groups).eq(0).all()   # nothing to explain
    campaign = measured_campaign(30, liquid=False)
    assert set(campaign.importance().index) == {'Mg', 'K', 'Promoter'}


def test_interactions_are_zero_for_additive_models():
    rng = np.random.default_rng(1)
    X = rng.random((60, 4))
    groups = [('a', [0]), ('b', [1]), ('c', [2, 3])]
    additive = analysis.interactions(Formula(lambda Z: Z[:, 0] + 2 * Z[:, 1] - Z[:, 2]), X, groups)
    assert additive['Interaction'].abs().max() < 1e-12 and len(additive) == 3
    product = analysis.interactions(Formula(lambda Z: 4 * Z[:, 0] * Z[:, 1] + Z[:, 2]), X, groups)
    assert tuple(product.iloc[0, :2]) == ('a', 'b') and product['Interaction'].iloc[0] > 0.1
    assert product['Interaction'].iloc[1:].abs().max() < 1e-12 and (product['Pairs compared'] == 30).all()
    assert analysis.interactions(Formula(lambda Z: Z[:, 0]), X[:1], groups).empty
    assert analysis.interactions(Formula(lambda Z: Z[:, 0]), X, groups[:1]).empty
    same = np.tile(X[:1], (5, 1))                                              # recipes that never differ
    assert analysis.interactions(Formula(lambda Z: Z[:, 0] * Z[:, 1]), same, groups)['Pairs compared'].eq(0).all()
    campaign = measured_campaign(20, liquid=False)
    assert campaign.interactions()['Interaction'].abs().max() < 1e-9          # the stand-in model is additive
    assert len(campaign.interactions(variables=['Mg', 'K'])) == 1


def test_pca_map():
    campaign = measured_campaign(15)
    pca = campaign.pca()
    assert pca.components.shape == (2, 5) and 0 < pca.explained[1] <= pca.explained[0] <= 1
    assert np.allclose(pca.components @ pca.components.T, np.eye(2), atol=1e-9)
    xy = campaign.project(pca, campaign.samples())
    assert xy.shape == (15, 2) and np.allclose(xy.mean(axis=0), 0, atol=1e-9)
    again = analysis.PCAMap.from_dict(pca.to_dict())
    assert np.allclose(again.transform(np.ones((1, 5))), pca.transform(np.ones((1, 5))))
    empty = Campaign(design(), model=FAST)
    assert not empty.pca().explained.any()
    recipes = empty.propose(6).recipes
    assert empty.pca(recipes).explained[0] > 0 and empty.project(empty.pca(recipes), recipes).shape == (6, 2)
    one = analysis.fit_pca(np.array([[0.0], [1.0], [3.0]]), ['a'])
    assert one.explained.tolist() == [1.0, 0.0] and one.transform([[2.0]])[0, 1] == 0
    assert not analysis.fit_pca(np.ones((4, 3)), list('abc')).components.any()   # identical recipes: no map


def test_batch_rules():
    assert make_selector().name == 'topk' and make_selector(fantasy=True).name == 'sequential_fantasy'
    assert make_selector(0.5).name == 'soft_penalty' and isinstance(make_selector(0.5, True), PenalizedFantasy)
    assert penalty_radius(0, 2.0) == 2.0 and penalty_radius(1, 2.0) == 6.0 and penalty_radius(5, 2.0) == 6.0
    assert spacing(np.array([[0.0], [1.0], [1.5], [4.0]])) == pytest.approx(0.75)
    assert spacing(np.zeros((1, 2))) == 1.0 and spacing(np.zeros((5, 2))) == 1.0
    with pytest.raises(ValueError):
        PenalizedFantasy(0)


# --------------------------------------------------------------------------- day 0 data and the tutorial
def test_day0_files_templates_and_guessed_variables():
    text = '﻿Mg (mM);K;Strain;ID;Yield\n1,5;40;wt;r1;0,5\n1,5;40;wt;r2;0,6\n3;80;ko;r3;1,1\n\n'
    frame = day0.read_table(text.encode('utf-8'), 'data.csv')
    assert list(frame.columns) == ['Mg (mM)', 'K', 'Strain', 'ID', 'Yield'] and len(frame) == 3
    assert day0.numeric_columns(frame) == ['Mg (mM)', 'K', 'Yield']
    variables, skipped = day0.infer_variables(frame, 'Yield')
    assert variables == [Discrete('Mg', [1.5, 3.0], unit='mM'), Discrete('K', [40, 80]), Categorical('Strain', ['wt', 'ko'])]
    assert skipped == [('ID', 'every row has a different text')]
    wide = pd.DataFrame({'x': np.linspace(0, 5, 30), 'empty': [None] * 30, 'y': range(30)})
    variables, skipped = day0.infer_variables(wide, 'y')
    assert variables == [Continuous('x', 0, 5)] and skipped == [('empty', 'it is empty')]
    assert day0.split_header(' Mg-glutamate (mM) ') == ('Mg-glutamate', 'mM') and day0.split_header('K') == ('K', '')

    template = day0.template(design(), 'sfGFP')
    back = day0.read_table(template.encode('utf-8-sig'), 'template.csv')
    assert list(back.columns) == ['Mg (mM)', 'K (mM)', 'Promoter', 'sfGFP'] and len(back) == 3
    assert back.iloc[0, :3].tolist() == back.iloc[1, :3].tolist()            # the replicate pair
    campaign = Campaign(design(), target='sfGFP', model=FAST)
    assert campaign.add_data(back) == 3 and campaign.data[col.ID].tolist() == [1, 1, 2]
    assert 'Mg-glutamate (mM),Promoter,Result' in day0.template()
    excel = io.BytesIO()
    back.to_excel(excel, index=False)
    assert day0.read_table(excel.getvalue(), 'data.xlsx').equals(back)


def test_tutorial_campaign_and_its_simulated_lab():
    data = tutorial.load()
    assert data.shape == (1017, 13) and len(tutorial.day0()) == 99
    campaign = tutorial.campaign(model=FAST)
    assert len(campaign.variables) == 11 and campaign.check() == [] and campaign.rounds == [0]
    assert all(v.kind == 'discrete' and len(v.levels) == 4 and v.stock == 25 * v.high for v in campaign.variables)
    space = campaign.space()
    assert np.array_equal(space.snap(space.encode(campaign.data[campaign.names])),
                          space.encode(campaign.data[campaign.names]))       # every measured level is pipettable
    proposal = campaign.propose(5)
    assert proposal.guided and proposal.round == 1
    means = data.groupby(campaign.names)[tutorial.TARGET].mean().reset_index()
    assert np.allclose(tutorial.simulate(means.iloc[:20]), np.round(means[tutorial.TARGET].iloc[:20], 4))
    simulated = tutorial.simulate(proposal.recipes)
    assert simulated.shape == (5,) and (simulated >= 0).all() and (simulated <= data[tutorial.TARGET].max()).all()
    twice = pd.concat([proposal.recipes, proposal.recipes], ignore_index=True)
    replicates = tutorial.simulate(twice, replicate=[1] * 5 + [2] * 5, seed=1)
    assert np.array_equal(replicates[:5], simulated) and not np.array_equal(replicates[5:], simulated)
    assert np.array_equal(replicates, tutorial.simulate(twice, replicate=[1] * 5 + [2] * 5, seed=1))


def test_model_registry():
    assert 'tabpfn35' in MODELS and MODELS['tabpfn35'].max_variables == 6000 and MODELS['tabpfn35'].label == 'TabPFN-3.5'
    assert make_model(FAST, seed=3, members=4).members == 4
    with pytest.raises(KeyError):
        make_model('missing')
    with pytest.raises(KeyError):
        Campaign(design(), model='missing')
    with pytest.raises(VariableError, match='at most'):
        Campaign([Continuous('v{}'.format(i), 0, 1) for i in range(6001)], model=FAST)


@pytest.mark.tabpfn
@pytest.mark.slow
def test_a_round_with_tabpfn35():
    """The real model: predictions, intervals from its own distribution, hill-climbing."""
    pytest.importorskip('tabpfn')
    campaign = measured_campaign(16, liquid=True)
    campaign.model, campaign.model_options = 'tabpfn35', {'n_estimators': 1, 'device': 'cpu'}
    try:
        proposal = campaign.propose(4, pool_size=40, hill_climb=True, climb_max_moves=3)
    except Exception as error:      # no licence, no weights, the v2 package: nothing this test can check
        pytest.skip('TabPFN-3.5 is not usable here: {}'.format(error))
    prediction = proposal.prediction
    assert len(proposal.recipes) == 4 and np.isfinite(prediction.to_numpy()).all()
    assert (prediction['low'] < prediction['high']).all() and (prediction['std'] > 0).all()
    assert (prediction['low'] <= prediction['mean'] + prediction['std']).all()
    assert proposal.settings['model_description']['version'] == 'v3.5'
    assert len(campaign.importance(proposal.fitted, repeats=1, max_rows=10)) == 3
