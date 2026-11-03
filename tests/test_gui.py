"""GUI: the logic (gui/core.py) and a headless run of the Streamlit page (gui/app.py)."""
import numpy as np
import pandas as pd
import pytest

from gui import core

CHEAP = {'type': 'xgb_ensemble', 'n_iter': 3, 'ensemble_len': 3}


@pytest.fixture(scope='module')
def table():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({'id': ['r{}'.format(i) for i in range(n)],
                       'a': rng.choice([0.0, 1.0, 2.0, 4.0], n), 'b': rng.choice([10.0, 20.0, 30.0], n),
                       'strain': rng.choice(['wt', 'k1', 'k2'], n)})
    df['yield'] = df.a * (df.strain == 'k1') + 0.01 * df.b + rng.normal(0, 0.05, n)
    df.loc[60:, 'yield'] = np.nan                    # unmeasured rows: candidates for 'rows' mode
    return df


def test_guess_columns_skips_ids(table):
    numeric, categorical = core.guess_columns(table, 'yield')
    assert numeric == ['a', 'b'] and categorical == ['strain']


@pytest.mark.parametrize('rule', core.BATCH_RULES)
def test_propose_every_batch_rule(table, rule):
    design = core.build_design(table, 'yield', ['a', 'b'], ['strain'])
    p = core.propose(design, CHEAP, batch_size=5, pool_size=30, rule=rule, seed=1)
    assert len(p.table) == 5 and set(p.table.strain) <= {'wt', 'k1', 'k2'}
    measured = set(design.space.keys(design.idx[design.measured]))
    assert not measured & set(design.space.keys(p.pool[p.chosen]))     # never re-proposes a measured recipe
    assert {'predicted yield', 'uncertainty (std)', 'from hill-climbing'} <= set(p.table.columns)


def test_propose_rows_mode_and_hill_climbing(table):
    design = core.build_design(table, 'yield', ['a', 'b'], ['strain'])
    rows = core.propose(design, CHEAP, batch_size=4, source='rows', seed=2)
    candidates = set(design.space.keys(design.idx[~design.measured]))
    assert set(design.space.keys(rows.pool[rows.chosen])) <= candidates
    climbed = core.propose(design, {'type': 'gp_matern'}, batch_size=4, pool_size=10, hill=True, n_starts=3,
                           max_steps=5, seed=2)
    assert climbed.climb is not None and (climbed.climb.steps <= 5).all()


def test_pca_and_importance(table):
    design = core.build_design(table, 'yield', ['a', 'b'], ['strain'])
    model, xy, loadings = core.pca(design)
    assert xy.shape == (len(design.frame), 2) and set(loadings.index) == {'a', 'b', 'strain'}
    imp = core.importance(design, CHEAP, repeats=2)
    assert set(imp.index) == {'a', 'b', 'strain'}


def test_streamlit_page_runs_headless():
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file('gui/app.py', default_timeout=300)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value('Example: Borkowski (cell-free, 11 factors)').run()
    assert not at.exception
    at.selectbox(key='model').set_value('Linear regression')    # fast; the GP on 1,007 recipes takes minutes
    at.number_input(key='pool').set_value(500)
    at.radio(key='rule').set_value('Distance penalty: low')
    at.button(key='go').click().run()
    assert not at.exception
    assert any('Suggested batch' in s.value for s in at.subheader)
