"""The app (gui/app.py), run headless with Streamlit's test client and a fast stand-in model.

The client cannot drive file uploaders or the editable table, so uploads are handed over
the way the uploaders do it (``inbox_workbook``, ``inbox_day0`` in the session state) and
table edits are tested through ``round_page.apply_edit``.
"""
import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest  # noqa: E402

from metis.campaign import Campaign, tutorial  # noqa: E402
from metis.campaign import columns as col  # noqa: E402

from .helpers import register_fast_model  # noqa: E402
from .test_campaign_api import lab, measured_campaign  # noqa: E402

FAST = register_fast_model()
FAST_LABEL = 'Fast test model'


def start():
    at = AppTest.from_file('gui/app.py', default_timeout=120)
    return ok(at.run())


def ok(at):
    """The run finished without an exception; returns the client for chaining."""
    assert not at.exception, [e.value for e in at.exception]
    return at


def keys(elements):
    return [element.key for element in elements]


def fill_row(at, r, kind, name, **values):
    """Fill variable row ``r``: ``kind`` and name, then number inputs / text inputs by their key suffix."""
    if kind != 'Continuous':
        ok(at.radio(key='row{}_kind'.format(r)).set_value(kind).run())
    at.text_input(key='row{}_name'.format(r)).set_value(name)
    if 'n' in values:
        ok(at.number_input(key='row{}_n'.format(r)).set_value(values.pop('n')).run())
    for suffix, value in values.items():
        key = 'row{}_{}'.format(r, suffix)
        (at.text_input if isinstance(value, str) else at.number_input)(key=key).set_value(value)
    return ok(at.run())


def new_campaign_with_three_variables(at):
    ok(at.button(key='b_new').click().run())
    rows = at.session_state['rows']
    fill_row(at, rows[0], 'Continuous', 'Mg', unit='mM', low=0.0, high=10.0, stock=500.0)
    ok(at.button(key='b_add').click().run())
    fill_row(at, at.session_state['rows'][1], 'Discrete', 'K', n=2, unit='mM', x0=40.0, x1=80.0, stock=4000.0)
    ok(at.button(key='b_add').click().run())
    fill_row(at, at.session_state['rows'][2], 'Categorical', 'Promoter', n=2, c0='T7', c1='T3')
    return ok(at.selectbox(key='f_model').set_value(FAST_LABEL).run())


def test_home_has_one_main_button_and_two_below():
    at = start()
    assert keys(at.button) == ['b_new', 'b_continue', 'b_tutorial']
    assert [b.label for b in at.button] == ['Start a new campaign', 'Continue a campaign', 'Tutorial']
    assert at.session_state['page'] == 'home' and at.title[0].value == 'METIS'


def test_generate_waits_for_one_complete_variable():
    at = start()
    ok(at.button(key='b_new').click().run())
    assert at.session_state['page'] == 'setup' and at.session_state['mode'] == 'new'
    assert at.selectbox(key='f_model').options == ['TabPFN-3.5', FAST_LABEL]
    assert at.number_input(key='f_samples').value == 10 and at.number_input(key='f_replicates').value == 1
    assert at.toggle(key='f_lh').value is True and at.number_input(key='f_droplet').value == 4.0
    assert at.slider(key='f_exploit').value == 0.5 and at.slider(key='f_exploit').disabled      # round 1 is random
    assert any('Latin hypercube' in info.value for info in at.info)
    generate = at.button(key='b_generate')
    assert generate.disabled and generate.label == 'Generate round 1 data'
    r = at.session_state['rows'][0]
    at.text_input(key='row{}_name'.format(r)).set_value('Mg')
    ok(at.number_input(key='row{}_low'.format(r)).set_value(0.0).run())
    assert at.button(key='b_generate').disabled
    assert [w.value for w in at.warning] == ['Mg: enter the minimum and the maximum.', 'Mg: enter the stock concentration.']
    at.number_input(key='row{}_high'.format(r)).set_value(10.0)
    ok(at.number_input(key='row{}_stock'.format(r)).set_value(5.0).run())
    assert at.button(key='b_generate').disabled                          # the maximum is above the stock
    assert any('above the stock concentration' in e.value for e in at.error)
    ok(at.number_input(key='row{}_stock'.format(r)).set_value(500.0).run())
    assert not at.button(key='b_generate').disabled and not at.warning and not at.error
    ok(at.button(key='b_add').click().run())                             # an untouched extra row is ignored
    assert len(at.session_state['rows']) == 2 and not at.button(key='b_generate').disabled
    ok(at.button(key='b_remove{}'.format(at.session_state['rows'][1])).click().run())
    assert len(at.session_state['rows']) == 1


def test_form_names_what_is_missing():
    at = start()
    ok(at.button(key='b_new').click().run())
    fill_row(at, at.session_state['rows'][0], 'Discrete', 'K', n=3, x0=40.0, x2=40.0, stock=4000.0)
    assert [w.value for w in at.warning] == ['K: fill in all 3 steps.']
    ok(at.number_input(key='row{}_x1'.format(at.session_state['rows'][0])).set_value(80.0).run())
    assert [w.value for w in at.warning] == ['K lists the same step twice.']
    ok(at.number_input(key='row{}_x2'.format(at.session_state['rows'][0])).set_value(120.0).run())
    assert not at.warning and not at.button(key='b_generate').disabled
    ok(at.button(key='b_add').click().run())
    r = at.session_state['rows'][1]
    fill_row(at, r, 'Categorical', 'k', n=2, c0='a')
    assert sorted(w.value for w in at.warning) == ['k: fill in all 2 categories.']
    ok(at.text_input(key='row{}_c1'.format(r)).set_value('b').run())
    assert [w.value for w in at.warning] == ['Two variables are called "k".'] and at.button(key='b_generate').disabled
    ok(at.toggle(key='f_lh').set_value(False).run())                     # without liquid handling: no stock boxes
    assert not [k for k in keys(at.number_input) if k.endswith('_stock')]


def test_new_campaign_round_one_is_a_latin_hypercube():
    at = new_campaign_with_three_variables(start())
    at.number_input(key='f_samples').set_value(6)
    ok(at.number_input(key='f_replicates').set_value(2).run())
    assert [w.value for w in at.warning] == [] and not at.button(key='b_generate').disabled
    ok(at.button(key='b_generate').click().run())
    assert at.session_state['page'] == 'round'
    rnd = at.session_state['round']
    proposal, table = rnd['proposal'], rnd['tables']['plain']
    assert not proposal.guided and proposal.round == 1 and list(table.columns) == [col.POINT, 'Mg', 'K', 'Promoter']
    assert len(table) == 6 and set(table['K']) <= {40.0, 80.0} and set(table['Promoter']) <= {'T7', 'T3'}
    assert [tab.label for tab in at.tabs] == ['Recipes', 'Pipetting information', 'PCA map']
    assert at.title[0].value == 'My campaign: round 1' and 'Latin hypercube' in at.caption[0].value
    assert 'r_hill' not in keys(at.toggle)                               # no model yet: nothing to climb
    assert set(keys(at.button)) == {'b_back', 'b_home_round'}            # the set-up screen is gone
    staged = rnd['staged']['plain']
    assert staged['error'] is None and len(staged['plan'].layout) == 12
    loaded = Campaign.from_workbook(staged['workbook'], require_results=False)
    assert len(loaded.data) == 12 and loaded.data[col.REPLICATE].tolist() == [1, 2] * 6 and loaded.name == 'My campaign'
    assert any('All volumes are in µL' in m.value for m in at.markdown)
    assert any('rows to fill in' in m.value for m in at.markdown)
    ok(at.text_input(key='f_initials').set_value('MoS').run())
    assert not at.error
    ok(at.button(key='b_back').click().run())                            # back: the form is as it was left
    assert at.session_state['page'] == 'setup' and at.number_input(key='f_samples').value == 6
    assert at.text_input(key='row{}_name'.format(at.session_state['rows'][2])).value == 'Promoter'


def filled_workbook(n=12):
    campaign = measured_campaign(n)
    return campaign, campaign.to_workbook()


def test_continue_needs_every_result_then_guides_the_next_round():
    at = start()
    ok(at.button(key='b_continue').click().run())
    assert at.session_state['page'] == 'upload'
    unfinished = measured_campaign(6)
    unfinished.add_round(unfinished.propose(4))
    at.session_state['inbox_workbook'] = unfinished.to_workbook()
    ok(at.run())
    assert at.session_state['page'] == 'upload' and '4 of the 10 rows have no value' in at.error[0].value
    at.session_state['inbox_workbook'] = b'not a workbook'
    ok(at.run())
    assert 'cannot be opened' in at.error[0].value

    campaign, data = filled_workbook()
    at.session_state['inbox_workbook'] = data
    ok(at.run())
    assert at.session_state['page'] == 'setup' and at.session_state['mode'] == 'continue'
    assert at.title[0].value == 'test: set up round 2' and at.selectbox(key='f_model').value == FAST_LABEL
    assert [tab.label for tab in at.tabs] == ['Round settings', 'PCA map', 'Progress', 'Importance', 'Interactions']
    assert len(at.session_state['rows']) == 3 and at.number_input(key='f_samples').value == 12
    slider = at.slider(key='f_exploit')
    assert (slider.value, slider.min, slider.max, slider.disabled) == (0.5, 0.5, 1.0, False)
    assert at.toggle(key='f_penalty_on').value is False and at.toggle(key='f_fantasy').value is False
    at.toggle(key='f_penalty_on').set_value(True)
    at.toggle(key='f_fantasy').set_value(True)
    ok(at.number_input(key='f_samples').set_value(5).run())
    ok(at.slider(key='f_penalty').set_value(1.0).run())
    ok(at.button(key='b_generate').click().run())
    rnd = at.session_state['round']
    settings = rnd['proposal'].settings
    assert at.session_state['page'] == 'round' and rnd['proposal'].guided and rnd['proposal'].round == 2
    assert (settings['exploitation'], settings['penalty'], settings['fantasy']) == (0.5, 1.0, True)
    table = rnd['tables']['plain']
    assert list(table.columns) == [col.POINT, 'Mg', 'K', 'Promoter'] + col.prediction_columns('sfGFP')
    assert len(table) == 5 and table[col.prediction_columns('sfGFP')].notna().all().all()
    assert [tab.label for tab in at.tabs] == ['Recipes', 'Pipetting information', 'PCA map', 'Progress', 'Importance',
                                              'Interactions']
    assert 'distance penalty, fantasy batch' in at.caption[0].value

    ok(at.toggle(key='r_hill').set_value(True).run())                    # hill-climbing: computed once, kept
    rnd = at.session_state['round']
    assert rnd['climbed'] is not None and rnd['climbed'].before is rnd['proposal'] and 'climbed' in rnd['tables']
    assert 'hill-climbing' in at.caption[0].value and 'climbed' in rnd['staged']
    second = Campaign.from_workbook(rnd['staged']['climbed']['workbook'], require_results=False)
    assert second.history[-1]['settings']['hill_climb'] is True and len(second.pending) == 5
    ok(at.toggle(key='r_hill').set_value(False).run())                   # and back
    assert 'hill-climbing' not in at.caption[0].value
    first = Campaign.from_workbook(at.session_state['round']['staged']['plain']['workbook'], require_results=False)
    assert first.history[-1]['settings']['hill_climb'] is False
    assert first.data.iloc[:12][campaign.names].equals(campaign.data[campaign.names])

    ok(at.button(key='b_importance_round').click().run())
    ok(at.button(key='b_interactions_round').click().run())
    analysis = at.session_state['analysis']
    assert set(analysis['importance'].index) == {'Mg', 'K', 'Promoter'} and len(analysis['interactions']) == 3
    assert 'b_importance_round' not in keys(at.button)                   # computed: the buttons are gone


def test_a_new_variable_needs_its_value_in_the_earlier_rounds():
    at = start()
    at.session_state['inbox_workbook'] = filled_workbook()[1]
    ok(at.run())
    ok(at.button(key='b_add').click().run())
    r = at.session_state['rows'][-1]
    fill_row(at, r, 'Continuous', 'DTT', unit='mM', low=0.0, high=2.0, stock=100.0)
    assert [w.value for w in at.warning] == ['DTT: enter the value it had in the data so far.']
    assert at.button(key='b_generate').disabled
    ok(at.number_input(key='row{}_past'.format(r)).set_value(0.0).run())
    assert not at.warning and not at.button(key='b_generate').disabled
    ok(at.button(key='b_remove{}'.format(at.session_state['rows'][1])).click().run())      # drop K
    ok(at.button(key='b_generate').click().run())
    campaign = at.session_state['round']['campaign']
    assert campaign.names == ['Mg', 'Promoter', 'DTT'] and campaign.extra == ['K'] and campaign.data['DTT'].eq(0).all()
    assert list(at.session_state['round']['tables']['plain'].columns)[1:4] == ['Mg', 'Promoter', 'DTT']


DAY0 = 'Mg (mM),K (mM),Promoter,Run,sfGFP\n' + ''.join(
    '{},{},{},run{},{}\n'.format(mg, k, p, i, 1 + 0.1 * mg + (p == 'T7') * 0.5 + k / 200)
    for i, (mg, k, p) in enumerate([(1, 40, 'T7'), (1, 40, 'T7'), (3, 80, 'T3'), (5, 40, 'T3'), (7, 80, 'T7'),
                                    (9, 40, 'T7'), (2, 80, 'T3'), (8, 80, 'T3')]))


def test_day0_data_makes_the_variables_and_guides_round_one():
    at = start()
    ok(at.button(key='b_new').click().run())
    at.session_state['inbox_day0'] = ('before.csv', DAY0.encode())
    ok(at.run())
    assert at.session_state['day0']['frame'].shape == (8, 5) and at.selectbox(key='f_day0_target').value == 'sfGFP'
    rows = at.session_state['rows']
    assert [at.text_input(key='row{}_name'.format(r)).value for r in rows] == ['Mg', 'K', 'Promoter']
    assert [at.radio(key='row{}_kind'.format(r)).value for r in rows] == ['Discrete', 'Discrete', 'Categorical']
    assert at.text_input(key='row{}_unit'.format(rows[0])).value == 'mM'
    assert any('"Run" was not made a variable' in c.value for c in at.caption)
    assert sorted(w.value for w in at.warning) == ['K: enter the stock concentration.', 'Mg: enter the stock concentration.']
    at.number_input(key='row{}_stock'.format(rows[0])).set_value(500.0)
    at.number_input(key='row{}_stock'.format(rows[1])).set_value(4000.0)
    ok(at.selectbox(key='f_model').set_value(FAST_LABEL).run())
    assert not at.warning and not at.slider(key='f_exploit').disabled and at.slider(key='f_exploit').min == 0.0
    assert [tab.label for tab in at.tabs][0] == 'Round settings' and at.toggle(key='f_use_day0').value is True
    ok(at.button(key='b_generate').click().run())
    rnd = at.session_state['round']
    assert rnd['proposal'].guided and rnd['proposal'].round == 1
    campaign = rnd['campaign']
    assert campaign.rounds == [0] and campaign.data[col.ID].tolist() == [1, 1, 2, 3, 4, 5, 6, 7]      # a replicate pair
    loaded = Campaign.from_workbook(rnd['staged']['plain']['workbook'], require_results=False)
    assert loaded.rounds == [0, 1] and len(loaded.pending) == 10 and loaded.target == 'sfGFP'

    ok(at.button(key='b_back').click().run())                            # the same data, round 1 as a Latin hypercube
    ok(at.toggle(key='f_use_day0').set_value(False).run())
    assert at.slider(key='f_exploit').disabled
    ok(at.button(key='b_generate').click().run())
    rnd = at.session_state['round']
    assert not rnd['proposal'].guided and rnd['campaign'].rounds == [0]

    ok(at.button(key='b_back').click().run())
    ok(at.button(key='b_remove_day0').click().run())
    assert at.session_state['day0'] is None and at.text_input(key='f_target').value == 'Result'
    at.session_state['inbox_day0'] = ('broken.csv', b'only text\nno numbers\n')
    ok(at.run())
    assert at.session_state['day0'] is None and 'Could not read "broken.csv"' in at.error[0].value


def test_tutorial_runs_two_rounds_with_the_simulated_lab():
    at = start()
    ok(at.button(key='b_tutorial').click().run())
    assert at.session_state['mode'] == 'tutorial' and len(at.session_state['rows']) == 11
    assert at.title[0].value == 'Tutorial: Borkowski tutorial' and 'step 1 of 3' in at.expander[0].markdown[0].value
    assert at.selectbox(key='f_day0_target').value == 'Yield' and not at.warning and not at.error
    assert not at.button(key='b_generate').disabled and at.slider(key='f_exploit').value == 0.5
    ok(at.selectbox(key='f_model').set_value(FAST_LABEL).run())
    ok(at.button(key='b_generate').click().run())
    rnd = at.session_state['round']
    assert rnd['proposal'].guided and rnd['proposal'].round == 1 and len(rnd['tables']['plain']) == 10
    assert 'step 2 of 3' in at.expander[0].markdown[0].value and 'b_simulate' in keys(at.button)
    ok(at.button(key='b_simulate').click().run())
    base = at.session_state['base']
    assert at.session_state['page'] == 'setup' and at.session_state['mode'] == 'tutorial'
    assert base.rounds == [0, 1] and len(base.data) == 109 and not len(base.pending)
    assert len(at.session_state['rows']) == 11 and 'step 3 of 3' in at.expander[0].markdown[0].value
    assert at.slider(key='f_exploit').value == 1.0 and at.slider(key='f_exploit').min == 0.5
    assert at.button(key='b_generate').label == 'Generate round 2 data' and not at.button(key='b_generate').disabled
    assert at.metric[0].value == '1' and at.metric[2].value == '109'
    ok(at.button(key='b_generate').click().run())
    rnd = at.session_state['round']
    assert rnd['proposal'].round == 2 and rnd['proposal'].settings['exploitation'] == 1.0
    simulated = tutorial.simulate(base.data[base.data[col.ROUND] == 1])
    assert simulated.tolist() == base.data.loc[base.data[col.ROUND] == 1, 'Yield'].tolist()
    ok(at.button(key='b_home_round').click().run())                      # home offers the way back to the round
    assert at.session_state['page'] == 'home' and 'b_resume' in keys(at.button)
    ok(at.button(key='b_resume').click().run())
    assert at.session_state['page'] == 'round'


def edit_script():
    """Runs inside the test client: change cells of a proposed round the way the table editor reports them."""
    import streamlit as st

    from gui import round_page
    from metis.campaign import Campaign, Categorical, Continuous, LiquidHandling
    from tests.helpers import register_fast_model

    campaign = Campaign([Continuous('a', 0, 60, unit='mM', stock=100), Continuous('b', 0, 60, unit='mM', stock=100),
                         Categorical('c', ['x', 'y'])], model=register_fast_model(),
                        liquid_handling=LiquidHandling(sample_volume_ul=10))
    campaign.add_round(campaign.propose(12))
    campaign.data['Result'] = campaign.data['a'] * 0.1 - campaign.data['b'] * 0.05 + (campaign.data['c'] == 'x')
    proposal = campaign.propose(4)
    rnd = {'campaign': campaign, 'proposal': proposal, 'climbed': None, 'replicates': 1,
           'tables': {'plain': proposal.table}, 'edits': {'plain': 0, 'climbed': 0}, 'version': 0,
           'staged': {'plain': 'kept until an edit'}}
    start = proposal.table.copy()
    log = []

    def edit(**cells):
        """Apply {column: (row, value)} and record what happened."""
        edited = rnd['tables']['plain'].copy()
        for column, (row, value) in cells.items():
            edited.loc[row, column] = value
        st.session_state.notice = None
        changed = round_page.apply_edit(rnd, 'plain', proposal, edited)
        log.append({'changed': changed, 'notice': st.session_state.notice, 'table': rnd['tables']['plain'].copy(),
                    'edits': rnd['edits']['plain'], 'version': rnd['version'], 'staged': dict(rnd['staged'])})

    edit()                                          # nothing changed
    edit(a=(0, 3.333), c=(0, 'y' if start.loc[0, 'c'] == 'x' else 'x'))      # rounded to whole droplets
    edit(a=(1, 999.0))                              # above the range: its maximum
    edit(a=(2, 60.0), b=(2, 60.0))                  # 12 uL in a 10 uL sample: undone
    edit(a=(0, 3.33))                               # rounds back to the value it has
    st.session_state.log, st.session_state.start = log, start


def test_edited_cells_are_rounded_checked_and_predicted_again():
    at = ok(AppTest.from_function(edit_script, default_timeout=120).run())
    log, start = at.session_state['log'], at.session_state['start']
    predicted = col.prediction_columns('Result')[0]
    assert log[0]['changed'] is False and log[0]['staged'] == {'plain': 'kept until an edit'}
    first = log[1]
    assert first['changed'] and first['table'].loc[0, 'a'] == 3.32 and first['table'].loc[0, 'c'] != start.loc[0, 'c']
    assert first['table'].loc[0, predicted] != start.loc[0, predicted]                    # predicted again
    assert first['table'].loc[1:, :].equals(start.loc[1:, :])                             # other recipes untouched
    assert (first['edits'], first['version'], first['staged'], first['notice']) == (1, 1, {}, None)
    assert log[2]['table'].loc[1, 'a'] == 60.0 and log[2]['edits'] == 2
    undone = log[3]
    assert undone['changed'] and undone['table'].equals(log[2]['table']) and undone['edits'] == 2
    assert 'Data point 3 would need 12 µL, more than the 10 µL sample' in undone['notice'] and undone['version'] == 3
    assert 'already the nearest' in log[4]['notice'] and log[4]['table'].equals(log[2]['table'])
