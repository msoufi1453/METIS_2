"""The round screen: the proposed recipes (editable), the pipetting information, the map and the downloads.

Nothing that only sometimes appears is drawn above the tabs: Streamlit forgets which tab
is open when the tabs move down or up the page. Waits and messages therefore have a fixed
slot (``st.empty``) or sit inside a tab. Widget keys differ from the set-up screen's,
because right after "Generate" both screens are drawn in the same run.
"""
import numpy as np
import pandas as pd
import streamlit as st

from metis.campaign import MODELS, CampaignError, VariableError, tutorial
from metis.campaign import columns as col
from metis.campaign.pipetting import safe_filename
from metis.idot import IdotError

from . import insight, state, texts

XLSX = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


def _changed_rows(old, new, names):
    """Positions of rows whose variable values differ between two tables."""
    differs = np.zeros(len(old), dtype=bool)
    for name in names:
        a, b = old[name].to_numpy(dtype=object), new[name].to_numpy(dtype=object)
        differs |= np.array([not (x == y or (pd.isna(x) and pd.isna(y))) for x, y in zip(a, b)])
    return np.flatnonzero(differs)


def apply_edit(rnd, variant, proposal, edited):
    """Take over cells the user changed: round them to allowed values, undo what does not fit
    the sample, and update the prediction. Returns True when the table changed."""
    campaign, names = rnd['campaign'], rnd['campaign'].names
    table = rnd['tables'][variant]
    edited = edited.reset_index(drop=True)
    rows = _changed_rows(table, edited, names)
    if not len(rows):
        return False
    notes = []
    try:
        snapped = campaign.snap(edited.loc[rows, names])
    except (CampaignError, VariableError) as error:
        st.session_state.notice = 'The change was undone: {}'.format(error)
        rnd['version'] += 1
        return True
    snapped.index = rows
    if campaign.liquid is not None:
        volume = campaign.volume_ul(snapped)
        over = volume > campaign.liquid.sample_volume_ul + 1e-9
        for position in rows[over]:
            notes.append('Data point {} would need {:.4g} µL, more than the {:g} µL sample: the change was '
                         'undone.'.format(position + 1, volume[list(rows).index(position)],
                                          campaign.liquid.sample_volume_ul))
        snapped = snapped.loc[rows[~over]]
    kept = snapped.index.to_numpy()
    kept = kept[_changed_rows(table.loc[kept, names].reset_index(drop=True), snapped.reset_index(drop=True), names)] \
        if len(kept) else kept
    if len(kept):
        table = table.copy()
        for name in names:
            table.loc[kept, name] = snapped.loc[kept, name].to_numpy()
        if proposal.guided:
            with st.spinner('Updating the prediction...'):
                prediction = proposal.predict(snapped.loc[kept])
            for column, key in zip(col.prediction_columns(campaign.target), ('mean', 'low', 'high')):
                table.loc[kept, column] = prediction[key].to_numpy()
        rnd['tables'][variant] = table
        rnd['edits'][variant] += len(kept)
        rnd['staged'].pop(variant, None)
    elif not notes:
        notes.append('The value was already the nearest one that can be used, so nothing changed.')
    st.session_state.notice = ' '.join(notes) or None
    rnd['version'] += 1
    return True


def stage(rnd, variant, proposal):
    """The campaign with this round added, its workbook and its pipetting plan (kept until the table changes)."""
    if variant in rnd['staged']:
        return rnd['staged'][variant]
    out = {'campaign': None, 'workbook': None, 'plan': None, 'error': None}
    campaign = rnd['campaign'].copy()
    try:
        campaign.add_round(rnd['tables'][variant], replicates=rnd['replicates'],
                           settings={**proposal.settings, 'replicates': rnd['replicates'],
                                     'recipes_edited_by_hand': rnd['edits'][variant]})
        out.update(campaign=campaign, workbook=campaign.to_workbook())
        if campaign.liquid is not None:
            out['plan'] = campaign.pipetting()
    except (CampaignError, VariableError) as error:
        out['error'] = str(error)
    rnd['staged'][variant] = out
    return out


def simulate_lab(variant):
    """Tutorial: fill in the round's results from the study's data and continue with the next round."""
    ss = st.session_state
    staged = ss.round['staged'].get(variant) or {}
    if staged.get('campaign') is None:
        return
    campaign = staged['campaign'].copy()
    pending = campaign.data[campaign.target].isna()
    rows = campaign.data.loc[pending]
    try:
        campaign.data.loc[pending, campaign.target] = tutorial.simulate(
            rows, rows[col.REPLICATE].to_numpy(), seed=int(rows[col.ROUND].max()))
    except (KeyError, VariableError, ValueError):
        ss.notice = ('The simulated lab only knows the tutorial\'s 11 components and their four steps. Go back, '
                     'restore the variables and generate the round again.')
        return
    state.open_campaign(campaign, mode='tutorial')


def _choose_proposal(rnd):
    """Draw the top of the Recipes tab (messages, the hill-climbing switch) and return
    ``(variant, proposal)``: the hill-climbed proposal when the switch is on, else the plain one.
    The climb is computed the first time the switch is turned on."""
    ss = st.session_state
    plain = rnd['proposal']
    notice, ss.notice = ss.notice, None             # shown once
    if notice:
        st.warning(notice)
    for warning in plain.warnings:
        st.warning(warning)
    if not plain.guided:
        return 'plain', plain
    st.toggle('Hill-climbing', key='r_hill', help=texts.HILL_HELP)
    slot = st.empty()
    if not ss.r_hill:
        return 'plain', plain
    if rnd['climbed'] is None:
        moves = int(ss.f_climb_moves)
        try:
            with slot, st.spinner('Hill-climbing from the {} best candidates (up to {} moves each)...'.format(
                    len(plain.recipes), moves)):
                rnd['climbed'] = plain.climb(max_steps=moves)
        except Exception as error:      # the model's own failures are shown, not thrown
            slot.error('Hill-climbing failed: {}: {}'.format(type(error).__name__, error))
            return 'plain', plain
        rnd['tables']['climbed'] = rnd['climbed'].table
    climbed = rnd['climbed']
    slot.caption('{} of the {} recipes were found by climbing ({} climbs; moves per climb: {}). On the PCA map, '
                 'dotted trails show where they started. Switch off to go back to the recipes chosen without '
                 'it.'.format(climbed.settings['climbed_in_batch'], len(climbed.recipes),
                              climbed.settings['climb_starts'],
                              ', '.join(str(m) for m in climbed.settings['climb_moves'])))
    return 'climbed', climbed


def _summary(campaign, proposal, variant):
    """One sentence on how the round was chosen."""
    if not proposal.guided:
        return '{} recipes spread over the design as a Latin hypercube (no model yet).'.format(len(proposal.recipes))
    settings = proposal.settings
    modules = [name for name, on in (('distance penalty', settings.get('penalty') is not None),
                                     ('fantasy batch', settings.get('fantasy')),
                                     ('hill-climbing', variant == 'climbed')) if on]
    return ('{} recipes chosen by {} from {:,} candidates, scored as {:.2f} x prediction + {:.2f} x uncertainty{}. '
            'The model learned from {} measurements.'.format(
                len(proposal.recipes), MODELS[campaign.model].label, settings['candidates_scored'],
                settings['exploitation'], settings['exploration'],
                '; modules: ' + ', '.join(modules) if modules else '', settings['training_rows']))


def _editor(rnd, variant):
    campaign, table = rnd['campaign'], rnd['tables'][variant]
    choices = campaign.space().choices()
    config = {col.POINT: st.column_config.NumberColumn(col.POINT, format='%d')}
    for v in campaign.variables:
        if choices[v.name] is not None:
            config[v.name] = st.column_config.SelectboxColumn(v.label, options=choices[v.name], required=True)
        else:
            config[v.name] = st.column_config.NumberColumn(v.label, min_value=v.low, max_value=v.high,
                                                           required=True, format='%.6g')
    predictions = [c for c in col.prediction_columns(campaign.target) if c in table.columns]
    for column in predictions:
        config[column] = st.column_config.NumberColumn(column, format='%.4g')
    shown = table.copy()
    shown.insert(0, col.ID, campaign.next_sample_id + shown[col.POINT] - 1)          # as the workbook will number them
    config[col.ID] = st.column_config.NumberColumn(col.ID, format='%d')
    return st.data_editor(shown, column_config=config, disabled=[col.ID, col.POINT] + predictions, hide_index=True,
                          column_order=[col.ID, col.POINT] + predictions + campaign.names,    # predictions in view
                          use_container_width=True, num_rows='fixed', height=insight.table_height(len(table)),
                          key='e_recipes_{}_{}'.format(variant, rnd['version']))


def _recipes_tab(rnd, variant, proposal, staged):
    ss = st.session_state
    campaign, target = rnd['campaign'], rnd['campaign'].target
    edited = _editor(rnd, variant)
    if apply_edit(rnd, variant, proposal, edited):
        st.rerun()
    notes = ['Change any cell to adjust a recipe: steps and categories from a list, continuous values by typing.']
    if proposal.guided:
        notes.append('"{}" is the model\'s prediction and the two interval columns the range it expects 80% of '
                     'measurements to fall in; they update when you change a recipe.'.format(col.predicted(target)))
    if campaign.liquid is not None:
        notes.append(texts.ROUNDING)
    st.caption(' '.join(notes))
    if staged['error']:
        st.error(staged['error'])
        return
    st.subheader('Download the round')
    recipes, replicates = len(rnd['tables'][variant]), rnd['replicates']
    st.markdown('The workbook lists {} recipe{} x {} replicate{} = **{} rows to fill in**{}.'.format(
        recipes, 's' if recipes != 1 else '', replicates, 's' if replicates != 1 else '', recipes * replicates,
        ', after the results of the earlier rounds' if len(campaign.data) else ''))
    st.markdown(texts.DOWNLOAD_STEPS.format(target=target, next=proposal.round + 1))
    st.download_button('Download the round {} workbook (.xlsx)'.format(proposal.round), staged['workbook'],
                       file_name='METIS_{}_round{}.xlsx'.format(safe_filename(campaign.name), proposal.round),
                       mime=XLSX, type='primary', key='d_workbook')
    if ss.mode == 'tutorial':
        st.button('Simulate the lab (tutorial): fill in the results and continue', key='b_simulate',
                  on_click=simulate_lab, args=(variant,))


def _pipetting_tab(rnd, proposal, staged):
    plan, campaign = staged['plan'], rnd['campaign']
    if plan is None:
        st.error(staged['error'] or 'No pipetting plan.')
        return
    st.markdown('**All volumes are in µL.** One column per well (1A = column 1, row A; wells fill column by column), '
                'one row per liquid. "{}" tops each well up to {:g} µL.'.format(
                    campaign.liquid.diluent, campaign.liquid.sample_volume_ul))
    plate = 1
    if len(plan.plates) > 1:
        plate = st.selectbox('Plate ({} wells on {} plates of 96)'.format(len(plan.layout), len(plan.plates)),
                             list(range(1, len(plan.plates) + 1)), key='r_plate')
    sheet = plan.table(plate)
    st.dataframe(sheet, use_container_width=True, height=insight.table_height(len(sheet)),
                 column_config={c: st.column_config.NumberColumn(format='%.3f') for c in sheet.columns})
    for note in plan.notes:
        st.caption(note)
    stem = '{}_round{}{}'.format(safe_filename(campaign.name), proposal.round,
                                 '_plate{}'.format(plate) if len(plan.plates) > 1 else '')
    left, right = st.columns(2)
    left.download_button('Download the pipetting table (.csv)', plan.csv(plate).encode('utf-8-sig'),
                         file_name='pipetting_{}.csv'.format(stem), mime='text/csv', key='d_pipetting')
    initials = right.text_input('Your initials, for the iDOT liquid names', key='f_initials', max_chars=10,
                                placeholder='e.g. MoS',
                                help='The iDOT file names each liquid <initials>_<name>, as the iDOT protocol '
                                     'builder does.')
    if initials.strip():
        try:
            idot = plan.idot(initials, 'iDOT_{}'.format(stem), plate)
        except IdotError as error:
            right.error(str(error))
        else:
            right.download_button('Download the iDOT protocol (.idotx, an XML file)', idot.data,
                                  file_name=idot.filename, mime='application/xml', key='d_idot')
            with st.expander('iDOT source plate: what to load, and the liquid names'):
                st.caption('Volumes in µL. The file was re-read and checked against the table before it was offered. '
                           'When the iDOT software meets a new liquid name it asks for a liquid class.')
                st.dataframe(idot.source, hide_index=True, use_container_width=True)
                st.dataframe(idot.liquids, hide_index=True, use_container_width=True)
                for warning in idot.warnings:
                    st.caption(warning)
    else:
        right.caption('Enter your initials to get the iDOT protocol file.')
    with st.expander('Plate layout: which sample is in which well'):
        st.dataframe(plan.layout, hide_index=True, use_container_width=True)


def render():
    ss = st.session_state
    rnd = ss.round
    campaign, plain = rnd['campaign'], rnd['proposal']
    head = st.columns([5, 1.3, 0.9])
    head[0].title('{}: round {}'.format(campaign.name, plain.round))
    head[1].button('Back to settings', key='b_back', on_click=state.go, args=('setup',), use_container_width=True)
    head[2].button('Home', key='b_home_round', on_click=state.go, args=('home',), use_container_width=True)
    summary = st.empty()
    if ss.mode == 'tutorial':
        with st.expander('Tutorial: what to do on this screen', expanded=True):
            st.markdown(texts.TUTORIAL_ROUND)
    has_results = len(campaign.measured) >= 2
    names = ['Recipes'] + (['Pipetting information'] if campaign.liquid is not None else []) + ['PCA map'] + \
        (['Progress', 'Importance', 'Interactions'] if has_results else [])
    tabs = dict(zip(names, st.tabs(names)))
    with tabs['Recipes']:
        variant, proposal = _choose_proposal(rnd)
    summary.caption(_summary(campaign, proposal, variant))
    staged = stage(rnd, variant, proposal)
    with tabs['Recipes']:
        _recipes_tab(rnd, variant, proposal, staged)
    if 'Pipetting information' in tabs:
        with tabs['Pipetting information']:
            _pipetting_tab(rnd, proposal, staged)
    with tabs['PCA map']:
        before = rnd['tables']['plain'] if variant == 'climbed' else None
        insight.pca_tab(campaign, proposed=rnd['tables'][variant], before=before, moves=proposal.moves,
                        round_number=plain.round, fitted=proposal.fitted, key='round')
    if has_results:
        with tabs['Progress']:
            insight.progress_tab(campaign)
        with tabs['Importance']:
            insight.importance_tab(campaign, proposal.fitted, key='round')
        with tabs['Interactions']:
            insight.interactions_tab(campaign, proposal.fitted, key='round')
