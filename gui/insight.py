"""The tabs that look at a campaign's data: PCA map, progress, importance and interactions.

They are shown on the set-up screen (the data so far) and on the round screen (with the
proposed recipes on the map). Importance and interactions need the fitted model, so they
are computed on request and kept until the data or the design changes.
"""
import numpy as np
import pandas as pd
import streamlit as st

from metis.campaign import has_gpu
from metis.campaign import columns as col

from . import charts, texts

CPU_ROWS_PER_SECOND, GPU_ROWS_PER_SECOND = 25.0, 2000.0      # TabPFN-3.5, 8 estimators (docs/APP.md)
MAX_INTERACTION_VARIABLES = 12


ROW_HEIGHT = 35         # of Streamlit's tables, to size them to their rows


def table_height(rows, most=800):
    return min(ROW_HEIGHT * (rows + 1) + 3, most)


def duration(rows):
    """A rough "about 40 s" for scoring ``rows`` recipes on this computer with the chosen number of estimators."""
    rate = (GPU_ROWS_PER_SECOND if has_gpu() else CPU_ROWS_PER_SECOND) * 8.0 / max(
        int(st.session_state.get('f_estimators') or 8), 1)
    seconds = rows / rate + 2
    return 'about {:.0f} s'.format(seconds) if seconds < 90 else 'about {:.0f} min'.format(seconds / 60)


def _cache(campaign):
    """The store for this campaign's computed analysis; a changed campaign starts an empty one."""
    ss = st.session_state
    key = campaign.fingerprint()
    if ss.analysis.get('key') != key:
        ss.analysis = {'key': key}
    return ss.analysis


def fitted_model(campaign, fitted=None):
    cache = _cache(campaign)
    if fitted is not None:
        cache['fitted'] = fitted
    if 'fitted' not in cache:
        cache['fitted'] = campaign.fit()
    return cache['fitted']


def _interval(mean, low, high):
    if mean is None or pd.isna(mean):
        return '–'
    return '{} ({} to {})'.format(charts.number(mean), charts.number(low), charts.number(high))


def round_name(number):
    return 'Round 0 (day 0 data)' if int(number) == 0 else 'Round {}'.format(int(number))


def map_points(campaign, pca, proposed=None, before=None, round_number=None):
    """One row per recipe for the PCA map: the measured ones, the proposed ones and, with
    hill-climbing, the recipes proposed before it."""
    names, target = campaign.names, campaign.target
    predicted, low, high = col.prediction_columns(target)
    samples = campaign.samples().dropna(subset=[target])
    frames = []
    if len(samples):
        xy = campaign.project(pca, samples[names])
        frame = samples[names].reset_index(drop=True)
        frame['uid'] = ['S{}'.format(i) for i in samples[col.ID]]
        frame['x'], frame['y'], frame['group'] = xy[:, 0], xy[:, 1], charts.MEASURED
        frame['round'] = [round_name(r) for r in samples[col.ROUND]]
        frame['label'] = ['{} · Data point {} (sample {})'.format(round_name(r), p, i) for r, p, i in
                          zip(samples[col.ROUND], samples[col.POINT], samples[col.ID])]
        frame['result'] = samples[target].to_numpy(dtype=float)
        frame['measured'] = ['{}{}'.format(charts.number(v), ' (mean of {})'.format(n) if n > 1 else '')
                             for v, n in zip(samples[target], samples['Results'])]
        frame['predicted'] = [_interval(m, a, b) for m, a, b in zip(samples[predicted], samples[low], samples[high])]
        frames.append(frame)
    for table, group, prefix in ((proposed, charts.PROPOSED, 'P'), (before, charts.BEFORE, 'B')):
        if table is None or not len(table):
            continue
        xy = campaign.project(pca, table[names])
        frame = table[names].reset_index(drop=True)
        frame['uid'] = ['{}{}'.format(prefix, p) for p in table[col.POINT]]
        frame['x'], frame['y'], frame['group'] = xy[:, 0], xy[:, 1], group
        frame['round'] = 'Round {}'.format(round_number)
        if group == charts.PROPOSED:            # the Sample IDs the workbook will give these recipes
            frame['label'] = ['Round {} · Data point {} (sample {})'.format(round_number, p,
                                                                            campaign.next_sample_id + p - 1)
                              for p in table[col.POINT]]
        else:
            frame['label'] = ['Before hill-climbing · Data point {}'.format(p) for p in table[col.POINT]]
        frame['result'] = np.nan
        frame['measured'] = 'not yet'
        frame['predicted'] = ([_interval(m, a, b) for m, a, b in zip(table[predicted], table[low], table[high])]
                              if predicted in table.columns else '–')
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _climb_paths(campaign, pca, proposed, moves):
    """Line segments from where each climbed recipe started to where it ended."""
    if moves is None or not len(moves):
        return None
    starts = moves[['start ' + name for name in campaign.names]].copy()
    starts.columns = campaign.names
    begin = campaign.project(pca, starts)
    ends = proposed.set_index(col.POINT).loc[moves[col.POINT], campaign.names]
    end = campaign.project(pca, ends)
    return pd.DataFrame({'x': begin[:, 0], 'y': begin[:, 1], 'x2': end[:, 0], 'y2': end[:, 1],
                         'label': ['Climb to data point {} ({} moves)'.format(p, m)
                                   for p, m in zip(moves[col.POINT], moves['Moves'])]})


def _details(campaign, points, uid, fitted):
    """Everything about one point of the map, as a two-column table."""
    row = points.loc[points['uid'] == uid].iloc[0]
    target = campaign.target
    rows = [('Recipe', row['label']), ('Measured ' + target, row['measured'])]
    if row['group'] == charts.MEASURED:
        rows.append(('Predicted before it was measured (80% interval)', row['predicted']))
        values = campaign.data.loc[campaign.data[col.ID] == int(uid[1:]), target].dropna()
        if len(values) > 1:
            rows.append(('Each replicate', ', '.join(charts.number(v) for v in values)))
        if fitted is not None:
            cache = _cache(campaign).setdefault('point_predictions', {})
            if uid not in cache:
                with st.spinner('Asking the model about this recipe...'):
                    now = campaign.predict(pd.DataFrame([row[campaign.names]]), fitted).iloc[0]
                cache[uid] = _interval(now['mean'], now['low'], now['high'])
            rows.append(("The model's prediction now (80% interval)", cache[uid]))
    else:
        rows.append(('Predicted {} (80% interval)'.format(target), row['predicted']))
    rows += [(v.label, charts.number(row[v.name], 6)) for v in campaign.variables]
    return pd.DataFrame(rows, columns=['', 'Value']).astype(str)


def pca_tab(campaign, proposed=None, before=None, moves=None, round_number=None, fitted=None, key='setup'):
    """``key`` names the screen, to keep widget keys apart (also for the two tabs below)."""
    samples = campaign.samples().dropna(subset=[campaign.target])
    if len(samples) + (len(proposed) if proposed is not None else 0) < 2:
        st.info('The map needs at least two recipes.')
        return
    pca = campaign.pca(recipes=proposed)
    points = map_points(campaign, pca, proposed, before, round_number)
    variables = [(v.name, v.label) for v in campaign.variables]
    paths = _climb_paths(campaign, pca, proposed, moves)
    chart = charts.pca_map(points, campaign.target, variables, pca.explained, paths)
    st.caption(texts.PCA + (' Darker blue = higher {}; the shape tells the round.'.format(campaign.target)
                            if len(samples) else ''))
    if proposed is not None and len(proposed):
        st.markdown(charts.map_legend('Proposed for round {}'.format(round_number), before is not None,
                                      paths is not None), unsafe_allow_html=True)
    event = st.altair_chart(chart, use_container_width=True, on_select='rerun', selection_mode='pick',
                            key='c_map_' + key)
    picked = [p.get('uid') for p in (event or {}).get('selection', {}).get('pick', []) if isinstance(p, dict)]
    picked = [uid for uid in picked if uid in set(points['uid'])]
    if picked:
        details = _details(campaign, points, picked[0], fitted)
        st.dataframe(details, hide_index=True, use_container_width=True, height=table_height(len(details)))
    else:
        st.caption('Click a point to see all its values, its measured result and the model\'s prediction.')
    with st.expander('All points as a table'):
        listing = points[['label', 'measured', 'predicted'] + campaign.names].rename(
            columns={'label': 'Recipe', 'measured': 'Measured ' + campaign.target,
                     'predicted': 'Predicted (80% interval)', **{v.name: v.label for v in campaign.variables}})
        st.dataframe(listing, hide_index=True, use_container_width=True)


def progress_tab(campaign):
    table = campaign.progress()
    if len(table) < 2:
        st.info('Progress over the rounds is shown once two rounds have results.')
        return
    samples = campaign.samples().dropna(subset=[campaign.target])
    st.caption('Every measured recipe by round (replicates averaged), and the best result so far.')
    st.altair_chart(charts.progress(samples, campaign.target, col.ROUND, col.ID), use_container_width=True)
    st.dataframe(table, hide_index=True, use_container_width=True)


def _settings():
    """Repeats and sample sizes: thorough with a GPU, lighter without one."""
    return {'repeats': 5, 'max_rows': 300, 'n_pairs': 40} if has_gpu() else \
        {'repeats': 2, 'max_rows': 100, 'n_pairs': 15}


def _compute_importance(campaign, fitted):
    cache = _cache(campaign)
    if 'importance' not in cache:
        setting = _settings()
        model = fitted_model(campaign, fitted)
        cache['importance'] = campaign.importance(model, repeats=setting['repeats'], max_rows=setting['max_rows'])
    return cache['importance']


def importance_tab(campaign, fitted=None, key='setup'):
    st.caption(texts.IMPORTANCE)
    if len(campaign.measured) < 2:
        st.info('Importance needs at least two measured results.')
        return
    cache, setting = _cache(campaign), _settings()
    rows = min(len(campaign.measured), setting['max_rows']) * (1 + len(campaign.variables) * setting['repeats'])
    if 'importance' not in cache:
        slot = st.empty()
        if not slot.button('Compute the importance ranking', key='b_importance_' + key):
            st.caption('One model call for {:,} recipes: {} on this computer.'.format(rows, duration(rows)))
            return
        slot.empty()
        try:
            with st.spinner('Fitting the model and shuffling each variable ({})...'.format(duration(rows))):
                _compute_importance(campaign, fitted)
        except Exception as error:      # shown to the user: a model that cannot run must not crash the page
            st.error('The model could not run: {}: {}'.format(type(error).__name__, error))
            return
    ranking = cache['importance']
    st.altair_chart(charts.ranking(ranking, 'Drop in R² when the variable is shuffled'), use_container_width=True)
    st.caption('R² of the model on the measured recipes it was fitted to: {:.2f}.'.format(ranking.attrs.get('r2', np.nan)))


def interactions_tab(campaign, fitted=None, key='setup'):
    st.caption(texts.INTERACTIONS.format(target=campaign.target))
    varying = [v for v in campaign.variables if v.varies]
    if len(campaign.measured) < 2 or len(varying) < 2:
        st.info('Interactions need at least two measured results and two variables that vary.')
        return
    cache, setting = _cache(campaign), _settings()
    g = min(len(varying), MAX_INTERACTION_VARIABLES)
    rows = setting['n_pairs'] * (1 + g + g * (g - 1) // 2)
    if 'interactions' not in cache:
        slot = st.empty()
        if not slot.button('Compute the interaction ranking', key='b_interactions_' + key):
            note = 'One model call for {:,} recipes: {} on this computer.'.format(rows, duration(rows))
            if len(varying) > MAX_INTERACTION_VARIABLES:
                note += ' Limited to the {} most important variables, so the importance ranking is computed ' \
                        'first.'.format(MAX_INTERACTION_VARIABLES)
            st.caption(note)
            return
        slot.empty()
        try:
            with st.spinner('Comparing pairs of variables ({})...'.format(duration(rows))):
                names = [v.name for v in varying]
                if len(names) > MAX_INTERACTION_VARIABLES:
                    ranked = _compute_importance(campaign, fitted)
                    names = [n for n in ranked.index if n in set(names)][:MAX_INTERACTION_VARIABLES]
                cache['interactions'] = campaign.interactions(fitted_model(campaign, fitted), variables=names,
                                                              n_pairs=setting['n_pairs'])
        except Exception as error:      # as above
            st.error('The model could not run: {}: {}'.format(type(error).__name__, error))
            return
    table = cache['interactions']
    pairs = pd.Series(table['Interaction'].to_numpy(), index=[
        '{} × {}'.format(a, b) for a, b in zip(table['Variable A'], table['Variable B'])])
    st.altair_chart(charts.ranking(pairs, 'Interaction (in units of {})'.format(campaign.target), top=15),
                    use_container_width=True)
    st.dataframe(table, hide_index=True, use_container_width=True)
