"""The set-up screen: the same form for a new campaign, the next round of a running one and the tutorial."""
import pandas as pd
import streamlit as st

from metis.campaign import MODELS, Campaign, CampaignError, VariableError, day0, has_gpu, model_spec

from . import form, insight, state, texts

MAX_SAMPLES, MAX_REPLICATES = 960, 12


def known_columns():
    """Columns of the data so far that a variable can take its past values from (None without data)."""
    ss = st.session_state
    if ss.base is not None:
        data = ss.base.data
        return {name for name in data.columns if data[name].notna().all()}
    if ss.day0 is not None:
        return set(ss.day0['frame'].columns) - {ss.get('f_day0_target')}
    return None


def model_options():
    ss = st.session_state
    return {'n_estimators': int(ss.f_estimators)} if 'n_estimators' in model_spec(state.model_key()).options else {}


def build_campaign(design):
    """``(campaign, message)``: the campaign the form describes with the data so far, or why it cannot be built."""
    ss = st.session_state
    if not design.ready:
        return None, None
    try:
        if ss.base is not None:
            campaign = ss.base.redesign(design.variables, design.liquid, fill=design.fill)
            campaign.model_options, campaign.seed = model_options(), int(ss.f_seed)
            return campaign, None
        target = ss.f_day0_target if ss.day0 is not None else ss.f_target
        campaign = Campaign(design.variables, target=target, liquid_handling=design.liquid, model=state.model_key(),
                            seed=int(ss.f_seed), name=ss.f_name, model_options=model_options())
        if ss.day0 is not None:
            campaign.add_data(ss.day0['frame'], fill=design.fill)
        return campaign, None
    except (VariableError, CampaignError) as error:
        return None, str(error)


def _day0_section(design):
    ss = st.session_state
    st.subheader('Day 0 data (optional)')
    st.caption('Results you already have. A CSV or Excel table: one column per variable, one for the measured '
               'value, and one row per measurement (replicates of a recipe are rows of their own).')
    left, right = st.columns([3, 1])
    upload = left.file_uploader('Upload day 0 data', type=['csv', 'xlsx'], key='u_day0_{}'.format(ss.uploads))
    if upload is not None and ss.get('day0_file') != upload.file_id:
        ss.day0_file = upload.file_id
        ss.inbox_day0 = (upload.name, upload.getvalue())
        st.rerun()
    target = ss.get('f_day0_target') if ss.day0 is not None else ss.f_target
    right.download_button('Download a template', day0.template(design.variables, target).encode('utf-8-sig'),
                          file_name='day0_template.csv', mime='text/csv', key='d_template',
                          help='A CSV with the right header for the variables below and three example rows '
                               '(the first two are a replicate pair). Replace the example rows with your data.',
                          use_container_width=True)
    if ss.day0_error:
        st.error(ss.day0_error)
    if ss.day0 is None:
        st.text_input('Name of the measured value (the result to maximise)', key='f_target',
                      help='For example sfGFP or DNA replication. It names the result column of the workbook.')
        return
    frame = ss.day0['frame']
    st.success('Day 0 data: {} ({} rows, {} columns).'.format(ss.day0['name'], len(frame), frame.shape[1]))
    columns = st.columns([2, 2, 1])
    columns[0].selectbox('Column with the measured value (the result to maximise)', day0.numeric_columns(frame),
                         key='f_day0_target')
    columns[1].button('Make the variables from the day 0 columns', key='b_from_day0', on_click=state.rows_from_day0,
                      help='Replaces the variable rows below: numeric columns with up to 10 different values '
                           'become discrete, other numeric columns continuous, text columns categorical.',
                      use_container_width=True)
    columns[2].button('Remove day 0 data', key='b_remove_day0', on_click=state.remove_day0, use_container_width=True)
    for column, reason in ss.day0.get('skipped') or []:
        st.caption('Column "{}" was not made a variable: {}.'.format(column, reason))
    with st.expander('Preview'):
        st.dataframe(frame.head(20), hide_index=True, use_container_width=True)
    st.toggle('Let the model choose round 1 from the day 0 data', key='f_use_day0',
              help='On: round 1 is model-guided. Off: round 1 is a Latin hypercube; the day 0 data is still '
                   'kept in the workbook as round 0 and used from round 2.')


def _data_summary(base):
    best = base.best()
    columns = st.columns(4)
    columns[0].metric('Rounds with results', len([r for r in base.rounds if r > 0]) or '–')
    columns[1].metric('Recipes measured', base.data['Sample ID'].nunique())
    columns[2].metric('Measurements', len(base.measured))
    columns[3].metric('Best {} so far'.format(base.target), '–' if best is None else '{:.4g}'.format(best[base.target]))


def _liquid_section():
    ss = st.session_state
    st.subheader('Liquid handling')
    st.toggle('I am using liquid handling (a droplet dispenser)', key='f_lh',
              help='On: each numeric variable needs a stock concentration, recipes are kept to whole droplets '
                   'and inside the sample volume, and a pipetting table is made for every round.')
    if not ss.f_lh:
        return
    columns = st.columns(3)
    columns[0].number_input('Smallest droplet (nL)', min_value=0.1, format='%g', key='f_droplet')
    columns[1].number_input('Sample volume (µL)', min_value=0.1, format='%g', key='f_volume',
                            help='The final volume of one sample. Whatever the recipe leaves free is topped up '
                                 'with the diluent.')
    columns[2].text_input('Diluent', key='f_diluent', help='The liquid that tops every sample up to its volume.')
    st.caption('Enter each stock concentration in the same unit as that variable\'s values (units may differ '
               'between variables, but keep them the same from round to round). To include a fixed component, '
               'such as an extract, add a discrete variable with one step.')


def _variables_section(known):
    ss = st.session_state
    st.subheader('Variables')
    limit = model_spec(state.model_key()).max_variables
    st.caption('What a recipe can change. Continuous: any value between a minimum and a maximum. Discrete: a list '
               'of steps. Categorical: a list of names.')
    for position, r in enumerate(ss.rows, start=1):
        form.variable_row(r, position, bool(ss.f_lh), known)
    st.button('Add a variable', key='b_add', on_click=state.add_row, disabled=len(ss.rows) >= limit,
              help='Up to {:,} variables with this model.'.format(limit))


def _issues_section(design, campaign, message):
    """Problems and warnings; returns True when a round can be generated."""
    for problem in design.problems:
        st.warning(problem)
    if message:
        st.error(message)
    if campaign is None:
        return False
    issues = campaign.check()
    for issue in issues:
        (st.error if issue.blocking else st.warning)(issue.message)
    if campaign.liquid is not None and not any(i.blocking for i in issues):
        with st.expander('Pipetting check: what one droplet adds, and the volume each variable takes'):
            table = campaign.liquid.summary(campaign.variables)
            st.dataframe(table, hide_index=True, use_container_width=True,
                         column_config={'One droplet adds': st.column_config.NumberColumn(format='%.4g'),
                                        'Stock': st.column_config.NumberColumn(format='%.4g'),
                                        'Volume at minimum (µL)': st.column_config.NumberColumn(format='%.3f'),
                                        'Volume at maximum (µL)': st.column_config.NumberColumn(format='%.3f')})
            share = campaign.space().feasible_fraction()
            if share < 1:
                st.caption('{:.0%} of random recipes fit in the sample volume.'.format(share))
    return not any(i.blocking for i in issues)


def _strategy_section(campaign, guided):
    ss = st.session_state
    st.subheader('Exploration and exploitation')
    low = 0.5 if ss.base is not None else 0.0        # a running campaign: between 0.5 and 1
    if ss.f_exploit < low:
        ss.f_exploit = low
    if not guided:
        st.info(texts.LATIN_HYPERCUBE)
    st.slider('Exploitation ratio (0 = explore, 1 = exploit)', min_value=low, max_value=1.0, step=0.05, key='f_exploit',
              disabled=not guided, help=texts.EXPLOIT_HELP)
    default = campaign.default_exploitation() if campaign is not None else 0.5
    st.caption(texts.EXPLOIT_DEFAULT.format(default))
    if not guided:
        return
    left, right = st.columns(2)
    with left:
        st.toggle('Distance penalty against clustered recipes', key='f_penalty_on', help=texts.PENALTY_HELP)
        st.slider('Penalty (0 = low, 1 = high)', min_value=0.0, max_value=1.0, step=0.1, key='f_penalty',
                  disabled=not ss.f_penalty_on)
    with right:
        st.toggle('Fantasy batch', key='f_fantasy', help=texts.FANTASY_HELP)
        if ss.f_fantasy and not has_gpu():
            st.caption('Without a GPU this adds {} per recipe.'.format(insight.duration(int(ss.f_pool))))


def _advanced_section(guided):
    ss = st.session_state
    with st.expander('Advanced settings'):
        columns = st.columns(4)
        columns[0].number_input('Candidate recipes to score', min_value=10, max_value=200000, step=500, key='f_pool',
                                help='Random recipes the model scores each round. More finds better recipes and '
                                     'takes longer. Default: 10,000 with a GPU, 1,000 without.')
        columns[1].number_input('TabPFN estimators', min_value=1, max_value=32, step=1, key='f_estimators',
                                help='8 is what the benchmark used. Fewer is faster and slightly less accurate.')
        columns[2].number_input('Random seed', min_value=0, max_value=10 ** 9, step=1, key='f_seed',
                                help='The same seed, data and settings give the same round.')
        columns[3].number_input('Hill-climbing: most moves per climb', min_value=1, max_value=200, step=1,
                                key='f_climb_moves')
        if guided:
            st.caption('{} A round with {:,} candidates takes {} on this computer; the first one also loads the '
                       'model.'.format('GPU found.' if has_gpu() else 'No GPU found.', int(ss.f_pool),
                                       insight.duration(int(ss.f_pool))))


def generate(campaign, guided):
    """Propose the round and open the round screen. Returns an error message, or None."""
    ss = st.session_state
    options = {'n_samples': int(ss.f_samples), 'guided': guided}
    if guided:
        options.update(exploitation=float(ss.f_exploit), fantasy=bool(ss.f_fantasy), pool_size=int(ss.f_pool),
                       penalty=float(ss.f_penalty) if ss.f_penalty_on else None)
    try:
        proposal = campaign.propose(**options)
    except (CampaignError, VariableError) as error:
        return str(error)
    except Exception as error:      # the model's own failures (no licence, no memory) are shown, not thrown
        return 'The model could not run: {}: {}'.format(type(error).__name__, error)
    ss.round = {'campaign': campaign, 'proposal': proposal, 'climbed': None, 'replicates': int(ss.f_replicates),
                'tables': {'plain': proposal.table}, 'edits': {'plain': 0, 'climbed': 0}, 'version': 0,
                'staged': {}}
    ss.r_hill = False
    ss.notice = None
    state.go('round')
    return None


def _form(design, campaign, message):
    """Draw the form. Returns True when the user has just generated a round (gui/app.py then shows it)."""
    ss = st.session_state
    base = ss.base
    top = st.columns(2)
    labels = [spec.label for spec in MODELS.values()]
    top[0].selectbox('Model', labels, key='f_model', disabled=base is not None,
                     help=model_spec(state.model_key()).note)
    top[1].text_input('Campaign name', key='f_name', disabled=base is not None)
    if base is not None:
        _data_summary(base)
    else:
        _day0_section(design)
    st.subheader('Size of the round')
    size = st.columns(2)
    size[0].number_input('Recipes per round', min_value=1, max_value=MAX_SAMPLES, step=1, key='f_samples')
    size[1].number_input('Replicates of each recipe', min_value=1, max_value=MAX_REPLICATES, step=1,
                         key='f_replicates', help='Each replicate is its own well and its own row of results.')
    _liquid_section()
    _variables_section(known_columns())
    ready = _issues_section(design, campaign, message)
    guided = (campaign is not None and len(campaign.measured) >= 2 and (base is not None or bool(ss.f_use_day0)))
    _strategy_section(campaign, guided)
    _advanced_section(guided)
    number = campaign.next_round if campaign is not None else (base.next_round if base is not None else 1)
    if st.button('Generate round {} data'.format(number), type='primary', key='b_generate',
                 disabled=not (ready and campaign is not None)):
        wait = ('Fitting the model and scoring {:,} candidates ({})...'.format(int(ss.f_pool),
                                                                                insight.duration(int(ss.f_pool)))
                if guided else 'Drawing the Latin hypercube...')
        with st.spinner(wait):
            error = generate(campaign, guided)
        if not error:
            return True
        st.error(error)
    if not design.variables and not design.problems:
        st.caption('Complete at least one variable to generate the round.')
    return False


def render():
    ss = st.session_state
    base = ss.base
    state.ensure_pool_size(st.empty())
    design = form.read(known_columns())
    campaign, message = build_campaign(design)
    head = st.columns([6, 1])
    if ss.mode == 'tutorial':
        head[0].title('Tutorial: {}'.format(ss.f_name))
    elif base is not None:
        head[0].title('{}: set up round {}'.format(base.name, base.next_round))
    else:
        head[0].title('New campaign')
    head[1].button('Home', key='b_home_setup', on_click=state.go, args=('home',), use_container_width=True)
    if ss.mode == 'tutorial':
        with st.expander('Tutorial: what to do on this screen', expanded=True):
            if base is None:
                st.markdown(texts.TUTORIAL_SETUP)
            else:
                st.markdown(texts.TUTORIAL_NEXT.format(last=max(base.rounds), next=base.next_round,
                                                       exploit=base.default_exploitation()))
    has_data = base is not None or ss.day0 is not None
    if not has_data:
        return _form(design, campaign, message)
    tabs = st.tabs(['Round settings', 'PCA map', 'Progress', 'Importance', 'Interactions'])
    with tabs[0]:
        if _form(design, campaign, message):
            return True
    if campaign is None:
        for tab in tabs[1:]:
            tab.info('Complete the variables on the first tab to see this.')
        return False
    with tabs[1]:
        insight.pca_tab(campaign, round_number=campaign.next_round, key='setup')
    with tabs[2]:
        insight.progress_tab(campaign)
    with tabs[3]:
        insight.importance_tab(campaign, key='setup')
    with tabs[4]:
        insight.interactions_tab(campaign, key='setup')
    return False


def render_upload():
    ss = st.session_state
    head = st.columns([6, 1])
    head[0].title('Continue a campaign')
    head[1].button('Home', key='b_home_upload', on_click=state.go, args=('home',), use_container_width=True)
    st.write('Upload the workbook you downloaded for the last round, with the result of every row filled in.')
    upload = st.file_uploader('Workbook (.xlsx)', type=['xlsx'], key='u_workbook')
    if upload is not None and ss.get('workbook_file') != upload.file_id:
        ss.workbook_file = upload.file_id
        ss.inbox_workbook = upload.getvalue()
        st.rerun()
    if ss.upload_error:
        st.error(ss.upload_error)
    st.caption('The workbook holds the whole campaign: variables, settings and every round so far. Nothing else '
               'is needed to carry on.')


def summary_table(campaign):
    """The variables of a campaign as a small table (used on the round screen)."""
    return pd.DataFrame([{'Variable': v.label, 'Type': v.kind,
                          'Values': ('{:g} to {:g}'.format(v.low, v.high) if v.kind == 'continuous' else
                                     ', '.join('{:g}'.format(x) for x in v.levels) if v.kind == 'discrete' else
                                     ', '.join(v.categories))} for v in campaign.variables])
