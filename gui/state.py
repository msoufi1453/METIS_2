"""What the app remembers between reruns: the screen, the form's fields and the round in progress.

Streamlit forgets a widget's value when the widget is not drawn, and the set-up form is
not drawn on the round screen. ``init`` therefore re-assigns every form key (prefixes
``f_`` and ``row``) on each run, which keeps them. Functions that change form keys run in
button callbacks, before the next run draws the widgets.
"""
import streamlit as st

from metis.campaign import DEFAULT_MODEL, MODELS, VariableError, day0, default_pool_size, tutorial

KINDS = {'Continuous': 'continuous', 'Discrete': 'discrete', 'Categorical': 'categorical'}
KIND_LABELS = {kind: label for label, kind in KINDS.items()}
FORM_PREFIXES = ('f_', 'row')
DEFAULT_STEPS = 3


def form_defaults():
    return {'f_model': MODELS[DEFAULT_MODEL].label, 'f_name': 'My campaign', 'f_target': 'Result',
            'f_samples': 10, 'f_replicates': 1, 'f_lh': True, 'f_droplet': 4.0, 'f_volume': 10.0,
            'f_diluent': 'Water', 'f_use_day0': True, 'f_exploit': 0.5, 'f_penalty_on': False, 'f_penalty': 0.0,
            'f_fantasy': False, 'f_estimators': 8, 'f_seed': 0, 'f_climb_moves': 20, 'f_initials': ''}


def init():
    ss = st.session_state
    for key, value in {'page': 'home', 'mode': None, 'rows': [], 'next_row': 1, 'base': None, 'day0': None,
                       'round': None, 'analysis': {}, 'notice': None, 'upload_error': None,
                       'day0_error': None, 'uploads': 0, 'scroll_top': False}.items():
        ss.setdefault(key, value)
    for key, value in form_defaults().items():
        ss.setdefault(key, value)
    for key in list(ss.keys()):
        if key.startswith(FORM_PREFIXES):
            ss[key] = ss[key]


def ensure_pool_size(slot):
    """Give the number of candidates its default. It depends on whether there is a GPU, and finding
    out imports PyTorch (seconds), so it waits until the set-up screen is first shown. The wait
    is announced in ``slot``, a place that exists on every run (see gui/round_page.py)."""
    ss = st.session_state
    if 'f_pool' not in ss:
        with slot, st.spinner('Checking for a GPU...'):
            ss.f_pool = default_pool_size(model_key())


def go(page):
    st.session_state.page = page
    st.session_state.scroll_top = True          # a new screen starts at its top (gui/app.py)


def model_key():
    """The registry key of the model chosen in the form."""
    label = st.session_state.f_model
    return next((key for key, spec in MODELS.items() if spec.label == label), DEFAULT_MODEL)


# --------------------------------------------------------------------------- the form's rows
def reset_form():
    ss = st.session_state
    for key in [k for k in ss.keys() if k.startswith(FORM_PREFIXES)]:
        del ss[key]
    ss.rows, ss.next_row = [], 1
    for key, value in form_defaults().items():
        ss[key] = value


def add_row(variable=None):
    """A new variable row, empty or filled from a ``metis.campaign`` variable."""
    ss = st.session_state
    r = ss.next_row
    ss.next_row = r + 1
    ss.rows = list(ss.rows) + [r]
    ss['row{}_kind'.format(r)] = 'Continuous'
    ss['row{}_n'.format(r)] = DEFAULT_STEPS
    if variable is None:
        return r
    ss['row{}_kind'.format(r)] = KIND_LABELS[variable.kind]
    ss['row{}_name'.format(r)] = variable.name
    if variable.numeric:
        ss['row{}_unit'.format(r)] = variable.unit
        ss['row{}_stock'.format(r)] = variable.stock
    if variable.kind == 'continuous':
        ss['row{}_low'.format(r)], ss['row{}_high'.format(r)] = variable.low, variable.high
    elif variable.kind == 'discrete':
        ss['row{}_n'.format(r)] = len(variable.levels)
        for k, level in enumerate(variable.levels):
            ss['row{}_x{}'.format(r, k)] = float(level)
    else:
        ss['row{}_n'.format(r)] = len(variable.categories)
        ss['row{}_volume'.format(r)] = float(variable.volume_ul)
        for k, category in enumerate(variable.categories):
            ss['row{}_c{}'.format(r, k)] = category
    return r


def remove_row(r):
    ss = st.session_state
    ss.rows = [x for x in ss.rows if x != r]
    for key in [k for k in ss.keys() if k.startswith('row{}_'.format(r))]:
        del ss[key]


def set_rows(variables):
    for r in list(st.session_state.rows):
        remove_row(r)
    for variable in variables:
        add_row(variable)


def rows_are_blank():
    """True when no row has a name or a value yet."""
    ss = st.session_state
    for r in ss.rows:
        prefix = 'row{}_'.format(r)
        for key in ss.keys():
            if key.startswith(prefix) and key[len(prefix):] not in ('kind', 'n', 'volume') and ss[key] not in (None, ''):
                return False
    return True


# --------------------------------------------------------------------------- starting points
def _clear_work():
    ss = st.session_state
    ss.base, ss.day0, ss.round, ss.analysis = None, None, None, {}
    ss.notice = ss.upload_error = ss.day0_error = None


def start_new():
    reset_form()
    _clear_work()
    add_row()
    st.session_state.mode = 'new'
    go('setup')


def start_upload():
    st.session_state.upload_error = None
    go('upload')


def start_tutorial():
    ss = st.session_state
    reset_form()
    _clear_work()
    frame = tutorial.day0()
    ss.day0 = {'name': 'Borkowski et al. 2020, first plate', 'frame': frame, 'raw': frame}
    ss.f_day0_target = tutorial.TARGET
    ss.f_name = tutorial.NAME
    ss.f_volume = tutorial.SAMPLE_VOLUME_UL
    set_rows(tutorial.variables())
    ss.mode = 'tutorial'
    go('setup')


def open_campaign(campaign, mode='continue'):
    """Show the set-up form for the next round of a campaign that already has data."""
    ss = st.session_state
    reset_form()
    _clear_work()
    ss.base = campaign
    ss.mode = mode
    ss.f_name, ss.f_target, ss.f_seed = campaign.name, campaign.target, campaign.seed
    ss.f_model = MODELS[campaign.model].label
    if 'n_estimators' in campaign.model_options:
        ss.f_estimators = int(campaign.model_options['n_estimators'])
    ss.f_lh = campaign.liquid is not None
    if campaign.liquid is not None:
        ss.f_droplet, ss.f_volume = campaign.liquid.droplet_nl, campaign.liquid.sample_volume_ul
        ss.f_diluent = campaign.liquid.diluent
    if campaign.history:
        ss.f_samples = int(campaign.history[-1].get('n_samples') or 10)
        ss.f_replicates = int(campaign.history[-1].get('replicates') or 1)
    ss.f_exploit = campaign.default_exploitation()
    set_rows(campaign.variables)
    go('setup')


# --------------------------------------------------------------------------- day 0 data
def load_day0(name, data):
    """Read an uploaded day 0 table. Columns are known by their name without the unit
    (``Mg (mM)`` -> ``Mg``); the unit goes to the variable row."""
    ss = st.session_state
    ss.day0_error = None
    try:
        raw = day0.read_table(data, name)
        names = [day0.split_header(c)[0] for c in raw.columns]
        if len(set(names)) < len(names):
            raise VariableError('Two columns have the same name.')
        frame = raw.copy()
        frame.columns = names
        numeric = day0.numeric_columns(frame)
        if not len(frame) or not numeric:
            raise VariableError('The file needs a header row, at least one row of data and a column of numbers for '
                                'the measured value.')
    except Exception as error:      # anything a broken upload can raise is shown to the user, not thrown
        ss.day0_error = 'Could not read "{}": {}'.format(name, error)
        return
    ss.day0 = {'name': name, 'frame': frame, 'raw': raw}
    ss.f_day0_target = ss.f_target if ss.f_target in numeric else numeric[-1]
    if rows_are_blank():
        rows_from_day0()


def rows_from_day0():
    """Replace the variable rows by a first guess from the day 0 columns."""
    ss = st.session_state
    raw, frame = ss.day0['raw'], ss.day0['frame']
    target = raw.columns[list(frame.columns).index(ss.f_day0_target)]
    variables, skipped = day0.infer_variables(raw, target)
    set_rows(variables)
    ss.day0['skipped'] = skipped


def remove_day0():
    ss = st.session_state
    ss.day0, ss.day0_error = None, None
    ss.uploads += 1             # a new key empties the file uploader
