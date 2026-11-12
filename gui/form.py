"""The set-up form: the variable rows and what they mean as ``metis.campaign`` objects."""
from dataclasses import dataclass, field

import streamlit as st

from metis.campaign import MAX_LEVELS, Categorical, Continuous, Discrete, LiquidHandling, VariableError

from . import state

PER_LINE = 6            # value boxes of a discrete or categorical variable per line


@dataclass
class Design:
    """What the form currently describes. ``problems`` lists what keeps it from being usable."""
    variables: list = field(default_factory=list)
    liquid: LiquidHandling = None
    fill: dict = field(default_factory=dict)        # value in the data so far, for variables that are new
    problems: list = field(default_factory=list)

    @property
    def ready(self):
        return bool(self.variables) and not self.problems


def _key(r, part):
    return 'row{}_{}'.format(r, part)


def _number(label, key, column, **options):
    return column.number_input(label, value=None, key=key, format='%g', **options)


def _value_boxes(r, n, numeric):
    """``n`` boxes for the steps of a discrete variable or the names of a categorical one."""
    for start in range(0, n, PER_LINE):
        columns = st.columns(PER_LINE)
        for k in range(start, min(start + PER_LINE, n)):
            if numeric:
                _number('Step {}'.format(k + 1), _key(r, 'x{}'.format(k)), columns[k - start])
            else:
                columns[k - start].text_input('Category {}'.format(k + 1), key=_key(r, 'c{}'.format(k)))


def _past_value(r, kind, name, column):
    help_text = ('The data so far has rows without a value for "{}". Enter the value this variable had in those '
                 'experiments (for example 0 if it was absent).'.format(name))
    if kind == 'categorical':
        column.text_input('Value in the data so far', key=_key(r, 'past'), help=help_text)
    else:
        _number('Value in the data so far', _key(r, 'past'), column, help=help_text)


def variable_row(r, position, liquid_on, known_columns):
    """Draw one variable row. ``known_columns``: columns of the data so far (None without data)."""
    ss = st.session_state
    with st.container(border=True):
        head = st.columns([6, 1])
        label = head[0].radio('Type of variable {}'.format(position), list(state.KINDS), key=_key(r, 'kind'),
                              horizontal=True, label_visibility='collapsed')
        head[1].button('Remove', key='b_remove{}'.format(r), on_click=state.remove_row, args=(r,),
                       use_container_width=True)
        kind = state.KINDS[label]
        name = (ss.get(_key(r, 'name')) or '').strip()
        is_new = known_columns is not None and bool(name) and name not in known_columns
        stock_help = 'Concentration of the stock you pipette from, in the same unit as the values.'
        if kind == 'continuous':
            widths = [2.6, 1.0, 1.5, 1.5] + ([2.2] if liquid_on else []) + ([2] if is_new else [])
            columns = st.columns(widths)
            columns[0].text_input('Name', key=_key(r, 'name'))
            columns[1].text_input('Unit', key=_key(r, 'unit'), placeholder='mM')
            _number('Minimum', _key(r, 'low'), columns[2])
            _number('Maximum', _key(r, 'high'), columns[3])
            if liquid_on:
                _number('Stock concentration', _key(r, 'stock'), columns[4], min_value=0.0, help=stock_help)
            if is_new:
                _past_value(r, kind, name, columns[-1])
            return
        count_label = 'Steps' if kind == 'discrete' else 'Categories'
        widths = ([1.1, 2.8] + ([1.0] if kind == 'discrete' else []) + ([2.2] if liquid_on else [])
                  + ([2] if is_new else []))
        columns = st.columns(widths)
        n = int(columns[0].number_input(count_label, min_value=1, max_value=MAX_LEVELS, step=1, key=_key(r, 'n')))
        columns[1].text_input('Name', key=_key(r, 'name'))
        at = 2
        if kind == 'discrete':
            columns[at].text_input('Unit', key=_key(r, 'unit'), placeholder='mM')
            at += 1
            if liquid_on:
                _number('Stock concentration', _key(r, 'stock'), columns[at], min_value=0.0, help=stock_help)
        elif liquid_on:
            columns[at].number_input(
                'Pipetted volume (µL)', min_value=0.0, key=_key(r, 'volume'), format='%g',
                help='If each category is its own liquid (for example a DNA template), the volume of it that '
                     'goes into every sample. Leave 0 if the category is not pipetted (a strain, a temperature).')
        if is_new:
            _past_value(r, kind, name, columns[-1])
        _value_boxes(r, n, kind == 'discrete')


def _read_row(r, position, liquid_on, known_columns, design):
    ss = st.session_state
    kind = state.KINDS[ss.get(_key(r, 'kind'), 'Continuous')]
    name = (ss.get(_key(r, 'name')) or '').strip()
    label = name or 'Variable {}'.format(position)
    unit = (ss.get(_key(r, 'unit')) or '').strip()
    stock = ss.get(_key(r, 'stock')) if liquid_on else None
    n = int(ss.get(_key(r, 'n')) or state.DEFAULT_STEPS)
    problems = []
    if kind == 'continuous':
        low, high = ss.get(_key(r, 'low')), ss.get(_key(r, 'high'))
        if not name and low is None and high is None and stock is None:
            return                              # an untouched row is ignored
        if low is None or high is None:
            problems.append('{}: enter the minimum and the maximum.'.format(label))
    elif kind == 'discrete':
        values = [ss.get(_key(r, 'x{}'.format(k))) for k in range(n)]
        if not name and all(v is None for v in values) and stock is None:
            return
        if any(v is None for v in values):
            problems.append('{}: fill in all {} steps.'.format(label, n))
    else:
        values = [(ss.get(_key(r, 'c{}'.format(k))) or '').strip() for k in range(n)]
        if not name and not any(values):
            return
        if not all(values):
            problems.append('{}: fill in all {} categories.'.format(label, n))
    if not name:
        problems.append('Variable {}: enter a name.'.format(position))
    if liquid_on and kind != 'categorical' and not stock:
        problems.append('{}: enter the stock concentration.'.format(label))
    if not problems:
        try:
            if kind == 'continuous':
                variable = Continuous(name, low, high, unit=unit, stock=stock)
            elif kind == 'discrete':
                variable = Discrete(name, values, unit=unit, stock=stock)
            else:
                variable = Categorical(name, values, volume_ul=ss.get(_key(r, 'volume')) or 0.0 if liquid_on else 0.0)
        except VariableError as error:
            problems.append(str(error))
    if not problems and known_columns is not None and name not in known_columns:
        past = ss.get(_key(r, 'past'))
        if past is None or str(past).strip() == '':
            problems.append('{}: enter the value it had in the data so far.'.format(label))
        else:
            design.fill[name] = past
    design.problems += problems
    if not problems:
        design.variables.append(variable)


def read(known_columns=None):
    """The form as a ``Design``. ``known_columns``: the columns of the data so far (None without data)."""
    ss = st.session_state
    design = Design()
    if ss.f_lh:
        try:
            design.liquid = LiquidHandling(sample_volume_ul=ss.f_volume, droplet_nl=ss.f_droplet,
                                           diluent=ss.f_diluent)
        except VariableError as error:
            design.problems.append(str(error))
    for position, r in enumerate(ss.rows, start=1):
        _read_row(r, position, bool(ss.f_lh), known_columns, design)
    names = [v.name.lower() for v in design.variables]
    for name in sorted({n for n in names if names.count(n) > 1}):
        design.problems.append('Two variables are called "{}".'.format(name))
    return design
