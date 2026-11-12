"""Day 0 data: measurements that exist before a campaign starts.

A table with one column per variable and one for the measured value. Every measurement
is a row; replicates of a recipe are rows with the same recipe and their own value.
"""
import csv
import io
import re

import pandas as pd

from .campaign import as_number
from .space import clean_category
from .variables import MAX_LEVELS, Categorical, Continuous, Discrete

MAX_DISCRETE = 10           # a numeric column with at most this many distinct values is read as discrete
_HEADER = re.compile(r'^(?P<name>.*\S)\s*\((?P<unit>[^()]*)\)$')


def read_table(source, name=None):
    """A CSV (comma, semicolon or tab separated) or Excel file as a table of text and numbers.
    ``source``: bytes, a path or a file object; ``name`` tells the format when it is not a path."""
    label = str(name or getattr(source, 'name', source)).lower()
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if label.endswith(('.xlsx', '.xlsm', '.xls')):
        frame = pd.read_excel(source)
    else:
        frame = pd.read_csv(source, sep=None, engine='python', encoding='utf-8-sig', skip_blank_lines=True)
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.loc[:, [not c.startswith('Unnamed:') for c in frame.columns]]
    return frame.dropna(how='all').reset_index(drop=True)


def split_header(header):
    """``'Mg-glutamate (mM)'`` -> ``('Mg-glutamate', 'mM')``; no brackets, no unit."""
    match = _HEADER.match(str(header).strip())
    return (match['name'], match['unit'].strip()) if match else (str(header).strip(), '')


def _column_numbers(values):
    """The column as floats, or None when a filled cell is not a number."""
    numbers = []
    for value in values:
        try:
            number = as_number(value)
        except (TypeError, ValueError):
            return None
        if number is not None:
            numbers.append(number)
    return numbers


def numeric_columns(frame):
    """Columns whose filled cells are all numbers: the candidates for the measured value."""
    return [c for c in frame.columns if _column_numbers(frame[c])]


def infer_variables(frame, target):
    """A first guess of the variables behind a day 0 table, for the user to adjust.

    Numeric columns with at most 10 distinct values become discrete (those values as
    steps), other numeric columns continuous (from their smallest to their largest value);
    text columns become categorical. Returns ``(variables, skipped)``; ``skipped`` lists
    columns that look like identifiers or notes (a different text in every row, or more
    than 100 different texts) with the reason.
    """
    variables, skipped = [], []
    for column in frame.columns:
        if column == target:
            continue
        name, unit = split_header(column)
        numbers = _column_numbers(frame[column])
        if numbers is not None:
            distinct = sorted(set(numbers))
            if not distinct:
                skipped.append((column, 'it is empty'))
            elif len(distinct) <= MAX_DISCRETE:
                variables.append(Discrete(name, distinct, unit=unit))
            else:
                variables.append(Continuous(name, distinct[0], distinct[-1], unit=unit))
            continue
        texts = [t for t in (clean_category(v) for v in frame[column]) if t]
        categories = list(dict.fromkeys(texts))
        if len(categories) > MAX_LEVELS:
            skipped.append((column, 'it has more than {} different texts'.format(MAX_LEVELS)))
        elif len(categories) > 1 and len(categories) == len(texts):
            skipped.append((column, 'every row has a different text'))
        else:
            variables.append(Categorical(name, categories))
    return variables, skipped


def _example(variable, k):
    if variable.kind == 'continuous':
        return (variable.low, variable.high)[k % 2]
    values = variable.levels if variable.kind == 'discrete' else variable.categories
    return values[k % len(values)]


def template(variables=(), target='Result'):
    """A CSV showing what day 0 data looks like: a header and three example rows, the
    first two a replicate pair (the same recipe measured twice). With no variables defined
    yet, two made-up ones show the layout."""
    variables = list(variables) or [Continuous('Mg-glutamate', 0, 10, unit='mM'),
                                    Categorical('Promoter', ['T7', 'T3'])]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow([v.label for v in variables] + [target or 'Result'])
    for k, result in ((0, 1.0), (0, 1.1), (1, 2.0)):
        writer.writerow(['{:g}'.format(x) if not isinstance(x, str) else x
                         for x in (_example(v, k) for v in variables)] + [result])
    return buffer.getvalue()
