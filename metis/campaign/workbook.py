"""A campaign as one Excel workbook.

* ``Experiments``: every recipe so far, one row per measurement. The new round's result
  cells are empty and highlighted: fill them in and load the workbook to continue.
* ``Instructions``: how to fill it in.
* ``Metadata``: the settings of every round in readable form.
* ``State``: the same, machine-readable (JSON split over cells). Do not edit it.

Text is always written as text, so a name that starts with "=" never becomes a formula.
"""
import io
import json
import math

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

EXPERIMENTS, INSTRUCTIONS, METADATA, STATE = 'Experiments', 'Instructions', 'Metadata', 'State'
STATE_TITLE = 'METIS campaign state. Do not edit this sheet.'
CHUNK = 30000                       # a cell holds at most 32,767 characters
PREFIX = 'json:'                    # keeps a chunk from starting with "=" or looking like a number
BOLD = Font(bold=True)
TO_FILL = PatternFill('solid', start_color='FFF2CC')
HEADER = PatternFill('solid', start_color='E7E6E6')


class WorkbookError(ValueError):
    """The file is not a workbook this package wrote, or it was changed in a way it cannot read."""


def _plain(value):
    """A cell value openpyxl can store; missing values become empty cells."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, 'item'):              # numpy scalars
        value = value.item()
        if isinstance(value, float) and math.isnan(value):
            return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _put(sheet, row, column, value):
    cell = sheet.cell(row=row, column=column, value=_plain(value))
    if isinstance(cell.value, str):
        cell.data_type = 's'                # text, even when it starts with "="
    return cell


def _widths(sheet, widths):
    for column, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column)].width = width


def write(table, headers, fill_column, state, instructions, metadata):
    """Workbook bytes.

    ``table``: the data, one row per measurement. ``headers``: the header for each of its
    columns. ``fill_column``: the column whose empty cells the user fills in.
    ``metadata``: ``[(key, value)]``; a row whose value is None is a section title.
    """
    book = Workbook()
    sheet = book.active
    sheet.title = EXPERIMENTS
    for c, name in enumerate(table.columns, start=1):
        cell = _put(sheet, 1, c, headers[name])
        cell.font, cell.fill = BOLD, HEADER
        cell.alignment = Alignment(wrap_text=True, vertical='top')
    fill_at = list(table.columns).index(fill_column) + 1
    for r, row in enumerate(table.itertuples(index=False), start=2):
        for c, value in enumerate(row, start=1):
            cell = _put(sheet, r, c, value)
            if c == fill_at and cell.value is None:
                cell.fill = TO_FILL
    sheet.freeze_panes = 'A2'
    _widths(sheet, [max(10, min(28, len(str(headers[name])) + 2)) for name in table.columns])

    sheet = book.create_sheet(INSTRUCTIONS)
    for r, line in enumerate(instructions, start=1):
        cell = _put(sheet, r, 1, line)
        cell.alignment = Alignment(wrap_text=True, vertical='top')
        if r == 1:
            cell.font = BOLD
    _widths(sheet, [110])

    sheet = book.create_sheet(METADATA)
    for r, (key, value) in enumerate(metadata, start=1):
        cell = _put(sheet, r, 1, key)
        if value is None:
            cell.font = BOLD
        else:
            _put(sheet, r, 2, value).alignment = Alignment(wrap_text=True, vertical='top')
    _widths(sheet, [34, 90])

    sheet = book.create_sheet(STATE)
    _put(sheet, 1, 1, STATE_TITLE).font = BOLD
    text = json.dumps(state, ensure_ascii=True, allow_nan=False)
    for r, start in enumerate(range(0, len(text), CHUNK), start=2):
        _put(sheet, r, 1, PREFIX + text[start:start + CHUNK])
    _widths(sheet, [60])

    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def read(source):
    """``(table, state)`` from workbook bytes, a path or a file object. The table's
    columns are the sheet's headers; empty rows are skipped; ``table.attrs['rows']`` gives
    each row's number in the sheet, for messages."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        book = load_workbook(source, data_only=True, read_only=True)
    except Exception as error:      # zipfile.BadZipFile, KeyError, ... : whatever openpyxl raises for a bad file
        raise WorkbookError('This file cannot be opened as an Excel workbook (.xlsx): {}.'.format(error)) from None
    try:
        missing = [name for name in (EXPERIMENTS, STATE) if name not in book.sheetnames]
        if missing:
            raise WorkbookError('This workbook has no "{}" sheet. Upload the workbook downloaded from this app, '
                                'with its sheets kept.'.format('" or "'.join(missing)))
        chunks = [row[0] for row in book[STATE].iter_rows(min_row=2, max_col=1, values_only=True)
                  if row and isinstance(row[0], str)]
        try:
            state = json.loads(''.join(c[len(PREFIX):] if c.startswith(PREFIX) else c for c in chunks))
        except ValueError:
            raise WorkbookError('The "State" sheet was changed and can no longer be read. Use the workbook as it '
                                'was downloaded and only fill in the results.') from None
        rows = list(book[EXPERIMENTS].iter_rows(values_only=True))
    finally:
        book.close()
    if not rows or not any(v is not None for v in rows[0]):
        raise WorkbookError('The "Experiments" sheet is empty.')
    header = [str(v).strip() if v is not None else None for v in rows[0]]
    while header and header[-1] is None:
        header.pop()
    if any(h is None for h in header):
        raise WorkbookError('The "Experiments" sheet has a column without a header.')
    duplicates = sorted({h for h in header if header.count(h) > 1})
    if duplicates:
        raise WorkbookError('The "Experiments" sheet has two columns called "{}".'.format(duplicates[0]))
    records, numbers = [], []
    for number, row in enumerate(rows[1:], start=2):
        values = list(row[:len(header)]) + [None] * (len(header) - len(row))
        if any(v is not None and not (isinstance(v, str) and not v.strip()) for v in values):
            records.append(values)
            numbers.append(number)
    table = pd.DataFrame(records, columns=header, dtype=object)
    table.attrs['rows'] = numbers
    return table, state
