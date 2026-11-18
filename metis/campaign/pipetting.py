"""From a round's recipes to a pipetting sheet and an iDOT protocol.

The sheet has the layout of ``iDOT_pipetting_sheet.csv`` in
https://github.com/msoufi1453/idot-protocol-builder: a header row of wells (``1A`` is
column 1, row A), then one row per liquid with the volume for each well in µL. Wells fill
column by column (1A, 1B, ... 1H, 2A, ...), 96 per plate, one well per measurement, in
the order of the Experiments sheet. The diluent tops every well up to the pipettable
volume: the whole sample, unless part of it is taken by liquids the dispenser does not add.

The protocol file is built by that repository's ``idot_builder`` (copied to
``metis.idot``), which re-reads and verifies the file before handing it over.
"""
import csv
import io
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pandas as pd

from ..idot import idot_builder
from . import columns
from .variables import VariableError

ROWS = 'ABCDEFGH'
WELLS = ['{}{}'.format(column, row) for column in range(1, 13) for row in ROWS]     # 1A, 1B, ..., 1H, 2A, ...


def assign_wells(n):
    """``[(plate, well)]`` for ``n`` measurements: plates of 96 wells, filled column by column."""
    return [(i // len(WELLS) + 1, WELLS[i % len(WELLS)]) for i in range(n)]


def _number(value):
    return '{:.6g}'.format(float(value))


def liquid_name(variable):
    """The row name of a numeric variable's stock, e.g. ``Mg-glutamate (500 mM stock)``."""
    stock = _number(variable.stock)
    return '{} ({} {} stock)'.format(variable.name, stock, variable.unit) if variable.unit else \
        '{} (stock {})'.format(variable.name, stock)


def _text(volume):
    """A Decimal volume as plain text: '4.14', never '4.140' or '4.14E+0'."""
    return format(volume.normalize(), 'f')


@dataclass
class IdotFile:
    """An iDOT protocol: the .idotx bytes, the liquid name mapping, the source-plate loading table, warnings."""
    filename: str
    data: bytes                 # the .idotx file (XML)
    liquids: pd.DataFrame       # liquid in the sheet -> its name in the iDOT software
    source: pd.DataFrame        # what to load into each source well
    warnings: list = field(default_factory=list)


@dataclass
class PipettingPlan:
    """Per plate, the volume of every liquid in every well (Decimal uL), the plate layout and notes."""
    plates: list                # one table per plate: liquids as rows, wells as columns, Decimal µL (0 = none)
    layout: pd.DataFrame        # Plate, Well, Sample ID, Data point, Replicate
    notes: list = field(default_factory=list)

    def table(self, plate=1):
        """The plate's sheet as numbers, for display (µL; empty where nothing is dispensed)."""
        frame = self.plates[plate - 1]
        return frame.apply(lambda column: column.map(lambda v: float(v) if v else None))

    def csv(self, plate=1):
        """The plate's sheet as CSV text, in the format ``idot_builder`` reads (µL)."""
        frame = self.plates[plate - 1]
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow([''] + list(frame.columns))
        for name, row in frame.iterrows():
            writer.writerow([name] + [_text(v) if v else '' for v in row])
        return buffer.getvalue()

    def idot(self, initials, name, plate=1):
        """The iDOT protocol for one plate (``initials`` prefix the liquid names)."""
        return idot_protocol(self.csv(plate), initials, name)


def pipetting_plan(rows, variables, liquid):
    """The plan for ``rows``: one row per measurement, with ``Plate`` and ``Well`` assigned
    and a column per variable."""
    if liquid is None:
        raise VariableError('Pipetting needs liquid handling to be switched on.')
    if not len(rows):
        raise VariableError('There are no recipes to pipette.')
    droplet = Decimal(repr(float(liquid.droplet_nl))) / 1000            # µL per droplet, exact
    volumes, notes, used = {}, [], pd.Series(0, index=rows.index, dtype='int64')

    def add(name, droplets):
        """Register a liquid row (raises if two liquids would share a name)."""
        if name in volumes:
            raise VariableError('Two liquids would be called "{}" in the pipetting sheet; rename a variable or '
                                'the diluent.'.format(name))
        volumes[name] = droplets

    for v in variables:
        if v.numeric:
            k = pd.Series(liquid.droplets(v, rows[v.name].to_numpy(dtype=float)), index=rows.index)
            add(liquid_name(v), k)
            used += k
        elif v.volume_ul > 0:
            k = liquid.category_droplets(v)
            for category in dict.fromkeys(rows[v.name]):
                add('{}: {}'.format(v.name, category), (rows[v.name] == category).astype('int64') * k)
            used += k
        else:
            notes.append('{} is not pipetted: set it by hand for each well (see the plate layout).'.format(v.name))
    spare = liquid.budget - used
    if (spare < 0).any():
        worst = rows.loc[spare.idxmin()]
        raise VariableError('Sample {} needs {} µL, more than {}.'.format(
            worst[columns.ID], _number(liquid.volume_ul(int(used.max()))), liquid.limit))
    add(liquid.diluent, spare)
    if liquid.reserved_ul > 0:
        notes.append('Each sample is {} µL: {} µL from this table and {} µL that the dispenser does not add.'.format(
            _number(liquid.sample_volume_ul), _number(liquid.pipettable_ul), _number(liquid.reserved_ul)))
    filled = float(liquid.volume_ul(liquid.budget))
    if filled < liquid.pipettable_ul - 1e-9:
        notes.append('Whole {} nL droplets cannot make exactly {} µL: each well gets {} µL from this table.'.format(
            _number(liquid.droplet_nl), _number(liquid.pipettable_ul), _number(filled)))
    empty = [name for name, k in volumes.items() if not k.any()]
    for name in empty:
        notes.append('{} is never dispensed in this round, so it is left out of the sheet.'.format(name))
    plates = []
    for plate in sorted(rows[columns.PLATE].unique()):
        on_plate = rows[columns.PLATE] == plate
        wells = list(rows.loc[on_plate, columns.WELL])
        sheet = pd.DataFrame({name: [int(n) * droplet for n in k[on_plate]] for name, k in volumes.items()
                              if name not in empty}, index=wells, dtype=object).T
        plates.append(sheet)
    layout = rows[[columns.PLATE, columns.WELL, columns.ID, columns.POINT, columns.REPLICATE]].reset_index(drop=True)
    return PipettingPlan(plates, layout, notes)


def safe_filename(name):
    """A file-system-safe name without an extension."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name)).strip('._') or 'METIS'


def idot_protocol(sheet_csv, initials, name):
    """An iDOT protocol (.idotx, an XML file) from a pipetting sheet in CSV text.

    ``initials`` prefix every liquid name, as the iDOT builder requires. Raises
    ``metis.idot.IdotError`` with a message for the user when the sheet cannot be used.
    """
    b = idot_builder
    initials = b.check_initials(initials)
    template = b.load_template(b.DEFAULT_TEMPLATE)
    sheet = b.parse_sheet(sheet_csv)
    plan = b.make_plan(sheet, initials)
    source = b.layout_source_plate(plan, template)
    filename = b.output_filename(safe_filename(name))
    title = Path(filename).stem
    data = b.encode(template, b.build_text(template, plan.wells, source.wells, title))
    b.verify_output(data, template, plan.wells, source.wells, title)
    liquids = pd.DataFrame([(sheet_name, idot_name) for idot_name, sheet_name in plan.liquids.items()],
                           columns=['Liquid in the sheet', 'Name in the iDOT software'])
    wells = pd.DataFrame([(w.position, w.liquid, float(w.volume_ul),
                           float(w.with_extra_ul.quantize(b.ONE_NL, rounding=b.ROUND_CEILING)))
                          for w in source.wells],
                         columns=['Source well', 'Liquid', 'Dispensed (µL)', 'Load at least (µL, with 5% extra)'])
    return IdotFile(filename, data, liquids, wells, sheet.warnings + plan.warnings + source.warnings)
