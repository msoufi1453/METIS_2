"""Variables of a campaign: what a recipe can change.

* ``Continuous``: any value between ``low`` and ``high``.
* ``Discrete``: one of up to 100 numeric ``levels``. One level makes a fixed component.
* ``Categorical``: one of up to 100 named ``categories`` (a strain, a promoter, ...).

Numeric variables carry a ``unit`` (free text; it only has to stay the same from round to
round) and, for liquid handling, the ``stock`` concentration in that unit.
"""
import math
from dataclasses import dataclass
from typing import ClassVar

from . import columns

MAX_LEVELS = 100


class VariableError(ValueError):
    """A problem with a variable or with the set of variables that the user can fix."""


def _name(value):
    name = ' '.join(str('' if value is None else value).split())
    if not name:
        raise VariableError('Every variable needs a name.')
    return name


def _number(value, what):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise VariableError('{} must be a number (got {!r}).'.format(what, value)) from None
    if not math.isfinite(number):
        raise VariableError('{} must be a finite number.'.format(what))
    return number


def _stock(value, name):
    if value is None:
        return None
    stock = _number(value, 'The stock concentration of {}'.format(name))
    if stock <= 0:
        raise VariableError('The stock concentration of {} must be above 0.'.format(name))
    return stock


def _set(obj, **values):
    for key, value in values.items():
        object.__setattr__(obj, key, value)


@dataclass(frozen=True)
class Continuous:
    name: str
    low: float
    high: float
    unit: str = ''
    stock: float = None
    kind: ClassVar[str] = 'continuous'
    numeric: ClassVar[bool] = True

    def __post_init__(self):
        name = _name(self.name)
        low = _number(self.low, 'The minimum of {}'.format(name))
        high = _number(self.high, 'The maximum of {}'.format(name))
        if not low < high:
            raise VariableError('The minimum of {} must be below its maximum.'.format(name))
        _set(self, name=name, low=low, high=high, unit=str(self.unit or '').strip(), stock=_stock(self.stock, name))

    @property
    def label(self):
        return '{} ({})'.format(self.name, self.unit) if self.unit else self.name

    @property
    def varies(self):
        return True

    def to_dict(self):
        return {'type': self.kind, 'name': self.name, 'low': self.low, 'high': self.high, 'unit': self.unit,
                'stock': self.stock}


@dataclass(frozen=True)
class Discrete:
    name: str
    levels: tuple
    unit: str = ''
    stock: float = None
    kind: ClassVar[str] = 'discrete'
    numeric: ClassVar[bool] = True

    def __post_init__(self):
        name = _name(self.name)
        try:
            raw = list(self.levels)
        except TypeError:
            raise VariableError('{} needs a list of levels.'.format(name)) from None
        levels = [_number(v, 'Every step of {}'.format(name)) for v in raw]
        if not 1 <= len(levels) <= MAX_LEVELS:
            raise VariableError('{} needs between 1 and {} steps.'.format(name, MAX_LEVELS))
        if len(set(levels)) != len(levels):
            raise VariableError('{} lists the same step twice.'.format(name))
        _set(self, name=name, levels=tuple(sorted(levels)), unit=str(self.unit or '').strip(),
             stock=_stock(self.stock, name))

    @property
    def label(self):
        return '{} ({})'.format(self.name, self.unit) if self.unit else self.name

    @property
    def low(self):
        return self.levels[0]

    @property
    def high(self):
        return self.levels[-1]

    @property
    def varies(self):
        return len(self.levels) > 1

    def to_dict(self):
        return {'type': self.kind, 'name': self.name, 'levels': list(self.levels), 'unit': self.unit,
                'stock': self.stock}


@dataclass(frozen=True)
class Categorical:
    """``volume_ul``: with liquid handling, the volume dispensed per sample of the chosen
    category's own liquid. 0 means the category is not pipetted (set by hand)."""
    name: str
    categories: tuple
    volume_ul: float = 0.0
    kind: ClassVar[str] = 'categorical'
    numeric: ClassVar[bool] = False

    def __post_init__(self):
        name = _name(self.name)
        if isinstance(self.categories, str):
            raise VariableError('{} needs a list of categories.'.format(name))
        try:
            raw = list(self.categories)
        except TypeError:
            raise VariableError('{} needs a list of categories.'.format(name)) from None
        categories = [' '.join(str('' if c is None else c).split()) for c in raw]
        if not 1 <= len(categories) <= MAX_LEVELS:
            raise VariableError('{} needs between 1 and {} categories.'.format(name, MAX_LEVELS))
        if any(not c for c in categories):
            raise VariableError('Every category of {} needs a name.'.format(name))
        if len(set(categories)) != len(categories):
            raise VariableError('{} lists the same category twice.'.format(name))
        volume = _number(0.0 if self.volume_ul is None else self.volume_ul, 'The volume of {}'.format(name))
        if volume < 0:
            raise VariableError('The volume of {} cannot be negative.'.format(name))
        _set(self, name=name, categories=tuple(categories), volume_ul=volume)

    unit = ''
    stock = None

    @property
    def label(self):
        return self.name

    @property
    def varies(self):
        return len(self.categories) > 1

    def to_dict(self):
        return {'type': self.kind, 'name': self.name, 'categories': list(self.categories),
                'volume_ul': self.volume_ul}


KINDS = {cls.kind: cls for cls in (Continuous, Discrete, Categorical)}


def from_dict(data):
    data = dict(data)
    kind = data.pop('type', None)
    if kind not in KINDS:
        raise VariableError('Unknown variable type {!r}.'.format(kind))
    return KINDS[kind](**data)


def check_variables(variables, target, max_variables=None):
    """Raise ``VariableError`` unless the variables form a usable design."""
    variables = list(variables)
    if not variables:
        raise VariableError('Add at least one variable.')
    if max_variables is not None and len(variables) > max_variables:
        raise VariableError('{} variables, but this model takes at most {:,}.'.format(len(variables), max_variables))
    target = ' '.join(str(target or '').split())
    if not target:
        raise VariableError('Name the value you measure (the result to maximise).')
    taken, labels = {}, {}
    for variable in variables:
        key = variable.name.lower()
        if key in columns.reserved(target):
            raise VariableError('"{}" is used for a column of the data table; choose another name.'.format(
                variable.name))
        if key in taken:
            raise VariableError('Two variables are called "{}".'.format(variable.name))
        if variable.label.lower() in labels or variable.label.lower() in taken or key in labels:
            raise VariableError('"{}" would be confused with another variable; rename one of them.'.format(
                variable.label))
        taken[key] = variable
        labels[variable.label.lower()] = variable
    if not any(v.varies for v in variables):
        raise VariableError('At least one variable needs more than one possible value.')
    return variables
