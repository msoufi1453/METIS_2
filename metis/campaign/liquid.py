"""Liquid handling: which concentrations a dispenser can make and the volume they take.

A dispenser adds whole droplets. With a stock of concentration ``S`` (in the variable's
own unit), droplets of ``d`` nL and a sample of ``V`` µL, one droplet adds
``S * d / (1000 * V)`` to the sample, so only multiples of that step can be pipetted. A
recipe fits when its droplets together are at most the pipettable volume ``P``; the rest
of ``P`` is topped up with the diluent. ``P`` is the whole sample unless part of every
sample is taken by liquids the dispenser does not add: with 1.75 µL of a 10 µL sample
taken, ``P`` = 8.25 µL. Concentrations are always per the whole sample ``V``.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .variables import VariableError

DEFAULT_DROPLET_NL = 4.0
TOLERANCE = 1e-9
FEW_LEVELS = 3              # fewer pipettable values than this in a continuous range draws a warning
ROUNDING_NOTICE = 0.001     # a discrete step moved by more than 0.1% draws a warning


@dataclass(frozen=True)
class Issue:
    """A design problem: ``level`` "error" (blocks a round) or "warning", a message, and the variable."""
    level: str              # 'error' (blocks a round) or 'warning'
    message: str
    variable: str = None

    @property
    def blocking(self):
        """True for errors."""
        return self.level == 'error'


def _fmt(value):
    return '{:.4g}'.format(float(value))


@dataclass(frozen=True)
class LiquidHandling:
    """A dispenser setting: sample volume (uL), droplet size (nL, default 4), diluent name and
    the pipettable volume (uL; None: the whole sample).

            Validated on construction (positive numbers, a pipettable volume no larger than
            the sample, a droplet smaller than the pipettable volume). A pipettable volume
            equal to the sample is stored as None.
    """
    sample_volume_ul: float
    droplet_nl: float = DEFAULT_DROPLET_NL
    diluent: str = 'Water'
    pipettable_volume_ul: float = None

    def __post_init__(self):
        for name, label in (('sample_volume_ul', 'The sample volume'), ('droplet_nl', 'The droplet size')):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError):
                raise VariableError('{} must be a number.'.format(label)) from None
            if not (math.isfinite(value) and value > 0):
                raise VariableError('{} must be above 0.'.format(label))
            object.__setattr__(self, name, value)
        if self.droplet_nl > self.sample_volume_ul * 1000.0:
            raise VariableError('One droplet is larger than the sample.')
        if self.pipettable_volume_ul is not None:
            try:
                value = float(self.pipettable_volume_ul)
            except (TypeError, ValueError):
                raise VariableError('The pipettable volume must be a number.') from None
            if not (math.isfinite(value) and value > 0):
                raise VariableError('The pipettable volume must be above 0.')
            if value > self.sample_volume_ul * (1 + TOLERANCE):
                raise VariableError('The pipettable volume ({} µL) cannot be more than the sample volume ({} µL).'
                                    .format(_fmt(value), _fmt(self.sample_volume_ul)))
            object.__setattr__(self, 'pipettable_volume_ul',
                               None if value >= self.sample_volume_ul * (1 - TOLERANCE) else value)
            if self.droplet_nl > self.pipettable_ul * 1000.0:
                raise VariableError('One droplet is larger than the pipettable volume.')
        object.__setattr__(self, 'diluent', ' '.join(str(self.diluent or '').split()) or 'Water')

    # ------------------------------------------------------------------ volumes
    @property
    def pipettable_ul(self):
        """Volume the dispenser fills in each sample (µL): the pipettable volume, or the whole sample."""
        return self.sample_volume_ul if self.pipettable_volume_ul is None else self.pipettable_volume_ul

    @property
    def reserved_ul(self):
        """Volume of each sample the dispenser does not fill (µL; 0 when it fills the whole sample)."""
        return self.sample_volume_ul - self.pipettable_ul

    @property
    def limit(self):
        """What a recipe must fit in, for messages: "the 10 µL sample" or "the 8.25 µL that can be
        pipetted into each 10 µL sample"."""
        if self.pipettable_volume_ul is None:
            return 'the {} µL sample'.format(_fmt(self.sample_volume_ul))
        return 'the {} µL that can be pipetted into each {} µL sample'.format(
            _fmt(self.pipettable_ul), _fmt(self.sample_volume_ul))

    # ------------------------------------------------------------------ droplets
    @property
    def budget(self):
        """Whole droplets the dispenser can add to one sample: components and diluent together."""
        return int(math.floor(self.pipettable_ul * 1000.0 / self.droplet_nl + TOLERANCE))

    def step(self, variable):
        """Concentration one droplet of the variable's stock adds to a sample (per the whole
        sample volume, also when only part of it is pipettable)."""
        if variable.stock is None:
            raise VariableError('Enter the stock concentration of {}.'.format(variable.name))
        return variable.stock * self.droplet_nl / (self.sample_volume_ul * 1000.0)

    def droplets(self, variable, values):
        """Nearest whole number of droplets for each concentration."""
        return np.rint(np.asarray(values, dtype=float) / self.step(variable)).astype(np.int64)

    def concentration(self, variable, droplets):
        """Concentration made by ``droplets`` droplets, rounded to hide floating-point noise."""
        step = self.step(variable)
        decimals = max(0, 12 - int(math.floor(math.log10(step))))
        return np.round(np.asarray(droplets, dtype=float) * step, decimals)

    def snap(self, variable, values):
        """Concentrations moved to the nearest whole number of droplets."""
        return self.concentration(variable, self.droplets(variable, values))

    def droplet_range(self, variable):
        """First and last droplet count inside a continuous variable's range (last < first: none)."""
        step = self.step(variable)
        return (int(math.ceil(variable.low / step - TOLERANCE)), int(math.floor(variable.high / step + TOLERANCE)))

    def category_droplets(self, variable):
        """Droplets of a categorical variable's own liquid per sample."""
        return int(round(variable.volume_ul * 1000.0 / self.droplet_nl))

    def volume_ul(self, droplets):
        """Droplet counts as volumes in uL."""
        return np.asarray(droplets, dtype=float) * self.droplet_nl / 1000.0

    # ------------------------------------------------------------------ persistence
    def to_dict(self):
        """JSON-ready settings. The pipettable volume is only listed when it is set, so a
        workbook without one can still be read by versions that do not know it."""
        data = {'sample_volume_ul': self.sample_volume_ul, 'droplet_nl': self.droplet_nl, 'diluent': self.diluent}
        if self.pipettable_volume_ul is not None:
            data['pipettable_volume_ul'] = self.pipettable_volume_ul
        return data

    @classmethod
    def from_dict(cls, data):
        """Settings from ``to_dict`` output (None or empty: no liquid handling)."""
        return cls(**data) if data else None

    # ------------------------------------------------------------------ checks
    def _extremes(self, variable):
        """(fewest, most) droplets the variable can take per sample."""
        if variable.kind == 'categorical':
            k = self.category_droplets(variable)
            return k, k
        if variable.kind == 'continuous':
            return self.droplet_range(variable)
        k = self.droplets(variable, variable.levels)
        return int(k.min()), int(k.max())

    def _check_numeric(self, v):
        unit = ' ' + v.unit if v.unit else ''
        if v.stock is None:
            return [Issue('error', 'Enter the stock concentration of {}.'.format(v.name), v.name)]
        if v.low < 0:
            return [Issue('error', '{}: a concentration cannot be negative when it is pipetted.'.format(v.name), v.name)]
        step, issues = self.step(v), []
        if v.high > v.stock * (1 + TOLERANCE):
            return [Issue('error', '{}: the maximum ({}{}) is above the stock concentration ({}{}), so no volume of '
                          'stock can reach it. Use a more concentrated stock or lower the maximum.'.format(
                              v.name, _fmt(v.high), unit, _fmt(v.stock), unit), v.name)]
        if v.kind == 'continuous':
            first, last = self.droplet_range(v)
            if last < first:
                issues.append(Issue('error', '{}: one droplet ({} nL) already adds {}{}, so nothing between {} and {} '
                                    'can be pipetted. Use a more dilute stock or widen the range.'.format(
                                        v.name, _fmt(self.droplet_nl), _fmt(step), unit, _fmt(v.low), _fmt(v.high)),
                                    v.name))
            elif last - first + 1 < FEW_LEVELS:
                issues.append(Issue('warning', '{}: only {} pipettable value(s) between {} and {}{} (one droplet adds '
                                    '{}). Use a more dilute stock for finer steps.'.format(
                                        v.name, last - first + 1, _fmt(v.low), _fmt(v.high), unit, _fmt(step)), v.name))
            return issues
        levels = np.asarray(v.levels, dtype=float)
        k = self.droplets(v, levels)
        lost = levels[(k == 0) & (levels > 0)]
        if len(lost):
            return [Issue('error', '{}: the step {}{} is less than half a droplet (one droplet adds {}{}), so it '
                          'would become 0. Use a more dilute stock.'.format(
                              v.name, _fmt(lost[0]), unit, _fmt(step), unit), v.name)]
        if len(np.unique(k)) < len(k):
            same = levels[np.flatnonzero(np.diff(k) == 0)[0]:][:2]
            return [Issue('error', '{}: the steps {} and {}{} would be the same number of droplets (one droplet adds '
                          '{}{}). Use a more dilute stock or move the steps apart.'.format(
                              v.name, _fmt(same[0]), _fmt(same[1]), unit, _fmt(step), unit), v.name)]
        snapped = self.concentration(v, k)
        moved = np.abs(snapped - levels) > ROUNDING_NOTICE * np.abs(levels)
        if moved.any():
            changes = ', '.join('{} -> {}'.format(_fmt(a), _fmt(b)) for a, b in
                                list(zip(levels[moved], snapped[moved]))[:5])
            issues.append(Issue('warning', '{}: steps are rounded to whole droplets ({}{}).'.format(
                v.name, changes, unit), v.name))
        return issues

    def check(self, variables):
        """Problems with pipetting these variables: errors block a round, warnings do not."""
        issues, extremes = [], {}
        for v in variables:
            if v.numeric:
                found = self._check_numeric(v)
                issues += found
                if not any(i.blocking for i in found):
                    extremes[v.name] = self._extremes(v)
            elif v.volume_ul > 0:
                k = self.category_droplets(v)
                if k == 0:
                    issues.append(Issue('error', '{}: {} µL is less than half a droplet.'.format(
                        v.name, _fmt(v.volume_ul)), v.name))
                    continue
                if abs(k * self.droplet_nl / 1000.0 - v.volume_ul) > 1e-9:
                    issues.append(Issue('warning', '{}: the volume is rounded to whole droplets ({} µL).'.format(
                        v.name, _fmt(k * self.droplet_nl / 1000.0)), v.name))
                extremes[v.name] = (k, k)
        if any(i.blocking for i in issues) or not extremes:
            return issues
        least = sum(lo for lo, _ in extremes.values())
        most = sum(hi for _, hi in extremes.values())
        if least > self.budget:
            # A larger sample does not help: the volumes a concentration needs grow with the sample.
            issues.append(Issue('error', 'Even at their lowest values the components need {} µL together, more than '
                                '{}. Use more concentrated stocks{} lower the minimums.'.format(
                                    _fmt(self.volume_ul(least)), self.limit,
                                    ' or' if self.pipettable_volume_ul is None else ', make more of each sample '
                                    'pipettable or')))
        elif most > self.budget:
            top = sorted(extremes.items(), key=lambda item: -item[1][1])[:5]
            listing = ', '.join('{} ({} µL)'.format(name, _fmt(self.volume_ul(hi))) for name, (_, hi) in top)
            issues.append(Issue('warning', 'At their maximum values the components need {} µL together, more than '
                                '{}, so the highest values cannot all be combined (recipes that do not fit are never '
                                'proposed). To make room, lower the maximum or raise the stock concentration of the '
                                'largest volumes: {}.'.format(_fmt(self.volume_ul(most)), self.limit, listing)))
        return issues

    def summary(self, variables):
        """One row per pipetted variable: the droplet step, how many values it allows and the volumes."""
        rows = []
        for v in variables:
            if v.numeric:
                if v.stock is None or v.low < 0:
                    continue
                first, last = self._extremes(v)
                count = max(last - first + 1, 0) if v.kind == 'continuous' else len(v.levels)
                rows.append({'Variable': v.label, 'Stock': v.stock, 'One droplet adds': self.step(v),
                             'Pipettable values': count,
                             'Volume at minimum (µL)': float(self.volume_ul(max(first, 0))),
                             'Volume at maximum (µL)': float(self.volume_ul(max(last, 0)))})
            elif v.volume_ul > 0:
                k = self.category_droplets(v)
                rows.append({'Variable': v.label, 'Stock': np.nan, 'One droplet adds': np.nan,
                             'Pipettable values': len(v.categories),
                             'Volume at minimum (µL)': float(self.volume_ul(k)),
                             'Volume at maximum (µL)': float(self.volume_ul(k))})
        return pd.DataFrame(rows, columns=['Variable', 'Stock', 'One droplet adds', 'Pipettable values',
                                           'Volume at minimum (µL)', 'Volume at maximum (µL)'])
