"""The recipes a campaign can propose.

A recipe is a row of numbers, one per variable: the value of a numeric variable, or the
position of the chosen category. ``RecipeSpace`` draws such rows (uniformly, or as a Latin
hypercube for a first round), keeps them on the pipettable grid and inside the sample
volume when liquid handling is on, lists a recipe's neighbours for hill-climbing, and turns
rows into model inputs and into coordinates for distances and the PCA map.

Measured recipes are encoded with the same space, so they may lie off the current grid
(after a range was changed) or use a category that is no longer offered.
"""
import math

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ..initialization import greedy_maximin
from ..space import Factor, _auto_log
from .variables import VariableError

CLIMB_STEPS = 20          # a continuous variable moves by 1/20 of its range per hill-climbing step


class _Axis:
    """One variable: one column of the recipe matrix."""

    width = 1             # columns it takes in the model inputs and in the embedding

    def __init__(self, variable):
        self.variable = variable
        self.name = variable.name

    def encode(self, series):
        """A column of a table as numbers (raises ``VariableError`` for non-numbers)."""
        values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bad.any():
            raise VariableError('{}: "{}" is not a number.'.format(self.name, series.to_numpy()[bad][0]))
        return values

    def decode(self, values):
        """Numbers back to table values."""
        return np.asarray(values, dtype=float)

    def features(self, values):
        """Model-input columns for these values."""
        return np.asarray(values, dtype=float)[:, None]

    def embedding(self, values):
        """Distance coordinates for these values (unit scale)."""
        return self.unit(values)[:, None]

    def droplets(self, values):
        """Droplets each value takes (0 when not pipetted)."""
        return np.zeros(len(values), dtype=np.int64)

    @property
    def feature_names(self):
        """Names of the model-input columns."""
        return [self.name]


class _Interval(_Axis):
    """Continuous, not pipetted: any value in the range."""

    cardinality = math.inf

    def __init__(self, variable):
        super().__init__(variable)
        self.low, self.high = variable.low, variable.high

    def sample(self, n, rng):
        """Uniform values in the range."""
        return rng.uniform(self.low, self.high, size=n)

    def from_unit(self, u):
        """Unit-cube values mapped onto the range."""
        return self.low + np.asarray(u, dtype=float) * (self.high - self.low)

    def snap(self, values):
        """Values clipped into the range."""
        return np.clip(np.asarray(values, dtype=float), self.low, self.high)

    def unit(self, values):
        """Values as positions in [0, 1]."""
        return (np.asarray(values, dtype=float) - self.low) / (self.high - self.low)

    def moves(self, value):
        """Hill-climbing neighbours: one twentieth of the range down and up, clipped."""
        step = (self.high - self.low) / CLIMB_STEPS
        targets = (max(value - step, self.low), min(value + step, self.high))
        return [t for t in targets if abs(t - value) > 1e-9 * (self.high - self.low)]


class _DropletGrid(_Axis):
    """Continuous and pipetted: every whole number of droplets inside the range."""

    def __init__(self, variable, liquid):
        super().__init__(variable)
        self.liquid = liquid
        self.first, self.last = liquid.droplet_range(variable)
        if self.last < self.first:
            raise VariableError('{}: nothing between its minimum and maximum can be pipetted.'.format(self.name))
        self.cardinality = self.last - self.first + 1
        self.stride = max(1, int(round(self.cardinality / CLIMB_STEPS)))

    def _values(self, droplets):
        return self.liquid.concentration(self.variable, droplets)

    def sample(self, n, rng):
        """Uniform whole droplet counts in the range, as concentrations."""
        return self._values(rng.integers(self.first, self.last + 1, size=n))

    def from_unit(self, u):
        """Unit-cube values mapped onto droplet counts."""
        offset = np.minimum((np.asarray(u, dtype=float) * self.cardinality).astype(np.int64), self.cardinality - 1)
        return self._values(self.first + offset)

    def snap(self, values):
        """Values moved to the nearest whole droplet inside the range."""
        return self._values(np.clip(self.droplets(values), self.first, self.last))

    def unit(self, values):
        """Values as positions in [0, 1] over the variable's range."""
        return (np.asarray(values, dtype=float) - self.variable.low) / (self.variable.high - self.variable.low)

    def droplets(self, values):
        """Droplets each concentration takes."""
        return self.liquid.droplets(self.variable, values)

    def moves(self, value):
        """Hill-climbing neighbours: ``stride`` droplets down and up (about 1/20 of the range)."""
        k = int(self.droplets([value])[0])
        targets = {min(max(k + d, self.first), self.last) for d in (-self.stride, self.stride)} - {k}
        return [float(v) for v in self._values(sorted(targets))]


class _Levels(_Axis):
    """Discrete: the listed steps (rounded to whole droplets when pipetted)."""

    def __init__(self, variable, liquid=None):
        super().__init__(variable)
        self.liquid = liquid
        levels = np.asarray(variable.levels, dtype=float)
        self.levels = np.unique(liquid.snap(variable, levels) if liquid else levels)
        self.cardinality = len(self.levels)
        self._scale = Factor(variable.name, self.levels, log_scale=bool(_auto_log(self.levels)))

    def _index(self, values):
        return np.abs(np.asarray(values, dtype=float)[:, None] - self.levels[None, :]).argmin(axis=1)

    def sample(self, n, rng):
        """Uniform choice of the steps."""
        return self.levels[rng.integers(0, self.cardinality, size=n)]

    def from_unit(self, u):
        """Unit-cube values mapped onto the steps."""
        return self.levels[np.minimum((np.asarray(u, dtype=float) * self.cardinality).astype(np.int64),
                                      self.cardinality - 1)]

    def snap(self, values):
        """Values moved to the nearest step."""
        return self.levels[self._index(values)]

    def unit(self, values):
        """Values as positions in [0, 1] (log scale when the steps span two decades or more)."""
        return np.asarray(self._scale.to_unit(values), dtype=float)

    def droplets(self, values):
        """Droplets each step takes (0 without liquid handling)."""
        if self.liquid is None:
            return np.zeros(len(values), dtype=np.int64)
        return self.liquid.droplets(self.variable, values)

    def moves(self, value):
        """Hill-climbing neighbours: the step below and the step above."""
        i = int(self._index([value])[0])
        return [float(self.levels[j]) for j in (i - 1, i + 1) if 0 <= j < self.cardinality]


class _Categories(_Axis):
    """Categorical. ``extra`` categories appear in the data but are no longer proposed."""

    def __init__(self, variable, liquid=None, extra=()):
        super().__init__(variable)
        self.active = list(variable.categories)
        self.all = self.active + [c for c in dict.fromkeys(extra) if c not in self.active]
        self.width = len(self.all)
        self.cardinality = len(self.active)
        self.cost = liquid.category_droplets(variable) if liquid is not None and variable.volume_ul > 0 else 0

    def encode(self, series):
        """Category names as positions in ``all`` (raises for an unknown category)."""
        position = {c: i for i, c in enumerate(self.all)}
        names = [clean_category(v) for v in series.to_numpy()]
        unknown = [n for n in names if n not in position]
        if unknown:
            raise VariableError('{}: "{}" is not one of its categories.'.format(self.name, unknown[0]))
        return np.array([position[n] for n in names], dtype=float)

    def decode(self, values):
        """Positions back to category names."""
        return np.array([self.all[int(v)] for v in values], dtype=object)

    def features(self, values):
        """One-hot columns over every category, including ``extra`` ones."""
        return np.eye(self.width)[np.asarray(values, dtype=np.int64)]

    def embedding(self, values):
        """One-hot / sqrt(2), so two different categories are exactly 1 apart."""
        return self.features(values) / math.sqrt(2.0)     # two different categories are exactly 1 apart

    def droplets(self, values):
        """The category volume in droplets (the same for every category)."""
        return np.full(len(values), self.cost, dtype=np.int64)

    def sample(self, n, rng):
        """Uniform choice of the active categories."""
        return rng.integers(0, self.cardinality, size=n).astype(float)

    def from_unit(self, u):
        """Unit-cube values mapped onto the active categories."""
        return np.minimum((np.asarray(u, dtype=float) * self.cardinality).astype(np.int64),
                          self.cardinality - 1).astype(float)

    def snap(self, values):
        """Unchanged (positions are already valid)."""
        return np.asarray(values, dtype=float)

    def moves(self, value):
        """Hill-climbing neighbours: every other active category."""
        return [float(c) for c in range(self.cardinality) if c != int(value)]

    @property
    def feature_names(self):
        """``"<variable> = <category>"`` for every category."""
        return ['{} = {}'.format(self.name, c) for c in self.all]


def clean_category(value):
    """A category as text, with surrounding and repeated spaces removed (as ``Categorical`` stores it)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    if isinstance(value, float) and value.is_integer():
        value = int(value)                # Excel turns the category "1" into 1.0
    return ' '.join(str(value).split())


def min_distance(E):
    """Smallest distance between two rows (inf with fewer than two rows)."""
    if len(E) < 2:
        return math.inf
    from sklearn.neighbors import NearestNeighbors
    return float(NearestNeighbors(n_neighbors=2).fit(E).kneighbors(E)[0][:, 1].min())


class RecipeSpace:
    """The recipes a campaign can propose, as rows of numbers (one per variable).

            Each variable becomes an axis: ``_Interval`` (continuous, no liquid handling),
            ``_DropletGrid`` (continuous, whole droplets), ``_Levels`` (discrete) or
            ``_Categories``. ``budget`` is the droplets one sample holds (None: no limit).
            ``extra_categories`` maps a categorical variable to categories that occur in the
            data but are no longer offered: they are encoded, never proposed.
    """
    def __init__(self, variables, liquid=None, extra_categories=None):
        self.variables = list(variables)
        self.liquid = liquid
        extra = extra_categories or {}
        self.axes = [self._axis(v, extra.get(v.name, ())) for v in self.variables]
        self.names = [v.name for v in self.variables]
        self.budget = liquid.budget if liquid is not None else None

    def _axis(self, variable, extra):
        if variable.kind == 'categorical':
            return _Categories(variable, self.liquid, extra)
        if variable.kind == 'discrete':
            return _Levels(variable, self.liquid)
        return _DropletGrid(variable, self.liquid) if self.liquid is not None else _Interval(variable)

    # ------------------------------------------------------------------ basic facts
    @property
    def n_vars(self):
        """Number of variables (columns of a recipe row)."""
        return len(self.axes)

    @property
    def size(self):
        """Number of distinct recipes before the volume limit (inf with a free continuous variable)."""
        total = 1.0
        for axis in self.axes:
            total *= axis.cardinality
        return total

    def _matrix(self, R):
        R = np.asarray(R, dtype=float)
        return R.reshape(-1, self.n_vars)

    def choices(self):
        """For each variable the values a recipe may take: a list for discrete and categorical
        variables, None for continuous ones (any value in the range, or any whole droplet)."""
        out = {}
        for axis in self.axes:
            if axis.variable.kind == 'categorical':
                out[axis.name] = list(axis.active)
            else:
                out[axis.name] = [float(v) for v in axis.levels] if axis.variable.kind == 'discrete' else None
        return out

    # ------------------------------------------------------------------ tables <-> rows
    def encode(self, frame):
        """A table with a column per variable as recipe rows."""
        missing = [name for name in self.names if name not in frame.columns]
        if missing:
            raise VariableError('No column for {}.'.format(', '.join(missing)))
        if not len(frame):
            return np.empty((0, self.n_vars))
        return np.column_stack([axis.encode(frame[axis.name]) for axis in self.axes])

    def decode(self, R):
        """Recipe rows as a table (category names restored)."""
        R = self._matrix(R)
        return pd.DataFrame({axis.name: axis.decode(R[:, j]) for j, axis in enumerate(self.axes)})

    # ------------------------------------------------------------------ model inputs and distances
    def features(self, R):
        """Model inputs: numeric values as is, categories one-hot."""
        R = self._matrix(R)
        return np.hstack([axis.features(R[:, j]) for j, axis in enumerate(self.axes)])

    @property
    def feature_names(self):
        """Names of the model-input columns."""
        return [name for axis in self.axes for name in axis.feature_names]

    @property
    def feature_groups(self):
        """``[(variable name, its columns in the model inputs)]``."""
        groups, start = [], 0
        for axis in self.axes:
            groups.append((axis.name, list(range(start, start + axis.width))))
            start += axis.width
        return groups

    def embedding(self, R):
        """Coordinates for distances: numeric variables on [0, 1] over their range, categories 1 apart."""
        R = self._matrix(R)
        return np.hstack([axis.embedding(R[:, j]) for j, axis in enumerate(self.axes)])

    # ------------------------------------------------------------------ volume
    def droplets(self, R):
        """Total droplets per recipe."""
        R = self._matrix(R)
        total = np.zeros(len(R), dtype=np.int64)
        for j, axis in enumerate(self.axes):
            total += axis.droplets(R[:, j])
        return total

    def feasible(self, R):
        """True for recipes that fit in the sample (always True without liquid handling)."""
        R = self._matrix(R)
        if self.budget is None:
            return np.ones(len(R), dtype=bool)
        return self.droplets(R) <= self.budget

    def feasible_fraction(self, rng=None, probe=4000):
        """Share of uniformly drawn recipes that fit in the sample volume."""
        if self.budget is None:
            return 1.0
        return float(self.feasible(self._draw(probe, rng or np.random.default_rng(0))).mean())

    # ------------------------------------------------------------------ identity, snapping, neighbours
    def keys(self, R):
        """Hashable row keys. Single precision, so values that differ only by rounding noise match."""
        rows = np.ascontiguousarray(self._matrix(R) + 0.0, dtype=np.float32)     # + 0.0 turns -0.0 into 0.0
        return [row.tobytes() for row in rows]

    def snap(self, R):
        """Nearest allowed value of every entry: inside the range, on a step, a whole number of droplets."""
        R = self._matrix(R)
        return np.column_stack([axis.snap(R[:, j]) for j, axis in enumerate(self.axes)]) if len(R) else R

    def neighbors(self, row):
        """Recipes one move away: a numeric variable one step up or down, or another category."""
        row = np.asarray(row, dtype=float)
        out = []
        for j, axis in enumerate(self.axes):
            for value in axis.moves(float(row[j])):
                neighbour = row.copy()
                neighbour[j] = value
                out.append(neighbour)
        out = np.array(out, dtype=float).reshape(-1, self.n_vars)
        return out[self.feasible(out)]

    # ------------------------------------------------------------------ sampling
    def _draw(self, m, rng):
        return np.column_stack([axis.sample(m, rng) for axis in self.axes])

    def _new(self, R, seen):
        """Rows of ``R`` whose key is not in ``seen`` (first occurrence only); ``seen`` is updated."""
        keep = []
        for i, key in enumerate(self.keys(R)):
            if key not in seen:
                seen.add(key)
                keep.append(i)
        return R[keep]

    def sample(self, n, rng, exclude=(), max_draws=2_000_000):
        """``n`` distinct recipes that fit the sample volume, uniform per variable.

        ``exclude``: keys that must not come back (measured recipes). Fewer than ``n`` rows
        return only when the space has no more.
        """
        seen, chosen, drawn, idle, rate = set(exclude), [], 0, 0, 1.0
        count = 0
        while count < n and drawn < max_draws and idle < 20:
            m = int(min(max(2 * (n - count) / max(rate, 1e-3), 64), 200_000))
            draw = self._draw(m, rng)
            drawn += m
            fits = draw[self.feasible(draw)]
            rate = max(len(fits) / m, 0.5 * rate)
            fresh = self._new(fits, seen)[:n - count]
            idle = 0 if len(fresh) else idle + 1
            chosen.append(fresh)
            count += len(fresh)
        return np.vstack(chosen) if chosen else np.empty((0, self.n_vars))

    def lhs(self, n, rng, exclude=(), n_designs=20):
        """A Latin hypercube of ``n`` recipes: every variable's range is cut into ``n`` slices
        and each slice is used once, so the recipes spread over the whole space. Of
        ``n_designs`` random hypercubes the one whose two closest recipes are farthest apart
        wins. When some recipes do not fit the sample volume, a larger hypercube is drawn and
        the ``n`` fitting recipes that spread best are kept.
        """
        exclude = set(exclude)
        rate = self.feasible_fraction(rng)
        if n < 1 or rate <= 0.0:
            return np.empty((0, self.n_vars))
        m = n if rate >= 1.0 else int(math.ceil(2.0 * n / rate))
        best, best_score = None, -math.inf
        for _ in range(n_designs):
            U = qmc.LatinHypercube(d=self.n_vars, seed=int(rng.integers(2 ** 31 - 1))).random(m)
            R = np.column_stack([axis.from_unit(U[:, j]) for j, axis in enumerate(self.axes)])
            seen = set(exclude)
            R = self._new(R[self.feasible(R)], seen)
            if len(R) < n:
                extra = self.sample(n - len(R), rng, exclude=seen)
                R = np.vstack([R, extra]) if len(extra) else R
            if len(R) > n:
                R = R[greedy_maximin(self.embedding(R), n)]
            score = min_distance(self.embedding(R))
            if best is None or len(R) > len(best) or (len(R) == len(best) and score > best_score):
                best, best_score = R, score
        return best
