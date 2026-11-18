"""An active-learning campaign: its variables, its data and the next round.

    from metis.campaign import Campaign, Categorical, Continuous, Discrete, LiquidHandling

    campaign = Campaign(
        [Continuous('Mg-glutamate', 0, 10, unit='mM', stock=500),
         Discrete('K-glutamate', [40, 80, 120, 160], unit='mM', stock=3000),
         Categorical('Promoter', ['T7', 'T3'])],
        target='sfGFP', liquid_handling=LiquidHandling(sample_volume_ul=10))

    batch = campaign.propose(10)            # round 1: a Latin hypercube
    campaign.add_round(batch)
    campaign.save('round1.xlsx')            # fill in the sfGFP column, then
    campaign = Campaign.load('round1.xlsx')
    batch = campaign.propose(10)            # model-guided: predictions and 80% intervals

Rounds are numbered from 1. Round 0 is data that existed before the campaign ("day 0",
``add_data``). Every measurement has its own row; replicates of a recipe share its
``Sample ID``.
"""
import copy
import datetime
import math
import platform
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import hillclimb
from ..acquisition import UCB
from ..optimize import suggest
from ..surrogates import Prediction
from . import analysis, workbook
from . import columns as col
from .batch import make_selector, penalty_radius, spacing
from .liquid import Issue, LiquidHandling
from .models import DEFAULT_MODEL, MODELS, default_pool_size, make_model, model_spec
from .pipetting import assign_wells, pipetting_plan
from .space import RecipeSpace, clean_category
from .variables import VariableError, check_variables, from_dict

FORMAT = 1
FIRST_GUIDED_EXPLOITATION = 0.5     # the schedule tuned on the replay data (docs/REPLAY.md, amendment 4):
LATER_EXPLOITATION = 1.0            # half exploration in the first model-guided round, none afterwards
MIN_FEASIBLE = 0.01                 # below this share of random recipes fitting the sample, a round is refused
CELL_LIMIT = 30000                  # Excel cells hold 32,767 characters
UNCHANGED = object()


class CampaignError(ValueError):
    """A problem with the campaign's data or settings that the user can fix."""


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def environment():
    """Versions of what produced a round, for the workbook's metadata."""
    import importlib.metadata as metadata

    import metis
    packages = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'openpyxl', 'torch', 'tabpfn', 'streamlit'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {'metis': metis.__version__, 'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages}


def _jsonable(value):
    """Plain Python for JSON: numpy scalars and arrays unwrapped, NaN and inf as None."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value if value is None or isinstance(value, str) else str(value)


def as_number(value):
    """A cell as a float: None for an empty cell, ValueError for anything that is not a finite number.
    A decimal comma ("1,5") is accepted."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.count(',') == 1 and '.' not in text:
            text = text.replace(',', '.')
        number = float(text)
    if not math.isfinite(number):
        raise ValueError(value)
    return number


def _numbers(values, what, where):
    """Floats for a column; NaN where a cell is empty. ``where[i]`` names row ``i`` in messages."""
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        try:
            number = as_number(value)
        except (TypeError, ValueError):
            raise CampaignError('{}: "{}" in the {} column is not a number.'.format(where[i], value, what)) from None
        if number is not None:
            out[i] = number
    return out


def _whole(values, what, where):
    """Whole numbers for a bookkeeping column ("Round 3" is read as 3)."""
    out = []
    for i, value in enumerate(values):
        text = str(value).strip() if value is not None else ''
        if isinstance(value, str) and text.lower().startswith('round'):
            text = text[5:].strip()
        try:
            number = float(text)
        except ValueError:
            number = math.nan
        if not (math.isfinite(number) and number == int(number) and number >= 0):
            raise CampaignError('{}: the {} column needs a whole number, not "{}".'.format(
                where[i], what, '' if value is None else value))
        out.append(int(number))
    return np.array(out, dtype=np.int64)


@dataclass
class Fitted:
    """A model fitted to the campaign's measurements, with the space that encodes recipes for it."""
    model: object
    space: RecipeSpace
    X: np.ndarray
    y: np.ndarray


@dataclass
class _Scored:
    """Everything a model-guided proposal was chosen from; kept for hill-climbing and re-prediction."""
    fitted: Fitted
    acquisition: UCB
    pool: np.ndarray
    prediction: Prediction
    exclude: set
    best_y: float
    radius: float
    fantasy: bool

    def pick(self, pool, prediction, k):
        """Batch of ``k`` rows of ``pool`` with the round's acquisition and batch rule (no refit)."""
        space = self.fitted.space
        chosen = suggest(self.fitted.model, self.fitted.X, self.fitted.y, space.features(pool), k,
                         acquisition=self.acquisition, selector=make_selector(self.radius, self.fantasy),
                         embedding=space.embedding(pool), fit=False, prediction=prediction)
        return chosen.indices


@dataclass
class Proposal:
    """The recipes proposed for a round.

    ``recipes``: one row per recipe, a column per variable. ``prediction`` (model-guided
    rounds): the model's ``mean``, ``std`` and 80% interval (``low``, ``high``) for each.
    """
    campaign: object
    round: int
    guided: bool
    recipes: pd.DataFrame
    prediction: pd.DataFrame = None
    settings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    before: object = None           # hill-climbing: the proposal it started from
    moves: pd.DataFrame = None      # hill-climbing: the batch recipes found by a climb, and where it began
    _scored: _Scored = None

    @property
    def fitted(self):
        """The fitted model behind a model-guided proposal (None for a Latin hypercube)."""
        return self._scored.fitted if self._scored is not None else None

    @property
    def table(self):
        """Recipes with their data point number and, when model-guided, the prediction columns."""
        return self.campaign.batch_table(self.recipes, self.prediction)

    def predict(self, recipes):
        """The model's prediction for any recipes (for example the batch after an edit by hand)."""
        if self._scored is None:
            raise CampaignError('This round was not chosen by a model, so it has no predictions.')
        return self.campaign.predict(recipes, self._scored.fitted)

    def climb(self, n_starts=None, max_steps=20):
        """The same round with hill-climbing: the best-scoring candidates move step by step to
        better-scoring neighbours, the end points join the candidates, and the batch is
        chosen again. ``before`` on the result is this proposal, to compare or go back."""
        if self._scored is None:
            raise CampaignError('Hill-climbing needs a model-guided round.')
        scored, space = self._scored, self._scored.fitted.space
        n = int(self.settings['n_samples'])
        n_starts = int(n_starts or n)
        climbed = hillclimb.climb(space, scored.fitted.model, scored.acquisition, scored.pool, scored.prediction,
                                  scored.exclude, best_y=scored.best_y, n_starts=n_starts, max_steps=int(max_steps))
        pool, prediction = scored.pool, scored.prediction
        if len(climbed.new):
            pool = np.vstack([pool, climbed.new])
            prediction = Prediction(np.concatenate([prediction.mean, climbed.new_prediction.mean]),
                                    np.concatenate([prediction.std, climbed.new_prediction.std]))
        chosen = scored.pick(pool, prediction, n)
        recipes = space.decode(pool[chosen])
        starts = dict(zip(reversed(space.keys(climbed.ends)), reversed(range(len(climbed.ends)))))   # first climb wins
        found = []
        for point, (index, key) in enumerate(zip(chosen, space.keys(pool[chosen])), start=1):
            if index >= len(scored.pool):
                start = starts[key]
                found.append({col.POINT: point, 'Moves': int(climbed.steps[start]),
                              **{'start ' + name: value for name, value in
                                 space.decode(climbed.starts[[start]]).iloc[0].items()}})
        settings = {**self.settings, 'hill_climb': True, 'climb_starts': n_starts, 'climb_max_moves': int(max_steps),
                    'climb_moves': [int(s) for s in climbed.steps], 'climb_new_recipes': int(len(climbed.new)),
                    'climbed_in_batch': len(found)}
        return Proposal(self.campaign, self.round, True, recipes, self.campaign.predict(recipes, scored.fitted),
                        settings, list(self.warnings), before=self, moves=pd.DataFrame(found), _scored=scored)


class Campaign:
    """A campaign: variables, target, optional liquid handling, model, and one data table.

            ``data`` has one row per measurement and the columns ``columns`` lists: bookkeeping
            (Sample ID, Round, Data point, Replicate, Plate, Well), one per variable, ``extra``
            (removed variables, notes), the prediction columns and the result (``target``).
            ``history`` holds one JSON-ready record per generated round (settings, versions,
            PCA map). Typical cycle: ``propose`` -> ``add_round`` -> ``save`` -> fill in results
            -> ``load`` -> ``propose`` ... ``add_data`` adds measurements that already exist.

            ``model``: a key of ``MODELS`` ('tabpfn35'); ``model_options`` override its defaults.
            ``seed`` makes every round reproducible (each round draws from ``[seed, round]``).
    """
    def __init__(self, variables, target='Result', liquid_handling=None, model=DEFAULT_MODEL, seed=0,
                 name='campaign', model_options=None):
        self.name = ' '.join(str(name or '').split()) or 'campaign'
        self.target = ' '.join(str(target or '').split())
        self.model = model_spec(model).key
        self.model_options = dict(model_options or {})
        self.seed = int(seed)
        self.liquid = liquid_handling
        self.variables = check_variables(variables, self.target, model_spec(model).max_variables)
        self.extra = []                 # other columns kept in the data (removed variables, notes)
        self.history = []               # one record per generated round
        self.created = _now()
        self.data = self._table({})

    # ------------------------------------------------------------------ the data table
    @property
    def names(self):
        """Variable names, in order."""
        return [v.name for v in self.variables]

    @property
    def columns(self):
        """Every column of ``data``, in order."""
        return list(col.BOOKKEEPING) + self.names + self.extra + col.prediction_columns(self.target) + [self.target]

    def _table(self, values, n=0):
        """A data table with every column, typed; columns missing from ``values`` are empty."""
        numeric = {v.name for v in self.variables if v.numeric} | set(col.prediction_columns(self.target)) | \
            {self.target}
        whole = {col.ID, col.ROUND, col.POINT, col.REPLICATE}
        table = {}
        for name in self.columns:
            if name in values:
                column = np.asarray(values[name], dtype=np.int64 if name in whole else float if name in numeric
                                    else object)
            elif name in whole:
                column = np.zeros(n, dtype=np.int64)
            else:
                column = np.full(n, np.nan) if name in numeric else np.full(n, None, dtype=object)
            table[name] = column
        return pd.DataFrame(table)

    def _append(self, block):
        self.data = block if not len(self.data) else pd.concat([self.data, block], ignore_index=True)

    def _rename(self, frame):
        """Columns under their variable names: ``Mg (mM)`` and `` Mg `` both become ``Mg``."""
        frame = pd.DataFrame(frame).copy()
        frame.columns = [str(c).strip() for c in frame.columns]
        labels = {v.label: v.name for v in self.variables}
        frame = frame.rename(columns={c: labels[c] for c in frame.columns if c in labels and labels[c] not in frame})
        return frame

    def _clean(self, frame, where=None, fill=None):
        """The variables' columns of ``frame``, checked: numbers for numeric variables, tidy
        text for categories. ``fill`` gives a value for variables the frame has no column for."""
        where = where or ['Row {}'.format(i + 1) for i in range(len(frame))]
        fill, out = fill or {}, {}
        for v in self.variables:
            if v.name in frame.columns:
                values = list(frame[v.name])
            elif v.name in fill:
                values = [fill[v.name]] * len(frame)
            else:
                raise CampaignError('The data has no column for {}.'.format(v.name))
            if v.numeric:
                numbers = _numbers(values, v.name, where)
                if np.isnan(numbers).any():
                    raise CampaignError('{}: {} has no value.'.format(where[int(np.flatnonzero(np.isnan(numbers))[0])],
                                                                      v.name))
                out[v.name] = numbers
            else:
                names = [clean_category(x) for x in values]
                if '' in names:
                    raise CampaignError('{}: {} has no value.'.format(where[names.index('')], v.name))
                out[v.name] = np.array(names, dtype=object)
        return pd.DataFrame(out, columns=self.names)

    def _recipe_codes(self, recipes):
        """A number per distinct recipe, in order of first appearance."""
        keys = [tuple(np.float32(x) if v.numeric else x for v, x in zip(self.variables, row))
                for row in recipes.itertuples(index=False)]
        return pd.factorize(pd.Series(keys, dtype=object))[0]

    def _next_id(self):
        return int(self.data[col.ID].max()) + 1 if len(self.data) else 1

    @property
    def next_sample_id(self):
        """The Sample ID the first recipe of the next round will get; the others follow in order."""
        return self._next_id()

    # ------------------------------------------------------------------ views of the data
    @property
    def rounds(self):
        """Round numbers present in the data (0 = day 0 data)."""
        return sorted(int(r) for r in self.data[col.ROUND].unique())

    @property
    def next_round(self):
        """Number the next generated round gets (day 0 data does not count)."""
        return max([r for r in self.rounds if r > 0], default=0) + 1

    @property
    def measured(self):
        """Rows with a result."""
        return self.data[self.data[self.target].notna()]

    @property
    def pending(self):
        """Rows still waiting for a result."""
        return self.data[self.data[self.target].isna()]

    def samples(self):
        """One row per recipe: its round, data point and variables, the mean of its results
        (``self.target``) and how many there are (``Results``)."""
        keep = [col.ID, col.ROUND, col.POINT] + self.names + col.prediction_columns(self.target)
        if not len(self.data):
            return pd.DataFrame(columns=keep + [self.target, 'Results'])
        first = self.data.drop_duplicates(col.ID).set_index(col.ID)
        results = self.data.groupby(col.ID)[self.target].agg(['mean', 'count'])
        out = first[keep[1:]].copy()
        out[self.target] = results['mean']
        out['Results'] = results['count']
        return out.reset_index()

    def best(self):
        """The best measured recipe so far (a row of ``samples()``), or None."""
        samples = self.samples().dropna(subset=[self.target])
        return None if samples.empty else samples.loc[samples[self.target].idxmax()]

    def progress(self):
        """Per round: recipes measured, best, median and best so far (``analysis.progress``)."""
        samples = self.samples().dropna(subset=[self.target])
        return analysis.progress(samples, col.ROUND, self.target)

    def fingerprint(self):
        """A short hash of the design, the model settings and the data: equal campaigns give equal
        fingerprints, so results computed for one can be reused for the other."""
        import hashlib
        import json
        state = {key: value for key, value in self._state().items() if key not in ('saved', 'created', 'history')}
        digest = hashlib.sha1(json.dumps(state, sort_keys=True).encode())
        digest.update(self.data.to_csv(index=False, float_format='%.12g').encode())
        return digest.hexdigest()[:16]

    def copy(self):
        """An independent copy (data, history and variable list copied; variables are immutable)."""
        other = copy.copy(self)
        other.data = self.data.copy()
        other.history = copy.deepcopy(self.history)
        other.extra = list(self.extra)
        other.variables = list(self.variables)
        other.model_options = dict(self.model_options)
        return other

    # ------------------------------------------------------------------ adding data
    def add_data(self, frame, round_number=0, fill=None):
        """Add measurements: a table with a column per variable (named ``name`` or
        ``name (unit)``) and the result column. Replicates of a recipe are separate rows.
        Rows without a result are skipped. Returns the number of rows added.

        ``round_number=0`` marks data that existed before the campaign (day 0). ``fill`` gives a
        value for variables the table has no column for.
        """
        frame = self._rename(frame).reset_index(drop=True)
        if self.target not in frame.columns:
            raise CampaignError('The data has no "{}" column.'.format(self.target))
        where = ['Line {}'.format(i + 2) for i in range(len(frame))]      # line 1 is the header
        results = _numbers(list(frame[self.target]), self.target, where)
        keep = np.flatnonzero(~np.isnan(results))
        if not len(keep):
            raise CampaignError('No row of the data has a value in the "{}" column.'.format(self.target))
        recipes = self._clean(frame.iloc[keep], [where[i] for i in keep], fill)
        codes = self._recipe_codes(recipes)
        earlier = self.data.loc[self.data[col.ROUND] == int(round_number), col.POINT]
        offset = int(earlier.max()) if len(earlier) else 0
        values = {name: recipes[name].to_numpy() for name in self.names}
        values.update({col.ID: self._next_id() + codes, col.ROUND: np.full(len(keep), int(round_number)),
                       col.POINT: offset + codes + 1,
                       col.REPLICATE: pd.Series(codes).groupby(codes).cumcount().to_numpy() + 1,
                       self.target: results[keep]})
        for name in self.extra:
            if name in frame.columns:
                values[name] = frame[name].to_numpy()[keep]
        self._append(self._table(values, len(keep)))
        return len(keep)

    def add_round(self, batch, replicates=1, settings=None):
        """Register a round's recipes; their results come later. ``batch`` is a ``Proposal``
        or a table of recipes (optionally with the prediction columns). Each recipe gets a
        new ``Sample ID`` and ``replicates`` rows. Returns the round's number.
        """
        if isinstance(batch, Proposal):
            recipes, prediction = batch.recipes, batch.prediction
            settings = batch.settings if settings is None else settings
        else:
            recipes = self._rename(batch)
            names = dict(zip(col.prediction_columns(self.target), ('mean', 'low', 'high')))
            prediction = recipes[list(names)].rename(columns=names) if set(names) <= set(recipes.columns) else None
        replicates, n = int(replicates), len(recipes)
        if n < 1 or replicates < 1:
            raise CampaignError('A round needs at least one recipe and one replicate.')
        where = ['Data point {}'.format(i + 1) for i in range(n)]
        recipes = self._clean(recipes.reset_index(drop=True), where)
        space = self.space()
        over = np.flatnonzero(~space.feasible(space.encode(recipes)))
        if len(over):
            raise CampaignError('{} needs more than {}.'.format(where[int(over[0])], self.liquid.limit))
        round_number = self.next_round
        rows = np.repeat(np.arange(n), replicates)
        values = {name: recipes[name].to_numpy()[rows] for name in self.names}
        values.update({col.ID: self._next_id() + rows, col.ROUND: np.full(len(rows), round_number),
                       col.POINT: rows + 1, col.REPLICATE: np.tile(np.arange(1, replicates + 1), n)})
        if self.liquid is not None:
            wells = assign_wells(len(rows))
            values[col.PLATE] = [plate for plate, _ in wells]
            values[col.WELL] = [well for _, well in wells]
        if prediction is not None:
            for name, key in zip(col.prediction_columns(self.target), ('mean', 'low', 'high')):
                values[name] = _numbers(list(prediction[key]), name, where)[rows]
        self._append(self._table(values, len(rows)))
        self.history.append(_jsonable({
            'round': round_number, 'created': _now(), 'n_samples': n, 'replicates': replicates,
            'settings': settings or {'method': 'entered by hand'},
            'variables': [v.to_dict() for v in self.variables],
            'liquid_handling': self.liquid.to_dict() if self.liquid is not None else None,
            'model': self.model, 'model_options': self.model_options, 'seed': self.seed,
            'environment': environment(), 'pca': self.pca().to_dict()}))
        return round_number

    def redesign(self, variables, liquid_handling=UNCHANGED, fill=None):
        """The same campaign with other variables or liquid handling for the coming rounds.

        Ranges and steps can change freely: measured recipes keep their values. A variable
        that is new, or that was dropped for some rounds and comes back, needs its value in
        the rows that have none (``fill``: for example 0 if it was absent). A variable that
        is dropped stays in the data table but the model no longer sees it; to keep a
        component fixed instead, make it discrete with one step.
        """
        fill = fill or {}
        other = self.copy()
        if liquid_handling is not UNCHANGED:
            other.liquid = liquid_handling
        other.variables = check_variables(variables, self.target, model_spec(self.model).max_variables)
        kept = set(other.names)
        other.extra = [n for n in self.names if n not in kept] + [n for n in self.extra if n not in kept]
        if len(self.data):
            data = self.data.copy()
            for v in other.variables:
                known = data[v.name].astype(object) if v.name in data.columns else \
                    pd.Series([None] * len(data), index=data.index, dtype=object)
                missing = known.isna() | known.astype(str).str.strip().eq('')
                if not missing.any():
                    continue
                value = fill.get(v.name)
                if value is None or str(value).strip() == '':
                    raise CampaignError(
                        '{} is new: enter the value it had in the rounds already measured.'.format(v.name)
                        if missing.all() else
                        '{} has no value in {} of the {} rows so far: enter the value it had there.'.format(
                            v.name, int(missing.sum()), len(data)))
                known[missing] = value
                data[v.name] = known
            where = ['Sample {}'.format(i) for i in data[col.ID]]
            cleaned = other._clean(data, where)
            for name in other.names:
                data[name] = cleaned[name].to_numpy()
            other.data = data[other.columns]
        else:
            other.data = other._table({})
        return other

    # ------------------------------------------------------------------ the design
    def space(self):
        """The recipe space of the current variables; categories that only occur in the data are encoded too."""
        extra = {}
        if len(self.data):
            for v in self.variables:
                if v.kind == 'categorical':
                    seen = [c for c in dict.fromkeys(self.data[v.name]) if c not in v.categories]
                    if seen:
                        extra[v.name] = seen
        return RecipeSpace(self.variables, self.liquid, extra)

    def check(self):
        """Problems with the design: ``Issue('error', ...)`` blocks a round, warnings do not."""
        if self.liquid is None:
            return []
        issues = list(self.liquid.check(self.variables))
        if not any(i.blocking for i in issues):
            share = self.space().feasible_fraction()
            if share < MIN_FEASIBLE:
                issues.append(Issue('error', 'Fewer than 1 in 100 random recipes fit in {}, so a round cannot be '
                                    'drawn. Lower the maximums{} use more concentrated stocks.'.format(
                                        self.liquid.limit, ' or' if self.liquid.pipettable_volume_ul is None else
                                        ', make more of each sample pipettable or')))
        return issues

    def snap(self, recipes):
        """Recipes moved to the nearest allowed values: inside each range, on a step, a whole number of droplets."""
        space = self.space()
        return space.decode(space.snap(space.encode(self._clean(self._rename(recipes).reset_index(drop=True)))))

    def volume_ul(self, recipes):
        """Volume each recipe's components take, without the diluent (liquid handling only)."""
        if self.liquid is None:
            raise CampaignError('Volumes need liquid handling.')
        space = self.space()
        return self.liquid.volume_ul(space.droplets(space.encode(self._clean(self._rename(recipes).reset_index(
            drop=True)))))

    # ------------------------------------------------------------------ the model
    def fit(self):
        """Fit the campaign's model to every measurement so far."""
        measured = self.measured
        if len(measured) < 2:
            raise CampaignError('The model needs at least two measured results.')
        space = self.space()
        X = space.features(space.encode(measured[self.names]))
        y = measured[self.target].to_numpy(dtype=float)
        model = make_model(self.model, seed=self.seed, **self.model_options)
        model.fit(X, y)
        return Fitted(model, space, X, y)

    def predict(self, recipes, fitted=None):
        """``mean``, ``std`` and the 80% interval (``low``, ``high``) the model expects for each recipe."""
        fitted = fitted or self.fit()
        recipes = self._clean(self._rename(recipes).reset_index(drop=True))
        X = fitted.space.features(fitted.space.encode(recipes))
        prediction, low, high = fitted.model.predict_interval(X, col.INTERVAL)
        return pd.DataFrame({'mean': np.asarray(prediction.mean, dtype=float),
                             'std': np.asarray(prediction.std, dtype=float),
                             'low': np.asarray(low, dtype=float), 'high': np.asarray(high, dtype=float)})

    def batch_table(self, recipes, prediction=None):
        """Recipes as the table shown for a round: data point, variables and, if given, the prediction."""
        table = recipes[self.names].reset_index(drop=True).copy()
        table.insert(0, col.POINT, np.arange(1, len(table) + 1))
        if prediction is not None:
            for name, key in zip(col.prediction_columns(self.target), ('mean', 'low', 'high')):
                table[name] = np.asarray(prediction[key], dtype=float)
        return table

    def default_exploitation(self):
        """The tuned schedule: 0.5 in the first model-guided round, 1 (prediction only) afterwards."""
        guided = [r for r in self.history if (r.get('settings') or {}).get('method') == 'model']
        return LATER_EXPLOITATION if guided else FIRST_GUIDED_EXPLOITATION

    def propose(self, n_samples=10, exploitation=None, penalty=None, fantasy=False, hill_climb=False,
                pool_size=None, climb_starts=None, climb_max_moves=20, seed=None, guided=None, fitted=None):
        """Recipes for the next round.

        Without measurements (or with ``guided=False``) the round is a Latin hypercube:
        ``n_samples`` recipes spread over the whole design. With measurements the model is
        fitted, ``pool_size`` random candidate recipes are scored with
        ``exploitation * prediction + (1 - exploitation) * uncertainty`` and the batch is
        the best of them. Optional modules: ``penalty`` (0 = low to 1 = high; None = off)
        pushes the batch's recipes apart, ``fantasy`` chooses them one by one, pretending
        each earlier pick was measured at its prediction, and ``hill_climb`` improves the
        best candidates step by step first (``Proposal.climb``).

        ``seed`` defaults to the campaign's seed and the round number, so a round is
        reproducible. The campaign is not changed: register the batch with ``add_round``.
        """
        n_samples = int(n_samples)
        if n_samples < 1:
            raise CampaignError('A round needs at least one recipe.')
        blocking = [issue for issue in self.check() if issue.blocking]
        if blocking:
            raise CampaignError(blocking[0].message)
        round_number = self.next_round
        entropy = [self.seed, round_number] if seed is None else int(seed)
        rng = np.random.default_rng(entropy)
        space = fitted.space if fitted is not None else self.space()
        exclude = set(space.keys(space.encode(self.data[self.names]))) if len(self.data) else set()
        can_guide = len(self.measured) >= 2
        guided = can_guide if guided is None else bool(guided)
        settings = {'round': round_number, 'n_samples': n_samples, 'seed': entropy}
        warnings = []
        if not guided:
            recipes = space.decode(space.lhs(n_samples, rng, exclude))
            settings['method'] = 'latin hypercube'
            proposal = Proposal(self, round_number, False, recipes, None, settings, warnings)
        else:
            if not can_guide:
                raise CampaignError('A model-guided round needs at least two measured results.')
            exploitation = self.default_exploitation() if exploitation is None else float(exploitation)
            if not 0.0 <= exploitation <= 1.0:
                raise CampaignError('The exploitation ratio must be between 0 and 1.')
            fitted = fitted or self.fit()
            pool_size = int(pool_size or default_pool_size(self.model))
            pool = space.sample(pool_size, rng, exclude)
            if not len(pool):
                raise CampaignError('Every recipe of this design has been measured: widen a range or add a step.')
            prediction = fitted.model.predict(space.features(pool))
            radius, between = None, None
            if penalty is not None:
                between = spacing(space.embedding(pool))
                radius = penalty_radius(penalty, between)
            best_y = float(self.samples()[self.target].max())
            scored = _Scored(fitted, UCB(exploitation, 1.0 - exploitation), pool, prediction, exclude, best_y,
                             radius, bool(fantasy))
            recipes = space.decode(pool[scored.pick(pool, prediction, n_samples)])
            describe = getattr(fitted.model, 'describe', dict)()
            settings.update({'method': 'model', 'model': self.model, 'model_description': describe,
                             'exploitation': exploitation, 'exploration': 1.0 - exploitation,
                             'candidates_requested': pool_size, 'candidates_scored': int(len(pool)),
                             'penalty': None if penalty is None else float(penalty), 'penalty_radius': radius,
                             'candidate_spacing': between, 'fantasy': bool(fantasy), 'hill_climb': False,
                             'training_rows': int(len(fitted.y)), 'best_measured': best_y})
            proposal = Proposal(self, round_number, True, recipes, self.predict(recipes, fitted), settings, warnings,
                                _scored=scored)
            if hill_climb:
                proposal = proposal.climb(climb_starts, climb_max_moves)
        if len(proposal.recipes) < n_samples:
            proposal.warnings.append('Only {} new recipe(s) could be drawn from this design, fewer than the {} asked '
                                     'for.'.format(len(proposal.recipes), n_samples))
        if not len(proposal.recipes):
            raise CampaignError('No new recipe can be drawn from this design: widen a range or add a step.')
        return proposal

    # ------------------------------------------------------------------ analysis
    def pca(self, recipes=None):
        """The two-dimensional map of the campaign's recipes. It is fitted to the measured
        recipes; before there are two of those, to all recipes so far or to ``recipes``."""
        space = self.space()
        samples = self.samples()
        measured = samples.dropna(subset=[self.target])
        frame = measured if len(measured) >= 2 else samples
        if len(frame) < 2 and recipes is not None and len(recipes):
            frame = recipes
        names = [name for axis in space.axes for name in
                 ([axis.name] if axis.width == 1 else axis.feature_names)]
        E = space.embedding(space.encode(frame[self.names])) if len(frame) else np.empty((0, len(names)))
        return analysis.fit_pca(E, names)

    def project(self, pca, recipes):
        """Map coordinates ``(n, 2)`` of recipes on a fitted ``PCAMap``."""
        space = self.space()
        recipes = self._clean(self._rename(recipes).reset_index(drop=True))
        return pca.transform(space.embedding(space.encode(recipes)))

    def importance(self, fitted=None, repeats=3, max_rows=150, seed=0):
        """How much the model relies on each variable (``analysis.importance``)."""
        fitted = fitted or self.fit()
        return analysis.importance(fitted.model, fitted.X, fitted.y, fitted.space.feature_groups, repeats=repeats,
                                   max_rows=max_rows, seed=seed)

    def interactions(self, fitted=None, variables=None, n_pairs=30, seed=0):
        """How strongly pairs of variables interact in the model (``analysis.interactions``).
        ``variables`` limits the pairs to these names (the cost grows with their square)."""
        fitted = fitted or self.fit()
        groups = fitted.space.feature_groups
        if variables is not None:
            groups = [g for g in groups if g[0] in set(variables)]
        return analysis.interactions(fitted.model, fitted.X, groups, n_pairs=n_pairs, seed=seed)

    # ------------------------------------------------------------------ liquid handling
    def pipetting(self, round_number=None):
        """The pipetting plan of a round (by default the latest one)."""
        if self.liquid is None:
            raise CampaignError('Pipetting needs liquid handling.')
        round_number = max(self.rounds, default=0) if round_number is None else int(round_number)
        rows = self.data[self.data[col.ROUND] == round_number]
        if not len(rows) or rows[col.WELL].isna().any():
            raise CampaignError('Round {} has no wells assigned.'.format(round_number))
        return pipetting_plan(rows.reset_index(drop=True), self.variables, self.liquid)

    # ------------------------------------------------------------------ the workbook
    def _state(self):
        return _jsonable({'format': FORMAT, 'name': self.name, 'target': self.target, 'model': self.model,
                          'model_options': self.model_options, 'seed': self.seed, 'created': self.created,
                          'saved': _now(), 'variables': [v.to_dict() for v in self.variables],
                          'liquid_handling': self.liquid.to_dict() if self.liquid is not None else None,
                          'extra_columns': self.extra, 'history': self.history})

    def _instructions(self):
        pending = self.pending
        round_text = 'round {}'.format(int(pending[col.ROUND].max())) if len(pending) else 'the next round'
        return [
            'METIS campaign "{}": how to use this workbook'.format(self.name),
            '1. Run the experiments of {}: the rows of the "{}" sheet whose "{}" cell is empty (highlighted).'.format(
                round_text, workbook.EXPERIMENTS, self.target),
            '2. Type each measured value into the "{}" column. Every row is one measurement: replicates of a recipe '
            'share its Sample ID and have their own rows.'.format(self.target),
            '3. Every row needs a value. If a measurement failed, delete its whole row.',
            '4. If what you pipetted differs from the recipe, correct the concentrations in the row (in all '
            'replicates of that Sample ID). Do not change the Sample ID, Round, Data point or Replicate columns.',
            '5. Save the file as .xlsx and upload it in the app under "Continue a campaign" (or '
            'Campaign.load in Python) to get the next round.',
            'The "{}" column is the model\'s prediction for the recipe and "{}" / "{}" the range it expects 80% of '
            'measurements to fall in; they are empty for rounds that were not chosen by the model.'.format(
                col.predicted(self.target), col.LOW, col.HIGH),
            'Concentrations are in the units given in the column headers. Wells are named column then row: 1A is '
            'column 1, row A.',
            'The "{}" sheet lists the settings of every round. Keep the "{}" sheet as it is: the app reads the '
            'campaign from it.'.format(workbook.METADATA, workbook.STATE)]

    def _metadata(self):
        rows = [('Campaign', None), ('Name', self.name), ('Result to maximise', self.target),
                ('Model', MODELS[self.model].label), ('Model options', str(self.model_options or 'defaults')),
                ('Random seed', self.seed), ('Created', self.created), ('Saved', _now()),
                ('Rounds in this workbook', ', '.join(str(r) for r in self.rounds)),
                ('Variables', None)]
        for v in self.variables:
            if v.kind == 'continuous':
                text = 'continuous, {:g} to {:g}'.format(v.low, v.high)
            elif v.kind == 'discrete':
                text = 'discrete: ' + ', '.join('{:g}'.format(x) for x in v.levels)
            else:
                text = 'categorical: ' + ', '.join(v.categories)
                if v.volume_ul:
                    text += '; {:g} µL per sample'.format(v.volume_ul)
            if v.numeric:
                text += ' {}'.format(v.unit) if v.unit else ''
                text += '; stock {:g}{}'.format(v.stock, ' ' + v.unit if v.unit else '') if v.stock else ''
            rows.append((v.name, text))
        rows.append(('Liquid handling', None))
        if self.liquid is None:
            rows.append(('Used', 'no'))
        else:
            rows += [('Sample volume (µL)', self.liquid.sample_volume_ul),
                     ('Pipettable volume (µL)', self.liquid.pipettable_ul),
                     ('Droplet size (nL)', self.liquid.droplet_nl), ('Diluent', self.liquid.diluent)]
        for record in self.history:
            rows.append(('Round {}'.format(record['round']), None))
            rows += [('Generated', record['created']), ('Recipes', record['n_samples']),
                     ('Replicates', record['replicates'])]
            rows += [(key.replace('_', ' ').capitalize(), value if isinstance(value, (int, float, str)) else str(value))
                     for key, value in (record.get('settings') or {}).items()
                     if value is not None and key not in ('round', 'n_samples', 'replicates')]     # listed just above
            env = record.get('environment') or {}
            rows.append(('Environment', 'METIS {}, Python {}, {}; {}'.format(
                env.get('metis'), env.get('python'), env.get('platform'),
                ', '.join('{} {}'.format(k, v) for k, v in (env.get('packages') or {}).items()))))
            pca = record.get('pca')
            if pca:
                rows.append(('PCA map: variation on axis 1, axis 2', ', '.join(
                    '{:.1%}'.format(v or 0.0) for v in pca['explained_variance_ratio'])))
                for axis, weights in enumerate(pca['components'], start=1):
                    rows.append(('PCA axis {} weights'.format(axis), ', '.join(
                        '{}: {:.4g}'.format(name, w) for name, w in zip(pca['columns'], weights))))
                rows.append(('PCA centre', ', '.join('{}: {:.4g}'.format(name, m) for name, m in
                                                     zip(pca['columns'], pca['mean']))))
        return [(key, value if not isinstance(value, str) or len(value) <= CELL_LIMIT else
                 value[:CELL_LIMIT] + ' ... (cut; the State sheet has everything)') for key, value in rows]

    def to_workbook(self):
        """The campaign as Excel workbook bytes (see ``metis.campaign.workbook``)."""
        headers = {name: name for name in self.columns}
        headers.update({v.name: v.label for v in self.variables})
        names = [c for c in self.columns if c not in (col.PLATE, col.WELL) or self.data[c].notna().any()]
        return workbook.write(self.data[names], headers, self.target, self._state(), self._instructions(),
                              self._metadata())

    def save(self, path):
        """Write the workbook to ``path`` (.xlsx)."""
        with open(path, 'wb') as handle:
            handle.write(self.to_workbook())

    @classmethod
    def load(cls, path, require_results=True):
        """Read a campaign from a workbook file (see ``from_workbook``)."""
        with open(path, 'rb') as handle:
            return cls.from_workbook(handle.read(), require_results)

    @classmethod
    def from_workbook(cls, source, require_results=True):
        """The campaign stored in a workbook. With ``require_results`` (the default) every
        row must have a result, as it must before the next round can be proposed."""
        table, state = workbook.read(source)
        if not isinstance(state, dict) or state.get('format') != FORMAT:
            raise CampaignError('This workbook was written by another version of METIS and cannot be read.')
        if state.get('model') not in MODELS:
            raise CampaignError('This campaign uses the model "{}", which is not available here.'.format(
                state.get('model')))
        try:
            variables = [from_dict(v) for v in state['variables']]
            campaign = cls(variables, target=state['target'],
                           liquid_handling=LiquidHandling.from_dict(state.get('liquid_handling')),
                           model=state['model'], seed=state.get('seed', 0), name=state.get('name'),
                           model_options=state.get('model_options'))
        except (KeyError, TypeError, VariableError) as error:
            raise CampaignError('The "State" sheet of this workbook is damaged ({}).'.format(error)) from None
        campaign.created = state.get('created') or campaign.created
        campaign.history = list(state.get('history') or [])
        campaign._read_table(table, require_results)
        return campaign

    def _read_table(self, table, require_results):
        numbers = table.attrs.get('rows') or list(range(2, len(table) + 2))
        table = self._rename(table)
        required = [col.ID, col.ROUND, col.POINT, col.REPLICATE] + self.names + [self.target]
        missing = [name for name in required if name not in table.columns]
        if missing:
            raise CampaignError('The "{}" sheet has no "{}" column.'.format(workbook.EXPERIMENTS, missing[0]))
        if not len(table):
            raise CampaignError('The "{}" sheet has no rows.'.format(workbook.EXPERIMENTS))
        where = ['Row {}'.format(n) for n in numbers]
        values = {name: _whole(list(table[name]), name, where)
                  for name in (col.ID, col.ROUND, col.POINT, col.REPLICATE)}
        recipes = self._clean(table, where)
        values.update({name: recipes[name].to_numpy() for name in self.names})
        results = _numbers(list(table[self.target]), self.target, where)
        empty = np.flatnonzero(np.isnan(results))
        if require_results and len(empty):
            listed = ', '.join('row {} (sample {}, replicate {})'.format(numbers[i], values[col.ID][i],
                                                                         values[col.REPLICATE][i])
                               for i in empty[:8])
            raise CampaignError(
                '{} of the {} rows have no value in the "{}" column: {}{}. Fill in every result, or delete the rows '
                'of measurements that failed, and upload the workbook again.'.format(
                    len(empty), len(table), self.target, listed, ' and more' if len(empty) > 8 else ''))
        values[self.target] = results
        for name in col.prediction_columns(self.target):
            if name in table.columns:
                values[name] = _numbers(list(table[name]), name, where)
        if col.PLATE in table.columns:
            plates = _numbers(list(table[col.PLATE]), col.PLATE, where)
            values[col.PLATE] = [None if math.isnan(p) else int(p) for p in plates]
        if col.WELL in table.columns:
            values[col.WELL] = [clean_category(v) or None for v in table[col.WELL]]
        known = set(required) | set(col.prediction_columns(self.target)) | {col.PLATE, col.WELL}
        self.extra = [name for name in table.columns if name not in known]
        for name in self.extra:
            values[name] = list(table[name])
        data = self._table(values, len(table))
        pairs = data[[col.ID, col.REPLICATE]].duplicated()
        if pairs.any():
            i = int(np.flatnonzero(pairs.to_numpy())[0])
            raise CampaignError('{}: sample {} already has a replicate {}.'.format(
                where[i], values[col.ID][i], values[col.REPLICATE][i]))
        codes = self._recipe_codes(data[self.names])
        mixed = pd.Series(codes).groupby(data[col.ID].to_numpy()).nunique()
        if (mixed > 1).any():
            raise CampaignError('The rows of sample {} have different recipes. Replicates share a recipe: give a '
                                'changed row its own Sample ID.'.format(int(mixed.index[mixed > 1][0])))
        self.data = data
