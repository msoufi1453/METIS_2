"""The GUI's logic, kept free of Streamlit so it can be tested.

A table of recipes (one column per factor, one target column) becomes a design space:
numeric columns are level factors (their observed values), text columns are categorical
factors. The next batch is proposed with the same parts the benchmark tests: a surrogate,
the UCB score with an exploration schedule, optional hill-climbing, and a batch rule (top
scores, a soft distance penalty or a fantasy batch).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from metis import hillclimb
from metis.acquisition import UCB, SequentialFantasy, SoftPenalty, TopK
from metis.optimize import suggest
from metis.space import DesignSpace, Factor, _auto_log
from metis.surrogates import Prediction

MODELS = {
    'TabPFN-3.5 (GPU recommended)': {'type': 'tabpfn', 'version': 'v3.5', 'n_estimators': 8, 'device': 'auto'},
    'XGBoost ensemble (METIS v1)': {'type': 'xgb_ensemble', 'n_iter': 200, 'ensemble_len': 20},
    'Gaussian process': {'type': 'gp_matern'},
    'Linear regression': {'type': 'linear_ensemble'},
}
BATCH_RULES = ('Top scores (no module)', 'Distance penalty: low', 'Distance penalty: high', 'Fantasy batch')
PENALTY_SCALE = {'Distance penalty: low': 1.0, 'Distance penalty: high': 3.0}


def read_table(source, name=None):
    """A CSV or Excel file (path or uploaded file object) as a DataFrame."""
    name = str(name or getattr(source, 'name', source)).lower()
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(source)
    return pd.read_csv(source)


def guess_columns(df, target):
    """Default factor columns: everything but the target and columns that look like IDs."""
    numeric, categorical = [], []
    for c in df.columns:
        if c == target:
            continue
        values = df[c].dropna()
        if pd.api.types.is_numeric_dtype(values):
            if values.nunique() > 1:
                numeric.append(c)
        elif 1 < values.nunique() <= 255 and values.nunique() < 0.5 * len(values):
            categorical.append(c)     # many distinct strings (one per row) are IDs, not factors
    return numeric, categorical


@dataclass
class Design:
    """A table mapped onto a design space."""
    frame: pd.DataFrame
    target: str
    numeric: list
    categorical: list
    space: DesignSpace = None
    idx: np.ndarray = None                  # grid indices of every row
    measured: np.ndarray = None             # rows with a target value
    columns: list = field(default_factory=list)


def build_design(df, target, numeric, categorical):
    df = df.dropna(subset=list(numeric) + list(categorical)).reset_index(drop=True)
    factors, blocks = [], []
    for c in numeric:
        values = df[c].to_numpy(float)
        levels = np.unique(values)
        if len(levels) > 255:
            raise ValueError('{} has {} distinct values; at most 255 levels per factor'.format(c, len(levels)))
        factors.append(Factor(c, levels, log_scale=_auto_log(levels)))
        blocks.append(np.searchsorted(factors[-1].levels, values)[:, None])
    for c in categorical:
        names = tuple(sorted(df[c].astype(str).unique()))
        factors.append(Factor(c, [1.0], alternatives=names))
        position = {n: i for i, n in enumerate(names)}
        blocks.append(np.zeros((len(df), 1), dtype=np.int64))
        blocks.append(df[c].astype(str).map(position).to_numpy()[:, None])
    space = DesignSpace(factors, name='uploaded')
    idx = np.hstack(blocks).astype(np.int64)
    measured = df[target].notna().to_numpy() if target in df else np.zeros(len(df), bool)
    return Design(df, target, list(numeric), list(categorical), space, idx, measured, space.columns)


def spacing(design, max_points=20000, seed=0):
    """Median distance from a recipe to its nearest other recipe (embedding units)."""
    from sklearn.neighbors import NearestNeighbors
    E = design.space.embedding(design.idx)
    if len(E) > max_points:
        E = E[np.random.default_rng(seed).choice(len(E), size=max_points, replace=False)]
    if len(E) < 2:
        return 1.0
    d = NearestNeighbors(n_neighbors=2).fit(E).kneighbors(E)[0][:, 1]
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def pca(design, idx=None):
    """PCA of the recipes in embedding coordinates: the PCA object, the rows' coordinates and,
    per factor, how strongly it loads on PC1 and PC2 (a factor's one-hot columns combined)."""
    from sklearn.decomposition import PCA
    E = design.space.embedding(design.idx)
    model = PCA(n_components=2).fit(E)
    xy = model.transform(design.space.embedding(design.idx if idx is None else idx))
    names = []
    for f, kind in design.space.dims:
        factor = design.space.factors[f]
        names += [factor.name] if kind == 'level' else [factor.name] * len(factor.alternatives)
    load = pd.DataFrame(model.components_.T, columns=['PC1', 'PC2'])
    load['factor'] = names
    loadings = load.groupby('factor', sort=False).agg(lambda v: float(np.sqrt(np.sum(v ** 2))))
    return model, xy, loadings


def schedule_weights(model_cfg, round_number, n_rounds):
    """(exploitation, exploration) for this round: TabPFN's tuned schedule, METIS's for other models."""
    from metis import schedules
    from metis_bench.arms import tabpfn_schedule
    if model_cfg['type'] == 'tabpfn':
        try:
            sched = schedules.make(tabpfn_schedule())
        except FileNotFoundError:
            sched = schedules.make({'type': 'legacy'})
    else:
        sched = schedules.make({'type': 'legacy'})
    exploitation, exploration = sched.weights(round_number, n_rounds, [])
    return (1.0 if exploitation is None else float(exploitation)), float(exploration)


def make_model(model_cfg, design, seed=0):
    from metis_bench.arms import make_surrogate
    return make_surrogate(dict(model_cfg), design.space, np.random.default_rng(seed), False)


def importance(design, model_cfg, seed=0, repeats=3):
    """How much the fitted model relies on each factor: the drop in R^2 on the measured
    recipes when that factor's column is shuffled (in-sample, so it describes the model)."""
    from sklearn.metrics import r2_score
    X = design.space.features(design.idx[design.measured])
    y = design.frame.loc[design.measured, design.target].to_numpy(float)
    model = make_model(model_cfg, design, seed).fit(X, y)
    base = r2_score(y, model.predict(X).mean)
    rng = np.random.default_rng(seed)
    out, col = {}, 0
    for factor in design.space.factors:
        width = 1 + (len(factor.alternatives) if factor.alternatives else 0)
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, col:col + width] = Xp[rng.permutation(len(X)), col:col + width]
            drops.append(base - r2_score(y, model.predict(Xp).mean))
        out[factor.name] = float(np.mean(drops))
        col += width
    return pd.Series(out, name='R2 drop when shuffled').sort_values(ascending=False)


def recipes(space, idx):
    """Grid indices as a readable table: the level of each numeric factor, the name of each category."""
    out = {}
    for d, (f, kind) in enumerate(space.dims):
        factor = space.factors[f]
        if not factor.alternatives:
            out[factor.name] = factor.levels[idx[:, d]]
        elif kind == 'alt':
            out[factor.name] = [factor.alternatives[k] for k in idx[:, d]]
    return pd.DataFrame(out)


@dataclass
class Proposal:
    table: pd.DataFrame            # the batch, in pick order, with predicted mean and std
    pool: np.ndarray               # candidate grid indices scored this round
    scores: np.ndarray             # UCB score of each candidate
    chosen: np.ndarray             # positions in ``pool``
    n_random: int                  # candidates before hill-climbing added its end points
    climb: object = None           # metis.hillclimb.Climb, if hill-climbing was on
    weights: tuple = None          # (exploitation, exploration)


def propose(design, model_cfg, batch_size=10, round_number=2, n_rounds=10, pool_size=10000, source='grid',
            rule=BATCH_RULES[0], hill=False, n_starts=10, max_steps=20, seed=0):
    """Next batch. ``source``: 'grid' (random unmeasured combinations of the observed levels)
    or 'rows' (the table's rows without a target value)."""
    space, rng = design.space, np.random.default_rng(seed)
    measured_idx = design.idx[design.measured]
    if source == 'rows':
        pool = design.idx[~design.measured]
        if not len(pool):
            raise ValueError('every row has a target value: no unmeasured rows to choose from')
        from metis_bench.oracles.base import PooledSpace
        space = PooledSpace(space.factors, pool, name='rows')
    else:
        pool = space.sample(pool_size, rng, exclude=space.keys(measured_idx))
    X_obs = space.features(measured_idx)
    y_obs = design.frame.loc[design.measured, design.target].to_numpy(float)
    exploitation, exploration = schedule_weights(model_cfg, round_number, n_rounds)
    acquisition = UCB(exploitation, exploration)
    model = make_model(model_cfg, design, seed)
    model.fit(X_obs, y_obs)
    pred = model.predict(space.features(pool))
    n_random, climbed = len(pool), None
    if hill:
        climbed = hillclimb.climb(space, model, acquisition, pool, pred, space.keys(measured_idx),
                                  best_y=float(np.max(y_obs)), n_starts=n_starts, max_steps=max_steps)
        if len(climbed.new):
            pool = np.vstack([pool, climbed.new])
            pred = Prediction(np.concatenate([pred.mean, climbed.new_prediction.mean]),
                              np.concatenate([pred.std, climbed.new_prediction.std]))
    if rule in PENALTY_SCALE:
        selector = SoftPenalty(PENALTY_SCALE[rule] * spacing(design))
    elif rule == 'Fantasy batch':
        selector = SequentialFantasy()
    else:
        selector = TopK()
    result = suggest(model, X_obs, y_obs, space.features(pool), batch_size, acquisition=acquisition,
                     selector=selector, embedding=space.embedding(pool), fit=False, prediction=pred)
    table = recipes(space, pool[result.indices])[list(design.numeric) + list(design.categorical)]
    table.insert(0, 'pick', np.arange(1, len(table) + 1))
    table['predicted ' + design.target] = result.prediction.mean
    table['uncertainty (std)'] = result.prediction.std
    table['from hill-climbing'] = result.indices >= n_random
    scores = np.asarray(acquisition(pred, best_y=float(np.max(y_obs))), float)
    return Proposal(table, pool, scores, result.indices, n_random, climbed, (exploitation, exploration))
