"""Gold-regressor oracles built from the Borkowski and PURE datasets.

Borkowski (Fig. 1b of Pandi et al. 2022): 1,017 measured recipes of 11 varied factors,
4 levels each. Gold regressor as in the paper's Methods and the Fig. 1b notebook
(Examples/Cell_Free_System_Simulation/v2.4_UCB.ipynb): XGBRegressor(n_estimators=500,
learning_rate=0.01, max_depth=6, min_child_weight=1, subsample=0.8), trained on 80% of
the data with the other 20% used for early stopping (15 rounds, MAE). The notebook
re-split the full data for its validation set, so its test points leaked into training;
here the 20% is held out cleanly. The split and model seeds are fixed.

The design space is the 4-level grid Borkowski pipetted. Every split threshold of the
gold model falls strictly between grid levels (checked, ``continuous_equivalent``), so
the model is piecewise constant on the 4^11 cells. Its maximum over the continuous box
that Fig. 1b sampled therefore equals the maximum over this grid, found exactly by
enumerating all 4,194,304 points.

PURE (Matsuura et al. 2009, as processed in Examples/PURE): 1,094 recipes of 69
components. The gold regressor's hyperparameters come from the PURE notebook's
RandomizedSearchCV ("All Data R2 CV5": 48 draws, 5-fold, MAE, seeded here). It is then
trained with the same 80/20 early-stopping protocol, using the notebook's split seed 41.
Two design spaces:

* ``pool`` (default): the 1,094 measured recipes, as in the PURE notebook's
  retrospective simulation. The maximum over the pool is exact.
* ``grid``: the product of each component's measured levels (about 1e75 points). Most
  of it is far from any data, and the maximum is only the best found by search.
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

from metis.space import DesignSpace, Factor

from .base import Oracle, PooledSpace

ROOT = pathlib.Path(__file__).resolve().parents[2]
BORKOWSKI_DIR = ROOT / 'Examples' / 'Cell_Free_System_Simulation' / 'no_controls'
PURE_CSV = ROOT / 'Examples' / 'PURE' / 'Data' / 'processed' / 'PURE_cleared.csv'
CACHE = ROOT / 'artifacts' / 'oracles'

BORKOWSKI_GOLD_PARAMS = dict(objective='reg:squarederror', n_estimators=500, learning_rate=0.01, max_depth=6,
                             min_child_weight=1, subsample=0.8)
PURE_SEARCH_GRID = {
    "learning_rate": [0.1, 0.3],
    "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
    "subsample": [0.9, 1.0],
    "n_estimators": [100, 300],
    "reg_lambda": [1.0],
    "gamma": [0, 0.1],
    "min_child_weight": [1, 2]}


# --------------------------------------------------------------------------- data
def load_borkowski():
    """All 1,017 recipes: 11 factor columns, ``yield`` and ``yield_std`` (triplicates)."""
    frames = [pd.read_csv(BORKOWSKI_DIR / 'plate_AL_{}_raw_yield_and_std.csv'.format(i)).assign(plate=i)
              for i in range(1, 11)]
    return pd.concat(frames, ignore_index=True)


def load_pure():
    return pd.read_csv(PURE_CSV)


BORKOWSKI_FACTORS = ['nad', 'folinic_acid', 'coa', 'nucleo_mix', 'spermidin', 'pga', 'aa', 'trna', 'mg_gluta',
                     'camp', 'K_gluta']


def _levels_space(X, name, log_scale='auto'):
    factors = []
    for col in X.columns:
        levels = np.unique(X[col].to_numpy(dtype=float))
        positive = levels[levels > 0]
        use_log = (len(positive) > 2 and positive.max() / positive.min() >= 100) if log_scale == 'auto' else log_scale
        factors.append(Factor(col, levels, drops=None, log_scale=bool(use_log)))
    return factors


def _fit_gold(X, y, params, seed, split_seed, early_stopping_rounds=15):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=split_seed)
    model = XGBRegressor(**params, early_stopping_rounds=early_stopping_rounds, eval_metric='mae',
                         random_state=seed, n_jobs=1)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model, r2_score(y_va, model.predict(X_va))


def _cv_r2(X, y, params, seed, folds=5):
    """5-fold CV R2 of the same recipe (80/20 early stopping inside each training fold)."""
    scores = []
    for k, (tr, te) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(X)):
        model, _ = _fit_gold(X[tr], y[tr], params, seed, split_seed=seed + k)
        scores.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(scores)), float(np.std(scores))


def _split_thresholds(model):
    trees = model.get_booster().trees_to_dataframe()
    splits = trees[trees.Feature != 'Leaf']
    return {f: np.unique(g.Split.to_numpy(dtype=np.float32)) for f, g in splits.groupby('Feature')}


def _cells_contain_levels(model, space):
    """True if the tree ensemble is constant on cells that each contain a grid level.

    Splits are ``x < t`` in float32, so per factor the box splits into cells
    ``[t_i, t_{i+1})``. If every cell within [min level, max level] holds a level, each
    cell's value is attained at a grid point and the continuous-box maximum equals the
    grid maximum.
    """
    thresholds = _split_thresholds(model)
    for f_index, factor in enumerate(space.factors):
        levels = factor.levels.astype(np.float32)
        t = thresholds.get('f{}'.format(f_index), np.array([], dtype=np.float32))
        cuts = np.unique(t[(t > levels.min()) & (t <= levels.max())])
        edges = np.concatenate([[levels.min()], cuts, [np.inf]])
        for a, b in zip(edges[:-1], edges[1:]):
            if not ((levels >= a) & (levels < b)).any():
                return False
    return True


class GoldRegressorOracle(Oracle):
    """Noise-free yield = the gold regressor's prediction at a grid point."""

    def __init__(self, oid, space, model, f_max, x_max, max_is_exact, meta):
        self.id = oid
        self.space = space
        self.model = model
        self.f_max = float(f_max)
        self.x_max = np.asarray(x_max, dtype=np.int64)
        self.max_is_exact = max_is_exact
        self.meta = meta

    def __call__(self, idx):
        return self.model.predict(self.space.features(np.atleast_2d(idx))).astype(float)

    # ------------------------------------------------------------------ caching
    def save(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path.with_suffix('.model.json')))
        json.dump({'id': self.id, 'f_max': self.f_max, 'x_max': self.x_max.tolist(),
                   'max_is_exact': self.max_is_exact, 'meta': self.meta}, open(path, 'w'), indent=1)

    @classmethod
    def load(cls, path, space):
        path = pathlib.Path(path)
        info = json.load(open(path))
        model = XGBRegressor()
        model.load_model(str(path.with_suffix('.model.json')))
        return cls(info['id'], space, model, info['f_max'], info['x_max'], info['max_is_exact'], info['meta'])


# --------------------------------------------------------------------------- Borkowski
def borkowski_space():
    data = load_borkowski()
    return DesignSpace(_levels_space(data[BORKOWSKI_FACTORS], 'borkowski'), name='borkowski')


def build_borkowski(seed=0, cache=True):
    path = CACHE / 'borkowski_gold.json'
    space = borkowski_space()
    if cache and path.exists():
        return GoldRegressorOracle.load(path, space)
    data = load_borkowski()
    X = data[BORKOWSKI_FACTORS].to_numpy(dtype=float)
    y = data['yield'].to_numpy(dtype=float)
    model, holdout_r2 = _fit_gold(X, y, BORKOWSKI_GOLD_PARAMS, seed=seed, split_seed=seed)
    cv_mean, cv_sd = _cv_r2(X, y, BORKOWSKI_GOLD_PARAMS, seed)

    # Exact maximum: enumerate the whole 4^11 grid.
    best, best_x = -np.inf, None
    for chunk in space.enumerate(chunk=500_000):
        pred = model.predict(space.features(chunk))
        i = int(np.argmax(pred))
        if pred[i] > best:
            best, best_x = float(pred[i]), chunk[i]

    between = _cells_contain_levels(model, space)
    meta = {'source': 'Borkowski et al. 2020 via METIS Examples/Cell_Free_System_Simulation', 'n_data': len(y),
            'n_factors': len(BORKOWSKI_FACTORS), 'gold_params': BORKOWSKI_GOLD_PARAMS,
            'best_iteration': int(model.best_iteration), 'holdout_r2': float(holdout_r2), 'cv5_r2_mean': cv_mean,
            'cv5_r2_sd': cv_sd, 'continuous_equivalent': bool(between), 'data_max_yield': float(y.max()),
            'grid_points_enumerated': space.size}
    oracle = GoldRegressorOracle('borkowski', space, model, best, best_x, True, meta)
    if cache:
        oracle.save(path)
    return oracle


# --------------------------------------------------------------------------- PURE
def _pure_xy():
    data = load_pure()
    X = data.drop(columns='yield')
    return X, data['yield'].to_numpy(dtype=float)


def pure_space(mode='pool'):
    X, _ = _pure_xy()
    factors = _levels_space(X, 'pure')
    if mode == 'grid':
        return DesignSpace(factors, name='pure_grid')
    proto = DesignSpace(factors)
    return PooledSpace(factors, proto.from_features(X.to_numpy(dtype=float)), name='pure_pool')


def _pure_gold_params(seed):
    X, y = _pure_xy()
    search = RandomizedSearchCV(XGBRegressor(objective='reg:squarederror', n_jobs=1), PURE_SEARCH_GRID, n_iter=48,
                                cv=5, scoring='neg_mean_absolute_error', random_state=seed, n_jobs=1)
    search.fit(X.to_numpy(dtype=float), y)
    params = dict(objective='reg:squarederror', **search.best_params_)
    # Early stopping decides the number of trees; the search value becomes a generous cap.
    params['n_estimators'] = 2000
    return params, {k: (v.item() if hasattr(v, 'item') else v) for k, v in search.best_params_.items()}


def _coordinate_ascent(oracle_fn, space, start, max_sweeps=20):
    x = start.copy()
    fx = float(oracle_fn(x[None])[0])
    for _ in range(max_sweeps):
        improved = False
        for d in range(space.n_dims):
            trial = np.repeat(x[None], space.cardinalities[d], axis=0)
            trial[:, d] = np.arange(space.cardinalities[d])
            values = oracle_fn(trial)
            j = int(np.argmax(values))
            if values[j] > fx + 1e-12:
                x, fx, improved = trial[j], float(values[j]), True
        if not improved:
            break
    return fx, x


def build_pure(mode='pool', seed=0, cache=True):
    path = CACHE / 'pure_gold_{}.json'.format(mode)
    space = pure_space(mode)
    if cache and path.exists():
        return GoldRegressorOracle.load(path, space)
    X_df, y = _pure_xy()
    X = X_df.to_numpy(dtype=float)
    params, searched = _pure_gold_params(seed)
    model, holdout_r2 = _fit_gold(X, y, params, seed=seed, split_seed=41)
    cv_mean, cv_sd = _cv_r2(X, y, params, seed)
    oracle = GoldRegressorOracle('pure_' + mode, space, model, 0.0, np.zeros(space.n_dims), True, {})
    data_idx = DesignSpace(space.factors).from_features(X)
    pred = oracle(data_idx)
    best_i = int(np.argmax(pred))
    best, best_x, exact = float(pred[best_i]), data_idx[best_i], True
    if mode == 'grid':
        # Best found: coordinate ascent from the 30 best measured recipes and 100 random grid points.
        exact = False
        starts = np.vstack([data_idx[np.argsort(pred)[::-1][:30]],
                            space.sample(100, np.random.default_rng(seed))])
        for s in starts:
            value, x = _coordinate_ascent(oracle, space, s)
            if value > best:
                best, best_x = value, x
    oracle.f_max, oracle.x_max, oracle.max_is_exact = best, np.asarray(best_x), exact
    oracle.meta = {'source': 'Matsuura et al. 2009 via METIS Examples/PURE', 'n_data': len(y),
                   'n_factors': X.shape[1], 'mode': mode, 'searched_params': searched, 'gold_params': params,
                   'best_iteration': int(model.best_iteration), 'holdout_r2': float(holdout_r2),
                   'cv5_r2_mean': cv_mean, 'cv5_r2_sd': cv_sd, 'data_max_yield': float(y.max())}
    if cache:
        oracle.save(path)
    return oracle


REAL_ORACLES = {
    'borkowski': build_borkowski,
    'pure_pool': lambda **kw: build_pure('pool', **kw),
    'pure_grid': lambda **kw: build_pure('grid', **kw),
}
