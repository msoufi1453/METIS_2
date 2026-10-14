"""METIS's incumbent surrogate: an ensemble of regressors whose spread is the uncertainty."""
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from .base import Prediction, Surrogate, as_1d_float, as_2d_float, inverse_variance_weights

# Hyperparameter space searched by the METIS notebooks (RandomCV = True).
XGB_PARAM_GRID_V1 = {
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
    "subsample": [0.6, 0.8, 0.9, 1.0],
    "max_depth": [2, 3, 4, 6, 8],
    "n_estimators": [10, 20, 40, 60, 80, 100, 300, 500],
    "reg_lambda": [1, 1.5, 2],
    "gamma": [0, 0.1, 0.4, 0.6],
    "min_child_weight": [1, 2, 4]}


def default_xgb_ensemble():
    """The notebooks' fallback ensemble (RandomCV = False): 20 fixed XGBoost settings."""
    return [XGBRegressor(objective='reg:squarederror', n_estimators=n, learning_rate=l, max_depth=6,
                         min_child_weight=4, subsample=0.9, gamma=0.4, colsample_bytree=0.9)
            for n in (10, 20, 30, 40, 50) for l in (0.01, 0.03, 0.1, 0.2)]


class RegressorEnsemble(Surrogate):
    """Any list of regressors with ``fit``/``predict``; mean and sample std across members.

    The statistics go through the same pandas reductions as v1 ``bayesian_optimization``
    so rankings, including float32 ties, are bit-identical to the original.

    ``noise_weighting``: with per-sample ``noise_var``, members are fitted with
    inverse-variance ``sample_weight``.
    """

    name = 'regressor_ensemble'

    def __init__(self, regressors, noise_weighting=False):
        self.regressors = list(regressors)
        self.noise_weighting = noise_weighting

    @property
    def accepts_noise(self):
        return self.noise_weighting

    def fit(self, X, y, noise_var=None):
        # X and y are passed through untouched: v1 handed label.values (shape (n, 1)) to fit.
        kwargs = {}
        if self.noise_weighting and noise_var is not None:
            kwargs['sample_weight'] = inverse_variance_weights(noise_var)
        for regressor in self.regressors:
            regressor.fit(X, y, **kwargs)
        return self

    def member_predictions(self, X):
        frame = pd.DataFrame(index=pd.RangeIndex(len(X)))
        for index, regressor in enumerate(self.regressors):
            frame['pred_yield_{}'.format(index)] = regressor.predict(X)
        return frame

    def predict(self, X):
        frame = self.member_predictions(X)
        return Prediction(mean=frame.mean(axis=1).to_numpy(), std=frame.std(axis=1).to_numpy())

    def describe(self):
        return {'name': self.name, 'members': len(self.regressors), 'noise_weighting': self.noise_weighting}


class XGBEnsemble(RegressorEnsemble):
    """The METIS incumbent as the notebooks run it.

    Every ``fit`` repeats ``RandomizedSearchCV`` (5-fold, MAE) over ``param_grid`` and
    keeps the ``ensemble_len`` best settings as ensemble members, all trained on the full
    data. The notebook default is ``n_iter=200``; benchmark pilots use fewer iterations
    and say so in their run metadata.
    """

    name = 'xgb_ensemble'

    def __init__(self, n_iter=200, ensemble_len=20, cv=5, param_grid=None, random_state=None,
                 n_jobs=1, noise_weighting=False):
        super().__init__([], noise_weighting=noise_weighting)
        self.n_iter = n_iter
        self.ensemble_len = ensemble_len
        self.cv = cv
        self.param_grid = copy.deepcopy(param_grid or XGB_PARAM_GRID_V1)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_params_ = None

    def fit(self, X, y, noise_var=None):
        X, y = as_2d_float(X), as_1d_float(y)
        fit_kwargs = {}
        if self.noise_weighting and noise_var is not None:
            fit_kwargs['sample_weight'] = inverse_variance_weights(noise_var)
        cv = min(self.cv, len(y))
        random_state = self.random_state
        if isinstance(random_state, np.random.Generator):
            # A generator gives every refit its own reproducible hyperparameter sample.
            random_state = int(random_state.integers(2 ** 31 - 1))
        search = RandomizedSearchCV(
            estimator=XGBRegressor(objective='reg:squarederror', n_jobs=1),
            param_distributions=self.param_grid, cv=cv, scoring='neg_mean_absolute_error',
            n_jobs=self.n_jobs, n_iter=self.n_iter, random_state=random_state)
        search.fit(X, y, **fit_kwargs)
        results = pd.DataFrame(search.cv_results_).sort_values('mean_test_score', ascending=False)
        self.best_params_ = list(results.params.iloc[:self.ensemble_len])
        self.regressors = [XGBRegressor(objective='reg:squarederror', n_jobs=1, **params)
                           for params in self.best_params_]
        return super().fit(X, y, noise_var=noise_var)

    def refit(self, X, y, noise_var=None):
        """Retrain the current members without a new hyperparameter search."""
        return RegressorEnsemble.fit(self, as_2d_float(X), as_1d_float(y), noise_var=noise_var)

    def describe(self):
        return {'name': self.name, 'n_iter': self.n_iter, 'ensemble_len': self.ensemble_len, 'cv': self.cv,
                'noise_weighting': self.noise_weighting}
