"""Choosing the next batch: the generic engine and the v1 entry points built on it."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .acquisition import UCB, SelectionContext, TopK, rank_descending
from .design import check_repetitive, feature_columns, random_combination_generator
from .surrogates import Prediction, RegressorEnsemble
from .space import DesignSpace


@dataclass
class Suggestion:
    indices: np.ndarray          # rows of the candidate matrix, in pick order
    prediction: Prediction       # surrogate prediction at the picked rows
    scores: np.ndarray           # acquisition score at pick time


def suggest(surrogate, X_obs, y_obs, candidates, k, acquisition=None, selector=None, noise_var=None,
            embedding=None, blocked=None, is_new=None, fit=True):
    """Fit ``surrogate`` and pick ``k`` rows of ``candidates``.

    ``noise_var`` (per-sample variance of the observed yields) reaches the surrogate only
    if it accepts noise. ``embedding`` gives candidate coordinates for distance-based
    selectors; ``blocked`` marks candidates that must not be picked (e.g. measured).
    """
    acquisition = acquisition or UCB()
    selector = selector or TopK()
    if fit:
        surrogate.fit(X_obs, y_obs, noise_var=noise_var if surrogate.accepts_noise else None)
    prediction = surrogate.predict(candidates)
    best_y = float(np.max(y_obs)) if y_obs is not None and len(y_obs) else None
    ctx = SelectionContext(candidates=candidates, prediction=prediction, surrogate=surrogate,
                           acquisition=acquisition, X_obs=X_obs, y_obs=y_obs, noise_var=noise_var,
                           embedding=embedding, blocked=blocked, is_new=is_new, best_y=best_y)
    selection = selector.select(ctx, k)
    return Suggestion(selection.indices, prediction.take(selection.indices), selection.scores)


def bayesian_optimization(regressors_list,
                          data, label,
                          concentrations_limits,
                          final_order,
                          df_main,
                          reaction_vol_nl=20000, max_nl=13200, drop_size_nl=100,
                          exploitation=1, exploration=1, test_size=100, pool_size=100000, verbose=0, day=1,
                          days_range=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                          batch_ucb=False):
    """Main bayesian optimization function

    Parameters
    ----------
    regressors_list:
        a list consists of more than one regressor that has .fit and .predict feature

    data: pandas.DataFrame
        all previous day data

    label: pandas.DataFrame
        all previous day label

    exploitation: 1
        coefficient of focus on higher yield query

    exploration: 1
        coefficient of focus on a more informative query

    test_size: 100
        output combinations number

    pool_size: 100000
        how many random combinations to ask from the regressor list each round
        caution: this parameter highly affects executions time

    Returns
    -------
    chosen_combinations: pandas.DataFrame
        combinations that expected to improve yield

    if batch_ucb == True
    Returns
    -------
    best sample based on ucb: pandas.Series, best sample's expected value: float

    Notes
    -----
    Thin wrapper over the pluggable engine: the regressors become a ``RegressorEnsemble``
    surrogate, scoring is ``UCB(exploitation, exploration)`` and selection is ``TopK``
    with v1's repeat check. To swap the surrogate, acquisition or selector, use
    ``suggest_combinations`` or ``metis.optimize.suggest``.
    """
    surrogate = RegressorEnsemble(regressors_list).fit(data.values, label.values)

    # make random test data
    pool = random_combination_generator(concentrations_limits, number_of_combination=pool_size,
                                        reaction_vol_nl=reaction_vol_nl,
                                        max_nl=max_nl, drop_size_nl=drop_size_nl, make_csv=False, return_df=True)
    desired_cols = list(pool.columns)
    candidates = pool.values
    prediction = surrogate.predict(candidates)
    acquisition = UCB(exploitation, exploration)

    if batch_ucb:
        top = rank_descending(acquisition(prediction))[0]
        return pool[final_order].iloc[top:top + 1, :], prediction.mean[top]

    # check to don`t make repeated combinations, but it is not likely
    def is_new(i):
        return check_repetitive(pd.DataFrame([candidates[i]], columns=desired_cols), df_main)

    ctx = SelectionContext(candidates=candidates, prediction=prediction, surrogate=surrogate,
                           acquisition=acquisition, is_new=is_new)
    chosen = TopK().select(ctx, test_size).indices
    chosen_combinations = pd.DataFrame(candidates[chosen], columns=desired_cols)
    return chosen_combinations[final_order]


# Batch UCB on top of bayesian optimization
def batch_ucb(regressors_list,
                data, label,
                concentrations_limits,
                final_order,
                df_main,
                reaction_vol_nl=20000, max_nl=13200, drop_size_nl=100,
                exploitation=1, exploration=1, test_size=100, pool_size=100000, verbose=0, day=1,
                days_range=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20]):
    """Batch UCB on top of bayesian optimization function

    Parameters
    ----------
    regressors_list:
        a list consists of more than one regressor that has .fit and .predict feature

    data: pandas.DataFrame
        all previous day data

    label: pandas.DataFrame
        all previous day label

    exploitation: 1
        coefficient of focus on higher yield query

    exploration: 1
        coefficient of focus on a more informative query

    test_size: 100
        output combinations number

    pool_size: 100000
        how many random combinations to ask from the regressor list each round
        caution: this parameter highly affects executions time

    Returns
    -------
    chosen_combinations: pandas.DataFrame
        combinations that expected to improve yield

    Notes
    -----
    ``metis.acquisition.SequentialFantasy`` is the same idea for any surrogate, choosing
    from one shared pool.
    """

    final_samples = []

    for i in range(test_size):
        sample, expected_value = bayesian_optimization(regressors_list, data, label, concentrations_limits,
                                            final_order=final_order,
                                            df_main = df_main,
                                            reaction_vol_nl=reaction_vol_nl, max_nl=max_nl,
                                            drop_size_nl=drop_size_nl,
                                            exploitation=exploitation, exploration=exploration, test_size=test_size, pool_size=pool_size, verbose=0, day=day, days_range = days_range,
                                            batch_ucb=True)
        final_samples.append(sample)
        data = pd.concat([data, sample], axis=0).reset_index(drop=True)
        label = pd.concat([label, pd.DataFrame({'yield': [expected_value]})], axis=0).reset_index(drop=True)

    return pd.concat(final_samples)


def suggest_combinations(surrogate, data, label, concentrations_limits, test_size=20, acquisition=None,
                         selector=None, reaction_vol_nl=20000, max_nl=None, drop_size_nl=100, pool_size=10000,
                         label_std=None, n_replicates=3, seed=None):
    """Notebook-level entry point with pluggable parts.

    Same inputs as ``bayesian_optimization`` (``data`` with the model columns, ``label``
    with a ``yield`` column), but any ``Surrogate``, ``Acquisition`` and
    ``BatchSelector`` can be plugged in. Candidates come from ``DesignSpace.sample``,
    which never re-proposes measured combinations. ``label_std`` (e.g. the ``yield_std``
    column of the Results files) is converted to the variance of the replicate mean and
    passed to surrogates that accept noise.

    Example
    -------
    >>> from metis.surrogates import GPSurrogate
    >>> from metis.acquisition import UCB, LocalPenalization
    >>> next_day = suggest_combinations(GPSurrogate(), data, label, concentrations_limits,
    ...                                 acquisition=UCB(1, 1.41), selector=LocalPenalization(),
    ...                                 max_nl=9300, test_size=20)
    """
    space = DesignSpace.from_limits(concentrations_limits, reaction_vol_nl=reaction_vol_nl,
                                    drop_size_nl=drop_size_nl, max_nl=max_nl)
    columns = feature_columns(concentrations_limits)
    X_obs = data[columns].to_numpy(dtype=float)
    y_obs = np.asarray(label['yield'] if isinstance(label, pd.DataFrame) else label, dtype=float)
    noise_var = None
    if label_std is not None:
        noise_var = np.asarray(label_std, dtype=float) ** 2 / n_replicates
    if hasattr(surrogate, 'input_transform') and surrogate.input_transform is None:
        surrogate.input_transform = space.feature_transform()
    rng = np.random.default_rng(seed)
    pool_idx = space.sample(pool_size, rng, exclude=space.keys(space.from_features(X_obs)))
    result = suggest(surrogate, X_obs, y_obs, space.features(pool_idx), test_size, acquisition=acquisition,
                     selector=selector, noise_var=noise_var, embedding=space.embedding(pool_idx))
    chosen = space.frame(pool_idx[result.indices])
    chosen['pred_yield'] = result.prediction.mean
    chosen['pred_std'] = result.prediction.std
    return chosen
