import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from metis import DesignSpace, suggest, suggest_combinations
from metis.acquisition import GPBUCB, UCB, LocalPenalization, SequentialFantasy, TopK
from metis.surrogates import GPSurrogate, RegressorEnsemble, Surrogate, XGBEnsemble
from tests.regression.scenario import LIMITS, MAX_NL, desired_columns, toy_yield


def peaked(X):
    return np.exp(-8 * ((X - 0.6) ** 2).sum(axis=1))


@pytest.fixture(scope='module')
def grid_problem():
    rng = np.random.default_rng(0)
    X_obs = rng.random((25, 2))
    g = np.linspace(0, 1, 40)
    candidates = np.array([[a, b] for a in g for b in g])
    return X_obs, peaked(X_obs), candidates


def test_regressor_ensemble_statistics():
    X = np.arange(10.0).reshape(-1, 1)
    model = RegressorEnsemble([LinearRegression(), LinearRegression()]).fit(X, 2 * X[:, 0])
    pred = model.predict(X)
    assert np.allclose(pred.mean, 2 * X[:, 0]) and np.allclose(pred.std, 0)


def test_gp_interpolates_and_covariance_is_consistent(grid_problem):
    X_obs, y_obs, candidates = grid_problem
    gp = GPSurrogate(n_restarts=1).fit(X_obs, y_obs)
    pred = gp.predict(X_obs)
    assert np.abs(pred.mean - y_obs).max() < 0.05
    sub = candidates[::97]
    cov = gp.posterior_covariance(sub, sub)
    assert np.allclose(np.sqrt(np.clip(np.diag(cov), 0, None)), gp.predict(sub).std, rtol=1e-3, atol=1e-6)


def test_gp_uses_replicate_noise(grid_problem):
    X_obs, y_obs, _ = grid_problem
    noisy = np.full(len(y_obs), 1e-6)
    noisy[0] = 10.0  # a point we barely trust
    gp = GPSurrogate(n_restarts=0).fit(X_obs, y_obs + np.eye(len(y_obs))[0] * 3, noise_var=noisy)
    assert abs(gp.predict(X_obs[:1]).mean[0] - y_obs[0]) < 1.0


def _min_pairwise(points):
    d = np.linalg.norm(points[:, None] - points[None], axis=-1)
    return d[np.triu_indices(len(points), 1)].min()


def test_selectors_return_distinct_unblocked_points(grid_problem):
    X_obs, y_obs, candidates = grid_problem
    blocked = np.zeros(len(candidates), bool)
    blocked[:100] = True
    picks = {}
    for selector in [TopK(), LocalPenalization(), LocalPenalization(min_radius=0.1), GPBUCB(), SequentialFantasy()]:
        result = suggest(GPSurrogate(n_restarts=0), X_obs, y_obs, candidates, 8, acquisition=UCB(1, 0.5),
                         selector=selector, embedding=candidates, blocked=blocked)
        assert len(set(result.indices)) == 8 and not blocked[result.indices].any()
        picks[selector.name + str(getattr(selector, 'min_radius', ''))] = candidates[result.indices]
    # Where the GP is confident around its optimum, paper LP and GP-BUCB may legitimately
    # cluster like top-k; they are never tighter. The radius floor enforces spacing.
    assert _min_pairwise(picks['local_penalization0.0']) >= _min_pairwise(picks['topk']) - 1e-12
    assert _min_pairwise(picks['local_penalization0.1']) > 0.08
    assert _min_pairwise(picks['gp_bucb']) >= _min_pairwise(picks['topk']) - 1e-12


def test_xgb_ensemble_incumbent_runs():
    rng = np.random.default_rng(3)
    X = rng.random((40, 3))
    y = X @ np.array([1.0, -2.0, 0.5])
    model = XGBEnsemble(n_iter=6, ensemble_len=4, random_state=rng).fit(X, y)
    assert len(model.regressors) == 4
    assert model.predict(X).std.shape == (40,)


def test_suggest_combinations_with_pluggable_parts():
    space = DesignSpace.from_limits(LIMITS, reaction_vol_nl=20000, drop_size_nl=100, max_nl=MAX_NL)
    idx = space.sample(20, np.random.default_rng(4))
    data = space.frame(idx)
    label = toy_yield(data)
    out = suggest_combinations(GPSurrogate(n_restarts=0), data, label, LIMITS, test_size=10, max_nl=MAX_NL,
                               selector=LocalPenalization(), label_std=np.full(20, 0.1), pool_size=2000, seed=0)
    assert len(out) == 10 and list(out.columns[:-2]) == desired_columns(LIMITS)
    measured = set(map(tuple, data.to_numpy()))
    assert not any(tuple(r) in measured for r in out[desired_columns(LIMITS)].to_numpy())


def test_custom_surrogate_plugs_in(grid_problem):
    class Constant(Surrogate):
        def fit(self, X, y, noise_var=None):
            self.c = float(np.mean(y))
            return self

        def predict(self, X):
            from metis.surrogates import Prediction
            return Prediction(np.full(len(X), self.c), np.linspace(0, 1, len(X)))

    X_obs, y_obs, candidates = grid_problem
    result = suggest(Constant(), X_obs, y_obs, candidates, 3)
    assert list(result.indices) == [len(candidates) - 1, len(candidates) - 2, len(candidates) - 3]
