"""Batch selectors: pick ``k`` candidates per round from scored candidates.

* ``TopK``: METIS's rule; the ``k`` best by acquisition score, skipping measured points.
* ``SequentialFantasy``: the v1 ``batch_ucb`` heuristic, generalised. Pick the best,
  pretend its yield equals the predicted mean, refit, repeat (a "kriging believer").
* ``LocalPenalization``: proper batch selection with a diversity penalty (Gonzalez et
  al., AISTATS 2016). Each pick multiplies the scores of nearby candidates by the
  probability that they lie outside the pick's Lipschitz exclusion ball.
* ``GPBUCB``: hallucinated GP-UCB (Desautels et al., JMLR 2014). Each pick shrinks the
  posterior variance around it as if it had been measured. GP surrogates only.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

from ..surrogates.base import Prediction


def rank_descending(scores):
    """Indices from best to worst score, ordered exactly as v1's ``DataFrame.sort_values``."""
    return pd.Series(scores).sort_values(ascending=False, kind='quicksort').index.to_numpy()


@dataclass
class SelectionContext:
    candidates: np.ndarray
    prediction: Prediction
    surrogate: object
    acquisition: object
    X_obs: np.ndarray = None
    y_obs: np.ndarray = None
    noise_var: np.ndarray = None
    embedding: np.ndarray = None
    blocked: np.ndarray = None
    is_new: object = None
    best_y: float = None

    def available(self, i):
        if self.blocked is not None and self.blocked[i]:
            return False
        return self.is_new is None or self.is_new(i)


@dataclass
class Selection:
    indices: np.ndarray
    scores: np.ndarray


class BatchSelector:
    name = 'selector'

    def select(self, ctx, k):
        raise NotImplementedError

    def describe(self):
        return {'name': self.name}


class TopK(BatchSelector):
    name = 'topk'

    def select(self, ctx, k):
        scores = np.asarray(ctx.acquisition(ctx.prediction, best_y=ctx.best_y))
        chosen = []
        for i in rank_descending(scores):
            if ctx.available(i):
                chosen.append(i)
                if len(chosen) == k:
                    break
        chosen = np.array(chosen, dtype=np.int64)
        return Selection(chosen, scores[chosen])


def _best_available(scores, ctx, taken):
    for i in rank_descending(scores):
        if i not in taken and ctx.available(i):
            return i
    return None


class SequentialFantasy(BatchSelector):
    """Kriging-believer batches, the generalisation of v1 ``batch_ucb``.

    v1 also drew a fresh random pool for every pick; here all picks come from the round's
    pool, so the arms in a comparison see the same candidates. The surrogate is refitted
    with ``refit`` (fixed hyperparameters), as v1 reused the tuned regressors.
    """

    name = 'sequential_fantasy'

    def select(self, ctx, k):
        prediction, chosen, fantasies, scores_at_pick = ctx.prediction, [], [], []
        X_obs = np.asarray(ctx.X_obs, dtype=float)
        y_obs = np.asarray(ctx.y_obs, dtype=float).reshape(-1)
        for j in range(k):
            if j:
                model = ctx.surrogate.clone()
                X_f = np.vstack([X_obs, ctx.candidates[chosen]])
                y_f = np.concatenate([y_obs, fantasies])
                noise = None
                if ctx.noise_var is not None:
                    # A fantasy carries the typical measurement noise of the real data.
                    fill = np.full(len(chosen), float(np.median(ctx.noise_var)))
                    noise = np.concatenate([np.asarray(ctx.noise_var, float).reshape(-1), fill])
                model.refit(X_f, y_f, noise_var=noise if model.accepts_noise else None)
                prediction = model.predict(ctx.candidates)
            scores = np.asarray(ctx.acquisition(prediction, best_y=ctx.best_y))
            i = _best_available(scores, ctx, set(chosen))
            if i is None:
                break
            chosen.append(i)
            fantasies.append(float(prediction.mean[i]))
            scores_at_pick.append(scores[i])
        return Selection(np.array(chosen, dtype=np.int64), np.array(scores_at_pick))


def lipschitz_estimate(embedding, mean, rng=None, max_points=3000, quantile=0.9):
    """Robust Lipschitz constant of the surrogate mean over the candidate set.

    Slopes ``|mean(a) - mean(b)| / ||a - b||`` between each candidate and its nearest
    neighbour, summarised by a high quantile. Tree ensembles are piecewise constant, so
    the raw maximum would come from a single split edge and make every exclusion ball
    vanish.
    """
    n = len(embedding)
    rows = np.arange(n)
    if n > max_points:
        rng = rng or np.random.default_rng(0)
        rows = rng.choice(n, size=max_points, replace=False)
    E, m = embedding[rows], np.asarray(mean, float)[rows]
    nn = NearestNeighbors(n_neighbors=2).fit(E)
    dist, nbr = nn.kneighbors(E)
    d, j = dist[:, 1], nbr[:, 1]
    ok = d > 1e-12
    if not ok.any():
        return 1.0
    slopes = np.abs(m[ok] - m[j[ok]]) / d[ok]
    return max(float(np.quantile(slopes, quantile)), 1e-9)


class LocalPenalization(BatchSelector):
    """Greedy batch with Lipschitz exclusion balls (Gonzalez et al. 2016).

    ``score(x) = g(acq(x)) * prod_j Phi((L * ||x - x_j|| - M + mu_j) / sigma_j)``, where
    ``g`` shifts the acquisition to be positive and ``L`` is the Lipschitz estimate.

    ``M`` is the optimistic maximum ``max(best measured, max_x mean(x) + kappa * std(x))``
    with ``kappa`` the acquisition's exploration / exploitation ratio (1 if it has none).
    The paper's ``M`` (best measured value) gives a zero-radius ball at the predicted
    optimum, so the first picks cluster exactly as in top-k. The optimistic ``M`` makes
    every ball's radius ``(M - mu_j) / L`` at least ``kappa * sigma_j / L``, i.e. it
    scales with the remaining uncertainty and the exploration weight.
    ``lipschitz_scale`` multiplies L: below 1 widens the balls, above 1 shrinks them.

    ``min_radius`` (embedding units, default 0 = the paper) is a floor on every ball's
    radius. Where the surrogate is confident around its optimum the Lipschitz radius
    shrinks below one grid step and the batch collapses onto neighbours of the best point;
    the floor guarantees picks at least that far apart (softly, via ``sigma_j / L``).
    """

    name = 'local_penalization'

    def __init__(self, lipschitz_scale=1.0, lipschitz_quantile=0.9, min_radius=0.0):
        self.lipschitz_scale = lipschitz_scale
        self.lipschitz_quantile = lipschitz_quantile
        self.min_radius = min_radius

    def select(self, ctx, k):
        mean = np.asarray(ctx.prediction.mean, float)
        std = np.maximum(np.asarray(ctx.prediction.std, float), 1e-9 * (np.abs(mean).max() + 1.0))
        scores = np.asarray(ctx.acquisition(ctx.prediction, best_y=ctx.best_y), float)
        E = ctx.embedding
        if E is None:
            lo, span = ctx.candidates.min(axis=0), np.ptp(ctx.candidates, axis=0)
            E = (ctx.candidates - lo) / np.where(span > 0, span, 1.0)
        L = self.lipschitz_scale * lipschitz_estimate(E, mean, quantile=self.lipschitz_quantile)
        exploitation = getattr(ctx.acquisition, 'exploitation', 1.0) or 1.0
        kappa = getattr(ctx.acquisition, 'exploration', 1.0) / exploitation
        M = float(np.max(mean + max(kappa, 0.0) * std))
        if ctx.y_obs is not None and len(ctx.y_obs):
            M = max(M, float(np.max(ctx.y_obs)))
        log_w = np.log(scores - scores.min() + 1e-12 * (np.abs(scores).max() + 1.0))
        mask = np.zeros(len(scores), dtype=bool)
        if ctx.blocked is not None:
            mask |= ctx.blocked
        chosen, picked_scores = [], []
        for _ in range(k):
            order = np.where(mask, -np.inf, log_w)
            i = None
            for c in rank_descending(order):
                if not np.isfinite(order[c]):
                    break
                if ctx.available(c):
                    i = c
                    break
                mask[c] = True
            if i is None:
                break
            chosen.append(i)
            picked_scores.append(scores[i])
            mask[i] = True
            dist = np.linalg.norm(E - E[i], axis=1)
            radius = max((M - mean[i]) / L, self.min_radius)
            log_w = log_w + norm.logcdf((dist - radius) / (std[i] / L))
        return Selection(np.array(chosen, dtype=np.int64), np.array(picked_scores))

    def describe(self):
        return {'name': self.name, 'lipschitz_scale': self.lipschitz_scale,
                'lipschitz_quantile': self.lipschitz_quantile, 'min_radius': self.min_radius}


class GPBUCB(BatchSelector):
    """Hallucinated GP-UCB. Needs a surrogate with ``posterior_covariance`` (``GPSurrogate``).

    After each pick the posterior variance is conditioned on a pretend measurement at that
    point. The mean does not change, so candidates near earlier picks lose their
    exploration bonus. Noise of the pretend measurement is the GP's fitted white noise.
    """

    name = 'gp_bucb'

    def select(self, ctx, k):
        model = ctx.surrogate
        if not hasattr(model, 'posterior_covariance'):
            raise TypeError('GPBUCB needs a surrogate with posterior_covariance (e.g. GPSurrogate)')
        mean = np.asarray(ctx.prediction.mean, float)
        var0 = np.asarray(ctx.prediction.std, float) ** 2
        noise = float(getattr(model, 'noise_std_', 0.0)) ** 2 + 1e-12
        cov_cols, chosen, picked_scores = [], [], []
        var = var0.copy()
        for _ in range(k):
            scores = np.asarray(ctx.acquisition(Prediction(mean, np.sqrt(np.maximum(var, 0.0))), best_y=ctx.best_y))
            i = _best_available(scores, ctx, set(chosen))
            if i is None:
                break
            chosen.append(i)
            picked_scores.append(scores[i])
            cov_cols.append(model.posterior_covariance(ctx.candidates, ctx.candidates[[i]])[:, 0])
            C = np.stack(cov_cols, axis=1)
            A = C[chosen, :] + noise * np.eye(len(chosen))
            var = var0 - np.einsum('ij,jk,ik->i', C, np.linalg.inv(A), C)
        return Selection(np.array(chosen, dtype=np.int64), np.array(picked_scores))
