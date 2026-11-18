"""Choosing a round's batch from scored candidates.

Two optional modules keep a batch from being near-copies of one recipe
(docs/MODULES.md): a soft distance penalty and a fantasy batch. Each works alone
(``metis.acquisition.SoftPenalty``, ``SequentialFantasy``); ``PenalizedFantasy`` is the
two together.
"""
import numpy as np

from ..acquisition import SequentialFantasy, SoftPenalty, TopK
from ..acquisition.selectors import BatchSelector, Selection, _best_available

PENALTY_LOW, PENALTY_HIGH = 1.0, 3.0      # radius in units of the candidates' spacing (docs/MODULES.md)


def spacing(E, max_points=20000, seed=0):
    """Typical distance between neighbouring recipes: the median distance to the nearest other one."""
    from sklearn.neighbors import NearestNeighbors
    E = np.asarray(E, dtype=float)
    if len(E) > max_points:
        E = E[np.random.default_rng(seed).choice(len(E), size=max_points, replace=False)]
    if len(E) < 2:
        return 1.0
    d = NearestNeighbors(n_neighbors=2).fit(E).kneighbors(E)[0][:, 1]
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def penalty_radius(strength, space_between):
    """``strength`` 0 (low) to 1 (high) -> radius between 1 and 3 times the spacing."""
    strength = min(max(float(strength), 0.0), 1.0)
    return (PENALTY_LOW + (PENALTY_HIGH - PENALTY_LOW) * strength) * space_between


class PenalizedFantasy(BatchSelector):
    """A fantasy batch whose picks also push each other apart.

    As ``SequentialFantasy``: after each pick the model is refitted as if the pick had
    been measured at its predicted value. As ``SoftPenalty``: every candidate's score,
    shifted to be positive, is multiplied by ``1 - exp(-d^2 / (2 r^2))`` for its distance
    ``d`` to each earlier pick.
    """

    name = 'penalized_fantasy'

    def __init__(self, radius):
        if not radius > 0:
            raise ValueError('radius must be positive')
        self.radius = float(radius)

    def select(self, ctx, k):
        """Greedy fantasy picks whose weights are also multiplied by the distance penalty."""
        if ctx.embedding is None:
            raise ValueError('PenalizedFantasy needs candidate coordinates (embedding)')
        E = np.asarray(ctx.embedding, dtype=float)
        X_obs = np.asarray(ctx.X_obs, dtype=float)
        y_obs = np.asarray(ctx.y_obs, dtype=float).reshape(-1)
        prediction, penalty = ctx.prediction, np.ones(len(E))
        chosen, fantasies, picked_scores = [], [], []
        for j in range(k):
            if j:
                model = ctx.surrogate.clone()
                model.refit(np.vstack([X_obs, ctx.candidates[chosen]]), np.concatenate([y_obs, fantasies]))
                prediction = model.predict(ctx.candidates)
            scores = np.asarray(ctx.acquisition(prediction, best_y=ctx.best_y), dtype=float)
            weight = (scores - scores.min() + 1e-12 * (np.abs(scores).max() + 1.0)) * penalty
            i = _best_available(weight, ctx, set(chosen))
            if i is None:
                break
            chosen.append(i)
            fantasies.append(float(prediction.mean[i]))
            picked_scores.append(scores[i])
            penalty = penalty * -np.expm1(-np.sum((E - E[i]) ** 2, axis=1) / (2.0 * self.radius ** 2))
        return Selection(np.array(chosen, dtype=np.int64), np.array(picked_scores))

    def describe(self):
        """Name and radius."""
        return {'name': self.name, 'radius': self.radius}


def make_selector(radius=None, fantasy=False):
    """The batch rule for the chosen modules; ``radius`` None means no distance penalty."""
    if radius is not None and fantasy:
        return PenalizedFantasy(radius)
    if fantasy:
        return SequentialFantasy()
    if radius is not None:
        return SoftPenalty(radius)
    return TopK()
