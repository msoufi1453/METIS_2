"""Discrete hill-climbing on the acquisition score: improve the best pool candidates step by step.

A random pool only offers what it happened to draw. Hill-climbing starts from the
``n_starts`` best-scoring candidates and repeatedly moves each one to its best-scoring
neighbour (``space.neighbors``: one level up or down, another alternative, or the
dataset's own notion of a neighbour), stopping at a local maximum of the score or after
``max_steps`` moves. The end points join the pool, so the batch selector can pick them.

Moves stay on the grid and inside the feasible set, so no rounding is needed, and the
score is the same one the selector uses (e.g. UCB). All neighbours of all climbers are
scored in one model call per step, and every scored recipe is cached.
"""
from dataclasses import dataclass

import numpy as np

from .acquisition import rank_descending
from .surrogates import Prediction


@dataclass
class Climb:
    starts: np.ndarray          # start recipes (grid indices), best-scoring first
    ends: np.ndarray            # where each climb stopped
    steps: np.ndarray           # accepted moves per climb
    capped: np.ndarray          # True if the climb was still improving when max_steps stopped it
    start_scores: np.ndarray
    end_scores: np.ndarray
    new: np.ndarray             # distinct end points that were not in the pool
    new_prediction: Prediction  # the surrogate's prediction at ``new``


def climb(space, surrogate, acquisition, pool, pool_prediction, measured_keys=(), best_y=None, n_starts=10,
          max_steps=20):
    """Climb from the ``n_starts`` best pool candidates. ``pool_prediction`` is the
    surrogate's prediction for ``pool`` (already computed by the caller)."""
    scores = np.asarray(acquisition(pool_prediction, best_y=best_y), float)
    order = rank_descending(scores)[:n_starts]
    starts = pool[order].copy()
    current, cur_scores = starts.copy(), scores[order].copy()
    steps = np.zeros(len(starts), dtype=int)
    active = np.ones(len(starts), dtype=bool)
    excluded = set(measured_keys)
    pool_keys = space.keys(pool)
    cache = {k: (pool_prediction.mean[i], pool_prediction.std[i], scores[i]) for i, k in enumerate(pool_keys)}
    tol = 1e-12 * (np.abs(scores).max() + 1.0)
    for _ in range(max_steps):
        owners, rows = [], []
        for i in np.flatnonzero(active):
            nb = space.neighbors(current[i])
            keep = [j for j, k in enumerate(space.keys(nb)) if k not in excluded] if len(nb) else []
            rows += list(nb[keep])
            owners += [i] * len(keep)
        if not rows:
            break
        rows, owners = np.array(rows, dtype=np.int64), np.array(owners)
        keys = space.keys(rows)
        todo = sorted({j for j, k in enumerate(keys) if k not in cache}, key=lambda j: keys[j])
        if todo:
            first = {}
            for j in todo:
                first.setdefault(keys[j], j)
            unique = list(first.values())
            p = surrogate.predict(space.features(rows[unique]))
            s = np.asarray(acquisition(p, best_y=best_y), float)
            for n, j in enumerate(unique):
                cache[keys[j]] = (p.mean[n], p.std[n], s[n])
        nb_scores = np.array([cache[k][2] for k in keys])
        moved = np.zeros(len(starts), dtype=bool)
        for i in np.flatnonzero(active):
            mine = np.flatnonzero(owners == i)
            if not len(mine):
                active[i] = False
                continue
            j = mine[int(np.argmax(nb_scores[mine]))]
            if nb_scores[j] > cur_scores[i] + tol:
                current[i], cur_scores[i] = rows[j], nb_scores[j]
                steps[i] += 1
                moved[i] = True
            else:
                active[i] = False
        if not moved.any():
            break
    capped = steps >= max_steps
    in_pool = set(pool_keys)
    new, seen = [], set()
    for row, key in zip(current, space.keys(current)):
        if key not in in_pool and key not in seen:
            seen.add(key)
            new.append(row)
    new = np.array(new, dtype=np.int64).reshape(-1, pool.shape[1])
    new_keys = space.keys(new)
    new_prediction = Prediction(np.array([cache[k][0] for k in new_keys], float),
                                np.array([cache[k][1] for k in new_keys], float))
    return Climb(starts, current, steps, capped, scores[order], cur_scores, new, new_prediction)
