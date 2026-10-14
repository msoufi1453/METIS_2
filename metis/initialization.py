"""Initial designs on the droplet grid.

* ``RandomInit``: METIS v1. Uniform level per factor, rejecting recipes over the volume
  budget.
* ``SobolInit``: scrambled Sobol points in the unit cube (each factor on its own scale,
  log for log-spaced factors), projected to the nearest grid level and filtered by
  volume. Points are taken in sequence order, so the design keeps Sobol's low
  discrepancy as far as the projection and filter allow.
* ``MaximinLHSInit``: Latin hypercube samples sized so enough survive the volume filter,
  projected and filtered. From the survivors, ``n`` points are chosen by greedy maximin
  (farthest point). The best of ``n_designs`` designs by minimum pairwise distance wins.
* ``FillUnsampledInit``: for imported data. Choose ``n`` feasible points that maximise
  the minimum distance to everything already measured, so the first batch covers the
  region the imported data never touched.

Distances use ``DesignSpace.embedding``: unit scale for levels, and alternatives one
apart. Every initialiser returns unique feasible points not in ``existing``, topping up
with random feasible points if a projection collapses too many points together.
"""
import numpy as np
from scipy.stats import qmc


class Initializer:
    name = 'init'

    def __call__(self, space, n, rng, existing=None):
        raise NotImplementedError

    def describe(self):
        return {'name': self.name, **{k: v for k, v in vars(self).items() if not k.startswith('_')}}


def _seed(rng):
    return int(rng.integers(2 ** 31 - 1))


def _unique_new(space, idx, existing):
    seen = set(space.keys(existing)) if existing is not None and len(existing) else set()
    keep = []
    for i, key in enumerate(space.keys(idx)):
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return idx[keep]


def _top_up(space, chosen, n, rng, existing):
    if len(chosen) >= n:
        return chosen[:n]
    taken = [existing] if existing is not None and len(existing) else []
    exclude = space.keys(np.vstack(taken + [chosen])) if (taken or len(chosen)) else None
    extra = space.sample(n - len(chosen), rng, exclude=exclude)
    return np.vstack([chosen, extra]) if len(chosen) else extra


def _feasible_rate(space, rng, probe=4000):
    draw = rng.integers(0, space.cardinalities, size=(probe, space.n_dims))
    return max(float(space.feasible(draw).mean()), 1.0 / probe)


def greedy_maximin(E_candidates, n, E_fixed=None):
    """Farthest-point selection of ``n`` rows of ``E_candidates`` (indices).

    With ``E_fixed`` the distances start from those points (fill mode); otherwise the
    first pick is the candidate farthest from the candidates' centroid.
    """
    if E_fixed is not None and len(E_fixed):
        d = np.min(np.linalg.norm(E_candidates[:, None, :] - E_fixed[None, :, :], axis=2), axis=1) \
            if len(E_fixed) * len(E_candidates) <= 4_000_000 else _chunked_min_dist(E_candidates, E_fixed)
        chosen = []
    else:
        first = int(np.argmax(np.linalg.norm(E_candidates - E_candidates.mean(axis=0), axis=1)))
        chosen = [first]
        d = np.linalg.norm(E_candidates - E_candidates[first], axis=1)
    while len(chosen) < min(n, len(E_candidates)):
        d_masked = d.copy()
        d_masked[chosen] = -np.inf
        i = int(np.argmax(d_masked))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(E_candidates - E_candidates[i], axis=1))
    return np.array(chosen, dtype=np.int64)


def _chunked_min_dist(A, B, chunk=2000):
    out = np.full(len(A), np.inf)
    for s in range(0, len(B), chunk):
        out = np.minimum(out, np.min(np.linalg.norm(A[:, None, :] - B[None, s:s + chunk, :], axis=2), axis=1))
    return out


def min_pairwise_distance(E):
    if len(E) < 2:
        return np.inf
    d = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=2)
    return float(d[np.triu_indices(len(E), 1)].min())


class RandomInit(Initializer):
    name = 'random'

    def __call__(self, space, n, rng, existing=None):
        exclude = space.keys(existing) if existing is not None and len(existing) else None
        return space.sample(n, rng, exclude=exclude)


class SobolInit(Initializer):
    name = 'sobol'

    def __init__(self, scramble=True, max_draw=2 ** 16):
        self.scramble = scramble
        self.max_draw = max_draw

    def __call__(self, space, n, rng, existing=None):
        sampler = qmc.Sobol(d=space.n_dims, scramble=self.scramble, seed=_seed(rng))
        rate = _feasible_rate(space, rng)
        m = int(2 ** np.ceil(np.log2(max(2 * n / rate, 2 * n))))
        chosen = np.empty((0, space.n_dims), dtype=np.int64)
        drawn = 0
        while len(chosen) < n and drawn < self.max_draw:
            U = sampler.random(m)
            drawn += m
            idx = space.from_unit(U)
            idx = idx[space.feasible(idx)]
            chosen = _unique_new(space, np.vstack([chosen, idx]), existing)
            m = int(min(m * 2, self.max_draw))
        return _top_up(space, chosen[:n], n, rng, existing)


class MaximinLHSInit(Initializer):
    name = 'lhs_maximin'

    def __init__(self, n_designs=20, oversample=2.0):
        self.n_designs = n_designs
        self.oversample = oversample

    def __call__(self, space, n, rng, existing=None):
        rate = _feasible_rate(space, rng)
        m = int(np.ceil(self.oversample * n / rate))
        best, best_score = None, -np.inf
        for _ in range(self.n_designs):
            U = qmc.LatinHypercube(d=space.n_dims, seed=_seed(rng)).random(m)
            idx = space.from_unit(U)
            idx = _unique_new(space, idx[space.feasible(idx)], existing)
            if len(idx) < n:
                idx = _top_up(space, idx, n, rng, existing)
            pick = idx[greedy_maximin(space.embedding(idx), n)]
            score = min_pairwise_distance(space.embedding(pick))
            if score > best_score:
                best, best_score = pick, score
        return best


class FillUnsampledInit(Initializer):
    """Space-filling batch that avoids the region already covered by imported data."""

    name = 'fill'

    def __init__(self, n_candidates=4096):
        self.n_candidates = n_candidates

    def __call__(self, space, n, rng, existing=None):
        candidates = SobolInit()(space, self.n_candidates, rng, existing=existing)
        E_fixed = space.embedding(existing) if existing is not None and len(existing) else None
        return candidates[greedy_maximin(space.embedding(candidates), n, E_fixed=E_fixed)]


REGISTRY = {'random': RandomInit, 'sobol': SobolInit, 'lhs_maximin': MaximinLHSInit, 'fill': FillUnsampledInit}


def make(cfg):
    """Build an initialiser from ``{'type': ..., **params}``; ``{'type': 'none'}`` gives None
    (the campaign starts model guided from imported data)."""
    cfg = dict(cfg or {'type': 'random'})
    kind = cfg.pop('type')
    return None if kind == 'none' else REGISTRY[kind](**cfg)
