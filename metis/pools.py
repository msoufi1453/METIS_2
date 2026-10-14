"""Candidate pools: where the recipes the acquisition scores come from.

METIS scores a pool of uniformly random recipes and proposes the best ones, so the pool
is the acquisition optimiser. In 12 or more factors random pools rarely contain anything
near the optimum: on a smooth 12-factor benchmark function even 100,000 random recipes
top out around 85% of the optimum, while a few local moves from the best recipe found
reach 98%. A pool can therefore cap what any surrogate or batch rule can find.

* ``RandomPool``: METIS v1, uniform per factor with volume rejection.
* ``LocalPool``: a share of the pool are mutations of the best measured recipes (1 to
  ``max_changes`` factors moved one level, or switched to another alternative), the rest
  random. This is a cheap trust-region-style optimiser that keeps global coverage.
"""
import numpy as np


class CandidatePool:
    name = 'pool'

    def generate(self, space, n, rng, observed=None, y_obs=None):
        raise NotImplementedError

    def describe(self):
        return {'name': self.name, **{k: v for k, v in vars(self).items() if not k.startswith('_')}}


class RandomPool(CandidatePool):
    name = 'random'

    def generate(self, space, n, rng, observed=None, y_obs=None):
        exclude = space.keys(observed) if observed is not None and len(observed) else None
        return space.sample(n, rng, exclude=exclude)


class LocalPool(CandidatePool):
    """``local_share`` of the pool from mutations of the ``n_parents`` best measured recipes."""

    name = 'local'

    def __init__(self, local_share=0.5, n_parents=8, max_changes=3):
        self.local_share = local_share
        self.n_parents = n_parents
        self.max_changes = max_changes

    def _mutate(self, space, parents, n, rng):
        rows = parents[rng.integers(0, len(parents), size=n)].copy()
        n_changes = rng.integers(1, self.max_changes + 1, size=n)
        for i in range(n):
            dims = rng.choice(space.n_dims, size=min(n_changes[i], space.n_dims), replace=False)
            for d in dims:
                card = space.cardinalities[d]
                if space.dims[d][1] == 'level':
                    rows[i, d] = np.clip(rows[i, d] + rng.choice([-1, 1]), 0, card - 1)
                elif card > 1:
                    rows[i, d] = (rows[i, d] + rng.integers(1, card)) % card
        return rows

    def generate(self, space, n, rng, observed=None, y_obs=None):
        if observed is None or not len(observed):
            return RandomPool().generate(space, n, rng)
        seen = set(space.keys(observed))
        parents = observed[np.argsort(np.asarray(y_obs))[::-1][:self.n_parents]]
        n_local = int(round(self.local_share * n))
        local, keys = [], set()
        for _ in range(20):
            if len(local) >= n_local:
                break
            draw = self._mutate(space, parents, 2 * (n_local - len(local)), rng)
            draw = draw[space.feasible(draw)]
            for row, key in zip(draw, space.keys(draw)):
                if key not in seen and key not in keys:
                    keys.add(key)
                    local.append(row)
                    if len(local) == n_local:
                        break
        local = np.array(local, dtype=np.int64).reshape(-1, space.n_dims)
        rest = space.sample(n - len(local), rng, exclude=seen | keys)
        return np.vstack([local, rest]) if len(local) else rest


REGISTRY = {'random': RandomPool, 'local': LocalPool}


def make(cfg):
    cfg = dict(cfg or {'type': 'random'})
    return REGISTRY[cfg.pop('type')](**cfg)
