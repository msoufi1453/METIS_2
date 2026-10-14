"""Benchmark problems ("oracles"): a design space, a noise-free yield, and its optimum."""
import numpy as np

from metis.space import DesignSpace

REFERENCE_SEED = 20241014
REFERENCE_SAMPLES = 10000


class Oracle:
    """``oracle(idx)`` returns noise-free yields for grid points of ``oracle.space``.

    ``f_max`` is the optimum over the feasible space. ``max_is_exact`` says whether it is
    proven (synthetic functions, enumerated grids, finite pools) or only the best value
    found. ``f_ref`` is the median yield of random feasible designs, the zero point that
    regret is normalised against. It is computed once per oracle with a fixed seed.
    """

    id = 'oracle'
    space = None
    f_max = None
    x_max = None
    max_is_exact = True
    meta = None

    def __call__(self, idx):
        raise NotImplementedError

    @property
    def f_ref(self):
        if getattr(self, '_f_ref', None) is None:
            sample = self.space.sample(REFERENCE_SAMPLES, np.random.default_rng(REFERENCE_SEED))
            self._f_ref = float(np.median(self(sample)))
        return self._f_ref

    def describe(self):
        return {'id': self.id, 'f_max': self.f_max, 'f_ref': self.f_ref, 'max_is_exact': self.max_is_exact,
                **self.space.describe(), **(self.meta or {})}


class SyntheticOracle(Oracle):
    def __init__(self, fn):
        self.fn = fn
        self.id = fn.meta['id']
        self.space = fn.space
        self.f_max = fn.f_max
        self.x_max = fn.x_max
        self.max_is_exact = True
        self.meta = {k: v for k, v in fn.meta.items() if k != 'id'}

    def __call__(self, idx):
        return self.fn(idx)


class PooledSpace(DesignSpace):
    """A design space restricted to a finite pool of grid points (e.g. measured recipes)."""

    def __init__(self, factors, pool, drop_nl=100.0, max_nl=None, name='pool'):
        super().__init__(factors, drop_nl=drop_nl, max_nl=max_nl, name=name)
        self.pool = np.asarray(pool, dtype=np.int64)
        self._pool_keys = self.keys(self.pool)
        self._pool_set = set(self._pool_keys)

    @property
    def size(self):
        return len(self.pool)

    def describe(self):
        out = super().describe()
        out['pool_size'] = len(self.pool)
        out['log10_size'] = float(np.log10(len(self.pool)))
        return out

    def feasible(self, idx):
        return np.array([k in self._pool_set for k in self.keys(idx)], dtype=bool)

    def sample(self, n, rng, exclude=None, unique=True, max_rounds=None):
        excluded = set(exclude or ())
        free = np.array([i for i, k in enumerate(self._pool_keys) if k not in excluded], dtype=np.int64)
        take = rng.choice(len(free), size=min(n, len(free)), replace=False)
        return self.pool[free[take]]

    def enumerate(self, feasible_only=True, chunk=1_000_000):
        for start in range(0, len(self.pool), chunk):
            yield self.pool[start:start + chunk]

    def neighbors(self, row, k=None, feasible_only=True):
        """The pool points closest to ``row`` (2 x n_dims of them): the pool's local graph."""
        k = k or 2 * self.n_dims
        E = self.embedding(self.pool)
        d = np.linalg.norm(E - self.embedding(np.asarray(row)[None]), axis=1)
        order = np.argsort(d, kind='stable')
        order = order[d[order] > 0][:k]
        return self.pool[order]
