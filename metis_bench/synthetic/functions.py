"""Composable synthetic yield functions with an exactly known maximum on the droplet grid.

A function is ``f(x) = scale * max(g(x) - floor, 0)`` with ``g(x) = max_c A_c * C_c(x)``.
Each component ``C_c`` is a sum or a product of lookup-table terms over *disjoint*
groups of one or two dimensions. Smooth, threshold and interaction landscapes use one
component. Peaks, decoys and needles use one component per peak, plus a flat
background component.

Why the maximum is exact: ``f`` is non-decreasing in ``g``, ``g`` is a max over
components, and each component is separable over its groups (in log space for
products). Its maximum under the volume budget ``sum(drops) <= capacity`` is a
multiple-choice knapsack over the groups. Drop counts are integers, so dynamic
programming over the capacity solves it exactly. ``tests/test_synthetic.py`` checks the
result against brute-force enumeration.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Term:
    dims: tuple
    table: np.ndarray
    label: str = ''

    def __call__(self, idx):
        return self.table[tuple(idx[:, d] for d in self.dims)]


@dataclass
class Component:
    """``amplitude * combine(terms) + offset`` with ``combine`` a product or a sum."""

    terms: list
    combine: str = 'prod'
    amplitude: float = 1.0
    offset: float = 0.0
    label: str = ''

    def __call__(self, idx):
        if not self.terms:
            return np.full(len(idx), self.amplitude * (1.0 if self.combine == 'prod' else 0.0) + self.offset)
        values = [t(idx) for t in self.terms]
        out = np.prod(values, axis=0) if self.combine == 'prod' else np.sum(values, axis=0)
        return self.amplitude * out + self.offset


def _integer_drops(space):
    """Per-dimension integer drop counts (zeros for alternatives and unconstrained factors)."""
    per_dim = []
    for d, (f, kind) in enumerate(space.dims):
        factor = space.factors[f]
        if kind == 'level' and factor.drops is not None and space.max_nl:
            drops = np.asarray(factor.drops, dtype=float)
            if not np.allclose(drops, np.round(drops)):
                raise ValueError('exact maxima need integer drop counts ({})'.format(factor.name))
            per_dim.append(np.round(drops).astype(np.int64))
        else:
            per_dim.append(np.zeros(space.cardinalities[d], dtype=np.int64))
    return per_dim


def component_max(component, space):
    """Exact max of ``component`` over feasible grid points, and one maximiser."""
    drops = _integer_drops(space)
    capacity = int(np.floor(space.max_nl / space.drop_nl + 1e-9)) if space.max_nl else None
    covered = {d for t in component.terms for d in t.dims}
    x = np.zeros(space.n_dims, dtype=np.int64)
    base = 0
    for d in range(space.n_dims):
        if d not in covered:
            x[d] = int(np.argmin(drops[d]))
            base += int(drops[d][x[d]])
    if capacity is None:
        capacity = base + sum(int(drops[d].max()) for d in covered)
    budget = capacity - base
    if budget < 0:
        raise ValueError('no feasible point')

    use_log = component.combine == 'prod'
    neutral = 0.0
    best = np.full(budget + 1, -np.inf)
    best[0] = neutral
    choices = []
    for term in component.terms:
        shape = term.table.shape
        combos = np.stack(np.unravel_index(np.arange(term.table.size), shape), axis=1)
        cost = np.zeros(len(combos), dtype=np.int64)
        for j, d in enumerate(term.dims):
            cost += drops[d][combos[:, j]]
        with np.errstate(divide='ignore'):
            value = np.log(term.table.ravel()) if use_log else term.table.ravel().astype(float)
        # For each cost keep only the best option.
        order = np.lexsort((-value, cost))
        cost_sorted, first = np.unique(cost[order], return_index=True)
        option = order[first]
        new = np.full(budget + 1, -np.inf)
        pick = np.full(budget + 1, -1, dtype=np.int64)
        for c_o, o in zip(cost_sorted, option):
            if c_o > budget:
                break
            cand = np.full(budget + 1, -np.inf)
            cand[c_o:] = best[:budget + 1 - c_o] + value[o]
            better = cand > new
            new[better] = cand[better]
            pick[better] = o
        choices.append((pick, combos, cost))
        best = new
    c_star = int(np.argmax(best))
    total = best[c_star]
    if not np.isfinite(total) and component.terms:
        # Every feasible point has value 0 for a product: any feasible point is a maximiser.
        return float(component.offset), x
    for term, (pick, combos, cost) in zip(reversed(component.terms), reversed(choices)):
        o = pick[c_star]
        x[list(term.dims)] = combos[o]
        c_star -= int(cost[o])
    raw = np.exp(total) if use_log else total
    if not component.terms:
        raw = 1.0 if use_log else 0.0
    return float(component.amplitude * raw + component.offset), x


@dataclass
class SyntheticFunction:
    """Noise-free yield ``f(idx)`` on ``space`` with its exact maximum."""

    space: object
    components: list
    floor: float = 0.0
    scale: float = 1.0
    meta: dict = field(default_factory=dict)

    def raw(self, idx):
        idx = np.atleast_2d(idx)
        return np.max(np.stack([c(idx) for c in self.components]), axis=0)

    def __call__(self, idx):
        return self.scale * np.maximum(self.raw(idx) - self.floor, 0.0)

    def raw_max(self):
        best, best_x = -np.inf, None
        for component in self.components:
            value, x = component_max(component, self.space)
            if value > best:
                best, best_x = value, x
        return best, best_x

    def finalize(self, zero_fraction=0.0, rng=None, n_ref=4096):
        """Set ``floor`` so ``zero_fraction`` of random feasible points yield exactly 0,
        and ``scale`` so the maximum is 1. Then record the exact optimum."""
        g_max, x_max = self.raw_max()
        if zero_fraction > 0:
            sample = self.space.sample(n_ref, rng or np.random.default_rng(0))
            values = np.sort(self.raw(sample))
            q = float(np.quantile(values, zero_fraction))
            # With ties (a flat background) the exact share is unreachable: take whichever of
            # "floor at the tied value" or "floor just below it" lands closer to the target.
            below = values[values < q]
            options = [q] + ([float(below.max())] if len(below) else [0.0])
            achieved = [np.mean(values <= o) if o > 0 else 0.0 for o in options]
            self.floor = options[int(np.argmin([abs(a - zero_fraction) for a in achieved]))]
            self.floor = min(self.floor, g_max * (1 - 1e-6))
        self.zero_fraction_target = zero_fraction
        self.scale = 1.0 / (g_max - self.floor)
        self.f_max = float(self(x_max[None])[0])
        self.x_max = x_max
        return self
