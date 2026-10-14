"""The droplet grid as an explicit, vectorised design space.

A design point is a row of integer indices, one per *dimension*. Each factor contributes
a concentration dimension (index into its levels) and, if it has alternatives (e.g.
promoters), a categorical dimension (index into its variants). Alternatives cost no
volume; concentrations cost ``drops[level] * drop_nl``. A point is feasible when the
total variable volume is at most ``max_nl``, the same rule
``random_combination_generator`` applies.

Model inputs ("features") use METIS's column layout: each factor's concentration
followed by one-hot columns ``<factor>_<alternative>``.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .design import allowed_output


@dataclass
class Factor:
    name: str
    levels: np.ndarray
    drops: np.ndarray = None
    alternatives: tuple = None
    log_scale: bool = False

    def __post_init__(self):
        self.levels = np.asarray(self.levels, dtype=float)
        order = np.argsort(self.levels, kind='stable')
        self.levels = self.levels[order]
        if self.drops is not None:
            self.drops = np.asarray(self.drops, dtype=float)[order]
        if self.alternatives is not None:
            self.alternatives = tuple(self.alternatives)
        positive = np.unique(self.levels[self.levels > 0])
        if self.log_scale and len(positive):
            # Positive levels sit exactly on a log scale; zero sits one typical step below the
            # smallest dose (the first ratio between doses, or one decade).
            self._pos_min = float(positive[0])
            self._zero_gap = float(np.log10(positive[1] / positive[0])) if len(positive) > 1 else 1.0
        else:
            self.log_scale = self.log_scale and len(positive) > 0
        scaled = self.scale(self.levels)
        self._lo, self._hi = float(scaled.min()), float(scaled.max())

    def scale(self, concentrations):
        c = np.asarray(concentrations, dtype=float)
        if not self.log_scale:
            return c
        top = np.log10(np.maximum(c, self._pos_min))
        # Between 0 and the smallest dose, interpolate linearly from the zero position.
        below = np.log10(self._pos_min) - self._zero_gap * (1 - np.clip(c, 0, None) / self._pos_min)
        return np.where(c >= self._pos_min, top, below)

    def to_unit(self, concentrations):
        span = self._hi - self._lo
        if span <= 0:
            return np.zeros_like(np.asarray(concentrations, dtype=float))
        return (self.scale(concentrations) - self._lo) / span

    @property
    def level_units(self):
        return self.to_unit(self.levels)


def _auto_log(levels, ratio=100.0):
    positive = np.asarray(levels, dtype=float)
    positive = positive[positive > 0]
    return len(positive) > 2 and positive.max() / positive.min() >= ratio


class DesignSpace:
    """Finite design space on the droplet grid with an optional volume budget."""

    def __init__(self, factors, drop_nl=100.0, max_nl=None, name='space'):
        self.factors = list(factors)
        self.drop_nl = float(drop_nl)
        self.max_nl = max_nl
        self.name = name
        self.dims = []
        for f_index, factor in enumerate(self.factors):
            self.dims.append((f_index, 'level'))
            if factor.alternatives:
                self.dims.append((f_index, 'alt'))
        self.cardinalities = np.array([len(self.factors[f].levels) if kind == 'level' else len(self.factors[f].alternatives)
                                       for f, kind in self.dims])
        if self.cardinalities.max() > 255:
            raise ValueError('at most 255 levels or alternatives per factor')
        self.level_dims = np.array([d for d, (_, kind) in enumerate(self.dims) if kind == 'level'])
        self.columns = []
        for factor in self.factors:
            self.columns.append(factor.name)
            if factor.alternatives:
                self.columns += ['{}_{}'.format(factor.name, a) for a in factor.alternatives]

    # ----------------------------------------------------------------- construction
    @classmethod
    def from_limits(cls, concentrations_limits, reaction_vol_nl=20000, drop_size_nl=100, max_nl=None,
                    log_scale='auto', name='space'):
        """Build from a METIS ``concentrations_limits`` dict.

        Levels and drop counts come from ``allowed_output``, so the grid is the one the
        notebooks sample. ``log_scale`` is 'auto' (log when positive levels span two or
        more decades), True, False, or a set of factor names.
        """
        factors = []
        for key, value in concentrations_limits.items():
            concs, vols = allowed_output(value, reaction_vol_nl=reaction_vol_nl, drop_size_nl=drop_size_nl)
            if log_scale == 'auto':
                use_log = _auto_log(concs)
            elif isinstance(log_scale, (set, frozenset, list, tuple)):
                use_log = key in log_scale
            else:
                use_log = bool(log_scale)
            factors.append(Factor(key, concs, np.asarray(vols, dtype=float) / drop_size_nl,
                                  value['Alternatives'] or None, use_log))
        return cls(factors, drop_nl=drop_size_nl, max_nl=max_nl, name=name)

    # ----------------------------------------------------------------- basic facts
    @property
    def n_dims(self):
        return len(self.dims)

    @property
    def n_factors(self):
        return len(self.factors)

    @property
    def size(self):
        """Number of grid points before the volume constraint (Python int, may be huge)."""
        total = 1
        for c in self.cardinalities:
            total *= int(c)
        return total

    def describe(self):
        return {'name': self.name, 'n_factors': self.n_factors, 'n_dims': self.n_dims,
                'log10_size': float(np.sum(np.log10(self.cardinalities))), 'max_nl': self.max_nl,
                'drop_nl': self.drop_nl}

    # ----------------------------------------------------------------- volumes
    def drops(self, idx):
        idx = np.atleast_2d(idx)
        total = np.zeros(len(idx))
        for d, (f, kind) in enumerate(self.dims):
            factor = self.factors[f]
            if kind == 'level' and factor.drops is not None:
                total += factor.drops[idx[:, d]]
        return total

    def volume_nl(self, idx):
        return self.drops(idx) * self.drop_nl

    def feasible(self, idx):
        idx = np.atleast_2d(idx)
        if not self.max_nl:
            return np.ones(len(idx), dtype=bool)
        # Small tolerance: drop counts from allowed_output can be inexact floats.
        return self.volume_nl(idx) <= self.max_nl + 1e-9

    # ----------------------------------------------------------------- encodings
    def features(self, idx):
        idx = np.atleast_2d(idx)
        blocks = []
        for d, (f, kind) in enumerate(self.dims):
            factor = self.factors[f]
            if kind == 'level':
                blocks.append(factor.levels[idx[:, d]][:, None])
            else:
                blocks.append(np.eye(len(factor.alternatives))[idx[:, d]])
        return np.hstack(blocks)

    def frame(self, idx):
        return pd.DataFrame(self.features(idx), columns=self.columns)

    def unit(self, idx):
        """Points in [0, 1]^n_dims on each factor's own scale (log for log-scale factors)."""
        idx = np.atleast_2d(idx)
        out = np.empty(idx.shape, dtype=float)
        for d, (f, kind) in enumerate(self.dims):
            if kind == 'level':
                out[:, d] = self.factors[f].level_units[idx[:, d]]
            else:
                card = self.cardinalities[d]
                out[:, d] = idx[:, d] / (card - 1) if card > 1 else 0.0
        return out

    def embedding(self, idx):
        """Coordinates for distances: unit scale for levels, one-hot / sqrt(2) for alternatives,
        so two different variants are exactly distance 1 apart."""
        idx = np.atleast_2d(idx)
        blocks = []
        for d, (f, kind) in enumerate(self.dims):
            if kind == 'level':
                blocks.append(self.factors[f].level_units[idx[:, d]][:, None])
            else:
                blocks.append(np.eye(self.cardinalities[d])[idx[:, d]] / np.sqrt(2.0))
        return np.hstack(blocks)

    def feature_transform(self):
        """Callable mapping METIS feature matrices to [0, 1] with each factor on its own scale."""
        plan = []
        col = 0
        for factor in self.factors:
            plan.append((col, factor))
            col += 1 + (len(factor.alternatives) if factor.alternatives else 0)

        def transform(X):
            Z = np.array(X, dtype=float, copy=True)
            for c, factor in plan:
                Z[:, c] = factor.to_unit(Z[:, c])
            return Z
        return transform

    def from_unit(self, U):
        """Project points of the unit cube onto the grid (nearest level on each factor's scale)."""
        U = np.clip(np.atleast_2d(np.asarray(U, dtype=float)), 0.0, 1.0)
        idx = np.empty(U.shape, dtype=np.int64)
        for d, (f, kind) in enumerate(self.dims):
            if kind == 'level':
                positions = self.factors[f].level_units
                idx[:, d] = np.abs(U[:, d][:, None] - positions[None, :]).argmin(axis=1)
            else:
                card = self.cardinalities[d]
                idx[:, d] = np.minimum((U[:, d] * card).astype(np.int64), card - 1)
        return idx

    def from_features(self, X):
        """Map model inputs (e.g. imported concentrations) to the nearest grid indices."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        idx = np.empty((len(X), self.n_dims), dtype=np.int64)
        col = 0
        for d, (f, kind) in enumerate(self.dims):
            factor = self.factors[f]
            if kind == 'level':
                units = factor.to_unit(X[:, col])
                idx[:, d] = np.abs(units[:, None] - factor.level_units[None, :]).argmin(axis=1)
                col += 1
            else:
                k = len(factor.alternatives)
                idx[:, d] = X[:, col:col + k].argmax(axis=1)
                col += k
        return idx

    def from_frame(self, df):
        return self.from_features(df[self.columns].to_numpy(dtype=float))

    def keys(self, idx):
        """Hashable row keys for de-duplication."""
        rows = np.ascontiguousarray(np.atleast_2d(idx).astype(np.uint8))
        return [r.tobytes() for r in rows]

    # ----------------------------------------------------------------- sampling
    def sample(self, n, rng, exclude=None, unique=True, max_rounds=200):
        """``n`` feasible points, uniform per dimension then filtered by volume (METIS's rule).

        ``exclude``: keys (from ``keys``) that must not be returned, e.g. points already
        measured. Returns fewer than ``n`` rows only if the space runs out.
        """
        seen = set(exclude or ())
        chosen = []
        need = n
        for _ in range(max_rounds):
            draw = rng.integers(0, self.cardinalities, size=(max(2 * need, 64), self.n_dims))
            draw = draw[self.feasible(draw)]
            for row, key in zip(draw, self.keys(draw)):
                if unique and key in seen:
                    continue
                seen.add(key)
                chosen.append(row)
                if len(chosen) == n:
                    return np.array(chosen, dtype=np.int64)
            need = n - len(chosen)
        return np.array(chosen, dtype=np.int64).reshape(-1, self.n_dims)

    def enumerate(self, feasible_only=True, chunk=1_000_000):
        """Yield every grid point in chunks. Only sensible for small spaces (e.g. 4^11)."""
        total = self.size
        for start in range(0, total, chunk):
            flat = np.arange(start, min(start + chunk, total), dtype=np.int64)
            idx = np.stack(np.unravel_index(flat, self.cardinalities), axis=1)
            if feasible_only:
                idx = idx[self.feasible(idx)]
            yield idx

    def neighbors(self, row):
        """Feasible points one step away: one level up or down, or a different alternative."""
        row = np.asarray(row, dtype=np.int64)
        out = []
        for d, (_, kind) in enumerate(self.dims):
            if kind == 'level':
                for step in (-1, 1):
                    v = row[d] + step
                    if 0 <= v < self.cardinalities[d]:
                        nb = row.copy()
                        nb[d] = v
                        out.append(nb)
            else:
                for v in range(self.cardinalities[d]):
                    if v != row[d]:
                        nb = row.copy()
                        nb[d] = v
                        out.append(nb)
        out = np.array(out, dtype=np.int64).reshape(-1, self.n_dims)
        return out[self.feasible(out)]
