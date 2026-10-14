"""Landscape classes, dials, and the function specs that make up the suite.

A ``FunctionSpec`` is a small, serialisable recipe: landscape subtype, the three dials
(active factors, zero fraction, categorical factors), a contrast and a seed.
``build_function(spec, n_dims)`` instantiates it on a fresh 12- or 27-factor droplet grid.
The same spec at 12 and at 27 dimensions gives the same kind of landscape, not the same
function.
"""
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from .functions import Component, SyntheticFunction, Term
from .shapes import Shape, sample_shape
from .spaces import make_space

LANDSCAPES = {
    'smooth': ('broad', 'plateau'),
    'threshold': ('switch_on', 'switch_off', 'rise_crash'),
    'needle': ('needle',),
    'multipeak': ('peaks',),
    'interaction': ('multiplicative', 'ratio'),
}
SUBTYPES = tuple(s for group in LANDSCAPES.values() for s in group)
LANDSCAPE_OF = {s: name for name, group in LANDSCAPES.items() for s in group}
DIMENSIONS = (12, 27)


@dataclass(frozen=True)
class FunctionSpec:
    subtype: str
    active: str = 'all'          # 'all' factors, or a 'subset' of 3 to 5 with the rest inert
    zero_fraction: float = 0.0   # share of random feasible recipes floored to zero yield
    n_categorical: int = 0       # factors with discrete alternatives (e.g. promoters)
    contrast: float = 10.0       # rough max / min yield ratio before the zero floor
    seed: int = 0

    @property
    def landscape(self):
        return LANDSCAPE_OF[self.subtype]

    @property
    def id(self):
        return '{}.{}.z{:02d}.c{}.r{:.3g}.s{}'.format(self.subtype, self.active, int(round(100 * self.zero_fraction)),
                                                     self.n_categorical, self.contrast, self.seed)

    def function_id(self, n_dims):
        return '{}.d{}'.format(self.id, n_dims)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        # Cast explicitly: rows read back from DataFrames carry NumPy scalars, which JSON rejects.
        return cls(subtype=str(d['subtype']), active=str(d['active']), zero_fraction=float(d['zero_fraction']),
                   n_categorical=int(d['n_categorical']), contrast=float(d['contrast']), seed=int(d['seed']))


# --------------------------------------------------------------------------- helpers
def _factor_dims(space):
    level, alt = {}, {}
    for d, (f, kind) in enumerate(space.dims):
        (level if kind == 'level' else alt)[f] = d
    return level, alt


def _baselines(n, contrast, rng, share=1.0):
    """Per-factor floors b_i with prod(b_i) = contrast ** -share, so the product of
    factor terms spans about ``contrast ** share`` whatever the number of factors."""
    if n == 0:
        return np.array([])
    w = rng.dirichlet(np.full(n, 2.0))
    return np.power(float(contrast), -share * w)


def _factor_term(space, f, shape, b, rng, level_dim, alt_dim):
    factor = space.factors[f]
    u = factor.level_units
    if alt_dim is None:
        return Term((level_dim,), b + (1 - b) * shape(u), label='{}:{}'.format(factor.name, shape.family))
    # Alternatives: each variant has its own strength and its own optimum within the family.
    n_alt = space.cardinalities[alt_dim]
    strengths = np.concatenate([[1.0], rng.uniform(0.35, 0.95, n_alt - 1)])
    rng.shuffle(strengths)
    columns = []
    for v in range(n_alt):
        variant_shape = shape if v == 0 else sample_shape(shape.family, rng, increasing=shape.params.get('increasing'))
        columns.append(b + (1 - b) * strengths[v] * variant_shape(u))
    return Term((level_dim, alt_dim), np.stack(columns, axis=1), label='{}:{}+alts'.format(factor.name, shape.family))


SMOOTH_FAMILIES = (('bell', 0.35), ('hill', 0.2), ('logarithmic', 0.15), ('biphasic', 0.15), ('linear', 0.15))
INCREASING_FAMILIES = (('linear', 0.3), ('logarithmic', 0.35), ('hill', 0.35))


def _draw(families, rng):
    names, probs = zip(*families)
    return str(rng.choice(names, p=np.array(probs) / sum(probs)))


def _smooth_shape(rng, families=SMOOTH_FAMILIES):
    family = _draw(families, rng)
    shape = sample_shape(family, rng, sharp=False, increasing=True)
    if family == 'bell':  # broad optimum
        shape = Shape('bell', {'mu': shape.params['mu'], 'width': float(rng.uniform(0.25, 0.6))})
    return shape


def _single_terms(space, factors, shapes, baselines, rng):
    level, alt = _factor_dims(space)
    return [_factor_term(space, f, s, b, rng, level[f], alt.get(f)) for f, s, b in zip(factors, shapes, baselines)]


def _random_feasible(space, rng, n=1):
    return space.sample(n, rng)


def _middle_point(space, rng):
    """Middle level of every factor, nudged down until it fits the volume budget."""
    x = (space.cardinalities - 1) // 2
    for _ in range(200):
        if space.feasible(x[None])[0]:
            return x
        d = rng.choice(space.level_dims)
        x[d] = max(x[d] - 1, 0)
    return _random_feasible(space, rng)[0]


def _bump_terms(space, factors, center, width, rng, alt_miss):
    level, alt = _factor_dims(space)
    terms = []
    for f in factors:
        u = space.factors[f].level_units
        bump = np.exp(-0.5 * ((u - u[center[level[f]]]) / width) ** 2)
        if f in alt:
            match = np.full(space.cardinalities[alt[f]], alt_miss)
            match[center[alt[f]]] = 1.0
            terms.append(Term((level[f], alt[f]), np.outer(bump, match)))
        else:
            terms.append(Term((level[f],), bump))
    return terms


# --------------------------------------------------------------------------- builders
def _smooth(space, active, spec, rng, families=SMOOTH_FAMILIES):
    shapes = [_smooth_shape(rng, families) for _ in active]
    terms = _single_terms(space, active, shapes, _baselines(len(active), spec.contrast, rng), rng)
    return [Component(terms, 'prod', label=spec.subtype)]


def _plateau(space, active, spec, rng):
    shapes = []
    for _ in active:
        if rng.random() < 0.6:
            shapes.append(Shape('hill', {'n': float(rng.uniform(1.0, 2.5)), 'K': float(rng.uniform(0.08, 0.35)),
                                         'increasing': True}))
        else:
            shapes.append(Shape('logarithmic', {'k': float(np.exp(rng.uniform(np.log(20), np.log(300))))}))
    terms = _single_terms(space, active, shapes, _baselines(len(active), spec.contrast, rng), rng)
    return [Component(terms, 'prod', label='plateau')]


def _threshold(space, active, spec, rng):
    n_switch = int(rng.integers(1, min(3, len(active)) + 1))
    switch = list(rng.choice(active, size=n_switch, replace=False))
    others = [f for f in active if f not in switch]
    shapes, b_switch = [], _baselines(n_switch, spec.contrast, rng, share=0.7)
    for _ in switch:
        if spec.subtype == 'switch_on':
            shapes.append(sample_shape('hill', rng, sharp=True, increasing=True))
        elif spec.subtype == 'switch_off':
            shapes.append(sample_shape('hill', rng, sharp=True, increasing=False))
        else:
            shapes.append(sample_shape('crash', rng))
    other_families = INCREASING_FAMILIES if spec.subtype == 'rise_crash' else SMOOTH_FAMILIES
    other_shapes = [_smooth_shape(rng, other_families) for _ in others]
    b_other = _baselines(len(others), spec.contrast, rng, share=0.3)
    terms = (_single_terms(space, switch, shapes, b_switch, rng)
             + _single_terms(space, others, other_shapes, b_other, rng))
    return [Component(terms, 'prod', label=spec.subtype)]


def _needle(space, active, spec, rng):
    center = _random_feasible(space, rng)[0]
    width = float(rng.uniform(0.03, 0.08))
    needle = Component(_bump_terms(space, active, center, width, rng, alt_miss=0.02), 'prod', label='needle')
    background = Component([], 'prod', amplitude=1.0 / spec.contrast, label='flat background')
    return [needle, background]


def _peaks(space, active, spec, rng):
    scale = np.sqrt(max(len(active), 1) / 4.0)
    decoy_center = _middle_point(space, rng)
    pool = _random_feasible(space, rng, 64)
    U = space.unit(pool)[:, space.level_dims]
    far = np.linalg.norm(U - space.unit(decoy_center[None])[:, space.level_dims], axis=1)
    global_center = pool[int(np.argmax(far))]
    peaks = [Component(_bump_terms(space, active, global_center, float(rng.uniform(0.06, 0.12)) * scale, rng, 0.3),
                       'prod', amplitude=1.0, label='global peak'),
             Component(_bump_terms(space, active, decoy_center, float(rng.uniform(0.3, 0.45)) * scale, rng, 0.6),
                       'prod', amplitude=float(rng.uniform(0.7, 0.88)), label='decoy')]
    for k in range(int(rng.integers(1, 4))):
        center = _random_feasible(space, rng)[0]
        peaks.append(Component(_bump_terms(space, active, center, float(rng.uniform(0.1, 0.25)) * scale, rng, 0.4),
                               'prod', amplitude=float(rng.uniform(0.35, 0.7)), label='minor peak'))
    peaks.append(Component([], 'prod', amplitude=1.0 / spec.contrast, label='background'))
    return peaks


def _pairs(space, active, rng):
    _, alt = _factor_dims(space)
    plain = [f for f in active if f not in alt]
    rng.shuffle(plain)
    pairs = [(plain[i], plain[i + 1]) for i in range(0, len(plain) - 1, 2)]
    singles = [f for f in active if f in alt] + (plain[-1:] if len(plain) % 2 else [])
    return pairs, singles


def _multiplicative(space, active, spec, rng):
    level, _ = _factor_dims(space)
    pairs, singles = _pairs(space, active, rng)
    weights = rng.dirichlet(np.full(len(pairs) + len(singles), 2.0))
    terms = []
    for (fi, fj), w in zip(pairs, weights):
        si, sj = _smooth_shape(rng), _smooth_shape(rng)
        table = np.outer(si(space.factors[fi].level_units), sj(space.factors[fj].level_units))
        terms.append(Term((level[fi], level[fj]), w * table, label='pair'))
    single_shapes = [_smooth_shape(rng) for _ in singles]
    terms += [Term(t.dims, w * t.table, t.label) for t, w in
              zip(_single_terms(space, singles, single_shapes, np.zeros(len(singles)), rng), weights[len(pairs):])]
    return [Component(terms, 'sum', offset=1.0 / max(spec.contrast - 1.0, 1e-3), label='multiplicative pairs')]


def _ratio_table(space, fi, fj, rng):
    xi, xj = space.factors[fi].levels, space.factors[fj].levels
    eps_i = xi[xi > 0].min() / 2 if (xi <= 0).any() else 0.0
    eps_j = xj[xj > 0].min() / 2 if (xj <= 0).any() else 0.0
    li, lj = np.log(xi + eps_i), np.log(xj + eps_j)
    a = int(rng.integers(1, len(li) - 1))
    b = int(rng.integers(1, len(lj) - 1))
    target = li[a] - lj[b]
    width = float(rng.uniform(0.08, 0.2)) * (np.ptp(li) + np.ptp(lj)) / 2
    ridge = np.exp(-0.5 * ((li[:, None] - lj[None, :] - target) / width) ** 2)
    ui, uj = space.factors[fi].level_units, space.factors[fj].level_units
    along = (ui[:, None] + uj[None, :]) / 2
    magnitude = 0.4 + 0.6 * np.exp(-0.5 * ((along - rng.uniform(0.3, 0.8)) / 0.3) ** 2)
    return ridge * magnitude


def _ratio(space, active, spec, rng):
    level, _ = _factor_dims(space)
    pairs, singles = _pairs(space, active, rng)
    baselines = _baselines(len(pairs) + len(singles), spec.contrast, rng)
    terms = []
    for (fi, fj), b in zip(pairs, baselines):
        terms.append(Term((level[fi], level[fj]), b + (1 - b) * _ratio_table(space, fi, fj, rng), label='ratio ridge'))
    terms += _single_terms(space, singles, [_smooth_shape(rng) for _ in singles], baselines[len(pairs):], rng)
    return [Component(terms, 'prod', label='required ratio')]


BUILDERS = {
    'broad': _smooth, 'plateau': _plateau,
    'switch_on': _threshold, 'switch_off': _threshold, 'rise_crash': _threshold,
    'needle': _needle, 'peaks': _peaks,
    'multiplicative': _multiplicative, 'ratio': _ratio,
}


def build_function(spec, n_dims):
    """Instantiate ``spec`` on a fresh ``n_dims``-factor grid; the exact optimum is computed."""
    seed = [spec.seed, n_dims, zlib.crc32(spec.id.encode())]
    rng = np.random.default_rng(seed)
    space = make_space(n_dims, rng, n_categorical=spec.n_categorical, name=spec.function_id(n_dims))
    if spec.active == 'subset':
        k = int(rng.integers(3, 6))
        if spec.subtype in ('multiplicative', 'ratio'):
            k = max(k, 4)
        active = sorted(rng.choice(space.n_factors, size=k, replace=False).tolist())
    else:
        active = list(range(space.n_factors))
    components = BUILDERS[spec.subtype](space, active, spec, rng)
    fn = SyntheticFunction(space, components, meta={
        'id': spec.function_id(n_dims), 'spec': spec.to_dict(), 'landscape': spec.landscape, 'n_dims': n_dims,
        'active_factors': [space.factors[f].name for f in active]})
    fn.finalize(spec.zero_fraction, rng=np.random.default_rng(seed + [1]))
    return fn
