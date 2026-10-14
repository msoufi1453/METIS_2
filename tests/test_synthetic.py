"""The synthetic suite's exact maxima are really exact."""
import numpy as np
import pytest

from metis.space import DesignSpace, Factor
from metis_bench.synthetic import FunctionSpec, SUBTYPES, build_function, make_space
from metis_bench.synthetic import suite as suite_module
from metis_bench.synthetic.functions import SyntheticFunction
from metis_bench.synthetic.shapes import FAMILIES, sample_shape


def _brute_force_max(fn):
    best = -np.inf
    for chunk in fn.space.enumerate():
        if len(chunk):
            best = max(best, float(fn(chunk).max()))
    return best


def _small_function(subtype, seed, n_factors=5, n_categorical=1, active='all'):
    """A suite builder applied to a tiny grid that can be enumerated."""
    rng = np.random.default_rng(seed)
    space = make_space(n_factors, rng, n_categorical=n_categorical)
    # Shrink every factor to at most 4 levels so the grid has a few thousand points.
    factors = []
    for f in space.factors:
        keep = np.unique(np.linspace(0, len(f.levels) - 1, min(4, len(f.levels))).round().astype(int))
        factors.append(Factor(f.name, f.levels[keep], f.drops[keep], f.alternatives, f.log_scale))
    space = DesignSpace(factors, drop_nl=space.drop_nl)
    totals = space.drops(space.sample(2000, rng))
    space.max_nl = float(np.quantile(totals, 0.5)) // 1 * space.drop_nl
    spec = FunctionSpec(subtype, active=active, zero_fraction=0.2, n_categorical=n_categorical, contrast=8, seed=seed)
    act = list(range(space.n_factors)) if active == 'all' else [0, 1, 2, 3]
    components = suite_module.BUILDERS[subtype](space, act, spec, rng)
    return SyntheticFunction(space, components).finalize(0.2, rng)


@pytest.mark.parametrize('subtype', SUBTYPES)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_dp_maximum_matches_brute_force(subtype, seed):
    fn = _small_function(subtype, seed)
    assert fn.space.size <= 5000
    assert fn.f_max == pytest.approx(_brute_force_max(fn), rel=1e-9, abs=1e-12)
    assert fn.space.feasible(fn.x_max[None])[0]
    assert fn(fn.x_max[None])[0] == pytest.approx(fn.f_max)


@pytest.mark.parametrize('subtype', SUBTYPES)
@pytest.mark.parametrize('n_dims', [12, 27])
def test_suite_functions_never_beaten_by_random_search(subtype, n_dims):
    spec = FunctionSpec(subtype, active='subset' if subtype == 'needle' else 'all', zero_fraction=0.2,
                        n_categorical=2, contrast=10, seed=3)
    fn = build_function(spec, n_dims)
    assert fn.space.n_factors == n_dims
    X = fn.space.sample(20000, np.random.default_rng(7))
    assert fn(X).max() <= fn.f_max + 1e-12
    assert fn.f_max == pytest.approx(1.0)


def test_build_is_deterministic():
    spec = FunctionSpec('ratio', seed=11)
    a, b = build_function(spec, 12), build_function(spec, 12)
    X = a.space.sample(500, np.random.default_rng(0))
    assert np.array_equal(a(X), b(X)) and np.array_equal(a.x_max, b.x_max)


@pytest.mark.parametrize('family', FAMILIES)
def test_shapes_are_normalised(family):
    rng = np.random.default_rng(0)
    for _ in range(20):
        shape = sample_shape(family, rng)
        u = np.linspace(0, 1, 2001)
        v = shape(u)
        assert v.min() >= 0 and v.max() == pytest.approx(1.0, abs=1e-9)


def test_zero_fraction_dial():
    fn = build_function(FunctionSpec('broad', zero_fraction=0.5, seed=4), 12)
    y = fn(fn.space.sample(5000, np.random.default_rng(1)))
    assert abs(np.mean(y == 0) - 0.5) < 0.05
