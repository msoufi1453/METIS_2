"""Landscape descriptors for calibrating the synthetic suite against real data.

The same descriptors are computed for every synthetic function (from the function
itself) and for each real dataset (from raw samples where possible, otherwise from
fitted XGBoost and GP models, both reported).

==================  =========================================================  ======================
descriptor          definition                                                 real data source
==================  =========================================================  ======================
zero_frac           share of random designs with yield <= 1% of the top        raw (Borkowski random
                    (99th percentile) yield                                    plate) / models
skew                skewness of random-design yields                           same
log_top_median      log10(q99 / median) of random-design yields                same
s_first             sum of first-order Sobol indices                           models
s_pair              sum of second-order (pairwise) Sobol indices               models
s_higher            1 - s_first - s_pair (>= 0): higher-order share            models
frac_factors_80     share of factors needed for 80% of total-effect variance   models
corr_length         distance where yield correlation falls to 1/e (variogram), raw pairs / models
                    / median random-pair distance
peak_frac           distinct local maxima / hill-climbing starts               models
threshold_max       max over influential factors of (SSE_smooth - SSE_step) /  raw level means /
                    SST on the main-effect profile                             models
learn_32/100/300    held-out R2 of XGBoost and GP trained on n designs (mean   raw subsamples
                    of the two models, each also reported)
noise_ratio         replicate variance / total variance of measured means      raw replicates
==================  =========================================================  ======================

Random designs and Sobol indices use a product measure over factor levels: uniform for
synthetic functions and the Borkowski grid (the classical definition; the volume budget
is ignored for Sobol), and each factor's empirical level frequencies for PURE. PURE's data
are one-factor-at-a-time titrations (98% of rows sit at each factor's modal level), so
uniform designs over its 69 factors would lie far outside anything measured and the
fitted models would only extrapolate. The empirical marginals keep model-based
descriptors near measured recipes. Estimators: Saltelli 2002 / 2010 and Jansen.
Distances use ``DesignSpace.embedding``.
"""
import numpy as np
from scipy.stats import skew as _skew
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from xgboost import XGBRegressor

REALISM_DESCRIPTORS = ('zero_frac', 'skew', 'log_top_median', 's_first', 's_pair', 'frac_factors_80',
                       'corr_length', 'peak_frac', 'threshold_max', 'learn_32', 'learn_100', 'learn_300',
                       'noise_ratio')
SCREENING_DESCRIPTORS = REALISM_DESCRIPTORS[:9]
LEARN_SIZES = (32, 100, 300)


# --------------------------------------------------------------------------- distribution
def distribution(y):
    y = np.asarray(y, float)
    top = float(np.quantile(y, 0.99))
    med = float(np.median(y))
    if not np.isfinite(top) or top <= 0:
        return {'zero_frac': 1.0, 'skew': 0.0, 'log_top_median': 0.0}
    eps = 0.01 * top
    return {'zero_frac': float(np.mean(y <= 0.01 * top)),
            'skew': float(_skew(y)) if np.ptp(y) > 0 else 0.0,
            'log_top_median': float(np.log10((top + eps) / (max(med, 0) + eps)))}


# --------------------------------------------------------------------------- sampling
def draw(space, n, rng, marginals=None):
    """``n`` grid points from the product measure: uniform levels, or ``marginals`` (one
    probability vector per dim)."""
    if marginals is None:
        return rng.integers(0, space.cardinalities, size=(n, space.n_dims))
    return np.stack([rng.choice(len(p), size=n, p=p) for p in marginals], axis=1)


def empirical_marginals(space, idx, smoothing=0.0):
    out = []
    for d in range(space.n_dims):
        counts = np.bincount(idx[:, d], minlength=space.cardinalities[d]).astype(float) + smoothing
        out.append(counts / counts.sum())
    return out


def random_designs(space, n, rng, marginals=None):
    """Random feasible designs: METIS's sampler, or the product of ``marginals``."""
    if marginals is None:
        return space.sample(n, rng)
    idx = draw(space, 2 * n, rng, marginals)
    return idx[space.feasible(idx)][:n]


# --------------------------------------------------------------------------- Sobol
def sobol(f, space, rng, n_base=1024, marginals=None):
    """First, total and second-order Sobol indices over the dims of ``space``."""
    d = space.n_dims
    A = draw(space, n_base, rng, marginals)
    B = draw(space, n_base, rng, marginals)
    fA, fB = f(A), f(B)
    var = float(np.var(np.concatenate([fA, fB])))
    if var <= 0:
        return {'S1': np.zeros(d), 'ST': np.zeros(d), 'S2_sum': 0.0, 'var': 0.0, 'A': A, 'fA': fA}
    fAB, fBA = np.empty((d, n_base)), np.empty((d, n_base))
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        BA = B.copy()
        BA[:, i] = A[:, i]
        fAB[i], fBA[i] = f(AB), f(BA)
    S1 = np.mean(fB * (fAB - fA), axis=1) / var                    # Saltelli 2010
    ST = 0.5 * np.mean((fA - fAB) ** 2, axis=1) / var               # Jansen 1999
    s2 = 0.0
    for i in range(d):                                               # Saltelli 2002
        closed = np.mean(fBA[i][None, :] * fAB[i + 1:] - (fA * fB)[None, :], axis=1) / var
        s2 += float(np.sum(closed - S1[i] - S1[i + 1:]))
    return {'S1': S1, 'ST': ST, 'S2_sum': s2, 'var': var, 'A': A, 'fA': fA}


def _factor_of_dim(space):
    return np.array([f for f, _ in space.dims])


def sobol_summary(sob, space):
    s_first = float(np.clip(np.sum(np.clip(sob['S1'], 0, None)), 0, 1))
    s_pair = float(np.clip(sob['S2_sum'], 0, 1 - s_first))
    ST_factor = np.bincount(_factor_of_dim(space), weights=np.clip(sob['ST'], 0, None), minlength=space.n_factors)
    total = ST_factor.sum()
    if total <= 0:
        n80 = space.n_factors
    else:
        cum = np.cumsum(np.sort(ST_factor)[::-1]) / total
        n80 = int(np.searchsorted(cum, 0.8) + 1)
    return {'s_first': s_first, 's_pair': s_pair, 's_higher': float(max(0.0, 1 - s_first - s_pair)),
            'n_factors_80': n80, 'frac_factors_80': n80 / space.n_factors}


# --------------------------------------------------------------------------- ruggedness
def _variogram_corr_length(E_a, E_b, y_a, y_b, ref_distance, n_bins=10):
    """Distance where the variogram correlation 1 - gamma(d) / var drops to 1/e."""
    dist = np.linalg.norm(E_a - E_b, axis=1)
    var = np.var(np.concatenate([y_a, y_b]))
    if var <= 0 or ref_distance <= 0:
        return 1.0
    edges = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
    centers, rho = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist <= hi)
        if m.sum() < 10:
            continue
        gamma = 0.5 * np.mean((y_a[m] - y_b[m]) ** 2)
        centers.append(float(np.mean(dist[m])))
        rho.append(1 - gamma / var)
    centers, rho = np.array(centers), np.array(rho)
    below = np.flatnonzero(rho < np.exp(-1))
    if len(centers) == 0:
        return 1.0
    if len(below) == 0:
        return float(centers[-1] / ref_distance)
    k = below[0]
    if k == 0:
        return float(centers[0] / ref_distance)
    x0, x1, r0, r1 = centers[k - 1], centers[k], rho[k - 1], rho[k]
    return float((x0 + (np.exp(-1) - r0) * (x1 - x0) / (r1 - r0)) / ref_distance)


def _random_pair_distance(space, rng, n=2000):
    a, b = space.sample(n, rng), space.sample(n, rng)
    return float(np.median(np.linalg.norm(space.embedding(a) - space.embedding(b), axis=1)))


def corr_length_model(f, space, rng, n_pairs=3000, marginals=None):
    """Pairs at a spread of distances: a random design and a copy moved k random level steps."""
    a = draw(space, n_pairs, rng, marginals)
    b = a.copy()
    steps = rng.integers(1, 2 * space.n_dims + 1, size=n_pairs)
    for i in range(n_pairs):
        for _ in range(steps[i]):
            d = rng.integers(space.n_dims)
            b[i, d] = np.clip(b[i, d] + rng.choice([-1, 1]), 0, space.cardinalities[d] - 1)
    if marginals is None:
        ref = _random_pair_distance(space, rng)
    else:
        u, v = draw(space, 2000, rng, marginals), draw(space, 2000, rng, marginals)
        ref = float(np.median(np.linalg.norm(space.embedding(u) - space.embedding(v), axis=1)))
    return _variogram_corr_length(space.embedding(a), space.embedding(b), f(a), f(b), ref)


def corr_length_raw(space, idx, y, rng, max_points=800):
    n = len(idx)
    keep = rng.choice(n, size=min(n, max_points), replace=False)
    i, j = np.triu_indices(len(keep), 1)
    E = space.embedding(idx[keep])
    ref = float(np.median(np.linalg.norm(E[i] - E[j], axis=1)))
    return _variogram_corr_length(E[i], E[j], y[keep][i], y[keep][j], ref)


def hill_climb_peaks(f, space, rng, n_starts=50, max_steps=300, flat=None):
    """Steepest-ascent climbs over ``space.neighbors``. Distinct local maxima / starts.

    Climbs that end at or below ``flat`` (the 10th percentile of random yields by
    default) got stuck on a flat floor and are not counted as peaks. Climbs move on the
    full product grid, ignoring the volume budget as the Sobol indices do: with the budget,
    every point on the budget surface would look like a peak to single-step moves. For a
    ``PooledSpace`` the neighbours are the nearest pool recipes.
    """
    starts = space.sample(n_starts, rng)
    if flat is None:
        flat = float(np.quantile(f(space.sample(min(500, space.size), rng)), 0.1))
    ends = set()
    for x in starts:
        fx = float(f(x[None])[0])
        for _ in range(max_steps):
            nb = space.neighbors(x, feasible_only=False)
            if len(nb) == 0:
                break
            values = f(nb)
            j = int(np.argmax(values))
            if values[j] <= fx + 1e-12 * max(1.0, abs(fx)):
                break
            x, fx = nb[j], float(values[j])
        if fx > flat + 1e-12:
            ends.add(space.keys(x[None])[0])
    return {'n_peaks': len(ends), 'peak_frac': len(ends) / n_starts}


# --------------------------------------------------------------------------- thresholds
def _profile_threshold(u, means, counts):
    """(SSE_smooth - SSE_step) / SST of a level-mean profile; >0 means step-like."""
    w = counts / counts.sum()
    mu = np.sum(w * means)
    sst = np.sum(w * (means - mu) ** 2)
    # A flat profile (variance at rounding level) has no shape to classify.
    if len(u) < 3 or sst <= 1e-12 * (mu ** 2 + np.max(np.abs(means)) ** 2 + 1e-300):
        return 0.0, 0.0
    degree = 2 if len(u) >= 4 else 1
    coef = np.polyfit(u, means, degree, w=np.sqrt(w))
    sse_smooth = np.sum(w * (means - np.polyval(coef, u)) ** 2)
    sse_step = np.inf
    for s in range(1, len(u)):
        left, right = slice(0, s), slice(s, None)
        ml = np.sum(w[left] * means[left]) / np.sum(w[left])
        mr = np.sum(w[right] * means[right]) / np.sum(w[right])
        sse_step = min(sse_step, np.sum(w[left] * (means[left] - ml) ** 2) + np.sum(w[right] * (means[right] - mr) ** 2))
    return float(np.clip((sse_smooth - sse_step) / sst, -1.0, 1.0)), float(sst)


def threshold_strength(space, idx, y):
    """Max step-vs-smooth score over factors whose main-effect variance is at least 10% of
    the largest one."""
    scores, variances = [], []
    for d, (f, kind) in enumerate(space.dims):
        if kind != 'level':
            continue
        levels = idx[:, d]
        present = np.unique(levels)
        means = np.array([y[levels == l].mean() for l in present])
        counts = np.array([np.sum(levels == l) for l in present], dtype=float)
        score, var = _profile_threshold(space.factors[f].level_units[present], means, counts)
        scores.append(score)
        variances.append(var)
    scores, variances = np.array(scores), np.array(variances)
    if not len(scores) or variances.max() <= 0:
        return {'threshold_max': 0.0, 'threshold_mean': 0.0}
    keep = variances >= 0.1 * variances.max()
    return {'threshold_max': float(scores[keep].max()), 'threshold_mean': float(scores[keep].mean())}


# --------------------------------------------------------------------------- learnability
def _xgb():
    return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                        random_state=0, n_jobs=1)


def _gp(n_features):
    kernel = ConstantKernel() * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=2.5) \
        + WhiteKernel(1e-2, (1e-6, 1.0))
    return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1, random_state=0)


def _r2(y, p):
    sst = np.sum((y - y.mean()) ** 2)
    return float(np.clip(1 - np.sum((y - p) ** 2) / sst, -1, 1)) if sst > 0 else 0.0


def learnability(Z_train_sets, y_train_sets, Z_test, y_test):
    """Held-out R2 of XGBoost and a Matern GP for each training set (``Z`` in [0, 1])."""
    out = {}
    for n, Z, y in zip(LEARN_SIZES, Z_train_sets, y_train_sets):
        for name, model in (('xgb', _xgb()), ('gp', _gp(Z.shape[1]))):
            try:
                model.fit(Z, y)
                out['learn_{}_{}'.format(name, n)] = _r2(y_test, model.predict(Z_test))
            except Exception:
                out['learn_{}_{}'.format(name, n)] = np.nan
        out['learn_{}'.format(n)] = float(np.nanmean([out['learn_xgb_{}'.format(n)], out['learn_gp_{}'.format(n)]]))
    return out


# --------------------------------------------------------------------------- whole landscapes
def landscape_descriptors(f, space, rng, noise=None, f_max=None, n_base=1024, n_starts=50, full=True,
                          n_random=4000):
    """All descriptors for a noise-free landscape ``f`` on ``space``.

    ``noise`` (a NoiseModel) is used for learnability and noise_ratio, which describe
    measured data. ``full=False`` skips both (the cheap screening set).
    """
    rand = space.sample(n_random, rng)
    y_rand = f(rand)
    out = distribution(y_rand)
    sob = sobol(f, space, rng, n_base=n_base)
    out.update(sobol_summary(sob, space))
    out['corr_length'] = corr_length_model(f, space, rng)
    out.update(hill_climb_peaks(f, space, rng, n_starts=n_starts))
    out.update(threshold_strength(space, sob['A'], sob['fA']))
    if full and noise is not None:
        f_max = f_max if f_max is not None else float(y_rand.max())
        transform = space.feature_transform()
        measured = lambda idx: noise.measure(f(idx), f_max, rng)  # noqa: E731
        test = space.sample(500, rng)
        y_test = measured(test)[0]
        Zs, ys = [], []
        for n in LEARN_SIZES:
            train = space.sample(n, rng)
            Zs.append(transform(space.features(train)))
            ys.append(measured(train)[0])
        out.update(learnability(Zs, ys, transform(space.features(test)), y_test))
        mean, std, _ = measured(space.sample(300, rng))
        out['noise_ratio'] = float(np.mean(std ** 2) / np.var(mean)) if np.var(mean) > 0 else 0.0
    return out
