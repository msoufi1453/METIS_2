"""Calibrate the synthetic suite against the Borkowski and PURE datasets.

1. ``real_references()`` computes descriptors for both datasets: raw samples where
   possible, otherwise fitted XGBoost and GP models. Each (dataset, model) pair becomes
   one reference vector, so the realism of a synthetic function never depends on a
   single model's quirks.
2. ``screen_candidates()`` builds many candidate specs per landscape subtype at 12 and 27
   dims and computes the cheap screening descriptors.
3. ``select_suite()`` keeps, per subtype, a mix of specs drawn toward the real region
   (probability proportional to exp(-d^2 / 2 tau^2)) and specs that cover the rest of
   the descriptor space (farthest point), while making sure every dial level appears.
4. ``finalize_suite()`` computes the full descriptors for the chosen functions, the final
   realism scores, the dev/test split and the 2D map, and writes the suite manifest.

Realism score = standardised Euclidean distance (RMS over descriptors) to the nearest
real reference. Lower is more realistic.
"""
import json
import pathlib

import numpy as np
import pandas as pd

from metis.space import DesignSpace
from metis.surrogates import GPSurrogate

from . import descriptors as D
from .noise import PRESETS
from .oracles.real import (BORKOWSKI_FACTORS, ROOT, _fit_gold, _pure_gold_params, borkowski_space, load_borkowski,
                           load_pure, pure_space)
from .synthetic import DIMENSIONS, SUBTYPES, FunctionSpec, build_function

PURE_RAW = ROOT / 'Examples' / 'PURE' / 'Data' / 'raw' / 'msb200950-sup-0003.xls'
DIALS = {'active': ('all', 'subset'), 'zero_fraction': (0.0, 0.2, 0.5), 'n_categorical': (0, 2)}


# --------------------------------------------------------------------------- real data
def _model_fn(model, space, transform=None):
    def f(idx):
        X = space.features(idx)
        return model.predict(transform(X) if transform else X).astype(float)
    return f


def _raw_learnability(space, idx, y, rng, repeats=3):
    transform = space.feature_transform()
    Z = transform(space.features(idx))
    rows = []
    for _ in range(repeats):
        order = rng.permutation(len(y))
        test = order[:500]
        rest = order[500:] if len(order) > 800 else order[min(500, len(order) // 2):]
        Zs = [Z[rest[:n]] for n in D.LEARN_SIZES]
        ys = [y[rest[:n]] for n in D.LEARN_SIZES]
        rows.append(D.learnability(Zs, ys, Z[test], y[test]))
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def _pure_noise_ratio():
    raw = pd.read_excel(PURE_RAW, sheet_name='data', header=0).iloc[:, :4]
    raw.columns = ['experiment', 'component', 'conc', 'fluor']
    raw = raw.dropna(subset=['fluor'])
    groups = raw.groupby(['experiment', 'component', 'conc']).fluor
    stats = groups.agg(['count', 'var'])
    # Groups with variance exactly 0 hold copied reference values, not re-measurements.
    rep = stats[(stats['count'] > 1) & (stats['var'] > 0)]
    pooled = float(np.sum(rep['var'] * (rep['count'] - 1)) / np.sum(rep['count'] - 1))
    return pooled / float(raw.fluor.var()), int(len(rep))


def _model_descriptors(f, space, rng, n_base, marginals=None, peak_space=None):
    out = D.distribution(f(D.random_designs(space, 4000, rng, marginals)))
    sob = D.sobol(f, space, rng, n_base=n_base, marginals=marginals)
    out.update(D.sobol_summary(sob, space))
    out['corr_length'] = D.corr_length_model(f, space, rng, marginals=marginals)
    out.update(D.hill_climb_peaks(f, peak_space or space, rng, n_starts=50))
    out.update(D.threshold_strength(space, sob['A'], sob['fA']))
    return out


def real_references(seed=0, n_base=1024, cache=True):
    """Descriptor table for the real datasets: rows (dataset, source) with source in
    {raw, xgb, gp}, plus the combined reference vectors (dataset, 'ref_xgb' / 'ref_gp').
    Each dataset is cached separately (artifacts/calibration/real_<dataset>.csv)."""
    path = ROOT / 'artifacts' / 'calibration' / 'real_descriptors.csv'
    if cache and path.exists():
        return pd.read_csv(path)
    parts = []
    for k, (dataset, builder) in enumerate((('borkowski', _borkowski_rows), ('pure', _pure_rows))):
        part_path = path.parent / 'real_{}.csv'.format(dataset)
        if cache and part_path.exists():
            parts.append(pd.read_csv(part_path))
            continue
        frame = pd.DataFrame(builder(np.random.default_rng([seed, k]), seed, n_base))
        part_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(part_path, index=False)
        parts.append(frame)
    table = pd.concat(parts, ignore_index=True)
    table = pd.concat([table, _combine_references(table)], ignore_index=True)
    table.to_csv(path, index=False)
    return table


def _borkowski_rows(rng, seed, n_base):
    rows = []
    data = load_borkowski()
    space = borkowski_space()
    idx = space.from_features(data[BORKOWSKI_FACTORS].to_numpy(float))
    y = data['yield'].to_numpy(float)
    raw = {**D.distribution(data.loc[data.plate == 1, 'yield'].to_numpy(float)),
           'corr_length': D.corr_length_raw(space, idx, y, rng),
           **D.threshold_strength(space, idx, y),
           **_raw_learnability(space, idx, y, rng),
           'noise_ratio': float(np.mean(data['yield_std'] ** 2) / np.var(y))}
    rows.append({'dataset': 'borkowski', 'source': 'raw', **raw})
    X = space.features(idx)
    from .oracles.real import BORKOWSKI_GOLD_PARAMS
    xgb, _ = _fit_gold(X, y, BORKOWSKI_GOLD_PARAMS, seed=seed, split_seed=seed)
    gp = GPSurrogate(input_transform=space.feature_transform(), n_restarts=1).fit(X, y)
    models = {'xgb': _model_fn(xgb, space), 'gp': lambda i, g=gp: g.predict(space.features(i)).mean}
    for name, f in models.items():
        rows.append({'dataset': 'borkowski', 'source': name, **_model_descriptors(f, space, rng, n_base)})
    return rows


def _pure_rows(rng, seed, n_base):
    rows = []
    pure = load_pure()
    Xp = pure.drop(columns='yield')
    yp = pure['yield'].to_numpy(float)
    grid = pure_space('grid')
    idx_p = grid.from_features(Xp.to_numpy(float))
    noise_ratio, n_groups = _pure_noise_ratio()
    raw_p = {**D.distribution(yp), 'corr_length': D.corr_length_raw(grid, idx_p, yp, rng),
             **D.threshold_strength(grid, idx_p, yp), **_raw_learnability(grid, idx_p, yp, rng),
             'noise_ratio': noise_ratio, 'noise_groups': n_groups}
    rows.append({'dataset': 'pure', 'source': 'raw', **raw_p})
    params, _ = _pure_gold_params(seed)
    xgb_p, _ = _fit_gold(Xp.to_numpy(float), yp, params, seed=seed, split_seed=41)
    sub = rng.choice(len(yp), size=min(600, len(yp)), replace=False)   # ARD GP in 69 dims: fit on 600 points
    gp_p = GPSurrogate(input_transform=grid.feature_transform(), n_restarts=0).fit(Xp.to_numpy(float)[sub], yp[sub])
    models = {'xgb': _model_fn(xgb_p, grid), 'gp': lambda i, g=gp_p: g.predict(grid.features(i)).mean}
    marginals = D.empirical_marginals(grid, idx_p)
    pool = pure_space('pool')
    for name, f in models.items():
        rows.append({'dataset': 'pure', 'source': name,
                     **_model_descriptors(f, grid, rng, n_base // 2, marginals=marginals, peak_space=pool)})
    return rows


# Which source feeds each descriptor of a real reference vector.
RAW_FIRST = {'borkowski': {'zero_frac', 'skew', 'log_top_median', 'corr_length', 'threshold_max', 'learn_32',
                           'learn_100', 'learn_300', 'noise_ratio'},
             'pure': {'zero_frac', 'skew', 'log_top_median', 'corr_length', 'threshold_max', 'learn_32', 'learn_100',
                      'learn_300', 'noise_ratio'}}


def _combine_references(table):
    """Reference vector per (dataset, model): raw values where measurable, the model elsewhere.

    Distribution descriptors come from raw samples for both datasets: Borkowski's first
    (random) plate, and PURE's titrations, whose distribution is the empirical-marginal
    product the PURE models are evaluated under anyway.
    """
    rows = []
    for dataset in ('borkowski', 'pure'):
        raw = table[(table.dataset == dataset) & (table.source == 'raw')].iloc[0]
        for model in ('xgb', 'gp'):
            fitted = table[(table.dataset == dataset) & (table.source == model)].iloc[0]
            row = {'dataset': dataset, 'source': 'ref_' + model}
            for name in D.REALISM_DESCRIPTORS:
                row[name] = raw[name] if name in RAW_FIRST[dataset] else fitted[name]
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- realism
def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


# Heavy-tailed descriptors are compressed before standardising: a flat landscape with one
# spike has skew in the tens, and without the transform those few values would set the
# scale and leave skew almost no weight for everything else.
TRANSFORMS = {'skew': _signed_log1p}


def descriptor_matrix(frame, columns):
    out = frame[list(columns)].astype(float).copy()
    for col, fn in TRANSFORMS.items():
        if col in out:
            out[col] = fn(out[col])
    return out


def standardize(frame, columns, reference=None):
    values = descriptor_matrix(frame, columns)
    ref = values if reference is None else descriptor_matrix(reference, columns)
    mu, sd = ref.mean(), ref.std(ddof=0).replace(0, 1.0)
    return (values - mu) / sd, mu, sd


def realism_scores(synthetic, real_refs, columns):
    """RMS standardised distance from each synthetic row to its nearest real reference.

    Standardisation uses the synthetic rows and the references together, so the scale
    reflects the whole space the suite spans. Skew is compressed first (``TRANSFORMS``).
    """
    refs = real_refs[real_refs.source.str.startswith('ref_')].reset_index(drop=True)
    S, R = descriptor_matrix(synthetic, columns), descriptor_matrix(refs, columns)
    union = pd.concat([S, R], ignore_index=True)
    mu = union.mean()
    sd = union.std(ddof=0).replace(0, 1.0)
    Zs = ((S - mu) / sd).fillna(0).to_numpy()
    Zr = ((R - mu) / sd).fillna(0).to_numpy()
    dist = np.sqrt(np.mean((Zs[:, None, :] - Zr[None, :, :]) ** 2, axis=2))
    nearest = dist.argmin(axis=1)
    return dist.min(axis=1), [refs.dataset[k] + '/' + refs.source[k][4:] for k in nearest]


# --------------------------------------------------------------------------- suite
def random_spec(subtype, rng, seed):
    return FunctionSpec(subtype=subtype,
                        active=str(rng.choice(DIALS['active'])),
                        zero_fraction=float(rng.choice(DIALS['zero_fraction'])),
                        n_categorical=int(rng.choice(DIALS['n_categorical'])),
                        contrast=float(np.round(np.exp(rng.uniform(np.log(2), np.log(1000))), 2)),
                        seed=int(seed))


def _describe_task(task):
    """Worker: descriptors of one spec at one dimensionality (pure function of its inputs)."""
    spec_dict, n_dims, rng_seed, kwargs = task
    spec = FunctionSpec.from_dict(spec_dict)
    fn = build_function(spec, n_dims)
    noise = kwargs.pop('noise', None)
    desc = D.landscape_descriptors(fn, fn.space, np.random.default_rng(rng_seed),
                                   noise=PRESETS[noise] if noise else None, f_max=fn.f_max, **kwargs)
    return {'spec_id': spec.id, 'function_id': spec.function_id(n_dims), 'n_dims': n_dims,
            'landscape': spec.landscape, **spec.to_dict(), **desc}


def _map(tasks, jobs, progress, label):
    """Run ``_describe_task`` over tasks, in order, optionally in worker processes.
    Results don't depend on ``jobs``: every task carries its own seed."""
    if jobs <= 1:
        out = []
        for i, t in enumerate(tasks, 1):
            out.append(_describe_task(t))
            if i % 20 == 0:
                progress('{} {}/{}'.format(label, i, len(tasks)))
        return out
    from concurrent.futures import ProcessPoolExecutor

    from .campaign import _limit_threads
    _limit_threads()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        out = list(pool.map(_describe_task, tasks, chunksize=2))
    progress('{} {}/{}'.format(label, len(out), len(tasks)))
    return out


def screen_candidates(n_per_subtype=24, seed=0, n_base=256, jobs=1, progress=print):
    rng = np.random.default_rng(seed)
    tasks = []
    for subtype in SUBTYPES:
        for k in range(n_per_subtype):
            spec = random_spec(subtype, rng, seed=1000 * SUBTYPES.index(subtype) + k)
            for n_dims in DIMENSIONS:
                tasks.append((spec.to_dict(), n_dims, [seed, SUBTYPES.index(subtype), k, n_dims],
                              {'n_base': n_base, 'n_starts': 20, 'full': False}))
    return pd.DataFrame(_map(tasks, jobs, progress, 'screened'))


def select_suite(screened, refs, per_subtype=5, realistic_share=0.6, seed=0):
    """Choose ``per_subtype`` specs per subtype: about ``realistic_share`` biased toward the
    real region, the rest for coverage. A spec's distance is the mean over its 12- and
    27-dim instances."""
    rng = np.random.default_rng(seed)
    cols = list(D.SCREENING_DESCRIPTORS)
    screened = screened.copy()
    screened['screen_distance'], _ = realism_scores(screened, refs, cols)
    by_spec = screened.groupby('spec_id').agg(subtype=('subtype', 'first'), distance=('screen_distance', 'mean'),
                                              **{c: (c, 'mean') for c in cols}).reset_index()
    Z = standardize(by_spec, cols)[0].fillna(0).to_numpy()
    chosen = []
    for subtype in SUBTYPES:
        pool = by_spec.index[by_spec.subtype == subtype].tolist()
        n_real = int(round(realistic_share * per_subtype))
        tau = float(np.median(by_spec.loc[pool, 'distance']))
        weights = np.exp(-0.5 * (by_spec.loc[pool, 'distance'].to_numpy() / tau) ** 2)
        picks = list(rng.choice(pool, size=n_real, replace=False, p=weights / weights.sum()))
        while len(picks) < per_subtype:   # coverage: farthest from everything already chosen
            rest = [i for i in pool if i not in picks]
            dmin = np.min(np.linalg.norm(Z[rest][:, None] - Z[chosen + picks][None], axis=2), axis=1)
            picks.append(rest[int(np.argmax(dmin))])
        chosen += picks
    selected = by_spec.loc[chosen].copy()
    selected['selection'] = ['realistic' if i % per_subtype < round(realistic_share * per_subtype) else 'coverage'
                             for i in range(len(selected))]
    return selected


def assign_split(spec_ids, subtypes):
    """Deterministic, stratified dev/test split: within each subtype (sorted ids), every
    third spec goes to dev. Phase 3 tuning sees only dev; confirmatory claims use test."""
    split = {}
    frame = pd.DataFrame({'spec_id': spec_ids, 'subtype': subtypes}).sort_values(['subtype', 'spec_id'])
    for _, g in frame.groupby('subtype'):
        for k, sid in enumerate(g.spec_id):
            split[sid] = 'dev' if k % 3 == 1 else 'test'
    return split


def finalize_suite(selected, screened, refs, out_dir, noise='realistic', n_base=1024, jobs=1, progress=print):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {r.spec_id: FunctionSpec.from_dict(screened[screened.spec_id == r.spec_id].iloc[0].to_dict())
             for r in selected.itertuples()}
    tasks = [(spec.to_dict(), n_dims, [7, k, n_dims], {'n_base': n_base, 'noise': noise})
             for k, spec in enumerate(specs.values()) for n_dims in DIMENSIONS]
    table = pd.DataFrame(_map(tasks, jobs, progress, 'described'))
    table['realism'], table['nearest_real'] = realism_scores(table, refs, D.REALISM_DESCRIPTORS)
    split = assign_split(list(specs), [s.subtype for s in specs.values()])
    table['split'] = table.spec_id.map(split)
    table['selection'] = table.spec_id.map(dict(zip(selected.spec_id, selected.selection)))
    table.to_csv(out_dir / 'suite_descriptors.csv', index=False)
    write_manifest(table, refs, out_dir, noise)
    return table


def write_manifest(table, refs, out_dir, noise='realistic'):
    """suite.json, the realism map and a copy of the real references, from the descriptor table."""
    out_dir = pathlib.Path(out_dir)
    manifest = {'noise_for_descriptors': noise, 'descriptors': list(D.REALISM_DESCRIPTORS),
                'functions': [{'function_id': r.function_id, 'spec': FunctionSpec.from_dict(r._asdict()).to_dict(),
                               'n_dims': int(r.n_dims), 'landscape': r.landscape, 'split': r.split,
                               'realism': float(r.realism), 'nearest_real': r.nearest_real, 'selection': r.selection}
                              for r in table.itertuples()]}
    (out_dir / 'suite.json').write_text(json.dumps(manifest, indent=1))
    refs.to_csv(out_dir / 'real_descriptors.csv', index=False)
    plot_map(table, refs, out_dir / 'descriptor_map.png')
    return manifest


def plot_map(table, refs, path):
    """2D PCA map of standardised descriptors: synthetic functions and real references."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cols = list(D.REALISM_DESCRIPTORS)
    ref_rows = refs[refs.source.str.startswith('ref_')].reset_index(drop=True)
    union = pd.concat([table[cols], ref_rows[cols]], ignore_index=True)
    Z = standardize(union, cols)[0].fillna(0).to_numpy()
    U, S, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    P = (Z - Z.mean(axis=0)) @ Vt[:2].T
    explained = S[:2] ** 2 / np.sum(S ** 2)
    n = len(table)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    palette = dict(zip(sorted(table.landscape.unique()), ['#2a6f97', '#e07a5f', '#81b29a', '#9b5de5', '#f2cc8f']))
    for landscape, color in palette.items():
        for n_dims, marker in ((12, 'o'), (27, 's')):
            m = ((table.landscape == landscape) & (table.n_dims == n_dims)).to_numpy()
            ax.scatter(P[:n][m, 0], P[:n][m, 1], c=color, marker=marker, s=28, alpha=0.8,
                       label='{} ({}d)'.format(landscape, n_dims), edgecolors='none')
    for k, row in ref_rows.iterrows():
        ax.scatter(P[n + k, 0], P[n + k, 1], marker='*', s=260, c='black', edgecolors='white', zorder=5)
        # Alternate label offsets: the two model versions of a dataset often sit close together.
        ax.annotate('{} ({})'.format(row.dataset, row.source[4:]), P[n + k], textcoords='offset points',
                    xytext=(8, 8) if k % 2 == 0 else (8, -14), fontsize=9)
    ax.set_xlabel('PC1 ({:.0%} of variance)'.format(explained[0]))
    ax.set_ylabel('PC2 ({:.0%} of variance)'.format(explained[1]))
    ax.set_title('Synthetic suite vs real datasets (standardised descriptors)')
    ax.legend(fontsize=7, ncol=2, frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    pd.DataFrame({'id': list(table.function_id) + list(ref_rows.dataset + '/' + ref_rows.source),
                  'pc1': P[:, 0], 'pc2': P[:, 1]}).to_csv(pathlib.Path(path).with_suffix('.csv'), index=False)
