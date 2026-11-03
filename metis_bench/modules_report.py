"""Figures and tables for the TabPFN module tests (experiments/tabpfn_modules.yaml).

    python -m metis_bench replay experiments/tabpfn_modules.yaml

Writes to ``results/<experiment>/replay/``:

* ``best_so_far.png``: best measured value so far as a fraction of the dataset's best,
  mean over seeds with a band of one standard error.
* ``yields_by_round.png``: the measured value of every recipe each arm chose, per round.
* ``batch_spread.png``: how spread out each batch is: the mean and the smallest distance
  between its recipes, in units of the dataset's typical spacing (the median distance
  from a recipe to its nearest neighbour). Low values mean a clustered batch.
* ``hillclimb_steps.png``: how many moves each climb took to reach a local top of the
  score, per round (cap 20).
* ``pca_<dataset>.png`` (seeds with snapshots): every candidate in the round's pool on
  the dataset's first two principal components, coloured by its score percentile within
  that pool; recipes measured before the round in grey, the chosen batch ringed, recipes
  found by hill-climbing as triangles. The PCA is fitted once per dataset, on all its
  measured recipes (or 10,000 random feasible ones for the synthetic function), so every
  panel shares the same axes.
* ``summary.csv`` / ``summary.md``: per dataset and arm, and paired against the control.
"""
import json
import pathlib

import numpy as np
import pandas as pd

from .replay import (DATASET_LABELS, _best_so_far, _colors, _label, _order, _paired_lines, _swarm_offsets,
                     collect, plot_yields)
from .report import INK, INK_2, PALETTE, SURFACE, _figure, _style

MEASURED_GREY = '#8a8984'
SCORE_CMAP = ('#eef3fa', '#123f7a')     # one hue, light to dark: low to high score percentile


def collect_modules(exp_dir):
    """One row per guided round: batch spread and hill-climbing diagnostics."""
    rows = []
    for path in sorted(pathlib.Path(exp_dir).rglob('seed_*.json')):
        r = json.loads(path.read_text())
        base = {'dataset': r['problem']['subtype'], 'arm': r['arm_name'], 'seed': r['seed'], 'path': str(path)}
        for x in r['rounds']:
            if 'batch_spread' not in x:
                continue
            rows.append({**base, 'round': x['round'], 'batch_spread': x['batch_spread'],
                         'batch_min_gap': x['batch_min_gap'], 'selector': x.get('selector'),
                         'hc_steps': x.get('hc_steps'), 'hc_capped': x.get('hc_capped'), 'hc_new': x.get('hc_new'),
                         'hc_chosen': x.get('hc_chosen')})
    return pd.DataFrame(rows)


def _sem(values):
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def _mean_se_panels(frame, metrics, path, datasets, arms, title):
    """Rows: ``metrics`` as (column, y label). Columns: datasets. Mean over seeds, band +-1 SE."""
    colors = _colors(arms)
    plt, fig, axes = _figure(len(metrics), len(datasets), figsize=(3.4 * len(datasets), 2.6 * len(metrics) + 0.9),
                             squeeze=False, sharex=True)
    handles = {}
    for i, (value, ylabel) in enumerate(metrics):
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            for arm in arms:
                sub = frame[(frame.dataset == ds) & (frame.arm == arm)]
                if sub.empty:
                    continue
                g = sub.groupby('round')[value].agg(['mean', _sem])
                handles[arm], = ax.plot(g.index, g['mean'], color=colors[arm], lw=2, zorder=3,
                                        solid_capstyle='round', solid_joinstyle='round')
                ax.fill_between(g.index, g['mean'] - g['_sem'], g['mean'] + g['_sem'], color=colors[arm], alpha=0.12,
                                lw=0, zorder=2)
            _style(ax)
            if i == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=10, color=INK, loc='left')
            if i == len(metrics) - 1:
                ax.set_xlabel('Round')
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
    shown = [a for a in arms if a in handles]
    fig.legend([handles[a] for a in shown], [_label(a) for a in shown], loc='lower center', ncol=len(shown),
               frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.08 if len(metrics) == 1 else 0.05, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_hillclimb_steps(diag, path, max_steps=20):
    climbs = diag[diag.hc_steps.notna()].explode('hc_steps')
    climbs = climbs.assign(hc_steps=climbs.hc_steps.astype(float))
    datasets = _order(climbs.dataset, list(DATASET_LABELS))
    color = _colors(['hillclimb'])['hillclimb']
    plt, fig, axes = _figure(1, len(datasets), figsize=(4.6 * len(datasets), 3.5), squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        sub = climbs[climbs.dataset == ds]
        ks = sorted(sub['round'].unique())
        values = [sub.loc[sub['round'] == k, 'hc_steps'].to_numpy() for k in ks]
        ax.boxplot(values, positions=ks, widths=0.7, showfliers=False, patch_artist=True, zorder=2,
                   boxprops={'facecolor': color + '2e', 'edgecolor': INK_2, 'linewidth': 0.7},
                   medianprops={'color': INK, 'linewidth': 1.4, 'zorder': 4},
                   whiskerprops={'color': INK_2, 'linewidth': 0.7}, capprops={'color': INK_2, 'linewidth': 0.7})
        for k, y in zip(ks, values):
            ax.scatter(k + _swarm_offsets(y, 0.5), y, s=6, color=color, alpha=0.8, linewidths=0, zorder=3)
        ax.axhline(max_steps, color=INK_2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(ks[0] - 0.4, max_steps + 0.3, 'cap ({} moves)'.format(max_steps), fontsize=7, color=INK_2, va='bottom')
        per_round = diag[(diag.dataset == ds) & diag.hc_steps.notna()]
        capped, total = int(per_round.hc_capped.sum()), len(sub)
        chosen = int(per_round.hc_chosen.sum())
        ax.set_title('{}\n{} climbs, {} hit the cap, {} climbed recipes picked'.format(
            DATASET_LABELS.get(ds, ds), total, capped, chosen), fontsize=9, color=INK, loc='left')
        _style(ax)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(-0.5, max_steps + 2)
        ax.set_xticks(ks)
        ax.set_xlabel('Round')
        if j == 0:
            ax.set_ylabel('moves to reach a local top\nof the score (one dot per climb)', fontsize=8)
    fig.suptitle('Hill-climbing: how far the best pool candidates were from a local top of the score '
                 '({} seeds pooled)'.format(climbs.seed.nunique()), fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _oracle_for_dataset(cfg, dataset):
    from .experiments import problem_configs
    from .problems import get_problem
    for problem_cfg in problem_configs(cfg):
        if problem_cfg['labels']['subtype'] == dataset:
            return get_problem(problem_cfg).oracle
    raise KeyError(dataset)


def _pca(oracle, n_reference=10000):
    """PCA fitted on the dataset's recipes (or random feasible designs), in embedding coordinates."""
    from sklearn.decomposition import PCA
    space = oracle.space
    if getattr(space, 'pool', None) is not None:
        reference = np.vstack([space.pool, getattr(oracle, 'test_idx', np.empty((0, space.n_dims), np.int64))])
    else:
        reference = space.sample(n_reference, np.random.default_rng(0))
    pca = PCA(n_components=2).fit(space.embedding(reference))
    return pca, pca.transform(space.embedding(reference))


def plot_pca(cfg, exp_dir, dataset, arms, out, seed=0, rounds=(2, 4, 6, 8, 10)):
    from matplotlib.colors import LinearSegmentedColormap
    oracle = _oracle_for_dataset(cfg, dataset)
    space = oracle.space
    pca, ref_xy = _pca(oracle)
    shown = []
    for arm in arms:
        files = [p for p in pathlib.Path(exp_dir).rglob('seed_{:03d}.npz'.format(seed))
                 if p.parent.name == arm and json.loads(p.with_suffix('.json').read_text())['problem']['subtype'] == dataset]
        if files:
            shown.append((arm, np.load(files[0]), json.loads(files[0].with_suffix('.json').read_text())))
    if not shown:
        return None
    available = sorted({int(k[1:].split('_')[0]) for _, snap, _ in shown for k in snap.files})
    cols = [k for k in rounds if k in available] or available[:5]
    cmap = LinearSegmentedColormap.from_list('score', SCORE_CMAP)
    plt, fig, axes = _figure(len(shown), len(cols), figsize=(2.55 * len(cols) + 0.9, 2.45 * len(shown) + 0.8),
                             squeeze=False, sharex=True, sharey=True)
    pad = 0.05 * np.ptp(ref_xy, axis=0)
    lo, hi = ref_xy.min(axis=0) - pad, ref_xy.max(axis=0) + pad
    scatter = None
    for i, (arm, snap, result) in enumerate(shown):
        log = result['evaluated']
        measured_idx, measured_round = np.array(log['idx'], dtype=np.int64), np.array(log['round'])
        for j, k in enumerate(cols):
            ax = axes[i][j]
            _style(ax)
            key = 'r{}_cand'.format(k)
            if key not in snap.files:
                ax.text(0.5, 0.5, 'no snapshot', transform=ax.transAxes, ha='center', fontsize=7, color=INK_2)
                continue
            cand = snap[key].astype(np.int64)
            xy = pca.transform(space.embedding(cand))
            score = snap['r{}_score'.format(k)]
            pct = (np.argsort(np.argsort(score, kind='stable'), kind='stable') + 0.5) / len(score)
            order = np.argsort(pct)
            scatter = ax.scatter(xy[order, 0], xy[order, 1], c=pct[order], cmap=cmap, vmin=0, vmax=1, s=3,
                                 linewidths=0, rasterized=True, zorder=2)
            before = measured_idx[measured_round < k]
            if len(before):
                m = pca.transform(space.embedding(before))
                ax.scatter(m[:, 0], m[:, 1], s=7, color=MEASURED_GREY, linewidths=0, zorder=3)
            chosen = snap['r{}_chosen'.format(k)]
            n_pool = int(snap['r{}_n_pool'.format(k)])
            pool_pick, climb_pick = chosen[chosen < n_pool], chosen[chosen >= n_pool]
            ax.scatter(xy[pool_pick, 0], xy[pool_pick, 1], s=34, facecolors='none', edgecolors=INK, linewidths=1.0,
                       zorder=4)
            if len(climb_pick):
                ax.scatter(xy[climb_pick, 0], xy[climb_pick, 1], s=40, marker='^', facecolors=PALETTE[2],
                           edgecolors=INK, linewidths=0.8, zorder=5)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title('Round {}'.format(k), fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(_label(arm), fontsize=9, color=INK)
    ev = pca.explained_variance_ratio_
    fig.suptitle('{}: the scored pool each round (seed {}); axes PC1 ({:.0%}) and PC2 ({:.0%}) of all recipes.\n'
                 'Dots: candidates, darker = higher score. Grey: measured before the round. Rings: the batch. '
                 'Triangles: picked after hill-climbing.'.format(DATASET_LABELS.get(dataset, dataset), seed, ev[0],
                                                                 ev[1]),
                 fontsize=9, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    cax = fig.add_axes([0.3, 0.045, 0.4, 0.011])
    bar = fig.colorbar(scatter, cax=cax, orientation='horizontal')
    bar.set_label("score percentile in the round's pool", fontsize=8, color=INK_2)
    bar.ax.tick_params(labelsize=7, colors=INK_2)
    path = out / 'pca_{}.png'.format(dataset)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_modules(cfg, exp_dir):
    exp_dir = pathlib.Path(exp_dir)
    out = exp_dir / 'replay'
    out.mkdir(parents=True, exist_ok=True)
    points, _, runs = collect(exp_dir)
    if runs.empty:
        raise FileNotFoundError('no results under {}'.format(exp_dir))
    diag = collect_modules(exp_dir)
    datasets = _order(runs.dataset, list(DATASET_LABELS))
    arms = _order(runs.arm, [a['name'] for a in cfg['arms']])
    _mean_se_panels(_best_so_far(points), [('best_fraction', "best measured value so far\n(fraction of the dataset's "
                                            "best)")], out / 'best_so_far.png', datasets, arms,
                    'Best recipe found so far (mean over {} seeds; band: one standard error)'.format(runs.seed.nunique()))
    plot_yields(points, out / 'yields_by_round.png', datasets, arms)
    if not diag.empty:
        _mean_se_panels(diag, [('batch_spread', 'mean distance between\nrecipes in the batch'),
                               ('batch_min_gap', 'smallest distance between\nrecipes in the batch')],
                        out / 'batch_spread.png', datasets, arms,
                        'How spread out each batch is, in units of the dataset\'s typical spacing (median distance '
                        'from a recipe to its nearest one). Low = clustered.')
        if diag.hc_steps.notna().any():
            max_steps = max((a.get('hillclimb') or {}).get('max_steps', 20) for a in cfg['arms'])
            plot_hillclimb_steps(diag, out / 'hillclimb_steps.png', max_steps=max_steps)
    for ds in datasets:
        plot_pca(cfg, exp_dir, ds, arms, out)
    # Table: per dataset and arm.
    stats = runs.groupby(['dataset', 'arm']).agg(
        seeds=('seed', 'nunique'), found=('round_found', 'count'), best_mean=('best_fraction', 'mean'),
        best_se=('best_fraction', _sem), auc_mean=('auc_norm_regret', 'mean'), auc_se=('auc_norm_regret', _sem))
    if not diag.empty:
        spread = diag.groupby(['dataset', 'arm']).agg(spread=('batch_spread', 'mean'), min_gap=('batch_min_gap', 'mean'))
        stats = stats.join(spread)
        climbs = diag[diag.hc_steps.notna()].explode('hc_steps')
        if not climbs.empty:
            climbs = climbs.assign(hc_steps=climbs.hc_steps.astype(float))
            stats = stats.join(climbs.groupby(['dataset', 'arm']).agg(
                climb_median=('hc_steps', 'median'), climb_q1=('hc_steps', lambda v: v.quantile(0.25)),
                climb_q3=('hc_steps', lambda v: v.quantile(0.75)), climbs=('hc_steps', 'size')))
            stats = stats.join(diag[diag.hc_steps.notna()].groupby(['dataset', 'arm']).agg(
                capped=('hc_capped', 'sum'), climbed_picked=('hc_chosen', 'sum')))
    stats.to_csv(out / 'summary.csv')
    lines = ['# {} (exploratory)'.format(cfg['name']), '',
             'Mean (standard error) over {} seeds. best fraction = best measured value found / best in the dataset; '
             'found = seeds that measured the best recipe; AUC = mean normalised regret over rounds (lower = good '
             'recipes sooner). Spread and gap: mean and smallest distance between recipes in a batch, in units of the '
             "dataset's typical spacing, averaged over rounds. Climb moves: median [quartiles] over all climbs, capped "
             'at 20.'.format(runs.seed.nunique()), '',
             '| dataset | arm | found | best fraction | AUC norm. regret | batch spread | smallest gap | climb moves '
             '| climbs capped | climbed recipes picked |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for ds in datasets:
        for arm in arms:
            if (ds, arm) not in stats.index:
                continue
            s = stats.loc[(ds, arm)]
            has_climb = 'climb_median' in stats.columns and pd.notna(s.get('climb_median'))
            lines.append('| {} | {} | {}/{} | {:.3f} ({:.3f}) | {:.3f} ({:.3f}) | {} | {} | {} | {} | {} |'.format(
                DATASET_LABELS.get(ds, ds), _label(arm), int(s.found), int(s.seeds), s.best_mean, s.best_se, s.auc_mean,
                s.auc_se, '{:.2f}'.format(s.spread) if 'spread' in s and pd.notna(s.spread) else '-',
                '{:.2f}'.format(s.min_gap) if 'min_gap' in s and pd.notna(s.min_gap) else '-',
                '{:g} [{:g}, {:g}]'.format(s.climb_median, s.climb_q1, s.climb_q3) if has_climb else '-',
                '{}/{}'.format(int(s.capped), int(s.climbs)) if has_climb else '-',
                int(s.climbed_picked) if has_climb else '-'))
    lines += _paired_lines(cfg, runs, datasets)
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out
