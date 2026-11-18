"""Figures and tables for the TabPFN module tests (experiments/tabpfn_modules.yaml).

    python -m metis_bench replay experiments/tabpfn_modules.yaml

Writes to ``results/<experiment>/replay/``:

* ``module_effects.png``: each module against the control, paired by seed: the change in
  AUC regret and in the best value found (one dot per seed, a bar at the mean).
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
  found by hill-climbing as triangles, and "xn" where n picks differ only in a categorical
  factor (on the cell-growth grid: one medium with n strains). The PCA
  is fitted once per dataset, on all its measured recipes (or 10,000 random feasible ones
  for the synthetic function), so every panel shares the same axes.
* ``summary.csv`` / ``summary.md``: per dataset and arm; each module paired by seed
  against the control; held-out accuracy; for the cell-growth grid, how many distinct
  media and strains each batch covers.
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


def paired_effects(runs, reference='control'):
    """One row per (dataset, module, seed): the module's AUC regret and best fraction minus the
    reference arm's, same seed (so the same round-1 recipes)."""
    ref = runs[runs.arm == reference].set_index(['dataset', 'seed'])
    rows = []
    for _, r in runs[runs.arm != reference].iterrows():
        if (r.dataset, r.seed) not in ref.index:
            continue
        base = ref.loc[(r.dataset, r.seed)]
        rows.append({'dataset': r.dataset, 'arm': r.arm, 'seed': r.seed,
                     'd_auc': r.auc_norm_regret - base.auc_norm_regret,
                     'd_best': r.best_fraction - base.best_fraction})
    return pd.DataFrame(rows)


def plot_effects(effects, path, datasets, arms):
    """Rows: change in AUC regret (lower = better) and in best value found (higher = better)."""
    from matplotlib.ticker import FuncFormatter
    colors = _colors(arms)
    modules = [a for a in arms if a in set(effects.arm)]
    metrics = [('d_auc', 'change in AUC regret\n(below 0 = better than no module)'),
               ('d_best', 'change in best value found\n(above 0 = better; fraction of best)')]
    plt, fig, axes = _figure(len(metrics), len(datasets), figsize=(3.8 * len(datasets), 6.4), squeeze=False)
    for i, (value, ylabel) in enumerate(metrics):
        lim = float(np.nanmax(np.abs(effects[value]))) * 1.15 or 0.1
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            here = [m for m in modules if ((effects.dataset == ds) & (effects.arm == m)).any()]
            ax.axhline(0, color=INK_2, lw=0.9, ls=(0, (4, 3)), zorder=1)
            for x, m in enumerate(here):
                v = effects[(effects.dataset == ds) & (effects.arm == m)][value].to_numpy()
                offsets = np.linspace(-0.12, 0.12, len(v)) if len(v) > 1 else np.zeros(1)
                ax.scatter(x + offsets, v, s=30, color=colors[m], edgecolors=SURFACE, linewidths=0.8, zorder=3)
                ax.plot([x - 0.28, x + 0.28], [v.mean()] * 2, color=INK, lw=2, solid_capstyle='round', zorder=4)
            _style(ax)
            ax.set_xlim(-0.6, max(len(here), 1) - 0.4)
            ax.set_ylim(-lim, lim)
            ax.set_xticks(range(len(here)))
            ax.set_xticklabels([_label(m).replace(', ', '\n').replace(' ', '\n').replace('-', '-\n') for m in here],
                               fontsize=7.5, color=INK_2)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: '{:+.2f}'.format(v).replace('-', '−')
                                                       if v else '0'))
            if i == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=10, color=INK, loc='left')
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle('Each module against TabPFN with no module, paired by seed (same first batch). Dots: seeds; '
                 'bar: mean.', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _same_levels(space, picks, xy):
    """Picks that differ only in categorical factors (e.g. one medium, several strains):
    [(position of the first, count)] for groups of 2 or more."""
    groups = {}
    for row, p in zip(picks, xy):
        key = row[space.level_dims].tobytes()
        if key in groups:
            groups[key][1] += 1
        else:
            groups[key] = [p, 1]
    return [(p, n) for p, n in groups.values() if n > 1]


def plot_hillclimb_steps(diag, path, max_steps=20):
    """Box plots of accepted hill-climbing moves per round and dataset."""
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
    """PCA maps of the scored pool and the chosen batch for chosen rounds (needs snapshot ``.npz`` files)."""
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
            # Random drawing order: where points overlap, the colour shows the local mix of scores
            # (drawing the best last would make a dense map look uniformly dark).
            order = np.random.default_rng(k).permutation(len(score))
            scatter = ax.scatter(xy[order, 0], xy[order, 1], c=pct[order], cmap=cmap, vmin=0, vmax=1,
                                 s=3 if len(score) <= 5000 else 1.5, linewidths=0, rasterized=True, zorder=2)
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
            placed = []
            for position, n in _same_levels(space, cand[chosen], xy[chosen]):
                # Labels of nearby groups take turns: above right, below right, above left, below left.
                near = sum(bool(np.all(np.abs(position - p) < 0.08 * (hi - lo))) for p in placed)
                offset = [(5, 4), (5, -11), (-20, 4), (-20, -11)][near % 4]
                ax.annotate('×{}'.format(n), position, xytext=offset, textcoords='offset points', fontsize=7.5,
                            color=INK, fontweight='bold', zorder=6)
                placed.append(position)
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
                 'Dots: candidates, darker = higher score. Grey: measured before the round. Rings: the batch '
                 '(×n: n picks differing only in a category, e.g. one medium with n strains). Triangles: picked '
                 'after hill-climbing.'.format(DATASET_LABELS.get(dataset, dataset), seed, ev[0],
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
    """All figures and tables of the TabPFN module tests into ``<exp_dir>/replay/`` (docs/MODULES.md)."""
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
    reference = cfg.get('reference', 'control')
    effects = paired_effects(runs, reference)
    if not effects.empty:
        plot_effects(effects, out / 'module_effects.png', datasets, arms)
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
    if not effects.empty:
        lines += ['', '## Each module against the control ({}), seed by seed'.format(_label(reference)), '',
                  'AUC regret of the module minus that of {} in the same seed (negative = better), and the '
                  'number of seeds in which the module was better.'.format(_label(reference)), '',
                  '| dataset | module | difference per seed | mean | better in |', '|---|---|---|---|---|']
        for ds in datasets:
            for arm in arms:
                e = effects[(effects.dataset == ds) & (effects.arm == arm)].sort_values('seed')
                if e.empty:
                    continue
                lines.append('| {} | {} | {} | {:+.3f} | {}/{} |'.format(
                    DATASET_LABELS.get(ds, ds), _label(arm), ', '.join('{:+.3f}'.format(v) for v in e.d_auc),
                    e.d_auc.mean(), int((e.d_auc < 0).sum()), len(e)))
    accuracy = runs.groupby(['dataset', 'arm'])[['final_test_r2', 'final_test_spearman', 'final_test_cov80']].mean()
    lines += ['', '## Held-out accuracy after the last round', '',
              "R2 and Spearman rank correlation on the held-out recipes (synthetic: 1,000 random ones), and the share "
              "inside the model's 80% interval. Mean over seeds.", '',
              '| dataset | arm | R2 | Spearman | 80% coverage |', '|---|---|---|---|---|']
    for ds in datasets:
        for arm in arms:
            if (ds, arm) in accuracy.index:
                a = accuracy.loc[(ds, arm)]
                lines.append('| {} | {} | {:.3f} | {:.3f} | {:.3f} |'.format(
                    DATASET_LABELS.get(ds, ds), _label(arm), a.final_test_r2, a.final_test_spearman,
                    a.final_test_cov80))
    cover = _media_per_batch(cfg, exp_dir) if 'cell_growth' in datasets else None
    if cover is not None:
        lines += ['', '## Cell growth: what a batch covers', '',
                  'Distinct media and strains among the 10 recipes of a batch (mean over rounds 2-10 and seeds).', '',
                  '| arm | media | strains |', '|---|---|---|']
        for arm in arms:
            if arm in cover.index:
                lines.append('| {} | {:.1f} | {:.1f} |'.format(_label(arm), cover.loc[arm, 'media'],
                                                               cover.loc[arm, 'strains']))
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


def _media_per_batch(cfg, exp_dir):
    """Cell growth: distinct media (composition) and strains per guided batch, mean per arm."""
    space = _oracle_for_dataset(cfg, 'cell_growth').space
    rows = []
    for path in sorted(pathlib.Path(exp_dir).rglob('seed_*.json')):
        r = json.loads(path.read_text())
        if r['problem']['subtype'] != 'cell_growth':
            continue
        idx, rnd = np.array(r['evaluated']['idx'], dtype=np.int64), np.array(r['evaluated']['round'])
        for k in sorted(set(rnd) - {1}):
            batch = idx[rnd == k]
            rows.append({'arm': r['arm_name'], 'media': len({b[:space.n_conc].tobytes() for b in batch}),
                         'strains': len(set(batch[:, space.strain_dim]))})
    return pd.DataFrame(rows).groupby('arm').mean() if rows else None
