"""Paper-style figures and table for the measured-data replay (experiments/replay_paper.yaml).

    python -m metis_bench replay experiments/replay_paper.yaml

Writes to ``results/<experiment>/replay/``:

* ``yields_by_round.png``: measured yield of every chosen recipe per round, one panel per
  dataset and model (all seeds pooled), as in Fig. 1 of the METIS paper.
* ``best_so_far.png``: best measured yield so far as a fraction of the pool's best, mean
  over seeds with the seed range shaded.
* ``heldout_r2.png``: R^2 on the held-out 20% of recipes after each round's fit.
* ``summary.csv`` / ``summary.md``: per dataset and model, mean (sd) over seeds.
"""
import json
import pathlib

import numpy as np
import pandas as pd

from .report import INK, INK_2, PALETTE, SURFACE, _figure, _style

DATASET_LABELS = {'pure': 'PURE', 'borkowski': 'Borkowski', 'cetch_yield': 'CETCH yield',
                  'cetch_efficiency': 'CETCH efficiency'}


def collect(exp_dir):
    """Chosen recipes (one row each), rounds (one row per fit) and runs (one row per campaign)."""
    points, rounds, runs = [], [], []
    for path in sorted(pathlib.Path(exp_dir).rglob('seed_*.json')):
        r = json.loads(path.read_text())
        base = {'dataset': r['problem']['subtype'], 'arm': r['arm_name'], 'seed': r['seed'],
                'f_max': r['problem']['f_max']}
        log = r['evaluated']
        points += [{**base, 'round': k, 'yield': y} for k, y in zip(log['round'], log['true'])]
        rounds += [{**base, 'round': x['round'], 'test_r2': x.get('test_r2'), 'test_spearman': x.get('test_spearman')}
                   for x in r['rounds'] if 'test_r2' in x]
        last = r['rounds'][-1] if r['rounds'] else {}
        found = [k for k, y in zip(log['round'], log['true']) if y >= r['problem']['f_max']]
        runs.append({**base, 'n_measured': len(log['true']), 'best_yield': max(log['true']),
                     'best_fraction': max(log['true']) / r['problem']['f_max'],
                     # First round that measured the pool's best recipe (NaN if never).
                     'round_found': min(found) if found else np.nan,
                     'auc_norm_regret': r['endpoints']['auc_norm_regret'],
                     'final_test_r2': last.get('test_r2'), 'final_test_spearman': last.get('test_spearman'),
                     'elapsed_s': r['elapsed_s']})
    return pd.DataFrame(points), pd.DataFrame(rounds), pd.DataFrame(runs)


def _order(values, preferred):
    return [v for v in preferred if v in set(values)] + sorted(set(values) - set(preferred))


def plot_yields(points, path, datasets, arms):
    plt, fig, axes = _figure(len(datasets), len(arms), figsize=(3.2 * len(arms), 2.4 * len(datasets)),
                             squeeze=False, sharex=True)
    rng = np.random.default_rng(0)
    for i, ds in enumerate(datasets):
        sub = points[points.dataset == ds]
        f_max = sub['f_max'].iloc[0]
        top = max(sub['yield'].max(), f_max)
        for j, arm in enumerate(arms):
            ax = axes[i][j]
            g = sub[sub.arm == arm]
            by_round = [g.loc[g['round'] == k, 'yield'].to_numpy() for k in sorted(g['round'].unique())]
            ks = sorted(g['round'].unique())
            ax.scatter(g['round'] + rng.uniform(-0.28, 0.28, len(g)), g['yield'], s=3, color=PALETTE[j % len(PALETTE)],
                       alpha=0.35, linewidths=0, zorder=2)
            ax.boxplot(by_round, positions=ks, widths=0.6, showfliers=False, zorder=3,
                       medianprops={'color': INK, 'lw': 1.2}, boxprops={'color': INK_2, 'lw': 0.8},
                       whiskerprops={'color': INK_2, 'lw': 0.8}, capprops={'color': INK_2, 'lw': 0.8})
            ax.axhline(f_max, color=INK_2, lw=0.8, ls='--', zorder=1)
            _style(ax)
            ax.set_ylim(top=top * 1.08)
            if i == 0:
                ax.set_title(arm, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel('{}\nmeasured yield'.format(DATASET_LABELS.get(ds, ds)), fontsize=8)
            if i == len(datasets) - 1:
                ax.set_xlabel('round')
                ax.set_xticks(ks)
    fig.suptitle('Recipes chosen each round ({} seed(s) pooled; dashed line: best recipe in the pool)'.format(
        points.seed.nunique()),
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _per_round_lines(frame, value, path, datasets, arms, ylabel, title):
    plt, fig, axes = _figure(1, len(datasets), figsize=(3.3 * len(datasets), 3.2), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[0][i]
        for j, arm in enumerate(arms):
            g = frame[(frame.dataset == ds) & (frame.arm == arm)].groupby('round')[value].agg(['mean', 'min', 'max'])
            if g.empty:
                continue
            color = PALETTE[j % len(PALETTE)]
            ax.plot(g.index, g['mean'], color=color, lw=2, label=arm)
            ax.fill_between(g.index, g['min'], g['max'], color=color, alpha=0.15, lw=0)
        _style(ax)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=10, color=INK, loc='left')
        ax.set_xlabel('round')
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=8)
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _best_so_far(points):
    rows = []
    for (ds, arm, seed), g in points.groupby(['dataset', 'arm', 'seed']):
        best = g.groupby('round')['yield'].max().cummax()
        rows += [{'dataset': ds, 'arm': arm, 'seed': seed, 'round': k, 'best_fraction': v / g['f_max'].iloc[0]}
                 for k, v in best.items()]
    return pd.DataFrame(rows)


def write_replay(cfg, exp_dir):
    exp_dir = pathlib.Path(exp_dir)
    out = exp_dir / 'replay'
    out.mkdir(parents=True, exist_ok=True)
    points, rounds, runs = collect(exp_dir)
    if runs.empty:
        raise FileNotFoundError('no results under {}'.format(exp_dir))
    datasets = _order(runs.dataset, list(DATASET_LABELS))
    arms = _order(runs.arm, [a['name'] for a in cfg['arms']])
    plot_yields(points, out / 'yields_by_round.png', datasets, arms)
    _per_round_lines(_best_so_far(points), 'best_fraction', out / 'best_so_far.png', datasets, arms,
                     'best measured yield so far\n(fraction of the pool\'s best)',
                     'Best recipe found so far (mean over seeds; band: seed range)')
    if not rounds.empty:
        _per_round_lines(rounds, 'test_r2', out / 'heldout_r2.png', datasets, arms,
                         'R$^2$ on held-out 20% of recipes', 'Prediction accuracy after each round\'s fit')
    cols = ['best_fraction', 'round_found', 'auc_norm_regret', 'final_test_r2', 'final_test_spearman', 'elapsed_s']
    table = runs.groupby(['dataset', 'arm'])[cols].agg(['mean', 'std'])
    table['n_seeds'] = runs.groupby(['dataset', 'arm']).size()
    table['n_found'] = runs.groupby(['dataset', 'arm'])['round_found'].count()
    table.to_csv(out / 'summary.csv')
    lines = ['# {} (exploratory)'.format(cfg['name']), '',
             'Mean (sd) over {} seed(s). best fraction = best measured yield found / best recipe in the pool. '
             'found = seeds that measured the best recipe in the pool; round found = first round it was measured, '
             'over those seeds (round 1 is random). '
             'AUC = mean normalised regret over rounds (0 = best recipe from the start, lower is faster). '
             'R2 and Spearman are on the held-out 20% after the last fit.'.format(runs.seed.nunique()), '',
             '| dataset | model | found | best fraction | round found | AUC norm. regret | held-out R2 | held-out Spearman '
             '| time (s) |',
             '|---|---|---|---|---|---|---|---|---|']
    for ds in datasets:
        for arm in arms:
            if (ds, arm) not in table.index:
                continue
            row = table.loc[(ds, arm)]
            cells = ['{}/{}'.format(int(row['n_found'].iloc[0]), int(row['n_seeds'].iloc[0]))]
            for c in cols[:-1]:
                mean, sd = row[(c, 'mean')], row[(c, 'std')]
                cells.append('-' if pd.isna(mean) else '{:.3f} ({:.3f})'.format(mean, 0.0 if pd.isna(sd) else sd))
            lines.append('| {} | {} | {} | {:.0f} |'.format(DATASET_LABELS.get(ds, ds), arm, ' | '.join(cells),
                                                           row[('elapsed_s', 'mean')]))
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out
