"""Paper-style figures and tables for the measured-data replay (experiments/replay_paper.yaml).

    python -m metis_bench replay experiments/replay_paper.yaml

Writes to ``results/<experiment>/replay/``:

* ``yields_by_round.png``: the measured yield of every recipe each model chose, per round
  (all seeds pooled): box plots with the points on top, in the style of the METIS paper's
  and notebooks' per-day figures. One row per dataset, one column per model.
* ``best_so_far.png``: best measured yield so far as a fraction of the pool's best, mean
  over seeds with the seed range shaded.
* ``model_accuracy.png``: R^2 and Spearman rank correlation on the held-out 20% of
  recipes after each round's fit.
* ``model_comparison.png``: one dot per seed: the round the best recipe was first
  measured, held-out R^2 after the last round, and how many held-out recipes fall inside
  the model's 80% interval (0.8 means its uncertainty is honest).
* ``summary.csv`` / ``summary.md``: per dataset and model, mean (sd) over seeds, and
  paired differences (same seed, so the same round-1 recipes) against each comparison
  family's reference arm.
"""
import json
import pathlib

import numpy as np
import pandas as pd

from .report import GRID, INK, INK_2, PALETTE, SURFACE, _figure, _style

DATASET_LABELS = {'pure': 'PURE', 'borkowski': 'Borkowski', 'cetch_yield': 'CETCH yield',
                  'cetch_efficiency': 'CETCH efficiency'}
# Display names; arms not listed keep their own name.
ARM_LABELS = {'incumbent': 'XGBoost (METIS)', 'linear': 'Linear', 'tabpfn35': 'TabPFN-3.5', 'random': 'Random'}
# Colour follows the model in every figure: the first three categorical slots (validated
# for colour-vision deficiency on all pairs) for the models, a neutral grey for the
# random baseline. Other arms take the remaining slots in order.
ARM_COLORS = {'incumbent': PALETTE[0], 'linear': PALETTE[1], 'tabpfn35': PALETTE[2], 'random': '#8a8984'}
NOT_FOUND = 11      # x position for "never measured the best recipe" in the comparison figure


def _label(arm):
    return ARM_LABELS.get(arm, arm)


def _colors(arms):
    spare = iter(PALETTE[3:])
    return {arm: ARM_COLORS.get(arm) or next(spare) for arm in arms}


def collect(exp_dir):
    """Chosen recipes (one row each), rounds (one row per fit) and runs (one row per campaign)."""
    points, rounds, runs = [], [], []
    for path in sorted(pathlib.Path(exp_dir).rglob('seed_*.json')):
        r = json.loads(path.read_text())
        base = {'dataset': r['problem']['subtype'], 'arm': r['arm_name'], 'seed': r['seed'],
                'f_max': r['problem']['f_max']}
        log = r['evaluated']
        points += [{**base, 'round': k, 'yield': y} for k, y in zip(log['round'], log['true'])]
        rounds += [{**base, 'round': x['round'], 'test_r2': x.get('test_r2'), 'test_spearman': x.get('test_spearman'),
                    'test_cov80': x.get('test_cov80')}
                   for x in r['rounds'] if 'test_r2' in x]
        last = r['rounds'][-1] if r['rounds'] else {}
        found = [k for k, y in zip(log['round'], log['true']) if y >= r['problem']['f_max']]
        runs.append({**base, 'n_measured': len(log['true']), 'best_yield': max(log['true']),
                     'best_fraction': max(log['true']) / r['problem']['f_max'],
                     # First round that measured the pool's best recipe (NaN if never).
                     'round_found': min(found) if found else np.nan,
                     'auc_norm_regret': r['endpoints']['auc_norm_regret'],
                     'final_test_r2': last.get('test_r2'), 'final_test_spearman': last.get('test_spearman'),
                     'final_test_cov80': last.get('test_cov80'),
                     'elapsed_s': r['elapsed_s']})
    return pd.DataFrame(points), pd.DataFrame(rounds), pd.DataFrame(runs)


def _order(values, preferred):
    return [v for v in preferred if v in set(values)] + sorted(set(values) - set(preferred))


def _number(x):
    return '{:g}'.format(x).replace('-', '−')


def _ranges(ks):
    """[2, 3, 4, 7] -> '2–4, 7'."""
    ks, out = sorted(int(k) for k in ks), []
    for k in ks:
        if out and k == out[-1][1] + 1:
            out[-1][1] = k
        else:
            out.append([k, k])
    return ', '.join(str(a) if a == b else '{}–{}'.format(a, b) for a, b in out)


def _swarm_offsets(y, bin_height, half_width=0.36, slots=9):
    """Deterministic beeswarm: points in the same ``bin_height`` band of y fan out sideways,
    alternating right and left, and wrap back to the centre after ``slots`` per side."""
    bins = np.floor(np.asarray(y) / bin_height).astype(int)
    step = half_width / slots
    offsets, used = np.zeros(len(bins)), {}
    for i in np.argsort(y, kind='stable'):
        k = used.get(bins[i], 0)
        used[bins[i]] = k + 1
        m = k % (2 * slots + 1)
        offsets[i] = ((m + 1) // 2) * step * (1 if m % 2 else -1)
    return offsets


def plot_yields(points, path, datasets, arms):
    colors = _colors(arms)
    plt, fig, axes = _figure(len(datasets), len(arms), figsize=(2.9 * len(arms) + 0.7, 2.3 * len(datasets) + 0.5),
                             squeeze=False, sharex=True, sharey='row')
    for i, ds in enumerate(datasets):
        sub = points[points.dataset == ds]
        f_max = sub['f_max'].iloc[0]
        bottom, top = min(0.0, sub['yield'].min()), max(sub['yield'].max(), f_max) * 1.06
        for j, arm in enumerate(arms):
            ax = axes[i][j]
            g = sub[sub.arm == arm]
            ks = sorted(g['round'].unique())
            values = [g.loc[g['round'] == k, 'yield'].to_numpy() for k in ks]
            ax.boxplot(values, positions=ks, widths=0.74, showfliers=False, patch_artist=True, zorder=2,
                       boxprops={'facecolor': colors[arm] + '2e', 'edgecolor': INK_2, 'linewidth': 0.7},
                       medianprops={'color': INK, 'linewidth': 1.4, 'zorder': 4},
                       whiskerprops={'color': INK_2, 'linewidth': 0.7}, capprops={'color': INK_2, 'linewidth': 0.7})
            for k, y in zip(ks, values):
                ax.scatter(k + _swarm_offsets(y, (top - bottom) / 60), y, s=4, color=colors[arm], alpha=0.75,
                           linewidths=0, zorder=3)
            ax.axhline(f_max, color=INK_2, lw=0.8, ls=(0, (4, 3)), zorder=1)
            _style(ax)
            ax.set_ylim(bottom, top)
            if i == 0:
                ax.set_title(_label(arm), fontsize=10, color=INK)
            if j == 0:
                measure = 'efficiency' if 'efficiency' in ds else 'yield'
                ax.set_ylabel('{}\nmeasured {}'.format(DATASET_LABELS.get(ds, ds), measure), fontsize=8)
            if i == len(datasets) - 1:
                ax.set_xlabel('Round')
                ax.set_xticks(ks)
    fig.suptitle('Every recipe each model chose, by round ({} seeds pooled). Round 1 is the same random batch for '
                 'all models; dashed line: best recipe in the pool.'.format(points.seed.nunique()),
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _per_round_panels(frame, metrics, path, datasets, arms, title):
    """Rows: ``metrics`` as (column, y label, y limits or None). Columns: datasets.

    Lines are means over seeds, bands the seed range. Where a mean falls below the lower
    limit the panel says so in text, so an off-scale model is never silently missing.
    """
    colors = _colors(arms)
    plt, fig, axes = _figure(len(metrics), len(datasets), figsize=(3.3 * len(datasets), 2.6 * len(metrics) + 0.9),
                             squeeze=False, sharex=True)
    handles = {}
    for r, (value, ylabel, limits) in enumerate(metrics):
        for c, ds in enumerate(datasets):
            ax = axes[r][c]
            notes = []
            for arm in arms:
                g = frame[(frame.dataset == ds) & (frame.arm == arm)].groupby('round')[value].agg(['mean', 'min', 'max'])
                if g.empty or g['mean'].isna().all():
                    continue
                handles[arm], = ax.plot(g.index, g['mean'], color=colors[arm], lw=2, solid_capstyle='round',
                                        solid_joinstyle='round', zorder=3)
                ax.fill_between(g.index, g['min'], g['max'], color=colors[arm], alpha=0.1, lw=0, zorder=2)
                if limits is not None and (g['mean'] < limits[0]).any():
                    off = g.index[g['mean'] < limits[0]]
                    notes.append('{}: mean below {} in round{} {}'.format(_label(arm), _number(limits[0]),
                                                                        's' if len(off) > 1 else '', _ranges(off)))
            _style(ax)
            if limits is not None:
                ax.set_ylim(*limits)
            if notes:
                ax.text(0.03, 0.97, '\n'.join(notes), transform=ax.transAxes, fontsize=7, color=INK_2, va='top')
            if r == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=10, color=INK, loc='left')
            if r == len(metrics) - 1:
                ax.set_xlabel('Round')
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8)
    shown = [a for a in arms if a in handles]
    fig.legend([handles[a] for a in shown], [_label(a) for a in shown], loc='lower center', ncol=len(shown),
               frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.07 if len(metrics) == 1 else 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_comparison(runs, path, datasets, arms):
    """One dot per seed: round the best recipe was first measured, final held-out R^2, final 80% coverage."""
    colors = _colors(arms)
    # (column, x label, x limits, floor): values below the floor are drawn as arrowheads at the edge.
    panels = [('round_found', 'Round the best recipe was first measured\n(lower = sooner)', (0.4, NOT_FOUND + 0.6),
               None),
              ('final_test_r2', 'Held-out R$^2$ after round 10\n(higher = more accurate)', (-0.6, 1.0), -0.5),
              ('final_test_cov80', "Share of held-out recipes inside the model's\n80% interval (0.8 = honest uncertainty)",
               (-0.03, 1.03), None)]
    plt, fig, axes = _figure(1, len(panels), figsize=(4.6 * len(panels), 0.95 * len(datasets) + 1.6), squeeze=False,
                             sharey=True)
    lanes = np.linspace(-0.3, 0.3, len(arms)) if len(arms) > 1 else np.zeros(1)
    seeds = sorted(runs.seed.unique())
    nudge = dict(zip(seeds, np.linspace(-0.045, 0.045, len(seeds)) if len(seeds) > 1 else [0.0]))
    handles = {}
    for c, (value, xlabel, limits, floor) in enumerate(panels):
        ax = axes[0][c]
        for i, ds in enumerate(datasets):
            for lane, arm in zip(lanes, arms):
                g = runs[(runs.dataset == ds) & (runs.arm == arm)]
                if g.empty or (value != 'round_found' and g[value].isna().all()):
                    continue
                x = g[value].to_numpy(float)
                if value == 'round_found':
                    x = np.where(np.isnan(x), NOT_FOUND, x)
                y = i + lane + g.seed.map(nudge).to_numpy()
                low = x < floor if floor is not None else np.zeros(len(x), dtype=bool)
                handles[arm] = ax.scatter(x[~low], y[~low], s=34, color=colors[arm], edgecolors=SURFACE, linewidths=1,
                                          zorder=3)
                if low.any():   # off-scale to the left: an arrowhead between the floor and the edge
                    ax.scatter(np.full(low.sum(), (floor + limits[0]) / 2), y[low], s=34, marker='<',
                               color=colors[arm], edgecolors=SURFACE, linewidths=1, zorder=3)
        if value == 'final_test_cov80':
            # A model whose members all agree reports zero spread, hence no interval at all.
            for lane, arm in zip(lanes, arms):
                g = runs[runs.arm == arm][value]
                if len(g) and g.notna().all() and (g == 0).all():
                    ax.text(0.04, lane, '{}: no spread, so no interval'.format(_label(arm)), fontsize=7, color=INK_2,
                            va='center')
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        for i in range(1, len(datasets)):
            ax.axhline(i - 0.5, color=GRID, lw=0.8, zorder=1)
        ax.set_xlim(*limits)
        ax.set_xlabel(xlabel, fontsize=8)
        if value == 'round_found':
            ticks = list(range(1, 11)) + [NOT_FOUND]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks[:-1]] + ['not\nfound'])
        if value == 'final_test_cov80':
            ax.axvline(0.8, color=INK_2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        if floor is not None:
            ax.axvline(floor, color=GRID, lw=0.8, zorder=1)
            ax.text(limits[0] + 0.01, len(datasets) - 0.55, '◀ = below {}'.format(_number(floor)), fontsize=7,
                    color=INK_2, va='bottom')
    axes[0][0].set_yticks(range(len(datasets)))
    axes[0][0].set_yticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets], fontsize=9, color=INK)
    axes[0][0].set_ylim(len(datasets) - 0.5, -0.5)
    shown = [a for a in arms if a in handles]
    fig.legend([handles[a] for a in shown], [_label(a) for a in shown], loc='lower center', ncol=len(shown),
               frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle('The models side by side, one dot per seed (same round-1 recipes for every model within a seed)',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _best_so_far(points):
    rows = []
    for (ds, arm, seed), g in points.groupby(['dataset', 'arm', 'seed']):
        best = g.groupby('round')['yield'].max().cummax()
        rows += [{'dataset': ds, 'arm': arm, 'seed': seed, 'round': k, 'best_fraction': v / g['f_max'].iloc[0]}
                 for k, v in best.items()]
    return pd.DataFrame(rows)


def _cell(mean, sd):
    return '-' if pd.isna(mean) else '{:.3f} ({:.3f})'.format(mean, 0.0 if pd.isna(sd) else sd)


def _paired_lines(cfg, runs, datasets):
    """Each family arm against its reference, paired by seed (same round-1 recipes)."""
    cols = ['round_found', 'auc_norm_regret', 'final_test_r2']
    lines = []
    for family in cfg.get('families', {}).values():
        ref = family['reference']
        for arm in family['arms']:
            a = runs[runs.arm == arm].set_index(['dataset', 'seed'])
            b = runs[runs.arm == ref].set_index(['dataset', 'seed'])
            both = a.index.intersection(b.index)
            if both.empty:
                continue
            if not lines:
                lines = ['', '## Paired by seed against the reference', '',
                         'Difference = model minus reference, same seed. Negative round found and AUC = better; '
                         'positive R2 = better. A seed that never found the best recipe counts as round {}.'.format(
                             NOT_FOUND), '',
                         '| model | reference | dataset | seeds | best found sooner / same / later | mean diff. round '
                         'found | mean diff. AUC | mean diff. held-out R2 |',
                         '|---|---|---|---|---|---|---|---|']
            for ds in datasets + ['all']:
                idx = [k for k in both if ds == 'all' or k[0] == ds]
                if not idx:
                    continue
                d = a.loc[idx, cols].fillna({'round_found': NOT_FOUND}) - b.loc[idx, cols].fillna(
                    {'round_found': NOT_FOUND})
                order = '{} / {} / {}'.format(int((d.round_found < 0).sum()), int((d.round_found == 0).sum()),
                                              int((d.round_found > 0).sum()))
                lines.append('| {} | {} | {} | {} | {} | {:+.2f} | {:+.3f} | {:+.3f} |'.format(
                    _label(arm), _label(ref), DATASET_LABELS.get(ds, ds), len(idx), order, d.round_found.mean(),
                    d.auc_norm_regret.mean(), d.final_test_r2.mean()))
    return lines


def write_schedule_tuning(cfg, exp_dir):
    """Figures and a table for a schedule tuning on measured data (experiments/tabpfn_schedule_tune.yaml).

    Writes to ``<exp_dir>/replay/``: ``schedules.png`` (what each candidate does per round),
    ``schedule_heatmap.png`` (each candidate against the reference schedule, per dataset),
    ``schedule_best_so_far.png`` and ``schedule_summary.md``.
    """
    from metis.schedules import make
    from matplotlib.colors import LinearSegmentedColormap
    from .report import CONTEXT

    exp_dir = pathlib.Path(exp_dir)
    out = exp_dir / 'replay'
    out.mkdir(parents=True, exist_ok=True)
    points, _, runs = collect(exp_dir)
    references = [a['name'] for a in cfg.get('reference_arms', [])]
    ref = references[0]
    candidates = [a for a in cfg['arms'] if a['name'] not in references]
    labels = {a['name']: '{:g}'.format(a['schedule']['slope']) if 'slope' in a['schedule'] else a['name']
              for a in candidates}
    tuned = json.loads(pathlib.Path(cfg['tuning']['output']).read_text()) if pathlib.Path(
        cfg['tuning']['output']).exists() else {}
    winner = tuned.get('winner', candidates[-1]['name'])
    datasets = _order(runs.dataset, list(DATASET_LABELS))
    n_rounds = int(points['round'].max())
    rounds = np.arange(2, n_rounds + 1)

    # 1. What the schedules do: the exploitation share of the UCB score each round.
    plt, fig, ax = _figure(figsize=(7.4, 4.0))
    for a in candidates:
        e = [make(a['schedule']).weights(r, n_rounds)[0] for r in rounds]
        if a['name'] != winner:
            ax.plot(rounds, e, color=CONTEXT, lw=1.2, zorder=2)
    ax.plot([], [], color=CONTEXT, lw=1.2, label='other ramps (slopes {})'.format(
        ', '.join(labels[a['name']] for a in candidates if a['name'] != winner)))
    ref_cfg = next(a for a in cfg['arms'] if a['name'] == ref)['schedule']
    ref_e = [1.0 / (1.0 + make(ref_cfg)(r, n_rounds)) for r in rounds]   # mean + b*std ranks like (mean + b*std)/(1+b)
    ax.step(rounds, ref_e, where='mid', color=INK_2, lw=2, ls=(0, (4, 3)), label="METIS's schedule (equivalent share)")
    win_cfg = next(a for a in candidates if a['name'] == winner)['schedule']
    ax.plot(rounds, [make(win_cfg).weights(r, n_rounds)[0] for r in rounds], color=ARM_COLORS['tabpfn35'], lw=2.5,
            marker='o', ms=5, label='winner: slope {}'.format(labels[winner]), zorder=4)
    _style(ax)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(rounds)
    ax.set_xlabel('Round')
    ax.set_ylabel('exploitation share of the score\n(0 = uncertainty only, 1 = prediction only)', fontsize=8)
    ax.set_title('The 10 candidate schedules: exploitation rises each round, exploration takes the rest',
                 fontsize=10, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc='lower right')
    fig.tight_layout()
    fig.savefig(out / 'schedules.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 2. Heat map: each candidate's mean AUC regret minus the reference's, per dataset.
    auc = runs.groupby(['dataset', 'arm']).auc_norm_regret.mean().unstack('arm')
    names = [a['name'] for a in candidates]
    diff = auc[names].sub(auc[ref], axis=0).loc[datasets]
    diff.loc['mean'] = diff.mean()
    lim = float(np.nanmax(np.abs(diff.to_numpy()))) or 1.0
    cmap = LinearSegmentedColormap.from_list('diverging', [PALETTE[0], '#f2f1ed', PALETTE[7]])
    plt, fig, ax = _figure(figsize=(9.2, 3.6))
    ax.imshow(diff.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim, aspect='auto')
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            v = diff.iat[i, j]
            ax.text(j, i, '{:+.3f}'.format(v).replace('-', '−'), ha='center', va='center', fontsize=7.5,
                    color='white' if abs(v) > 0.6 * lim else INK, fontweight='bold' if names[j] == winner else 'normal')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([labels[n] + ('\nwinner' if n == winner else '') for n in names], fontsize=8, color=INK_2)
    ax.set_yticks(range(diff.shape[0]))
    ylabels = ['{} ({:.3f})'.format(DATASET_LABELS.get(d, d), auc.loc[d, ref]) for d in datasets]
    ax.set_yticklabels(ylabels + ['mean of the 4 ({:.3f})'.format(auc.loc[datasets, ref].mean())], fontsize=8,
                       color=INK)
    ax.axhline(len(datasets) - 0.5, color=SURFACE, lw=3)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel('slope of the exploitation ramp (higher = pure exploitation sooner)', fontsize=8, color=INK_2)
    ax.set_title("Each ramp's average regret minus that of METIS's schedule (in brackets): blue = finds good "
                 'recipes sooner, red = later', fontsize=9, color=INK, loc='left')
    fig.tight_layout()
    fig.savefig(out / 'schedule_heatmap.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 3. Best yield so far: the winner, a slow ramp and the reference.
    slow = next((a['name'] for a in candidates if a['schedule'].get('slope') == 1.0), candidates[0]['name'])
    shown = [(winner, 'winner: slope {}'.format(labels[winner]), ARM_COLORS['tabpfn35'], '-'),
             (slow, 'slope {} (pure exploitation only in round 10)'.format(labels[slow]), PALETTE[1], '-'),
             (ref, "METIS's schedule", INK_2, (0, (4, 3)))]
    best = _best_so_far(points)
    plt, fig, axes = _figure(1, len(datasets), figsize=(3.3 * len(datasets), 3.4), squeeze=False)
    handles = []
    for c, ds in enumerate(datasets):
        ax = axes[0][c]
        for arm, label, color, ls in shown:
            g = best[(best.dataset == ds) & (best.arm == arm)].groupby('round').best_fraction.mean()
            line, = ax.plot(g.index, g.to_numpy(), color=color, lw=2, ls=ls, label=label)
            if c == 0:
                handles.append(line)
        _style(ax)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=10, color=INK, loc='left')
        ax.set_xlabel('Round')
        if c == 0:
            ax.set_ylabel("best measured yield so far\n(fraction of the pool's best)", fontsize=8)
    fig.legend(handles, [h.get_label() for h in handles], loc='lower center', ncol=3, frameon=False, fontsize=8,
               labelcolor=INK_2)
    fig.suptitle('Best recipe found so far with TabPFN-3.5 (mean over {} seeds)'.format(runs.seed.nunique()),
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / 'schedule_best_so_far.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 4. Table: winner against the reference, paired by seed.
    lines = ['# {}: the winning ramp against METIS\'s schedule'.format(cfg['name']), '',
             'Mean over {} seeds. AUC = mean normalised regret over rounds (lower = good recipes sooner). '
             'Difference = winner minus METIS\'s schedule, same seed (mean and standard error).'.format(
                 runs.seed.nunique()), '',
             '| dataset | AUC, winner (slope {}) | AUC, METIS\'s schedule | difference | best recipe found (winner / METIS) '
             '| round found (winner / METIS) |'.format(labels[winner]),
             '|---|---|---|---|---|---|']
    a = runs[runs.arm == winner].set_index(['dataset', 'seed'])
    b = runs[runs.arm == ref].set_index(['dataset', 'seed'])
    for ds in datasets + ['all']:
        idx = [k for k in a.index.intersection(b.index) if ds == 'all' or k[0] == ds]
        d = a.loc[idx, 'auc_norm_regret'] - b.loc[idx, 'auc_norm_regret']
        lines.append('| {} | {:.3f} | {:.3f} | {:+.3f} ({:.3f}) | {} / {} | {:.1f} / {:.1f} |'.format(
            DATASET_LABELS.get(ds, 'all four'), a.loc[idx, 'auc_norm_regret'].mean(),
            b.loc[idx, 'auc_norm_regret'].mean(), d.mean(), d.std(ddof=1) / np.sqrt(len(d)),
            int(a.loc[idx, 'round_found'].notna().sum()), int(b.loc[idx, 'round_found'].notna().sum()),
            a.loc[idx, 'round_found'].mean(), b.loc[idx, 'round_found'].mean()))
    (out / 'schedule_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


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
    _per_round_panels(_best_so_far(points), [('best_fraction', "best measured yield so far\n(fraction of the pool's "
                                              "best)", None)], out / 'best_so_far.png', datasets, arms,
                      'Best recipe found so far (mean over seeds; band: seed range)')
    if not rounds.empty:
        model_arms = [a for a in arms if a in set(rounds.arm)]
        _per_round_panels(rounds, [('test_r2', 'R$^2$ on held-out recipes', (-0.5, 1.0)),
                                   ('test_spearman', 'Spearman rank correlation\non held-out recipes', (0.0, 1.0))],
                          out / 'model_accuracy.png', datasets, model_arms,
                          "How well each model predicts the held-out 20% of recipes after each round's fit "
                          '(mean over seeds; band: seed range)')
        plot_comparison(runs, out / 'model_comparison.png', datasets, arms)
    cols = ['best_fraction', 'round_found', 'auc_norm_regret', 'final_test_r2', 'final_test_spearman',
            'final_test_cov80', 'elapsed_s']
    table = runs.groupby(['dataset', 'arm'])[cols].agg(['mean', 'std'])
    table['n_seeds'] = runs.groupby(['dataset', 'arm']).size()
    table['n_found'] = runs.groupby(['dataset', 'arm'])['round_found'].count()
    table.to_csv(out / 'summary.csv')
    lines = ['# {} (exploratory)'.format(cfg['name']), '',
             'Mean (sd) over {} seed(s). best fraction = best measured yield found / best recipe in the pool. '
             'found = seeds that measured the best recipe in the pool; round found = first round it was measured, '
             'over those seeds (round 1 is random). '
             'AUC = mean normalised regret over rounds (0 = best recipe from the start, lower is faster). '
             'R2, Spearman and 80% coverage (share of held-out recipes inside the model\'s 80% interval; 0.8 is '
             'honest) are on the held-out 20% after the last fit. time = wall-clock seconds per campaign on the '
             'machine that ran it.'.format(runs.seed.nunique()), '',
             '| dataset | model | found | best fraction | round found | AUC norm. regret | held-out R2 | held-out Spearman '
             '| 80% coverage | time (s) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for ds in datasets:
        for arm in arms:
            if (ds, arm) not in table.index:
                continue
            row = table.loc[(ds, arm)]
            cells = ['{}/{}'.format(int(row['n_found'].iloc[0]), int(row['n_seeds'].iloc[0]))]
            cells += [_cell(row[(c, 'mean')], row[(c, 'std')]) for c in cols[:-1]]
            lines.append('| {} | {} | {} | {:.0f} |'.format(DATASET_LABELS.get(ds, ds), _label(arm), ' | '.join(cells),
                                                           row[('elapsed_s', 'mean')]))
    lines += _paired_lines(cfg, runs, datasets)
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out
