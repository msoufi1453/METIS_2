"""Collect campaign results and write the comparison report for one experiment."""
import json
import pathlib

import numpy as np
import pandas as pd

from . import metrics, stats

SECONDARY = ('final_norm_rec_regret', 'auc_norm_regret', 'final_norm_cum_regret', 'rounds_to_target')
SURROGATE_COLUMNS = ('test_r2', 'test_spearman', 'test_rmse', 'test_cov50', 'test_cov80', 'test_cov95', 'test_mace',
                     'test_nlpd', 'test_z_sd', 'batch_spearman', 'batch_cov80')


def collect(exp_dir):
    """Runs table (one row per campaign) and rounds table (one row per guided round)."""
    runs, rounds, curves = [], [], []
    for path in sorted(pathlib.Path(exp_dir).rglob('seed_*.json')):
        r = json.loads(path.read_text())
        p = r['problem']
        base = {'problem_id': p['id'], 'arm': r['arm_name'], 'seed': r['seed'], 'kind': p.get('kind'),
                'landscape': p.get('landscape'), 'subtype': p.get('subtype'), 'n_dims': p.get('n_dims'),
                'noise': p['noise']['name'], 'split': p.get('split'), 'realism': p.get('realism'),
                'nearest_real': p.get('nearest_real'), 'max_is_exact': p.get('max_is_exact')}
        runs.append({**base, **r['endpoints'], 'elapsed_s': r['elapsed_s']})
        for rr in r['rounds']:
            rounds.append({**base, **{k: rr.get(k) for k in ('round', 'exploration', 'select_s') + SURROGATE_COLUMNS}})
        for t in r['trajectory']:
            curves.append({**base, 'round': t['round'], 'norm_regret': t['norm_regret'],
                           'norm_rec_regret': t['norm_rec_regret']})
    return pd.DataFrame(runs), pd.DataFrame(rounds), pd.DataFrame(curves)


# Chart style: the reference categorical palette in fixed order (slots 1-3 are validated
# for colour-vision deficiency on all pairs, which matters when lines cross), neutral ink
# for the reference arm, a light grey for context lines, recessive grid.
PALETTE = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')
INK, INK_2, CONTEXT, SURFACE, GRID = '#0b0b0b', '#52514e', '#d9d8d1', '#fcfcfb', '#ecebe6'


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _figure(*args, **kwargs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return plt, fig, axes


def _direct_labels(ax, ends, x):
    """Label line ends in ink (identity comes from the line it sits on), nudged apart."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ends = sorted(ends)
    placed = []
    for y, text in ends:
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.annotate(text, (x, y), xytext=(x + 0.25, y_text), textcoords='data', fontsize=8, color=INK_2,
                    va='center', annotation_clip=False)


def _summarise(curves, value):
    per_problem = curves.groupby(['arm', 'problem_id', 'round'])[value].mean().reset_index()
    return {arm: g.groupby('round')[value].agg(['mean', 'sem']) for arm, g in per_problem.groupby('arm')}


def _plot_curves(curves, path, value='norm_regret', title='', reference=None, order=None, highlight=None):
    """Mean over problems (per-problem means over seeds) by round.

    At most four highlighted arms plus the reference get colour; with more arms (or an
    explicit ``highlight`` list) the rest become thin grey context lines. No error bands:
    the spread across problems reflects how hard the problems are, not the paired arm
    differences the statistics test (those are in the family tables).
    """
    summary = _summarise(curves, value)
    arms = [a for a in (order or sorted(summary)) if a in summary]
    others = [a for a in arms if a != reference]
    if highlight is None:
        highlight = others if len(others) <= 4 else sorted(
            others, key=lambda a: summary[a]['mean'].iloc[-1])[:3]
    context = [a for a in others if a not in highlight]
    plt, fig, ax = _figure(figsize=(7.4, 4.4))
    for arm in context:
        ax.plot(summary[arm].index, summary[arm]['mean'], color=CONTEXT, lw=1, zorder=1)
    series = [(reference, INK_2)] if reference in summary else []
    series += [(arm, PALETTE[k % len(PALETTE)]) for k, arm in enumerate(highlight)]
    ends = []
    for arm, color in series:
        m = summary[arm]
        ax.plot(m.index, m['mean'], color=color, lw=2, label=arm, zorder=3)
        ends.append((float(m['mean'].iloc[-1]), arm))
    if context:
        ax.plot([], [], color=CONTEXT, lw=1, label='other arms ({})'.format(len(context)))
    _style(ax)
    ax.set_xlabel('round')
    ax.set_ylabel('mean ' + value.replace('_', ' ') + '\n(0 = optimum, 1 = median random recipe)')
    ax.set_title(title, fontsize=10, color=INK, loc='left')
    last = max(m.index.max() for m in summary.values())
    first = min(m.index.min() for m in summary.values())
    ax.set_xticks(range(first, last + 1))
    ax.set_xlim(first - 0.3, last + (3.2 if len(series) <= 4 else 0.3))
    if len(series) <= 4:
        _direct_labels(ax, ends, last)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _plot_surrogate(rounds, path, reference=None, order=None):
    panels = [('test_r2', 'held-out R2'), ('test_spearman', 'held-out Spearman'),
              ('test_cov80', '80% interval coverage (target 0.8)'), ('test_mace', 'calibration error (MACE)')]
    arms = [a for a in (order or sorted(rounds.arm.unique())) if a in set(rounds.arm)]
    colors, k = {}, 0
    for arm in arms:
        if arm == reference:
            colors[arm] = INK_2
        else:
            colors[arm] = PALETTE[k % len(PALETTE)]
            k += 1
    plt, fig, axes = _figure(1, 4, figsize=(14, 3.6))
    for ax, (col, label) in zip(axes, panels):
        for arm in arms:
            s = rounds[rounds.arm == arm].groupby('round')[col].median()
            ax.plot(s.index, s.values, color=colors[arm], lw=2, label=arm)
        if col == 'test_cov80':
            ax.axhline(0.8, color=INK_2, ls=(0, (4, 3)), lw=1)
        _style(ax)
        ax.set_title(label, fontsize=9, color=INK, loc='left')
        ax.set_xlabel('round')
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle('Surrogate quality: median over problems and seeds', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _ablation_highlight(runs, interventions, metric=metrics.PRIMARY_ENDPOINT):
    """Best single intervention, best subset short of everything, and the full combination."""
    means = runs.groupby('arm')[metric].mean()
    full = 'abl_' + '+'.join(sorted(interventions))
    singles = [a for a in means.index if a.startswith('abl_') and '+' not in a]
    partial = [a for a in means.index if a.startswith('abl_') and a != full]
    picks = []
    if singles:
        picks.append(means[singles].idxmin())
    if partial:
        best = means[partial].idxmin()
        if best not in picks:
            picks.append(best)
    if full in means.index:
        picks.append(full)
    return picks


def _fmt(table):
    return table.to_markdown(index=False, floatfmt='.4g') if hasattr(table, 'to_markdown') else table.to_string(index=False)


def write_report(cfg, exp_dir, n_boot=stats.N_BOOTSTRAP):
    exp_dir = pathlib.Path(exp_dir)
    out = exp_dir / 'report'
    out.mkdir(parents=True, exist_ok=True)
    runs, rounds, curves = collect(exp_dir)
    if runs.empty:
        raise RuntimeError('no results under {}'.format(exp_dir))
    (out / 'config.json').write_text(json.dumps(cfg, indent=1, default=str))
    runs.to_csv(out / 'runs.csv', index=False)
    rounds.to_csv(out / 'rounds.csv', index=False)
    reference = cfg.get('reference', 'incumbent')
    order = [a['name'] for a in cfg['arms']]   # colours follow arms in config order, never their rank
    arms = [a['name'] for a in cfg['arms'] if a['name'] != reference]
    confirm = runs[runs.max_is_exact.fillna(True).astype(bool)]
    lines = ['# {}'.format(cfg['name']), '']
    if cfg.get('pilot'):
        lines += ['**Pilot run.** Machinery check only; not confirmatory (docs/PREREGISTRATION.md).', '']
    lines += ['Problems: {}; arms: {}; seeds per problem: {}; campaigns: {}.'.format(
        runs.problem_id.nunique(), runs.arm.nunique(), runs.groupby(['problem_id', 'arm']).seed.nunique().min(), len(runs)),
        '', 'Reference arm: `{}`. Negative differences favour the arm (lower regret).'.format(reference), '']
    families = cfg.get('families') or {cfg.get('family', 'main'): arms}
    for family, members in families.items():
        family_ref = reference
        if isinstance(members, dict):
            family_ref, members = members.get('reference', reference), members['arms']
        lines += ['## Family {}: primary endpoint `{}` (vs `{}`)'.format(family, metrics.PRIMARY_ENDPOINT, family_ref), '']
        table = stats.compare(confirm, members, family_ref, metrics.PRIMARY_ENDPOINT, n_boot=n_boot)
        table.to_csv(out / 'family_{}_primary.csv'.format(family), index=False)
        lines += [_fmt(table.round(4)), '']
        for metric in SECONDARY:
            sec = stats.compare(confirm, members, family_ref, metric, n_boot=min(n_boot, 2000))
            sec.to_csv(out / 'family_{}_{}.csv'.format(family, metric), index=False)
        lines += ['Secondary endpoints: `family_{}_<metric>.csv`.'.format(family), '']
    lines += ['## Mean primary endpoint by stratum', '']
    for by in (['landscape'], ['n_dims'], ['noise']):
        strat = runs.groupby(by + ['arm'])[metrics.PRIMARY_ENDPOINT].mean().unstack('arm')
        lines += [_fmt(strat.reset_index().round(3)), '']
    if runs.realism.notna().any() and (runs.kind == 'synthetic').any():
        syn = runs[runs.kind == 'synthetic'].copy()
        syn['realism_tercile'] = pd.qcut(syn.realism.rank(method='first'), 3, labels=['most', 'middle', 'least'])
        strat = syn.groupby(['realism_tercile', 'arm'], observed=True)[metrics.PRIMARY_ENDPOINT].mean().unstack('arm')
        lines += ['By realism tercile (most realistic first):', '', _fmt(strat.reset_index().round(3)), '']
    if cfg.get('interventions'):
        from .ablation import analyze
        coef, shap, per_subset = analyze(confirm, cfg['interventions'], n_boot=min(n_boot, 2000))
        coef.to_csv(out / 'ablation_factorial.csv', index=False)
        shap.to_csv(out / 'ablation_shapley.csv', index=False)
        per_subset.to_csv(out / 'ablation_subsets.csv', index=False)
        lines += ['## Ablation', '',
                  'Mean paired difference to the incumbent per subset of interventions (negative = better):', '',
                  _fmt(per_subset.round(4)), '',
                  'Factorial regression (main effects and pairwise interactions on regret, bootstrap CIs over problems):',
                  '', _fmt(coef.round(4)), '',
                  "Shapley shares of the full combination's improvement (they sum to its total improvement):", '',
                  _fmt(shap.round(4)) if not shap.empty else '(not all subsets present)', '']
    if not rounds.empty:
        # Medians: one degenerate problem (e.g. a flat needle test set) must not dominate the table.
        surr = rounds.groupby('arm')[list(SURROGATE_COLUMNS)].median()
        surr.to_csv(out / 'surrogate_quality.csv')
        lines += ['## Surrogate quality (median over problems, seeds and guided rounds)', '',
                  _fmt(surr.reset_index().round(3)), '']
        _plot_surrogate(rounds, out / 'surrogate_quality.png', reference=reference, order=order)
    highlight = _ablation_highlight(runs, cfg['interventions']) if cfg.get('interventions') else None
    _plot_curves(curves, out / 'regret_curves.png', title='{}: best true yield so far'.format(cfg['name']),
                 reference=reference, order=order, highlight=highlight)
    _plot_curves(curves, out / 'rec_regret_curves.png', value='norm_rec_regret',
                 title='{}: recipe with the best measured mean'.format(cfg['name']), reference=reference, order=order,
                 highlight=highlight)
    lines += ['![regret curves](regret_curves.png)', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return out / 'report.md'
