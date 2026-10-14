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


def _plot_curves(curves, path, value='norm_regret', title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    per_problem = curves.groupby(['arm', 'problem_id', 'round'])[value].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for arm, g in per_problem.groupby('arm'):
        summary = g.groupby('round')[value].agg(['mean', 'sem'])
        ax.plot(summary.index, summary['mean'], marker='o', ms=3, label=arm)
        ax.fill_between(summary.index, summary['mean'] - 1.96 * summary['sem'], summary['mean'] + 1.96 * summary['sem'],
                        alpha=0.15)
    ax.set_xlabel('round')
    ax.set_ylabel(value.replace('_', ' '))
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_surrogate(rounds, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [('test_r2', 'held-out R2'), ('test_spearman', 'held-out Spearman'),
              ('test_cov80', '80% interval coverage'), ('test_mace', 'calibration error (MACE)')]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (col, label) in zip(axes, panels):
        for arm, g in rounds.groupby('arm'):
            s = g.groupby('round')[col].median()
            ax.plot(s.index, s.values, marker='o', ms=3, label=arm)
        if col == 'test_cov80':
            ax.axhline(0.8, color='grey', ls='--', lw=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('round')
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
        _plot_surrogate(rounds, out / 'surrogate_quality.png')
    _plot_curves(curves, out / 'regret_curves.png', title=cfg['name'])
    _plot_curves(curves, out / 'rec_regret_curves.png', value='norm_rec_regret', title=cfg['name'])
    lines += ['![regret curves](regret_curves.png)', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return out / 'report.md'
