"""One markdown overview of every experiment's report (``python -m metis_bench summary``)."""
import json
import pathlib

import pandas as pd

ORDER = ['fig1b_check', 'phase2_init', 'phase2_import', 'phase2_surrogates', 'phase2_batch',
         'phase3_schedule_tune', 'phase3_schedule', 'phase3_hparams_tune', 'phase3_hparams', 'phase3_ablation']
FAMILY_COLUMNS = ['arm', 'reference', 'n_problems', 'n_pairs', 'mean_diff', 'ci_low', 'ci_high', 'share_better',
                  'share_worse', 'p_value', 'p_holm', 'significant']
SURROGATE_COLUMNS = ['arm', 'test_r2', 'test_spearman', 'test_cov80', 'test_cov95', 'test_mace', 'test_z_sd',
                     'batch_spearman']


def _md(frame, digits=3):
    return frame.round(digits).to_markdown(index=False) if len(frame) else '(none)'


def _experiment_section(name, report):
    lines = ['## {}'.format(name), '']
    runs = pd.read_csv(report / 'runs.csv')
    lines += ['{} campaigns: {} problems x {} arms, {} seeds each.'.format(
        len(runs), runs.problem_id.nunique(), runs.arm.nunique(),
        int(runs.groupby(['problem_id', 'arm']).seed.nunique().min())), '']
    overview = runs.groupby('arm').agg(final_norm_regret=('final_norm_regret', 'mean'),
                                       rec_regret=('final_norm_rec_regret', 'mean'),
                                       auc=('auc_norm_regret', 'mean'),
                                       found_optimum=('found_optimum', 'mean'),
                                       minutes=('elapsed_s', lambda s: s.mean() / 60)).reset_index()
    lines += ['Mean endpoints per arm (normalised regret: 0 = optimum, 1 = median random recipe):', '',
              _md(overview), '']
    for path in sorted(report.glob('family_*_primary.csv')):
        table = pd.read_csv(path)
        family = path.stem[len('family_'):-len('_primary')]
        lines += ['Family {} (paired differences, negative = arm better):'.format(family), '',
                  _md(table[[c for c in FAMILY_COLUMNS if c in table]]), '']
    by_landscape = runs.groupby(['landscape', 'arm']).final_norm_regret.mean().unstack('arm').reset_index()
    lines += ['Final normalised regret by landscape:', '', _md(by_landscape), '']
    quality = report / 'surrogate_quality.csv'
    if quality.exists():
        q = pd.read_csv(quality)
        lines += ['Surrogate quality, median over guided rounds (held-out R2 / Spearman; coverage of nominal 80% and '
                  '95% intervals; calibration error; z_sd > 1 = overconfident):', '',
                  _md(q[[c for c in SURROGATE_COLUMNS if c in q]]), '']
    for table_name, title in (('ablation_subsets.csv', 'Ablation: mean paired difference per subset'),
                              ('ablation_factorial.csv', 'Ablation: main effects and interactions'),
                              ('ablation_shapley.csv', 'Ablation: Shapley shares of the full improvement')):
        if (report / table_name).exists():
            lines += [title + ':', '', _md(pd.read_csv(report / table_name), 4), '']
    lines += ['![regret curves]({})'.format((report / 'regret_curves.png').as_posix()), '']
    return lines


def write_summary(results_dir, out_path, pilot=False, artifacts='artifacts'):
    results_dir = pathlib.Path(results_dir)
    suffix = '-pilot' if pilot else ''
    lines = ['# METIS benchmark: {} results'.format('pilot' if pilot else 'confirmatory'), '']
    if pilot:
        lines += ['**Pilot runs check the machinery.** They use a few problems and seeds and a cheaper incumbent '
                  '(15 hyperparameter draws instead of 200), and the decision step ignores significance so Phase 3 '
                  'has something to run. None of this is evidence for or against an intervention; see '
                  'docs/PREREGISTRATION.md.', '']
    for name in ORDER:
        report = results_dir / (name + suffix) / 'report'
        if (report / 'runs.csv').exists():
            lines += _experiment_section(name, report)
    winners = pathlib.Path(artifacts) / 'winners{}.json'.format(suffix)
    if winners.exists():
        w = json.loads(winners.read_text())
        lines += ['## Decision step', '', 'Chosen interventions (`{}`, significance required: {}):'.format(
            winners.as_posix(), w.get('significance_required', True)), '', '```json',
            json.dumps(w['interventions'], indent=1), '```', '']
    for part in ('schedule', 'surrogate'):
        tuned = pathlib.Path(artifacts) / 'tuning' / '{}{}.json'.format(part, suffix)
        if tuned.exists():
            t = json.loads(tuned.read_text())
            table = pd.DataFrame(t['table']).head(8)
            lines += ['## Tuning: {}'.format(part), '', 'Winner `{}`: `{}`'.format(t['winner'], json.dumps(t['fragment'])),
                      '', 'Best candidates on the dev split (mean final normalised regret):', '', _md(table, 4), '']
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text('\n'.join(lines), encoding='utf-8')
    return out_path
