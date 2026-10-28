"""Phase 3 tuning on the dev split (steps 11 and 12) and the winner bookkeeping.

A tuning file lists candidate values for one part of an arm (``schedule`` or
``surrogate``) on top of ``base_arm``. Every combination becomes an arm, all arms run
on the dev problems with tuning seeds (disjoint from the confirmatory seeds 0-49), and
the arm with the lowest mean primary endpoint (problems weighted equally) wins.
``tuning: {metric: ...}`` ranks by another endpoint instead, and ``reference_arms`` run
alongside for comparison but cannot win. The winning fragment goes to a JSON file that
confirmatory experiments reference with ``from_tuning``.

``decide_winners`` applies the preregistered decision rule (section 6) to Phase 2
reports and writes ``artifacts/winners.json``, which Phase 3 files use through
``base_arm: {from_winners: ...}`` and ``arms: {ablation_of: ...}``.
"""
import copy
import itertools
import json
import pathlib

import numpy as np
import pandas as pd

from . import metrics


def _expand(entry):
    """``{type: linear, start: [1, 2], end: [0.1]}`` -> one dict per combination."""
    lists = {k: (v if isinstance(v, list) else [v]) for k, v in entry.items()}
    keys = list(lists)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(lists[k] for k in keys))]


def _arm_name(part, fragment):
    bits = [fragment['type']] + ['{}{}'.format(k, v) for k, v in fragment.items() if k != 'type']
    return '{}_{}'.format(part, '_'.join(str(b) for b in bits))


def candidate_arms(cfg):
    """Every candidate combination as an arm on top of ``base_arm``.

    ``candidates`` is a list of grids; ``candidates_by_type`` maps a surrogate type to its
    grids, and the base arm's surrogate type (default xgb_ensemble) picks which applies.
    """
    part = cfg['tuning']['part']
    base = copy.deepcopy(cfg.get('base_arm', {}))
    entries = cfg.get('candidates')
    if not entries:
        kind = base.get('surrogate', {}).get('type', 'xgb_ensemble')
        entries = cfg['candidates_by_type'][kind]
    arms = []
    for entry in entries:
        for fragment in _expand(entry):
            arm = copy.deepcopy(base)
            if part == 'surrogate' and 'type' not in fragment:
                fragment = {**base.get('surrogate', {}), **fragment}
            arm[part] = fragment
            arm['name'] = _arm_name(part, fragment)
            arms.append(arm)
    for reference in cfg.get('reference_arms', []):
        arm = copy.deepcopy(base)
        arm.update(copy.deepcopy(reference))
        arms.append(arm)
    return arms


def select_winner(runs, metric=metrics.PRIMARY_ENDPOINT, exclude=()):
    """Arm with the lowest mean over problems of the per-problem mean ``metric``.

    Arms in ``exclude`` (reference arms) appear in the table but cannot win.
    """
    per_problem = runs.groupby(['arm', 'problem_id'])[metric].mean().reset_index()
    table = per_problem.groupby('arm')[metric].agg(['mean', 'sem', 'count']).sort_values('mean')
    eligible = [arm for arm in table.index if arm not in set(exclude)]
    return eligible[0], table


def write_winner(path, cfg, winner_name, table):
    part = cfg['tuning']['part']
    arm = next(a for a in cfg['arms'] if a['name'] == winner_name)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'part': part, 'fragment': arm[part], 'winner': winner_name,
                                'pilot': bool(cfg.get('pilot')), 'experiment': cfg.get('name'),
                                'metric': cfg['tuning'].get('metric', metrics.PRIMARY_ENDPOINT),
                                'seeds': cfg.get('seeds'), 'base_arm': cfg.get('base_arm'),
                                'table': table.reset_index().round(5).to_dict(orient='records')}, indent=1))
    return path


PARAM_HINTS = {'slope': 'slope of the exploitation ramp (higher = pure exploitation sooner)'}


def _candidate_label(fragment):
    params = ['{:g}'.format(v) if isinstance(v, (int, float)) else str(v) for k, v in fragment.items() if k != 'type']
    return ', '.join(params) or fragment['type']


def tuning_report(cfg, runs, winner, metric, exclude, out_dir):
    """``tuning_report.md`` and ``tuning_curve.png`` in ``out_dir``.

    The table gives every arm's mean ``metric`` per problem and over problems, and each
    arm's paired difference to the winner (same problem and seed) with its standard error
    over problem x seed pairs: differences within about two standard errors of zero are
    not distinguishable from the winner with these seeds.
    """
    from .replay import DATASET_LABELS
    from .report import INK, INK_2, PALETTE, SURFACE, _figure, _style

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    part = cfg['tuning']['part']
    order = [a['name'] for a in cfg['arms'] if a['name'] in set(runs.arm)]
    candidates = [a for a in order if a not in set(exclude)]
    fragments = {a['name']: a.get(part, {}) for a in cfg['arms']}
    per_problem = runs.groupby(['arm', 'problem_id'])[metric].mean().unstack('problem_id')
    problems = list(per_problem.columns)
    names = {p: DATASET_LABELS.get(p.replace('measured_', ''), p) for p in problems}
    ref = runs[runs.arm == winner].set_index(['problem_id', 'seed'])[metric]
    rows = []
    for arm in order:
        diff = runs[runs.arm == arm].set_index(['problem_id', 'seed'])[metric] - ref
        diff = diff.dropna()
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float('nan')
        rows.append({'arm': arm, 'label': _candidate_label(fragments[arm]) if arm in candidates else arm,
                     'role': 'candidate' if arm in candidates else 'reference', 'mean': per_problem.loc[arm].mean(),
                     'diff': diff.mean(), 'se': se, **{names[p]: per_problem.loc[arm, p] for p in problems}})
    table = pd.DataFrame(rows)
    table['rank'] = table['mean'].rank(method='min').astype(int)
    table.to_csv(out_dir / 'tuning_table.csv', index=False)
    edge = winner in (candidates[0], candidates[-1]) if candidates else False
    lines = ['# {}: tuning ({}, lower is better)'.format(cfg['name'], metric), '',
             'Winner: **{}** ({}). Mean over {} problem(s) of the per-problem mean over {} seed(s). Paired '
             'difference to the winner: mean (standard error) over problem x seed pairs.'.format(
                 winner, _candidate_label(fragments[winner]), len(problems), runs.seed.nunique()), '']
    if edge:
        lines += ['Note: the winner is at the edge of the candidate grid, so a value beyond it might do better.',
                  '']
    lines += ['| candidate | role | ' + ' | '.join(names[p] for p in problems) + ' | mean | diff. to winner | rank |',
              '|---|---|' + '---|' * len(problems) + '---|---|---|']
    # Records, not itertuples: problem names such as "CETCH yield" are not valid field names.
    for row in table.to_dict(orient='records'):
        cells = ['{:.3f}'.format(row[names[p]]) for p in problems]
        star = ' (winner)' if row['arm'] == winner else ''
        lines.append('| {}{} | {} | {} | {:.3f} | {:+.3f} ({:.3f}) | {} |'.format(
            row['label'], star, row['role'], ' | '.join(cells), row['mean'], row['diff'],
            0.0 if pd.isna(row['se']) else row['se'], row['rank']))
    (out_dir / 'tuning_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # Figure: candidates in file order on x; one thin line per problem (when few), the mean in ink.
    cand = table[table.role == 'candidate'].reset_index(drop=True)
    plt, fig, ax = _figure(figsize=(7.6, 4.2))
    x = np.arange(len(cand))
    if len(problems) <= 8:
        for k, p in enumerate(problems):
            ax.plot(x, cand[names[p]], color=PALETTE[k], lw=1.2, alpha=0.9, marker='o', ms=3.5, label=names[p])
    ax.plot(x, cand['mean'], color=INK, lw=2.2, marker='o', ms=5, label='mean over problems', zorder=4)
    best = cand.index[cand.arm == winner]
    if len(best):
        ax.scatter(x[best], cand.loc[best, 'mean'], s=150, facecolors='none', edgecolors=INK, linewidths=1.5, zorder=5)
    for row in table[table.role == 'reference'].itertuples(index=False):
        ax.axhline(row.mean, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=2, label='{} (reference, mean)'.format(row.arm))
    _style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(cand['label'], fontsize=8)
    varied = sorted({k for a in candidates for k in fragments[a] if k != 'type'})
    ax.set_xlabel(', '.join(PARAM_HINTS.get(k, k) for k in varied) if varied else 'candidate')
    ax.set_ylabel('{} (lower is better)'.format(metric.replace('_', ' ')), fontsize=8)
    ax.set_title('{}: winner circled'.format(cfg['name']), fontsize=10, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out_dir / 'tuning_curve.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_dir


PARTS = ('init', 'surrogate', 'selector', 'schedule', 'pool', 'use_noise')


def _differing_parts(arm, reference):
    """Parts in which two arms differ once resolved. A schedule difference neither arm asks
    for comes from the surrogate's own default (TabPFN's tuned schedule, amendment 4), so it
    belongs to the surrogate intervention."""
    from .arms import resolve
    a, r = resolve(arm), resolve(reference)
    parts = [p for p in PARTS if a.get(p) != r.get(p)]
    if 'schedule' in parts and 'surrogate' in parts and 'schedule' not in arm and 'schedule' not in reference:
        parts.remove('schedule')
    return parts


def decide_winners(report_dirs, out_path, pilot=False, require_significance=True):
    """Preregistered rule (section 6), applied per intervention part.

    Arms significantly better than their family reference (Holm, mean difference < 0)
    that differ from it in exactly one part compete for that part; the lowest mean
    difference wins. Parts with no significant arm keep the incumbent's choice.

    ``require_significance=False`` (pilots only) drops the significance requirement so a
    pilot exercises Phase 3 with some interventions; the output is marked accordingly.
    """
    if not require_significance and not pilot:
        raise ValueError('the significance requirement can only be dropped for pilots')
    best = {}
    evidence = {}
    for report_dir in report_dirs:
        cfg = json.loads((pathlib.Path(report_dir) / 'config.json').read_text())
        arms = {a['name']: a for a in cfg['arms']}
        for family, spec in (cfg.get('families') or {}).items():
            table_path = pathlib.Path(report_dir) / 'family_{}_primary.csv'.format(family)
            if not table_path.exists():
                continue
            table = pd.read_csv(table_path)
            evidence[family] = table.round(5).to_dict(orient='records')
            reference = arms[spec['reference'] if isinstance(spec, dict) else cfg.get('reference', 'incumbent')]
            eligible = (table.mean_diff < 0) & (table.significant if require_significance else True)
            for row in table[eligible].itertuples():
                parts = _differing_parts(arms[row.arm], reference)
                if len(parts) != 1:
                    continue
                part = parts[0]
                if part not in best or row.mean_diff < best[part][0]:
                    best[part] = (row.mean_diff, family, row.arm, arms[row.arm][part])
    winners = {'pilot': pilot, 'significance_required': require_significance, 'evidence': evidence,
               'interventions': {'{}:{}'.format(fam, part): {part: value} for part, (_, fam, _, value) in best.items()},
               'chosen_arms': {part: arm for part, (_, _, arm, _) in best.items()}}
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(json.dumps(winners, indent=1))
    return winners
