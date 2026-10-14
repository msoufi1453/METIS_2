"""Phase 3 tuning on the dev split (steps 11 and 12) and the winner bookkeeping.

A tuning file lists candidate values for one part of an arm (``schedule`` or
``surrogate``) on top of ``base_arm``. Every combination becomes an arm, all arms run
on the dev problems with tuning seeds (disjoint from the confirmatory seeds 0-49), and
the arm with the lowest mean primary endpoint (problems weighted equally) wins.
The winning fragment goes to a JSON file that confirmatory experiments reference with
``from_tuning``.

``decide_winners`` applies the preregistered decision rule (section 6) to Phase 2
reports and writes ``artifacts/winners.json``, which Phase 3 files use through
``base_arm: {from_winners: ...}`` and ``arms: {ablation_of: ...}``.
"""
import copy
import itertools
import json
import pathlib

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
    return arms


def select_winner(runs, metric=metrics.PRIMARY_ENDPOINT):
    """Arm with the lowest mean over problems of the per-problem mean endpoint."""
    per_problem = runs.groupby(['arm', 'problem_id'])[metric].mean().reset_index()
    table = per_problem.groupby('arm')[metric].agg(['mean', 'sem', 'count']).sort_values('mean')
    return table.index[0], table


def write_winner(path, cfg, winner_name, table):
    part = cfg['tuning']['part']
    arm = next(a for a in cfg['arms'] if a['name'] == winner_name)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'part': part, 'fragment': arm[part], 'winner': winner_name,
                                'pilot': bool(cfg.get('pilot')),
                                'table': table.reset_index().round(5).to_dict(orient='records')}, indent=1))
    return path


PARTS = ('init', 'surrogate', 'selector', 'schedule', 'pool', 'use_noise')


def _differing_parts(arm, reference):
    from .arms import resolve
    a, r = resolve(arm), resolve(reference)
    return [p for p in PARTS if a.get(p) != r.get(p)]


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
