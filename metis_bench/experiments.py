"""Experiment files (``experiments/*.yaml``) expanded into campaign tasks.

Pilot mode (``--pilot``) applies the file's ``pilot`` overrides and writes to
``<results>/<name>-pilot``, so pilot runs can never mix with confirmatory ones.
"""
import copy
import json
import pathlib
import zlib

import yaml

from .campaign import Settings, result_path

ROOT = pathlib.Path(__file__).resolve().parents[1]


PART_NAMES = {'init': 'init', 'surrogate': 'surrogate', 'selector': 'batch', 'use_noise': 'noise',
              'schedule': 'schedule', 'pool': 'pool'}


def _read_json(path):
    path = ROOT / path if not pathlib.Path(path).is_absolute() else pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError('{} not found (produced by an earlier phase)'.format(path))
    return json.loads(path.read_text())


def winner_interventions(path, extra_tuning=()):
    """``{short name: arm fragment}`` from winners.json plus any tuning results."""
    out = {}
    for key, fragment in _read_json(path)['interventions'].items():
        part = next(iter(fragment))
        out[PART_NAMES[part]] = {**out.get(PART_NAMES[part], {}), **fragment}
    for tuning_path in extra_tuning:
        tuned = _read_json(tuning_path)
        out[PART_NAMES[tuned['part']]] = {tuned['part']: tuned['fragment']}
    return out


def _resolve_arms(cfg):
    from . import ablation, tuning

    base = cfg.get('base_arm') or {}
    if 'from_winners' in base:
        merged = {}
        for fragment in winner_interventions(base.pop('from_winners')).values():
            merged.update(fragment)
        base = {**merged, **base}
        cfg['base_arm'] = base
    for tuning_path in cfg.get('base_tuning', []):
        tuned = _read_json(tuning_path)
        base[tuned['part']] = tuned['fragment']
        cfg['base_arm'] = base
    if cfg.get('candidates') or cfg.get('candidates_by_type'):
        cfg['arms'] = tuning.candidate_arms(cfg)
    arms = cfg.get('arms')
    if isinstance(arms, dict) and 'ablation_of' in arms:
        interventions = winner_interventions(arms['ablation_of'], arms.get('tuning', []))
        if arms.get('only'):
            interventions = {k: v for k, v in interventions.items() if k in arms['only']}
        cfg['interventions'] = sorted(interventions)
        cfg['arms'] = ablation.ablation_arms(interventions)
    resolved = []
    for arm in cfg['arms']:
        arm = dict(arm)
        if 'from_tuning' in arm:
            tuned = _read_json(arm.pop('from_tuning'))
            arm[tuned['part']] = tuned['fragment']
        resolved.append({**{k: v for k, v in base.items() if k != 'name'}, **arm})
    cfg['arms'] = resolved


def load(path, pilot=False):
    cfg = yaml.safe_load(pathlib.Path(path).read_text())
    cfg.setdefault('settings', {})
    if pilot:
        over = cfg.get('pilot', {})
        cfg['name'] = cfg['name'] + '-pilot'
        cfg['seeds'] = over.get('seeds', cfg.get('seeds'))
        cfg['settings'] = {**cfg['settings'], **over.get('settings', {})}
        cfg['problems'] = {**cfg['problems'], **over.get('problems', {})}
        for key in ('candidates', 'candidates_by_type'):
            if key in over:
                cfg[key] = over[key]
        cfg['pilot'] = True
        cfg['pilot_overrides'] = over.get('arm_overrides', {})
        for key in ('output',):
            if key in cfg.get('tuning', {}):
                cfg['tuning'][key] = cfg['tuning'][key].replace('.json', '-pilot.json')
        for arm in cfg['arms'] if isinstance(cfg.get('arms'), list) else []:
            if 'from_tuning' in arm:
                arm['from_tuning'] = arm['from_tuning'].replace('.json', '-pilot.json')
        if isinstance(cfg.get('arms'), dict):
            cfg['arms'] = {k: (v.replace('.json', '-pilot.json') if isinstance(v, str) else
                               [x.replace('.json', '-pilot.json') for x in v]) for k, v in cfg['arms'].items()}
            if over.get('ablation_only'):
                cfg['arms']['only'] = over['ablation_only']
        cfg['base_tuning'] = [p.replace('.json', '-pilot.json') for p in cfg.get('base_tuning', [])]
        if 'from_winners' in (cfg.get('base_arm') or {}):
            cfg['base_arm']['from_winners'] = cfg['base_arm']['from_winners'].replace('.json', '-pilot.json')
    _resolve_arms(cfg)
    if pilot:
        # Cheaper surrogates for pilots, applied to every resolved arm using that surrogate.
        for arm in cfg['arms']:
            target = arm.get('surrogate', {'type': 'xgb_ensemble'})
            params = cfg.get('pilot_overrides', {}).get(target.get('type'), {})
            if params:
                arm['surrogate'] = {**target, **params}
    return cfg


def load_suite(path):
    path = ROOT / path if not pathlib.Path(path).is_absolute() else pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError('{} not found: run `python -m metis_bench calibrate` first'.format(path))
    return json.loads(path.read_text())['functions']


def stratified_sample(functions, n):
    """``n`` functions spread evenly over (landscape subtype, dimension) strata.

    Round-robin over the strata in sorted order, taking one function per stratum per
    pass, so every stratum is represented before any gets a second. Within a stratum the
    order is a fixed hash of the function id: neutral with respect to realism and the
    selection reason, and identical on every machine.
    """
    strata = {}
    for f in functions:
        strata.setdefault((f['spec']['subtype'], f['n_dims']), []).append(f)
    queues = [sorted(group, key=lambda f: zlib.crc32(f['function_id'].encode())) for _, group in sorted(strata.items())]
    chosen = []
    while len(chosen) < n and any(queues):
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop(0))
    return chosen


def problem_configs(cfg):
    p = cfg['problems']
    out = []
    noises = p.get('noise', ['realistic'])
    imported = p.get('imported')
    if p.get('suite'):
        functions = load_suite(p['suite'])
        functions = [f for f in functions if p.get('split', 'all') in ('all', f['split'])]
        functions = [f for f in functions if f['n_dims'] in p.get('dims', [12, 27])]
        if p.get('subtypes'):
            functions = [f for f in functions if f['spec']['subtype'] in p['subtypes']]
        if p.get('stratified_sample'):
            functions = stratified_sample(functions, int(p['stratified_sample']))
        if p.get('limit_per_subtype'):
            kept, counts = [], {}
            for f in sorted(functions, key=lambda f: f['realism']):
                key = (f['spec']['subtype'], f['n_dims'])
                if counts.get(key, 0) < p['limit_per_subtype']:
                    counts[key] = counts.get(key, 0) + 1
                    kept.append(f)
            functions = kept
        for f in functions:
            for noise in noises:
                out.append({'synthetic': f['spec'], 'n_dims': f['n_dims'], 'noise': noise, 'imported': imported,
                            'labels': {'kind': 'synthetic', 'landscape': f['landscape'], 'subtype': f['spec']['subtype'],
                                       'split': f['split'], 'realism': f['realism'],
                                       'nearest_real': f['nearest_real'], 'selection': f.get('selection')}})
    for entry in p.get('measured', []):
        # A measured dataset may set its own batch size: {name: pure, settings: {batch_size: 44}}.
        entry = {'name': entry} if isinstance(entry, str) else entry
        out.append({'measured': entry['name'], 'noise': 'none', 'settings': entry.get('settings', {}),
                    'labels': {'kind': 'measured', 'landscape': 'measured', 'subtype': entry['name'], 'split': 'measured',
                               'realism': 0.0, 'nearest_real': entry['name']}})
    for name in p.get('real', []):
        for noise in noises:
            out.append({'real': name, 'noise': noise, 'imported': imported,
                        'labels': {'kind': 'real', 'landscape': 'real', 'subtype': name, 'split': 'real',
                                   'realism': 0.0, 'nearest_real': name}})
    return out


def seeds_of(cfg):
    start, stop = cfg.get('seeds', [0, 50])
    return list(range(start, stop))


def tasks(cfg, out_dir):
    from .problems import get_problem

    settings = {**vars(Settings()), **cfg.get('settings', {})}
    out = []
    for problem_cfg in problem_configs(cfg):
        pid = get_problem(problem_cfg).id
        problem_settings = {**settings, **problem_cfg.get('settings', {})}
        for arm in cfg['arms']:
            for seed in seeds_of(cfg):
                path = result_path(out_dir, cfg['name'], pid, arm['name'], seed)
                out.append((copy.deepcopy(problem_cfg), copy.deepcopy(arm), seed, problem_settings, str(path)))
    return out
