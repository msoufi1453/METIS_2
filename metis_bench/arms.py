"""Arm configurations: which initialiser, surrogate, acquisition, selector and schedule.

An arm is a plain dict, so it can live in YAML experiment files and be logged verbatim:

    name: incumbent
    init: {type: random}
    surrogate: {type: xgb_ensemble, n_iter: 200, ensemble_len: 20}
    acquisition: {type: ucb, exploitation: 1.0}
    selector: {type: topk}
    schedule: {type: legacy}
    pool: {type: random}    # where candidate recipes come from (metis.pools)
    use_noise: false        # feed replicate variances to the surrogate

Missing keys take METIS v1 defaults (``INCUMBENT``).
"""
import copy

from metis import initialization, pools, schedules
from metis.acquisition import GPBUCB, UCB, ExpectedImprovement, LocalPenalization, RandomSelector, SequentialFantasy, TopK
from metis.surrogates import GPSurrogate, TabPFNSurrogate, XGBEnsemble

INCUMBENT = {
    'name': 'incumbent',
    'init': {'type': 'random'},
    'surrogate': {'type': 'xgb_ensemble', 'n_iter': 200, 'ensemble_len': 20},
    'acquisition': {'type': 'ucb', 'exploitation': 1.0},
    'selector': {'type': 'topk'},
    'schedule': {'type': 'legacy'},
    'pool': {'type': 'random'},
    'use_noise': False,
}


def resolve(arm):
    """Fill missing parts of an arm from the incumbent (one level deep)."""
    out = copy.deepcopy(INCUMBENT)
    for key, value in arm.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and value.get('type', out[key].get('type')) == out[key].get('type'):
            out[key] = {**out[key], **value}
        else:
            out[key] = copy.deepcopy(value)
    return out


def _params(cfg):
    return {k: v for k, v in cfg.items() if k != 'type'}


def make_surrogate(cfg, space, rng, use_noise):
    kind, p = cfg['type'], _params(cfg)
    if kind == 'xgb_ensemble':
        return XGBEnsemble(random_state=rng, noise_weighting=use_noise, **p)
    if kind == 'gp_matern':
        return GPSurrogate(input_transform=space.feature_transform(), random_state=rng, use_noise=use_noise, **p)
    if kind == 'tabpfn':
        return TabPFNSurrogate(random_state=rng, **p)
    raise ValueError('unknown surrogate {}'.format(kind))


def make_acquisition(cfg):
    kind, p = cfg['type'], _params(cfg)
    if kind == 'ucb':
        return UCB(**p)
    if kind == 'ei':
        return ExpectedImprovement(**p)
    raise ValueError('unknown acquisition {}'.format(kind))


def make_selector(cfg, rng=None):
    kind, p = cfg['type'], _params(cfg)
    if kind == 'random':
        return RandomSelector(seed=int(rng.integers(2 ** 31 - 1)) if rng is not None else 0)
    return {'topk': TopK, 'sequential_fantasy': SequentialFantasy, 'local_penalization': LocalPenalization,
            'gp_bucb': GPBUCB}[kind](**p)


def make_pool(cfg):
    return pools.make(cfg)


def make_schedule(cfg):
    return schedules.make(cfg)


def make_initializer(cfg):
    return initialization.make(cfg)
