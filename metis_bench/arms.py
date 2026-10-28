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

Missing keys take METIS v1 defaults (``INCUMBENT``), except that a TabPFN arm without
its own ``schedule`` takes TabPFN's tuned schedule (``TABPFN_SCHEDULE``, amendment 4).
"""
import copy
import functools
import json
import pathlib

from metis import initialization, pools, schedules
from metis.acquisition import GPBUCB, UCB, ExpectedImprovement, LocalPenalization, RandomSelector, SequentialFantasy, TopK
from sklearn.linear_model import LinearRegression

from metis.surrogates import GPSurrogate, RegressorEnsemble, TabPFNSurrogate, XGBEnsemble

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


# TabPFN's own exploration schedule, tuned on the measured-data replay by
# experiments/tabpfn_schedule_tune.yaml. XGBoost keeps the v1 schedule it was tuned with.
TABPFN_SCHEDULE = pathlib.Path(__file__).resolve().parents[1] / 'experiments' / 'tuned' / 'tabpfn_schedule.json'


@functools.lru_cache(maxsize=4)
def _tuned_fragment(path, mtime_ns):
    tuned = json.loads(pathlib.Path(path).read_text())
    if tuned.get('part') != 'schedule':
        raise ValueError('{} does not hold a schedule'.format(path))
    return tuned['fragment']


def tabpfn_schedule():
    """The schedule a TabPFN arm uses when it sets none. Refuses to guess if it is not tuned yet."""
    if not TABPFN_SCHEDULE.exists():
        raise FileNotFoundError(
            '{} not found. TabPFN arms use the exploration schedule tuned on the replay data: run '
            '`python -m metis_bench tune experiments/tabpfn_schedule_tune.yaml` (TabPFN-3.5, GPU) and commit '
            'the file, or give the arm an explicit schedule.'.format(TABPFN_SCHEDULE))
    return copy.deepcopy(_tuned_fragment(str(TABPFN_SCHEDULE), TABPFN_SCHEDULE.stat().st_mtime_ns))


def resolve(arm):
    """Fill missing parts of an arm from the incumbent (one level deep).

    A TabPFN arm that sets no ``schedule`` gets ``tabpfn_schedule()`` instead of the
    incumbent's v1 schedule.
    """
    out = copy.deepcopy(INCUMBENT)
    for key, value in arm.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and value.get('type', out[key].get('type')) == out[key].get('type'):
            out[key] = {**out[key], **value}
        else:
            out[key] = copy.deepcopy(value)
    if 'schedule' not in arm and out['surrogate'].get('type') == 'tabpfn':
        out['schedule'] = tabpfn_schedule()
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
    if kind == 'linear_ensemble':
        # The paper's linear arm: identical copies, so the ensemble spread is zero and UCB ranks by the mean.
        return RegressorEnsemble([LinearRegression() for _ in range(p.get('members', 20))], noise_weighting=use_noise)
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
