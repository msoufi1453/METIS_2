"""The models a campaign can use.

Only TabPFN-3.5 is registered. Another model is one call away: anything with the
``metis.surrogates.Surrogate`` interface works, for example METIS v1's XGBoost ensemble::

    from metis.campaign import register_model
    from metis.surrogates import XGBEnsemble
    register_model('xgboost', 'XGBoost ensemble (METIS v1)',
                   lambda seed=0, **options: XGBEnsemble(random_state=seed, **options),
                   max_variables=1000, options={'n_iter': 200, 'ensemble_len': 20})
"""
import functools
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    build: object               # build(seed=0, **options) -> Surrogate
    max_variables: int          # most variables the app lets the user define for this model
    options: dict = field(default_factory=dict)
    pool_size: tuple = (10000, 10000)       # default candidates to score: (with a GPU, without)
    note: str = ''


MODELS = {}
DEFAULT_MODEL = 'tabpfn35'


def register_model(key, label, build, max_variables, options=None, pool_size=(10000, 10000), note=''):
    MODELS[key] = ModelSpec(key, label, build, int(max_variables), dict(options or {}), tuple(pool_size), note)
    return MODELS[key]


def model_spec(key):
    if key not in MODELS:
        raise KeyError('unknown model {!r}; registered: {}'.format(key, ', '.join(MODELS)))
    return MODELS[key]


def make_model(key, seed=0, **options):
    spec = model_spec(key)
    return spec.build(seed=seed, **{**spec.options, **options})


@functools.lru_cache(maxsize=None)
def has_gpu():
    """True when PyTorch finds a CUDA GPU. Importing PyTorch takes seconds, so the answer is kept."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def default_pool_size(key):
    with_gpu, without = model_spec(key).pool_size
    return with_gpu if has_gpu() else without


def _tabpfn35(seed=0, n_estimators=8, device='auto'):
    from ..surrogates import TabPFNSurrogate
    return TabPFNSurrogate(n_estimators=n_estimators, device=device, version='v3.5', random_state=seed)


register_model('tabpfn35', 'TabPFN-3.5', _tabpfn35, max_variables=6000,
               options={'n_estimators': 8, 'device': 'auto'}, pool_size=(10000, 1000),
               note='Non-commercial licence: accept it once or set TABPFN_TOKEN before starting. Fast on a GPU; '
                    'on a CPU it scores about 25 recipes a second.')
