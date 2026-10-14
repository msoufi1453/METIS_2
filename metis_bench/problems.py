"""Problems = oracle x noise preset (x optional imported data). Built lazily and cached per process."""
import functools
import json
import zlib
from dataclasses import dataclass

import numpy as np

from .noise import get_noise
from .oracles import REAL_ORACLES, SyntheticOracle
from .synthetic import FunctionSpec, build_function


@dataclass
class Problem:
    id: str
    oracle: object
    noise: object
    imported: dict = None
    labels: dict = None


@functools.lru_cache(maxsize=256)
def _oracle_from_key(key):
    cfg = json.loads(key)
    if 'real' in cfg:
        return REAL_ORACLES[cfg['real']]()
    return SyntheticOracle(build_function(FunctionSpec.from_dict(cfg['synthetic']), cfg['n_dims']))


def oracle_for(cfg):
    key = {'real': cfg['real']} if 'real' in cfg else {'synthetic': cfg['synthetic'], 'n_dims': cfg['n_dims']}
    return _oracle_from_key(json.dumps(key, sort_keys=True))


def get_problem(cfg):
    """``cfg``: ``{'real': 'borkowski'}`` or ``{'synthetic': spec_dict, 'n_dims': 12}``, plus
    ``noise`` (preset name or dict) and optional ``imported`` and ``labels``."""
    oracle = oracle_for(cfg)
    noise = get_noise(cfg.get('noise', 'realistic'))
    pid = '{}|{}'.format(oracle.id, noise.name)
    if cfg.get('imported'):
        pid += '|imported-{}'.format(cfg['imported'].get('kind', 'ofat'))
    return Problem(pid, oracle, noise, cfg.get('imported'), cfg.get('labels') or {})


def imported_design(space, n, rng, kind='ofat', n_bases=2):
    """Historical data concentrated in one corner of the space, as when importing a
    previous study: one-factor-at-a-time titrations around ``n_bases`` base recipes (the
    PURE data looks like this). The bases share a random region covering the lower half
    of each factor's levels."""
    lower = np.maximum(space.cardinalities // 2, 1)
    bases = []
    for _ in range(200):
        b = rng.integers(0, lower)
        if space.feasible(b[None])[0]:
            bases.append(b)
        if len(bases) == n_bases:
            break
    points, keys = [], set()
    for _ in range(200 * n):
        if len(points) == n or not bases:
            break
        base = bases[len(points) % len(bases)].copy()
        d = rng.integers(space.n_dims)
        base[d] = rng.integers(space.cardinalities[d])
        key = space.keys(base[None])[0]
        if key not in keys and space.feasible(base[None])[0]:
            keys.add(key)
            points.append(base)
    if len(points) < n:
        # Pooled spaces rarely contain one-factor variations of a random base: top up randomly.
        extra = space.sample(n - len(points), rng, exclude=keys)
        points += list(extra)
    return np.array(points, dtype=np.int64)


def problem_seed(problem_id):
    return zlib.crc32(problem_id.encode())
