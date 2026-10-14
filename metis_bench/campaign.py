"""Campaign runner: one arm on one problem with one seed, and experiments over many.

Protocol (docs/PREREGISTRATION.md, section 3): ``n_rounds`` rounds of ``batch_size``
recipes. Round 1 is the arm's initial design, unless the problem starts from imported
data and the arm's initialiser is ``none``. Every later round fits the surrogate to
all replicate means so far and scores a fresh pool of ``pool_size`` unmeasured
feasible recipes. The selector picks the batch, which is measured in triplicate under
the problem's noise model.

Seeds give common random numbers. A seed and a problem fix the initial design for
initialisers that share a type, the imported data, and every noise draw (plate effects
and per-well noise are drawn per round in a fixed order). Paired arms therefore differ
only through the recipes they choose.
"""
import json
import os
import pathlib
import platform
import subprocess
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import numpy as np

from metis.optimize import suggest

from . import arms as arm_lib
from . import metrics
from .problems import get_problem, imported_design, problem_seed


@dataclass
class Settings:
    batch_size: int = 32
    n_rounds: int = 10
    pool_size: int = 10000
    test_points: int = 1000


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              cwd=pathlib.Path(__file__).parent).stdout.strip()
    except OSError:
        return None


def run_campaign(problem_cfg, arm_cfg, seed, settings=None):
    settings = settings or Settings()
    t_start = time.time()
    problem = get_problem(problem_cfg)
    oracle, noise, space = problem.oracle, problem.noise, problem.oracle.space
    arm = arm_lib.resolve(arm_cfg)
    ss = np.random.SeedSequence([seed, problem_seed(problem.id)])
    s_init, s_noise, s_pool, s_model, s_import = ss.spawn(5)
    noise_rngs = [np.random.default_rng(s) for s in s_noise.spawn(settings.n_rounds + 1)]
    pool_rngs = [np.random.default_rng(s) for s in s_pool.spawn(settings.n_rounds + 1)]
    model_rng = np.random.default_rng(s_model)

    # Held-out accuracy set: fixed per problem, independent of seed and arm.
    test_rng = np.random.default_rng(zlib.crc32((problem.id + '|test').encode()))
    test_idx = space.sample(settings.test_points, test_rng)
    test_X, test_f = space.features(test_idx), oracle(test_idx)

    log = {'idx': [], 'round': [], 'true': [], 'mean': [], 'std': []}

    def measure(idx, r):
        f = oracle(idx)
        mean, std, _ = noise.measure(f, oracle.f_max, noise_rngs[r])
        log['idx'] += idx.tolist()
        log['round'] += [r] * len(idx)
        log['true'] += f.tolist()
        log['mean'] += mean.tolist()
        log['std'] += std.tolist()

    if problem.imported:
        n_imp = problem.imported.get('n', settings.batch_size)
        measure(imported_design(space, n_imp, np.random.default_rng(s_import), problem.imported.get('kind', 'ofat')), 0)

    initializer = arm_lib.make_initializer(arm['init'])
    first_guided = 1
    if initializer is not None:
        existing = np.array(log['idx'], dtype=np.int64) if log['idx'] else None
        measure(initializer(space, settings.batch_size, np.random.default_rng(s_init), existing=existing), 1)
        first_guided = 2

    schedule = arm_lib.make_schedule(arm['schedule'])
    selector = arm_lib.make_selector(arm['selector'], np.random.default_rng(s_model.spawn(1)[0]))
    use_noise = bool(arm.get('use_noise'))
    rounds_log = []
    # Best measured yield after each completed round, for adaptive schedules.
    history = [float(np.max(log['mean']))] if log['mean'] else []
    for r in range(first_guided, settings.n_rounds + 1):
        t0 = time.time()
        observed = np.array(log['idx'], dtype=np.int64)
        X_obs, y_obs = space.features(observed), np.array(log['mean'])
        noise_var = np.array(log['std']) ** 2 / noise.n_replicates if use_noise else None
        surrogate = arm_lib.make_surrogate(arm['surrogate'], space, model_rng, use_noise)
        acquisition = arm_lib.make_acquisition(arm['acquisition'])
        exploration = None
        if hasattr(acquisition, 'exploration'):
            exploration = float(schedule(r, settings.n_rounds, history))
            acquisition.exploration = exploration
        pool = space.sample(settings.pool_size, pool_rngs[r], exclude=space.keys(observed))
        if len(pool) == 0:
            break
        if not selector.needs_model:
            # Random search: no surrogate to fit or evaluate.
            chosen = pool[selector.rng.choice(len(pool), size=min(settings.batch_size, len(pool)), replace=False)]
            measure(chosen, r)
            history.append(float(np.max(log['mean'])))
            rounds_log.append({'round': r, 'exploration': None, 'select_s': time.time() - t0, 'batch_size': len(chosen)})
            continue
        result = suggest(surrogate, X_obs, y_obs, space.features(pool), settings.batch_size, acquisition=acquisition,
                         selector=selector, noise_var=noise_var, embedding=space.embedding(pool))
        t_select = time.time() - t0
        chosen = pool[result.indices]

        test_pred = surrogate.predict(test_X)
        chosen_f = oracle(chosen)
        record = {'round': r, 'exploration': exploration, 'select_s': t_select, 'batch_size': len(chosen),
                  **{'test_' + k: v for k, v in metrics.accuracy(test_pred.mean, test_f).items()},
                  **{'test_' + k: v for k, v in metrics.calibration(test_pred.mean, test_pred.std, test_f).items()},
                  **{'batch_' + k: v for k, v in metrics.accuracy(result.prediction.mean, chosen_f).items()},
                  **{'batch_' + k: v for k, v in metrics.calibration(result.prediction.mean, result.prediction.std,
                                                                      chosen_f).items()}}
        measure(chosen, r)
        history.append(float(np.max(log['mean'])))
        rounds_log.append(record)

    trajectory = metrics.trajectory(log['round'], log['true'], log['mean'], oracle.f_max, oracle.f_ref,
                                    settings.n_rounds)
    by_round = {row['round']: row for row in trajectory}
    for record in rounds_log:
        record.update({k: v for k, v in by_round.get(record['round'], {}).items() if k != 'round'})
    return {
        'problem': {'id': problem.id, **oracle.describe(), 'noise': noise.to_dict(), 'imported': problem.imported,
                    **(problem.labels or {})},
        'arm': arm, 'arm_name': arm['name'], 'seed': seed, 'settings': asdict(settings),
        'trajectory': trajectory, 'rounds': rounds_log,
        'endpoints': metrics.endpoints(trajectory, oracle.f_max),
        'evaluated': log,
        'environment': {'git_commit': _git_commit(), 'python': platform.python_version()},
        'elapsed_s': time.time() - t_start,
    }


# --------------------------------------------------------------------------- experiments
def _slug(text):
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in text)


def result_path(out_dir, experiment, problem_id, arm_name, seed):
    return pathlib.Path(out_dir) / _slug(experiment) / _slug(problem_id) / _slug(arm_name) / 'seed_{:03d}.json'.format(seed)


def _worker(task):
    problem_cfg, arm_cfg, seed, settings, path = task
    try:
        result = run_campaign(problem_cfg, arm_cfg, seed, Settings(**settings))
        tmp = pathlib.Path(str(path) + '.tmp')
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(result, default=_jsonable))
        os.replace(tmp, path)
        return path, None, result['elapsed_s']
    except Exception:  # keep the experiment going; the failure is reported and logged
        return path, traceback.format_exc(), None


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _limit_threads():
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'


def run_tasks(tasks, jobs=1, progress=print):
    """Run (problem_cfg, arm_cfg, seed, settings_dict, path) tasks; skips finished ones."""
    todo = [t for t in tasks if not pathlib.Path(t[4]).exists()]
    progress('{} tasks, {} already done, {} to run with {} job(s)'.format(len(tasks), len(tasks) - len(todo), len(todo), jobs))
    failures = []
    if jobs <= 1:
        for i, task in enumerate(todo, 1):
            path, error, elapsed = _worker(task)
            failures += [(path, error)] if error else []
            progress('[{}/{}] {} {}'.format(i, len(todo), 'FAILED' if error else '{:.1f}s'.format(elapsed), path))
        return failures
    _limit_threads()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_worker, t) for t in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            path, error, elapsed = fut.result()
            failures += [(path, error)] if error else []
            progress('[{}/{}] {} {}'.format(i, len(todo), 'FAILED' if error else '{:.1f}s'.format(elapsed), path))
    return failures
