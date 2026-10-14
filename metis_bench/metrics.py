"""Benchmark metrics. FROZEN: see docs/PREREGISTRATION.md, section 4.

Any change needs a dated amendment in the preregistration and a new hash in
``metis_bench/FROZEN.json``; tests/test_frozen.py enforces this.

Conventions
-----------
* ``f_max``: optimum of the oracle over its feasible space (exact for synthetic
  functions, the enumerated Borkowski grid and the PURE pool).
* ``f_ref``: median noise-free yield of 10,000 random feasible designs (fixed seed).
* Normalised regret ``(f_max - v) / (f_max - f_ref)``: 0 at the optimum, 1 at the
  median random design. Lower is better. It is comparable across problems with
  different yield scales.
* Round 1 is the initial design; rounds 2 to R are model guided.
"""
import numpy as np
from scipy.stats import norm, spearmanr

PRIMARY_ENDPOINT = 'final_norm_regret'
TARGET_NORM_REGRET = 0.05
COVERAGE_LEVELS = (0.5, 0.8, 0.95)
CALIBRATION_CURVE = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))


def normalized_regret(value, f_max, f_ref):
    return (f_max - np.asarray(value, dtype=float)) / (f_max - f_ref)


def trajectory(rounds, true_yields, measured_means, f_max, f_ref, n_rounds):
    """Per-round regret trajectory of one campaign.

    ``rounds[i]`` is the round in which evaluated point ``i`` was measured,
    ``true_yields[i]`` its noise-free yield and ``measured_means[i]`` its replicate mean.

    * best-so-far: best noise-free yield among points measured up to that round
      (in-sample simple regret, the standard BO measure).
    * recommended: the point with the best *measured* mean so far, i.e. what an
      experimenter would pick; its noise-free yield gives the recommendation regret.
    * cumulative regret: sum of ``f_max - f(x)`` over all measured points; also as a mean
      normalised regret per measured point.
    """
    rounds = np.asarray(rounds)
    true_yields = np.asarray(true_yields, dtype=float)
    measured_means = np.asarray(measured_means, dtype=float)
    scale = f_max - f_ref
    rows = []
    for r in range(1, n_rounds + 1):
        seen = rounds <= r
        if not seen.any():
            continue
        best_true = float(true_yields[seen].max())
        rec = np.flatnonzero(seen)[int(np.argmax(measured_means[seen]))]
        cum = float(np.sum(f_max - true_yields[seen]))
        rows.append({'round': r, 'n_evaluated': int(seen.sum()), 'best_true': best_true,
                     'simple_regret': f_max - best_true, 'norm_regret': (f_max - best_true) / scale,
                     'rec_true': float(true_yields[rec]), 'norm_rec_regret': (f_max - true_yields[rec]) / scale,
                     'cum_regret': cum, 'norm_cum_regret': cum / (seen.sum() * scale)})
    return rows


def endpoints(trajectory_rows, f_max):
    """Per-campaign endpoints, all from the trajectory. Primary: ``final_norm_regret``."""
    norm_regret = np.array([row['norm_regret'] for row in trajectory_rows])
    final = trajectory_rows[-1]
    hit = np.flatnonzero(norm_regret <= TARGET_NORM_REGRET)
    return {
        'final_norm_regret': float(final['norm_regret']),
        'final_norm_rec_regret': float(final['norm_rec_regret']),
        'auc_norm_regret': float(norm_regret.mean()),
        'final_norm_cum_regret': float(final['norm_cum_regret']),
        'rounds_to_target': int(trajectory_rows[hit[0]]['round']) if len(hit) else len(trajectory_rows) + 1,
        'found_optimum': bool(final['best_true'] >= f_max - 1e-9 * max(1.0, abs(f_max))),
        'final_best_true': float(final['best_true']),
    }


def accuracy(pred_mean, truth):
    """R2, Spearman rho and RMSE of predicted means against noise-free truth."""
    pred_mean, truth = np.asarray(pred_mean, float), np.asarray(truth, float)
    resid = truth - pred_mean
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    rho = spearmanr(pred_mean, truth).statistic if np.ptp(truth) > 0 and np.ptp(pred_mean) > 0 else np.nan
    return {'r2': r2, 'spearman': float(rho), 'rmse': float(np.sqrt(np.mean(resid ** 2)))}


def calibration(pred_mean, pred_std, truth):
    """Calibration of central Gaussian intervals ``mean +/- z * std`` against truth.

    * ``cov50``/``cov80``/``cov95``: share of truths inside the 50/80/95% intervals.
    * ``mace``: mean |empirical - nominal| coverage over levels 0.1 to 0.9.
    * ``nlpd``: mean Gaussian negative log predictive density (std floored at 1e-9 *
      the truth's spread, so a zero-spread ensemble scores very badly rather than
      infinitely so).
    * ``sharpness``: mean predicted std; ``z_sd``: std of standardised errors (1 if
      calibrated, above 1 if overconfident).
    """
    m, s, t = (np.asarray(a, float) for a in (pred_mean, pred_std, truth))
    s = np.maximum(s, 1e-9 * (np.std(t) + 1e-12))
    z = (t - m) / s
    out = {}
    for level in COVERAGE_LEVELS:
        out['cov{}'.format(int(round(level * 100)))] = float(np.mean(np.abs(z) <= norm.ppf(0.5 + level / 2)))
    curve = [abs(np.mean(np.abs(z) <= norm.ppf(0.5 + lv / 2)) - lv) for lv in CALIBRATION_CURVE]
    out['mace'] = float(np.mean(curve))
    out['nlpd'] = float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * z ** 2))
    out['sharpness'] = float(np.mean(s))
    out['z_sd'] = float(np.std(z))
    return out
