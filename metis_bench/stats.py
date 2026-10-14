"""Statistical comparison of arms. FROZEN: see docs/PREREGISTRATION.md, section 5.

Design: every arm runs the same problems with the same seeds. A seed fixes the initial
design (for arms sharing an initialiser) and every noise draw, so arms are compared on
paired runs.

Unit of analysis for claims about the suite: the problem (function x noise preset).
Seeds are replicates within a problem.

* Per problem: the paired difference ``arm - reference`` in the endpoint for each seed,
  their mean, and a two-sided Wilcoxon signed-rank test across seeds.
* Across problems: two-sided Wilcoxon signed-rank test on the per-problem mean
  differences. Effect sizes: Hodges-Lehmann estimate (median of Walsh averages),
  share of problems where the arm is better, and a hierarchical bootstrap 95% CI of the
  mean difference (resample problems, then seeds within each problem).
* Multiplicity: Holm-Bonferroni over each preregistered family of comparisons.
* Lower endpoint values are better (regret), so a negative difference favours the arm.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ALPHA = 0.05
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20241014
PRACTICAL_THRESHOLD = 0.05


def paired_differences(runs, arm, reference, metric, keys=('problem_id',)):
    """One row per (problem, seed) present for both arms: ``diff = arm - reference``."""
    cols = list(keys) + ['seed', metric]
    a = runs.loc[runs.arm == arm, cols]
    b = runs.loc[runs.arm == reference, cols]
    merged = a.merge(b, on=list(keys) + ['seed'], suffixes=('_arm', '_ref'))
    merged['diff'] = merged[metric + '_arm'] - merged[metric + '_ref']
    return merged


def _wilcoxon_p(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, zero_method='wilcox', alternative='two-sided').pvalue)


def hodges_lehmann(values):
    v = np.asarray(values, float)
    i, j = np.triu_indices(len(v))
    return float(np.median((v[i] + v[j]) / 2)) if len(v) else np.nan


def hierarchical_bootstrap_ci(diffs, keys=('problem_id',), n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, level=0.95):
    groups = [g['diff'].to_numpy() for _, g in diffs.groupby(list(keys))]
    if not groups:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, len(groups), len(groups))
        stats[b] = np.mean([groups[c][rng.integers(0, len(groups[c]), len(groups[c]))].mean() for c in chosen])
    tail = (1 - level) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def per_problem(diffs, keys=('problem_id',)):
    rows = []
    for key, g in diffs.groupby(list(keys)):
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))), 'n_seeds': len(g),
                     'mean_diff': float(g['diff'].mean()), 'median_diff': float(g['diff'].median()),
                     'p_seeds': _wilcoxon_p(g['diff'])})
    return pd.DataFrame(rows)


def compare(runs, arms, reference, metric, keys=('problem_id',), n_boot=N_BOOTSTRAP):
    """Suite-level comparison of each arm against ``reference`` on ``metric``.

    ``arms`` is one preregistered family; Holm's correction is applied within it.
    """
    rows = []
    for arm in arms:
        diffs = paired_differences(runs, arm, reference, metric, keys)
        table = per_problem(diffs, keys)
        if table.empty:
            continue
        lo, hi = hierarchical_bootstrap_ci(diffs, keys, n_boot=n_boot)
        rows.append({'arm': arm, 'reference': reference, 'metric': metric, 'n_problems': len(table),
                     'n_pairs': len(diffs), 'mean_diff': float(table.mean_diff.mean()),
                     'hodges_lehmann': hodges_lehmann(table.mean_diff), 'ci_low': lo, 'ci_high': hi,
                     'share_better': float(np.mean(table.mean_diff < 0)),
                     'share_worse': float(np.mean(table.mean_diff > 0)),
                     'p_value': _wilcoxon_p(table.mean_diff)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out['p_holm'] = holm(out.p_value)
        out['significant'] = out.p_holm < ALPHA
        out['practically_relevant'] = out.mean_diff.abs() >= PRACTICAL_THRESHOLD
    return out
