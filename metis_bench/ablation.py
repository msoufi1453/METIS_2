"""Step 13: full factorial ablation of the winning interventions.

``ablation_arms`` turns k interventions into the 2^k arms: every subset applied on top
of the incumbent, the empty subset being the incumbent itself. ``analyze`` attributes the
effect:

* **Factorial regression** on per-problem paired differences to the incumbent:
  ``d = sum_i b_i x_i + sum_{i<j} b_ij x_i x_j``. The empty subset is the reference, so
  there is no intercept. CIs come from a bootstrap over problems.
* **Shapley shares** of the full combination's improvement, exact over all subsets.
  ``v(S)`` = mean improvement over the incumbent of the arm with subset ``S``. Shares
  sum to the full combination's improvement.
"""
import itertools
import math

import numpy as np
import pandas as pd

from . import metrics


def ablation_arms(interventions, base=None):
    """``interventions``: ``{name: arm fragment}``. One arm per subset."""
    names = sorted(interventions)
    arms = []
    for k in range(len(names) + 1):
        for subset in itertools.combinations(names, k):
            arm = dict(base or {})
            for name in subset:
                arm.update(interventions[name])
            arm['name'] = 'incumbent' if not subset else 'abl_' + '+'.join(subset)
            arm['ablation_subset'] = list(subset)
            arms.append(arm)
    return arms


def _subset_of(arm_name, names):
    if arm_name == 'incumbent':
        return frozenset()
    return frozenset(arm_name[len('abl_'):].split('+'))


def paired_effects(runs, names, metric=metrics.PRIMARY_ENDPOINT):
    """Per (problem, subset): mean over seeds of (arm - incumbent) on paired seeds."""
    ref = runs[runs.arm == 'incumbent'][['problem_id', 'seed', metric]]
    rows = []
    for arm, g in runs.groupby('arm'):
        merged = g[['problem_id', 'seed', metric]].merge(ref, on=['problem_id', 'seed'], suffixes=('', '_ref'))
        merged['diff'] = merged[metric] - merged[metric + '_ref']
        for pid, h in merged.groupby('problem_id'):
            rows.append({'problem_id': pid, 'subset': _subset_of(arm, names), 'diff': h['diff'].mean()})
    return pd.DataFrame(rows)


def _design(subsets, names):
    pairs = list(itertools.combinations(names, 2))
    X = [[1.0 if n in s else 0.0 for n in names] + [1.0 if (a in s and b in s) else 0.0 for a, b in pairs]
         for s in subsets]
    return np.array(X), [*names, *['{}x{}'.format(a, b) for a, b in pairs]]


def _shapley(values, names):
    """values: {frozenset: improvement}. Exact Shapley values."""
    n = len(names)
    out = {}
    for name in names:
        total = 0.0
        others = [m for m in names if m != name]
        for k in range(n):
            for subset in itertools.combinations(others, k):
                s = frozenset(subset)
                weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                total += weight * (values[s | {name}] - values[s])
        out[name] = total
    return out


def analyze(runs, names, metric=metrics.PRIMARY_ENDPOINT, n_boot=2000, seed=0):
    effects = paired_effects(runs, names, metric)
    problems = sorted(effects.problem_id.unique())

    def estimate(problem_ids):
        sub = pd.concat([effects[effects.problem_id == p] for p in problem_ids])
        mean_by_subset = sub.groupby('subset')['diff'].mean()
        X, labels = _design(list(mean_by_subset.index), names)
        coef = np.linalg.lstsq(X, mean_by_subset.to_numpy(), rcond=None)[0] if len(X) else np.array([])
        improvement = {s: -float(mean_by_subset.get(s, 0.0)) for s in mean_by_subset.index}
        improvement.setdefault(frozenset(), 0.0)
        shap = _shapley(improvement, names) if len(improvement) == 2 ** len(names) else {}
        return dict(zip(labels, coef)), shap

    coef, shap = estimate(problems)
    rng = np.random.default_rng(seed)
    boot_coef, boot_shap = [], []
    for _ in range(n_boot):
        sample = list(rng.choice(problems, size=len(problems), replace=True))
        c, s = estimate(sample)
        boot_coef.append(c)
        boot_shap.append(s)
    coef_table = pd.DataFrame([{'term': k, 'effect_on_regret': v,
                                'ci_low': float(np.quantile([b[k] for b in boot_coef], 0.025)),
                                'ci_high': float(np.quantile([b[k] for b in boot_coef], 0.975))}
                               for k, v in coef.items()])
    shap_table = pd.DataFrame([{'intervention': k, 'shapley_improvement': v,
                                'ci_low': float(np.quantile([b[k] for b in boot_shap if b], 0.025)),
                                'ci_high': float(np.quantile([b[k] for b in boot_shap if b], 0.975))}
                               for k, v in shap.items()]) if shap else pd.DataFrame()
    per_subset = effects.groupby('subset')['diff'].agg(['mean', 'sem', 'count']).reset_index()
    per_subset['subset'] = per_subset.subset.map(lambda s: '+'.join(sorted(s)) or 'incumbent')
    return coef_table, shap_table, per_subset
