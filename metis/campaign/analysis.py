"""What the data and the model say: a PCA map, variable importance, pairwise interactions, progress.

Importance and interactions question the fitted model on the measured recipes, so they
describe what the model has learned from the data so far; they are not proof of cause.
Each needs one model call, whatever the number of variables.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAMap:
    """A two-dimensional map of recipes: fitted once, then used to place any other recipe."""
    names: list                 # the embedding's columns
    mean: np.ndarray
    components: np.ndarray      # (2, n columns); a zero row when the recipes do not vary that far
    explained: np.ndarray       # share of the variation between the fitted recipes on each axis

    def transform(self, E):
        """Map coordinates (n, 2) of embedding rows."""
        E = np.asarray(E, dtype=float).reshape(-1, len(self.mean))
        return (E - self.mean) @ self.components.T

    def to_dict(self):
        """JSON-ready form (stored in the round history)."""
        return {'columns': list(self.names), 'mean': [float(v) for v in self.mean],
                'components': [[float(v) for v in row] for row in self.components],
                'explained_variance_ratio': [float(v) for v in self.explained]}

    @classmethod
    def from_dict(cls, data):
        """A map from ``to_dict`` output."""
        return cls(list(data['columns']), np.asarray(data['mean'], dtype=float),
                   np.asarray(data['components'], dtype=float).reshape(2, -1),
                   np.asarray(data['explained_variance_ratio'], dtype=float))


def fit_pca(E, names):
    """Principal components of the rows of ``E`` (centred, not rescaled: the embedding already is)."""
    E = np.asarray(E, dtype=float)
    if E.ndim != 2:
        E = E.reshape(len(E), -1)
    n, d = E.shape
    mean = E.mean(axis=0) if n else np.zeros(d)
    components, explained = np.zeros((2, d)), np.zeros(2)
    if n >= 2 and d >= 1:
        _, S, Vt = np.linalg.svd(E - mean, full_matrices=False)
        total = float(np.sum(S ** 2))
        for c in range(min(2, len(S))):
            if total <= 0 or S[c] <= 1e-10 * S[0]:
                break
            axis = Vt[c]
            components[c] = axis if axis[np.argmax(np.abs(axis))] >= 0 else -axis     # a fixed sign
            explained[c] = S[c] ** 2 / total
    return PCAMap(list(names), mean, components, explained)


def _r2(y, predictions):
    """R^2 of each row of ``predictions`` against ``y`` (0 when ``y`` does not vary)."""
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0:
        return np.zeros(len(predictions))
    return 1.0 - np.sum((predictions - y) ** 2, axis=1) / total


def importance(model, X, y, groups, repeats=3, max_rows=150, seed=0):
    """How much the fitted model relies on each variable: the drop in R^2 on the measured
    recipes when that variable's values are shuffled between recipes (mean of ``repeats``).

    ``groups``: ``[(variable, its columns of X)]``. At most ``max_rows`` recipes are used.
    Returns a Series, largest first, with the unshuffled R^2 in ``.attrs['r2']``.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        rows = np.sort(rng.choice(len(X), size=max_rows, replace=False))
        X, y = X[rows], y[rows]
    names = [name for name, _ in groups]
    if len(X) < 2 or not groups:
        out = pd.Series(0.0, index=names, name='R2 drop when shuffled')
        out.attrs['r2'] = float('nan')
        return out
    blocks = [X]
    for _, columns in groups:
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[:, columns] = X[rng.permutation(len(X))][:, columns]
            blocks.append(shuffled)
    predictions = np.asarray(model.predict(np.vstack(blocks)).mean, dtype=float).reshape(len(blocks), len(X))
    r2 = _r2(y, predictions)
    drops = r2[0] - r2[1:].reshape(len(groups), repeats).mean(axis=1)
    out = pd.Series(drops, index=names, name='R2 drop when shuffled').sort_values(ascending=False, kind='stable')
    out.attrs['r2'] = float(r2[0])
    return out


def interactions(model, X, groups, n_pairs=30, seed=0):
    """How strongly pairs of variables interact in the fitted model.

    Take two measured recipes ``a`` and ``b``. Give ``a`` variable ``i`` from ``b``, variable
    ``j`` from ``b``, and both. If the two variables act independently, the change from
    both equals the sum of the single changes; the interaction is how far off that is:
    ``|f(both) - f(i) - f(j) + f(a)|``, averaged over ``n_pairs`` random pairs of recipes
    that differ in both variables. It is in the units of the result, and exactly 0 for a
    model in which the two variables simply add up.

    Returns one row per pair of variables, strongest first.
    """
    X = np.asarray(X, dtype=float)
    columns = ['Variable A', 'Variable B', 'Interaction', 'Pairs compared']
    g, n = len(groups), len(X)
    if n < 2 or g < 2:
        return pd.DataFrame(columns=columns)
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, size=n_pairs)
    b = (a + 1 + rng.integers(0, n - 1, size=n_pairs)) % n          # never the same recipe as ``a``
    base, other = X[a], X[b]
    differs = np.array([np.any(base[:, cols] != other[:, cols], axis=1) for _, cols in groups])
    pairs = [(i, j) for i in range(g) for j in range(i + 1, g)]
    blocks = [base]
    for _, cols in groups:
        single = base.copy()
        single[:, cols] = other[:, cols]
        blocks.append(single)
    for i, j in pairs:
        both = base.copy()
        both[:, groups[i][1]] = other[:, groups[i][1]]
        both[:, groups[j][1]] = other[:, groups[j][1]]
        blocks.append(both)
    f = np.asarray(model.predict(np.vstack(blocks)).mean, dtype=float).reshape(len(blocks), n_pairs)
    rows = []
    for p, (i, j) in enumerate(pairs):
        informative = differs[i] & differs[j]
        mixed = np.abs(f[1 + g + p] - f[1 + i] - f[1 + j] + f[0])
        strength = float(mixed[informative].mean()) if informative.any() else 0.0
        rows.append((groups[i][0], groups[j][0], strength, int(informative.sum())))
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values('Interaction', ascending=False, kind='stable').reset_index(drop=True)


def progress(samples, round_column, target):
    """Per round: recipes measured, their best and median result, and the best so far."""
    if not len(samples):
        return pd.DataFrame(columns=[round_column, 'Recipes', 'Best', 'Median', 'Best so far'])
    grouped = samples.groupby(round_column)[target]
    out = pd.DataFrame({'Recipes': grouped.size(), 'Best': grouped.max(), 'Median': grouped.median()}).sort_index()
    out['Best so far'] = out['Best'].cummax()
    return out.reset_index()
