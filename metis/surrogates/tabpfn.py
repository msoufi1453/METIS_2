"""TabPFN surrogate (optional dependency: ``uv sync --extra tabpfn``)."""
import numpy as np

from .base import Prediction, Surrogate, as_1d_float, as_2d_float


def tabpfn_available():
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        return False
    return True


class TabPFNSurrogate(Surrogate):
    """TabPFN v2 regressor; ``std`` comes from its predictive bar distribution.

    TabPFN infers in context, so ``fit`` is cheap and ``predict`` does the work. Inputs
    are chunked to bound memory. TabPFN cannot take per-sample noise variances, so
    ``accepts_noise`` is False and replicate stds are ignored.

    The weights (``Prior-Labs/TabPFN-v2-reg``) download on first use; pass ``model_path``
    to use a local copy on an offline cluster.
    """

    name = 'tabpfn'
    accepts_noise = False

    def __init__(self, n_estimators=8, device='cpu', model_path='auto', chunk_size=2000, random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.model_path = model_path
        self.chunk_size = chunk_size
        self.random_state = random_state

    def fit(self, X, y, noise_var=None):
        from tabpfn import TabPFNRegressor

        random_state = self.random_state
        if isinstance(random_state, np.random.Generator):
            random_state = int(random_state.integers(2 ** 31 - 1))
        self.model_ = TabPFNRegressor(n_estimators=self.n_estimators, device=self.device,
                                      model_path=self.model_path, random_state=random_state,
                                      ignore_pretraining_limits=True)
        self.model_.fit(as_2d_float(X), as_1d_float(y))
        return self

    def predict(self, X):
        X = as_2d_float(X)
        means, stds = [], []
        for start in range(0, len(X), self.chunk_size):
            out = self.model_.predict(X[start:start + self.chunk_size], output_type='full')
            means.append(np.asarray(out['mean'], dtype=float))
            variance = out['criterion'].variance(out['logits']).detach().cpu().numpy()
            stds.append(np.sqrt(np.maximum(variance.astype(float), 0.0)))
        return Prediction(mean=np.concatenate(means), std=np.concatenate(stds))

    def describe(self):
        return {'name': self.name, 'n_estimators': self.n_estimators}
