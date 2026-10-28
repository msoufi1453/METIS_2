"""TabPFN surrogate.

``version='v2'`` (default, the main benchmark) needs ``uv sync --extra tabpfn``
(package 2.0.x). ``version='v3.5'`` needs ``uv sync --extra tabpfn35`` (package 9.x,
non-commercial weights, one-time licence acceptance with a Prior Labs account). The two
extras are mutually exclusive, and ``fit`` refuses to run on the wrong one, so a run
never silently uses a different model than its configuration says.
"""
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

    The std is the spread of TabPFN's predictive distribution for a *measurement*, so it
    includes the noise TabPFN infers; it cannot be split into latent and noise parts.
    Judged against noise-free truth (metis_bench calibration metrics), its intervals
    therefore over-cover, while the GP reports latent std. Keep this in mind when
    comparing calibration.

    The weights (``Prior-Labs/TabPFN-v2-reg``) download on first use; pass ``model_path``
    to use a local copy on an offline cluster.
    """

    name = 'tabpfn'
    accepts_noise = False

    def __init__(self, n_estimators=8, device='cpu', model_path='auto', chunk_size=2000, random_state=0, version='v2'):
        self.version = version
        self.n_estimators = n_estimators
        self.device = device
        self.model_path = model_path
        self.chunk_size = chunk_size
        self.random_state = random_state

    def fit(self, X, y, noise_var=None):
        import tabpfn
        from tabpfn import TabPFNRegressor

        random_state = self.random_state
        if isinstance(random_state, np.random.Generator):
            random_state = int(random_state.integers(2 ** 31 - 1))
        major = int(str(getattr(tabpfn, '__version__', '0')).split('.')[0])
        if self.version == 'v2':
            if major >= 3:
                raise RuntimeError('TabPFN v2 arm needs the tabpfn 2.0.x package (uv sync --extra tabpfn); '
                                   'installed: {}'.format(tabpfn.__version__))
            self.model_ = TabPFNRegressor(n_estimators=self.n_estimators, device=self.device,
                                          model_path=self.model_path, random_state=random_state,
                                          ignore_pretraining_limits=True)
        elif self.version == 'v3.5':
            if major < 9:
                raise RuntimeError('TabPFN-3.5 arm needs the tabpfn 9.x package (uv sync --extra tabpfn35); '
                                   'installed: {}'.format(getattr(tabpfn, '__version__', '?')))
            from tabpfn.constants import ModelVersion
            options = {'device': self.device, 'random_state': random_state, 'ignore_pretraining_limits': True}
            if self.n_estimators != 'auto':
                options['n_estimators'] = self.n_estimators
            self.model_ = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, **options)
        else:
            raise ValueError('unknown TabPFN version {}'.format(self.version))
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
        out = {'name': self.name, 'version': self.version, 'n_estimators': self.n_estimators}
        model = getattr(self, 'model_', None)
        if model is not None:   # the device the model actually ran on, e.g. cuda:0 or cpu
            devices = getattr(model, 'devices_', None) or getattr(model, 'device_', None)
            out['device_used'] = str(devices[0] if isinstance(devices, (tuple, list)) else devices)
        return out
