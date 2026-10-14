"""Gaussian process surrogate with a Matern kernel."""
import warnings

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .base import Prediction, Surrogate, as_1d_float, as_2d_float


class MinMaxTransform:
    """Scales each input column to [0, 1] using ranges seen at fit time."""

    def fit(self, X):
        self.lo = X.min(axis=0)
        span = X.max(axis=0) - self.lo
        self.span = np.where(span > 0, span, 1.0)
        return self

    def __call__(self, X):
        return (X - self.lo) / self.span


class GPSurrogate(Surrogate):
    """GP regression on standardised yields with an ARD Matern kernel.

    Kernel: ``C * Matern(nu, ARD length scales) + White``. The white term absorbs noise
    that per-sample variances don't explain (outliers, batch effects). With
    ``use_noise=True``, per-sample ``noise_var`` (triplicate variance of the mean) is added
    to the diagonal as known heteroscedastic noise.

    ``input_transform`` maps model inputs to the unit cube; the benchmark passes the design
    space's own transform (log scale for log-spaced factors). Without one, columns are
    min-max scaled on the training data.

    ``predict`` returns the latent-yield std, i.e. without measurement noise, which is the
    quantity UCB should explore on. ``noise_std_`` holds the fitted white-noise std in
    yield units for anyone who needs predictive intervals for new measurements.
    """

    name = 'gp_matern'

    def __init__(self, nu=2.5, ard=True, n_restarts=2, use_noise=True, input_transform=None,
                 length_scale_bounds=(1e-2, 1e2), noise_bounds=(1e-6, 1.0), random_state=0):
        self.nu = nu
        self.ard = ard
        self.n_restarts = n_restarts
        self.use_noise = use_noise
        self.input_transform = input_transform
        self.length_scale_bounds = length_scale_bounds
        self.noise_bounds = noise_bounds
        self.random_state = random_state

    @property
    def accepts_noise(self):
        return self.use_noise

    def _transform(self, X):
        return self.transform_(as_2d_float(X))

    def fit(self, X, y, noise_var=None, kernel=None):
        X, y = as_2d_float(X), as_1d_float(y)
        self.transform_ = self.input_transform or MinMaxTransform().fit(X)
        Z = self.transform_(X)
        self.y_mean_ = float(y.mean())
        self.y_scale_ = float(y.std()) or 1.0
        ys = (y - self.y_mean_) / self.y_scale_

        alpha = np.full(len(y), 1e-8)
        if self.use_noise and noise_var is not None:
            alpha = alpha + np.asarray(noise_var, dtype=float).reshape(-1) / self.y_scale_ ** 2
        optimizer = 'fmin_l_bfgs_b'
        if kernel is None:
            length_scale = np.ones(Z.shape[1]) if self.ard else 1.0
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * Matern(length_scale=length_scale, length_scale_bounds=self.length_scale_bounds, nu=self.nu)
                      + WhiteKernel(1e-2, self.noise_bounds))
        else:
            optimizer = None
        random_state = self.random_state
        if isinstance(random_state, np.random.Generator):
            random_state = int(random_state.integers(2 ** 31 - 1))
        self.gp_ = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=False, optimizer=optimizer,
                                            n_restarts_optimizer=self.n_restarts if optimizer else 0,
                                            random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            self.gp_.fit(Z, ys)
        self.white_ = float(self.gp_.kernel_.k2.noise_level)
        self.noise_std_ = np.sqrt(self.white_) * self.y_scale_
        return self

    def refit(self, X, y, noise_var=None):
        """Condition on new data with the already fitted kernel hyperparameters."""
        return self.fit(X, y, noise_var=noise_var, kernel=self.gp_.kernel_)

    def predict(self, X):
        Z = self._transform(X)
        mean, std = self.gp_.predict(Z, return_std=True)
        latent_var = np.maximum(std ** 2 - self.white_, 1e-12)
        return Prediction(mean=mean * self.y_scale_ + self.y_mean_, std=np.sqrt(latent_var) * self.y_scale_)

    def posterior_covariance(self, XA, XB):
        """Latent posterior covariance between two input sets, in yield units squared."""
        ZA, ZB = self._transform(XA), self._transform(XB)
        signal = self.gp_.kernel_.k1
        X_train = self.gp_.X_train_
        VA = solve_triangular(self.gp_.L_, signal(X_train, ZA), lower=True)
        VB = solve_triangular(self.gp_.L_, signal(X_train, ZB), lower=True)
        return (signal(ZA, ZB) - VA.T @ VB) * self.y_scale_ ** 2

    def length_scales(self):
        return np.atleast_1d(self.gp_.kernel_.k1.k2.length_scale)

    def describe(self):
        return {'name': self.name, 'nu': self.nu, 'ard': self.ard, 'n_restarts': self.n_restarts,
                'use_noise': self.use_noise}
