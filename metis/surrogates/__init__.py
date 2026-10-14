"""Pluggable surrogate models. Each implements ``fit(X, y, noise_var=None)`` and ``predict(X)``."""
from .base import Prediction, Surrogate, inverse_variance_weights
from .ensemble import XGB_PARAM_GRID_V1, RegressorEnsemble, XGBEnsemble, default_xgb_ensemble
from .gp import GPSurrogate, MinMaxTransform
from .tabpfn import TabPFNSurrogate, tabpfn_available

__all__ = ['Prediction', 'Surrogate', 'RegressorEnsemble', 'XGBEnsemble', 'GPSurrogate', 'TabPFNSurrogate',
           'MinMaxTransform', 'XGB_PARAM_GRID_V1', 'default_xgb_ensemble', 'inverse_variance_weights',
           'tabpfn_available']
