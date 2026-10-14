"""METIS: machine-learning guided experimental trials for improvement of systems.

The v1 notebook API (``bayesian_optimization``, ``random_combination_generator``, ...)
is importable from here and from the ``utils.py`` shim. The pluggable parts live in
``metis.surrogates``, ``metis.acquisition``, ``metis.space``, ``metis.initialization``
and ``metis.schedules``.
"""
from .compat import *  # noqa: F401,F403
from .compat import (allowed_output, batch_ucb, bayesian_optimization, check_repetitive, concentration_to_volume,
                     day_finder, find_stock, percentage_possible, put_volumes_to_96_wells, put_volumes_to_384_wells,
                     put_volumes_to_wells, random_combination_generator, result_preprocess, source_to_destination)
from .design import feature_columns
from .optimize import suggest, suggest_combinations
from .space import DesignSpace, Factor

__version__ = '0.2.0'
