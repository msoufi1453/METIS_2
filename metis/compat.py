"""The complete v1 ``utils`` namespace.

``utils.py`` at the repository root re-exports this module, so notebooks that do
``from utils import *`` keep getting the same names, including the modules v1 leaked
(``np``, ``pd``, ``os``, ``product``, ``Iterable``).
"""
import os  # noqa: F401
from collections.abc import Iterable  # noqa: F401
from itertools import product  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

from .design import (allowed_output, check_repetitive, concentration_to_volume, find_stock,  # noqa: F401
                     percentage_possible, random_combination_generator)
from .echo import (put_volumes_to_96_wells, put_volumes_to_384_wells, put_volumes_to_wells,  # noqa: F401
                   source_to_destination)
from .io import day_finder, result_preprocess  # noqa: F401
from .optimize import batch_ucb, bayesian_optimization  # noqa: F401
