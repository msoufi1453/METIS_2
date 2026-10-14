"""Synthetic yield landscapes with exactly known maxima on the droplet grid."""
from .functions import Component, SyntheticFunction, Term, component_max
from .shapes import FAMILIES, Shape, sample_shape
from .spaces import make_space
from .suite import DIMENSIONS, LANDSCAPES, SUBTYPES, FunctionSpec, build_function
