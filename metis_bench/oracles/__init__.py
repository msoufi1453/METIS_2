"""Benchmark problems: synthetic functions and gold-regressor oracles from real data."""
from .base import Oracle, PooledSpace, SyntheticOracle
from .real import REAL_ORACLES, GoldRegressorOracle, build_borkowski, build_pure, load_borkowski, load_pure
