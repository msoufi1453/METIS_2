"""Anti-breakage guard: one seeded METIS round must match the v1 output byte for byte.

Two checks:

* ``test_live_utils_matches_golden``: the refactored package, reached through the
  ``utils.py`` shim the notebooks import, reproduces the golden files.
* ``test_frozen_original_matches_golden``: the frozen v1 module still reproduces them.
  If only this one fails, the environment drifted (see pyproject.toml / uv.lock), not
  the code.
"""
import importlib.util
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REGRESSION = HERE / 'regression'
GOLDEN = REGRESSION / 'golden'
sys.path.insert(0, str(REGRESSION))
sys.path.insert(0, str(HERE.parent))

from scenario import run_scenario  # noqa: E402


def _golden():
    return {p.stem: p.read_bytes().decode('utf-8') for p in sorted(GOLDEN.glob('*.txt'))}


def _assert_matches(outputs):
    golden = _golden()
    assert set(outputs) == set(golden), 'scenario outputs changed: {}'.format(set(outputs) ^ set(golden))
    mismatched = [name for name in golden if outputs[name] != golden[name]]
    assert not mismatched, 'outputs differ from v1 golden files: {}'.format(mismatched)


@pytest.fixture(scope='module')
def live_outputs():
    import utils
    return run_scenario(utils)


def test_live_utils_matches_golden(live_outputs):
    _assert_matches(live_outputs)


def test_frozen_original_matches_golden():
    spec = importlib.util.spec_from_file_location('utils_v1_original', REGRESSION / 'utils_v1_original.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _assert_matches(run_scenario(module))


def test_shim_exports_v1_names():
    """Every public name the v1 module exported via ``from utils import *`` still exists."""
    import utils
    spec = importlib.util.spec_from_file_location('utils_v1_names', REGRESSION / 'utils_v1_original.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    v1_names = {n for n in dir(original) if not n.startswith('_')}
    missing = sorted(v1_names - set(dir(utils)))
    assert not missing, missing
