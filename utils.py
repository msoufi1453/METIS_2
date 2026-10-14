"""Compatibility shim: the METIS v1 helper module.

Everything moved into the ``metis`` package; this file re-exports the v1 names so
``from utils import *`` in the notebooks keeps working unchanged. New code should
import from ``metis`` directly.
"""
from metis.compat import *  # noqa: F401,F403
