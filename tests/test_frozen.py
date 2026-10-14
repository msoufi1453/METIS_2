"""Frozen metric and test definitions must not change silently (docs/PREREGISTRATION.md)."""
from metis_bench.freeze import current_hashes, recorded_hashes


def test_frozen_definitions_unchanged():
    assert current_hashes() == recorded_hashes(), (
        'metrics.py or stats.py changed after the preregistration freeze. Add an amendment to '
        'docs/PREREGISTRATION.md and update metis_bench/FROZEN.json deliberately.')
