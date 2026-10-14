"""Hashes of the frozen benchmark definitions (docs/PREREGISTRATION.md).

``python -m metis_bench.freeze`` prints the current hashes. Updating FROZEN.json is a
deliberate act that must come with an amendment in the preregistration.
"""
import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
FROZEN_FILES = ('metrics.py', 'stats.py')


def current_hashes():
    # Normalise line endings so Windows checkouts (CRLF) hash like everyone else's.
    return {name: hashlib.sha256((HERE / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in FROZEN_FILES}


def recorded_hashes():
    return json.loads((HERE / 'FROZEN.json').read_text())['sha256']


if __name__ == '__main__':
    print(json.dumps(current_hashes(), indent=1))
