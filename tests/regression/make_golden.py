"""Regenerate the golden files from the frozen original ``utils`` (v1).

Only run this when the pinned environment changes on purpose. Then commit the new
golden files together with the uv.lock change that caused them.

    uv run python tests/regression/make_golden.py
"""
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scenario import run_scenario  # noqa: E402


def load_original():
    spec = importlib.util.spec_from_file_location('utils_v1_original', HERE / 'utils_v1_original.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    golden = HERE / 'golden'
    golden.mkdir(exist_ok=True)
    for old in golden.glob('*.txt'):
        old.unlink()
    outputs = run_scenario(load_original())
    for name, text in outputs.items():
        (golden / '{}.txt'.format(name)).write_text(text, encoding='utf-8', newline='\n')
    print('wrote {} golden files to {}'.format(len(outputs), golden))


if __name__ == '__main__':
    main()
