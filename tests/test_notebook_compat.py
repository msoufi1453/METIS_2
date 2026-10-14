"""Every call that a repository notebook makes into ``utils`` still binds to the shim's signature.

Notebooks run on Colab and can't execute here (``!wget``, uploads), so this checks the
static contract: each ``utils`` function they call exists and accepts the positional and
keyword arguments they pass. Functions a notebook redefines locally are skipped.
"""
import ast
import inspect
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import utils  # noqa: E402

UTILS_FUNCS = {n for n, f in vars(utils).items() if inspect.isfunction(f)}


def _notebooks():
    for path in sorted(ROOT.rglob('*.ipynb')):
        cells = json.loads(path.read_text(encoding='utf-8'))['cells']
        code = [''.join(c['source']) for c in cells if c['cell_type'] == 'code']
        if any('from utils import' in c for c in code):
            yield path, code


def _parse(cell):
    lines = [l for l in cell.splitlines() if not l.lstrip().startswith(('!', '%'))]
    try:
        return ast.parse('\n'.join(lines))
    except SyntaxError:
        return None


NOTEBOOKS = list(_notebooks())


def test_notebooks_found():
    assert len(NOTEBOOKS) >= 10


@pytest.mark.parametrize('path,code', NOTEBOOKS, ids=[str(p.relative_to(ROOT)) for p, _ in NOTEBOOKS])
def test_notebook_calls_bind(path, code):
    trees = [t for t in (_parse(c) for c in code) if t is not None]
    redefined = {n.name for t in trees for n in ast.walk(t) if isinstance(n, ast.FunctionDef)}
    problems, checked = [], 0
    for tree in trees:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            name = node.func.id
            if name not in UTILS_FUNCS or name in redefined:
                continue
            if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
                continue
            try:
                inspect.signature(getattr(utils, name)).bind(*([None] * len(node.args)),
                                                             **{k.arg: None for k in node.keywords})
                checked += 1
            except TypeError as exc:
                problems.append('{} line {}: {}'.format(name, node.lineno, exc))
    assert not problems, problems
    assert checked > 0
