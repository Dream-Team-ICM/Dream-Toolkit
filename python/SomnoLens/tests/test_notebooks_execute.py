"""Quick level: every Voila notebook executes from top to bottom, without clicking anything. It catches what breaks a
notebook before any button is pressed: a syntax error, a broken import (a missing package, a shared module not found),
a misspelt name, a widget built wrongly.

The Jupyter (code-visible) twins are not run: they hold paths to edit and start working as they execute. The Curry twins
are gone (phase 6 of tools/plan_brainvision.md): the tools read Curry themselves."""
from pathlib import Path

import pytest

from nbdriver import NotebookSession

pytestmark = pytest.mark.quick

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(p.relative_to(REPO).as_posix() for p in (REPO / 'tools').glob('*_voila.ipynb'))


def test_every_voila_notebook_is_listed():
    """Guard: the glob found the toolkit (an empty list would make this level pass silently)."""
    assert len(NOTEBOOKS) >= 13


@pytest.mark.parametrize('notebook', NOTEBOOKS)
def test_notebook_executes(notebook):
    with NotebookSession(notebook) as nb:            # raises on any cell error, with the cell's message
        text = '\n'.join(nb.log)
    assert 'Traceback (most recent call last)' not in text, text[-3000:]
    assert 'Import error' not in text, text[-3000:]     # the tools print this instead of raising
