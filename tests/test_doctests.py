"""Run the doctests in the package's own docstrings.

The examples in ``fastjsd/_api.py`` print numeric output, and nothing else in
the suite executes them: ``pyproject.toml`` sets ``testpaths = ["tests"]``
without ``--doctest-modules``. A worked example that prints a different number
than the code returns is exactly the kind of error a reader trusts, so it is
checked here rather than left to review.
"""

import doctest
import importlib

import pytest

MODULES = ["fastjsd", "fastjsd._api", "fastjsd._common"]


@pytest.mark.parametrize("name", MODULES)
def test_module_doctests(name):
    mod = importlib.import_module(name)
    result = doctest.testmod(mod, verbose=False, report=True)
    assert result.failed == 0, (
        f"{name}: {result.failed} of {result.attempted} doctests failed"
    )
