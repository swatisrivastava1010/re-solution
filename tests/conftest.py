import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

E2E_ENV = "RESOLUTION_E2E"


def pytest_configure(config):
    # Registered here rather than in pyproject.toml, next to the rule that keeps them opt-in.
    config.addinivalue_line(
        "markers",
        f"e2e: drives a real browser against a real server; opt-in with -m e2e or {E2E_ENV}=1 (needs Google Chrome)",
    )


def pytest_collection_modifyitems(config, items):
    """End-to-end browser tests take minutes and need Google Chrome, so they only run when asked for."""
    asked = "e2e" in (config.getoption("markexpr", "") or "") or os.environ.get(E2E_ENV) == "1"
    if asked:
        return
    skip = pytest.mark.skip(reason=f"end-to-end browser test: run with -m e2e or {E2E_ENV}=1")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)
