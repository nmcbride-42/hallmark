import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FROZEN = ROOT / "data" / "frozen"


def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true",
                     help="run tests that call public astronomy APIs")
    parser.addoption("--run-slow", action="store_true",
                     help="run the full end-to-end validation (~2-3 min)")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "network" in item.keywords and not config.getoption("--run-network"):
            item.add_marker(pytest.mark.skip(reason="network test; use --run-network"))
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(pytest.mark.skip(reason="slow test; use --run-slow"))


@pytest.fixture
def frozen():
    return FROZEN


@pytest.fixture
def ledger_conn(tmp_path):
    from hallmark import ledger
    conn = ledger.connect(tmp_path / "ledger.db")
    yield conn
    conn.close()
