"""The full validation run. Slow (~2-3 min); opt in with --run-slow."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.mark.slow
def test_validation_passes(tmp_path, capsys):
    import validate_contract
    rc = validate_contract.main(["--db", str(tmp_path / "ledger.db")])
    out = capsys.readouterr().out
    assert rc == 0, out[-2000:]
    assert "nulls promoted             : 0/10" in out
    assert "known positives replicated : 7/9" in out
    assert "unconfirmed candidate TIC 220481009 : replicated" in out
