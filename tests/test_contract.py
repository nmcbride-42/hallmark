"""The contract and ledger: fail-closed by construction, not by convention."""

import json

import pytest

from hallmark import contract, ledger
from hallmark.verifier import Verdict, VerificationHandle
from hallmark.verifiers.ephemeris import HeldOutEphemeris


def _declare(conn, cid="c1", handle=None, sha="0" * 64, key="128193913_s92"):
    handle = handle or HeldOutEphemeris()
    ledger.declare(conn, cid, kind="eclipsing_binary", dataset_key=key,
                   dataset_sha256=sha, verifier_name=handle.name,
                   verifier_spec_sha256=handle.spec_sha256)
    return handle


# ── Verdict: graded judgment cannot enter ──────────────────────────────────

def test_verdict_rejects_non_bool():
    with pytest.raises(TypeError):
        Verdict(passed=0.93, detail="looks good", spec_sha256="x")


def test_indeterminate_cannot_pass():
    with pytest.raises(ValueError):
        Verdict(passed=True, detail="?", spec_sha256="x", indeterminate=True)


def test_clause6_raises_on_non_bool_clause():
    with pytest.raises(contract.ContractViolation):
        contract.clause6_mechanical_promotion({"x": ("probably", "llm says yes")})


# ── Spec hashing: any parameter change is a different verifier ─────────────

def test_spec_hash_changes_with_any_parameter():
    assert HeldOutEphemeris().spec_sha256 == HeldOutEphemeris().spec_sha256
    assert HeldOutEphemeris().spec_sha256 != HeldOutEphemeris(n_sigma=3.9).spec_sha256


def test_verifier_spec_drift_is_rejected(ledger_conn):
    declared = _declare(ledger_conn)
    loosened = HeldOutEphemeris(tolerance_frac=0.2)
    v = Verdict(True, "passed under a looser spec", loosened.spec_sha256)
    ok, detail = contract.clause4_predeclared_verifier(ledger_conn, "c1", v)
    assert ok is False and "drift" in detail
    assert declared.spec_sha256 != loosened.spec_sha256


def test_undeclared_verifier_fails_clause4(ledger_conn):
    ok, _ = contract.clause4_predeclared_verifier(ledger_conn, "never-declared")
    assert ok is False


# ── Clause 1: frozen data, pinned by hash ──────────────────────────────────

def test_clause1_verifies_shipped_hash(frozen):
    idx = json.loads((frozen / "index.json").read_text())
    key = "128193913_s92"
    ok, detail = contract.clause1_frozen_dataset(key, idx[key]["sha256"])
    assert ok, detail


def test_clause1_detects_hash_drift():
    ok, detail = contract.clause1_frozen_dataset("128193913_s92", "f" * 64)
    assert ok is False and "drift" in detail


def test_clause1_missing_dataset_fails():
    ok, _ = contract.clause1_frozen_dataset("does_not_exist", "0" * 64)
    assert ok is False


def test_every_shipped_dataset_matches_its_index(frozen):
    import numpy as np
    from hallmark.hashing import dataset_sha256
    idx = json.loads((frozen / "index.json").read_text())
    for key, meta in idx.items():
        d = np.load(frozen / f"{key}.npz")
        keys = ("phase", "flux") if "phase" in d.files else ("time", "flux")
        assert dataset_sha256(**{k: d[k] for k in keys}) == meta["sha256"], key


# ── Clause 2: the code that computes the result exists ─────────────────────

def test_clause2_resolves_real_module():
    ok, detail = contract.clause2_executable_code(
        "hallmark.verifiers.ephemeris:HeldOutEphemeris")
    assert ok, detail


def test_clause2_rejects_missing_module():
    assert contract.clause2_executable_code("hallmark.nope:X")[0] is False
    assert contract.clause2_executable_code("")[0] is False


# ── Clause 3: novelty needs a named catalogue and a real query ─────────────

def test_clause3_no_catalogue_shape_stops_at_replicated():
    ok, detail = contract.clause3_novelty_boundary(None)
    assert ok is False and detail.startswith("n/a")


def test_clause3_failed_query_is_not_absence():
    # A network error yields a record with no 'hits' key: incomplete, not novel.
    rec = {"catalogue": "AAVSO VSX", "queried_at": "t", "cone_arcsec": 60,
           "error": "URLError"}
    ok, detail = contract.clause3_novelty_boundary(rec)
    assert ok is False and "incomplete" in detail


def test_clause3_catalogued_object_is_rejected():
    rec = {"catalogue": "AAVSO VSX", "queried_at": "t", "cone_arcsec": 60, "hits": 1}
    assert contract.clause3_novelty_boundary(rec)[0] is False


def test_external_confirmation_is_never_self_awarded(ledger_conn):
    rec = {"catalogue": "AAVSO VSX", "queried_at": "t", "cone_arcsec": 60, "hits": 0}
    outcome, _ = contract.to_externally_confirmed(ledger_conn, "c1", rec, False)
    assert outcome == "awaiting_human_gate"
    outcome, _ = contract.to_externally_confirmed(ledger_conn, "c1", rec, True)
    assert outcome == "cleared_for_deposition"     # not "externally_confirmed"


# ── Ledger: a state machine, append-only ───────────────────────────────────

def test_illegal_transition_raises(ledger_conn):
    _declare(ledger_conn)
    with pytest.raises(ledger.IllegalTransition):
        ledger.transition(ledger_conn, "c1", ledger.REPLICATED, {})


def test_rejected_is_terminal(ledger_conn):
    _declare(ledger_conn)
    ledger.transition(ledger_conn, "c1", ledger.REJECTED, {})
    with pytest.raises(ledger.IllegalTransition):
        ledger.transition(ledger_conn, "c1", ledger.COMPUTED_CANDIDATE, {})


def test_rejected_candidate_cannot_be_redeclared(ledger_conn):
    """A rejected candidate cannot be reopened under a different verifier:
    the second declare() raises, and the original declaration survives."""
    first = _declare(ledger_conn)
    ledger.transition(ledger_conn, "c1", ledger.COMPUTED_CANDIDATE, {})
    ledger.transition(ledger_conn, "c1", ledger.REJECTED, {})
    with pytest.raises(ledger.IllegalTransition):
        _declare(ledger_conn, handle=HeldOutEphemeris(n_sigma=3.0))
    assert ledger.state(ledger_conn, "c1") == ledger.REJECTED
    row = ledger_conn.execute("SELECT verifier_spec_sha256 FROM candidates "
                              "WHERE cid='c1'").fetchone()
    assert row["verifier_spec_sha256"] == first.spec_sha256
    declared = ledger.history(ledger_conn, "c1")[0]
    assert json.loads(declared["clauses"])["verifier_spec_sha256"] == first.spec_sha256
    assert len(ledger.history(ledger_conn, "c1")) == 3


def test_externally_confirmed_needs_an_acceptance_record(ledger_conn):
    _declare(ledger_conn)
    ledger.transition(ledger_conn, "c1", ledger.COMPUTED_CANDIDATE, {})
    ledger.transition(ledger_conn, "c1", ledger.REPLICATED, {})
    with pytest.raises(ledger.IllegalTransition):
        ledger.transition(ledger_conn, "c1", ledger.EXTERNALLY_CONFIRMED, {})
    with pytest.raises(ledger.IllegalTransition):
        ledger.record_external_acceptance(ledger_conn, "c1", {"body": "AAVSO VSX"})
    assert ledger.state(ledger_conn, "c1") == ledger.REPLICATED
    ledger.record_external_acceptance(
        ledger_conn, "c1", {"body": "AAVSO VSX", "reference": "example-id",
                            "accepted_at": "2026-01-01"})
    assert ledger.state(ledger_conn, "c1") == ledger.EXTERNALLY_CONFIRMED


def test_transition_without_declare_raises(ledger_conn):
    with pytest.raises(ledger.IllegalTransition):
        ledger.transition(ledger_conn, "ghost", ledger.COMPUTED_CANDIDATE, {})


def test_history_is_append_only_audit(ledger_conn):
    _declare(ledger_conn)
    ledger.transition(ledger_conn, "c1", ledger.COMPUTED_CANDIDATE, {"a": [True, "x"]})
    h = ledger.history(ledger_conn, "c1")
    assert [r["to_state"] for r in h] == [ledger.PROPOSAL, ledger.COMPUTED_CANDIDATE]


# ── End to end on one real positive ────────────────────────────────────────

def test_known_positive_walks_to_replicated(ledger_conn, frozen):
    idx = json.loads((frozen / "index.json").read_text())
    key = "14069011_s92"
    h = _declare(ledger_conn, cid="pos", sha=idx[key]["sha256"], key=key)
    st, c = contract.to_computed_candidate(
        ledger_conn, "pos", key, idx[key]["sha256"],
        "hallmark.verifiers.ephemeris:HeldOutEphemeris")
    assert st == ledger.COMPUTED_CANDIDATE, c
    v = h.verify(frozen / f"{key}.npz")
    bundle = {"dataset_sha256": idx[key]["sha256"], "code_ref": "x",
              "parameters": h.spec, "outputs": {},
              "falsification_attempts": [{"name": "controls", "result": "ok"}]}
    st, c = contract.to_replicated(ledger_conn, "pos", v, bundle)
    assert st == ledger.REPLICATED, c


def test_bundle_without_falsification_attempts_rejects(ledger_conn, frozen):
    idx = json.loads((frozen / "index.json").read_text())
    key = "14069011_s92"
    h = _declare(ledger_conn, cid="pos", sha=idx[key]["sha256"], key=key)
    contract.to_computed_candidate(ledger_conn, "pos", key, idx[key]["sha256"],
                                   "hallmark.verifiers.ephemeris:HeldOutEphemeris")
    v = h.verify(frozen / f"{key}.npz")
    bundle = {"dataset_sha256": "x", "code_ref": "x", "parameters": {},
              "outputs": {}, "falsification_attempts": []}
    st, _ = contract.to_replicated(ledger_conn, "pos", v, bundle)
    assert st == ledger.REJECTED


class _Graded(VerificationHandle):
    name = "graded"

    def _run(self, path):
        return {}

    def _decide(self, stats):
        return Verdict(0.9, "an opinion", self.spec_sha256)


def test_a_verifier_that_returns_a_score_cannot_construct_a_verdict():
    with pytest.raises(TypeError):
        _Graded().verify("unused")
