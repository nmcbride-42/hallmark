"""The discovery contract: six clauses, mapped to three transitions.

Conjunctive and fail-closed. **No LLM judgment anywhere in the
promotion path**: every function here is deterministic Python over frozen
inputs, and `promote()` raises if handed anything that is not a bool.

Clause -> transition mapping:

  proposal -> computed_candidate     : 1, 2 hold; 4 DECLARED
  computed_candidate -> replicated   : 4 PASSES; 5 assembled; 6 mechanical
  replicated -> externally_confirmed : 3 (novelty, re-queried); human gate

A result with no catalogue-claim shape reaches `replicated` and stops. That is
the mapping working, not an exception to it.
"""

import hashlib
import importlib.util
from pathlib import Path

from . import config, ledger


class ContractViolation(RuntimeError):
    pass


# ── Clause predicates. Each returns (passed: bool, detail: str). ────────────

def clause1_frozen_dataset(dataset_key, expected_sha256, path=None):
    """A frozen observational dataset, pinned by hash and access date.

    `path` defaults to `$HALLMARK_DATA_DIR/frozen/<dataset_key>.npz`."""
    import numpy as np
    from .hashing import dataset_sha256
    path = Path(path) if path else config.frozen_dir() / f"{dataset_key}.npz"
    if not path.exists():
        return False, f"dataset {dataset_key} absent"
    d = np.load(path)
    keys = ("phase", "flux") if "phase" in d.files else ("time", "flux")
    got = dataset_sha256(**{k: d[k] for k in keys})
    if got != expected_sha256:
        return False, f"hash drift: {got[:12]} != {expected_sha256[:12]}"
    return True, f"sha256 {got[:12]} verified, {len(d[keys[0]])} samples"


def clause2_executable_code(code_ref):
    """Executable code that computes the result from that dataset."""
    if not code_ref:
        return False, "no code reference"
    mod, _, fn = code_ref.partition(":")
    try:
        mspec = importlib.util.find_spec(mod)
    except (ImportError, ValueError):
        mspec = None
    if mspec is None or not mspec.origin or not Path(mspec.origin).exists():
        return False, f"code {mod} not on disk"
    src_hash = hashlib.sha256(Path(mspec.origin).read_bytes()).hexdigest()
    return True, f"{code_ref} @ {src_hash[:12]}"


def clause3_novelty_boundary(novelty):
    """A declared novelty boundary — absent from WHICH catalogue, queried WHEN,
    with what cone/tolerance. 'Novel' with no named catalogue is meaningless.

    Returns not-applicable as a FAILURE with a distinct detail so the caller
    can tell 'has no catalogue shape' (stop at replicated) from 'is catalogued'
    (rejected for novelty)."""
    if novelty is None:
        return False, "n/a: result has no catalogue-claim shape"
    required = {"catalogue", "queried_at", "cone_arcsec", "hits"}
    missing = required - set(novelty)
    if missing:
        return False, f"novelty boundary incomplete: missing {sorted(missing)}"
    if novelty["hits"] != 0:
        return False, (f"present in {novelty['catalogue']} "
                       f"({novelty['hits']} hit(s) within "
                       f"{novelty['cone_arcsec']}\")")
    return True, (f"absent from {novelty['catalogue']} within "
                  f"{novelty['cone_arcsec']}\" as of {novelty['queried_at']}")


def clause4_predeclared_verifier(conn, cid, verdict=None):
    """A predeclared verifier — chosen and written down BEFORE the result is
    computed. At proposal time we check it was declared; at replication time we
    check the SAME verifier spec produced the verdict."""
    row = conn.execute("SELECT verifier_name, verifier_spec_sha256, declared_at "
                       "FROM candidates WHERE cid=?", (cid,)).fetchone()
    if not row or not row["verifier_name"]:
        return False, "no verifier declared"
    if verdict is None:
        return True, (f"declared {row['verifier_name']} "
                      f"@ {row['verifier_spec_sha256'][:12]} at {row['declared_at']}")
    if verdict.spec_sha256 != row["verifier_spec_sha256"]:
        return False, (f"verifier spec drift: verdict from "
                       f"{verdict.spec_sha256[:12]}, declared "
                       f"{row['verifier_spec_sha256'][:12]}")
    if not verdict.passed:
        return False, f"verifier rejected: {verdict.detail}"
    return True, f"{row['verifier_name']} passed: {verdict.detail}"


def clause5_reproducibility_bundle(bundle):
    """Data hash, code commit, parameters, outputs, and the falsification
    attempts that failed to kill it."""
    required = {"dataset_sha256", "code_ref", "parameters", "outputs",
                "falsification_attempts"}
    missing = required - set(bundle or {})
    if missing:
        return False, f"bundle incomplete: missing {sorted(missing)}"
    if not bundle["falsification_attempts"]:
        return False, "no falsification attempts recorded"
    return True, (f"bundle complete, {len(bundle['falsification_attempts'])} "
                  f"falsification attempt(s) survived")


def clause6_mechanical_promotion(clause_results):
    """The pass/fail decision is made by code. No LLM judgment anywhere in the
    promotion path. If an LLM opinion is needed to decide, the candidate is not
    promotable by definition."""
    for name, (passed, _) in clause_results.items():
        if not isinstance(passed, bool):
            raise ContractViolation(
                f"clause {name} returned {type(passed).__name__}, not bool — "
                "a non-boolean verdict implies judgment, which the contract "
                "forbids in the promotion path")
    return True, f"{len(clause_results)} clause(s) decided mechanically by code"


# ── Transitions ────────────────────────────────────────────────────────────

def to_computed_candidate(conn, cid, dataset_key, dataset_sha256, code_ref,
                          ds_path=None):
    """proposal -> computed_candidate: clauses 1, 2 hold; 4 is declared."""
    c = {
        "1_frozen_dataset": clause1_frozen_dataset(dataset_key, dataset_sha256,
                                                   ds_path),
        "2_executable_code": clause2_executable_code(code_ref),
        "4_verifier_declared": clause4_predeclared_verifier(conn, cid),
    }
    c["6_mechanical"] = clause6_mechanical_promotion(c)
    ok = all(p for p, _ in c.values())
    target = ledger.COMPUTED_CANDIDATE if ok else ledger.REJECTED
    ledger.transition(conn, cid, target, c,
                      "clauses 1,2,4-declared" if ok else _why(c))
    return target, c


def to_replicated(conn, cid, verdict, bundle):
    """computed_candidate -> replicated: verifier PASSES, bundle assembled,
    decision mechanical."""
    c = {
        "4_verifier_passed": clause4_predeclared_verifier(conn, cid, verdict),
        "5_bundle": clause5_reproducibility_bundle(bundle),
    }
    c["6_mechanical"] = clause6_mechanical_promotion(c)
    ok = all(p for p, _ in c.values())
    if ok:
        ledger.write_bundle(conn, cid, bundle)
    target = ledger.REPLICATED if ok else ledger.REJECTED
    ledger.transition(conn, cid, target, c,
                      "verifier passed, bundle assembled" if ok else _why(c))
    return target, c


def to_externally_confirmed(conn, cid, novelty, human_gate_approved):
    """replicated -> externally_confirmed: clause 3 (re-queried), human gate,
    then the external acceptor's verdict.

    NOTE: this function can only ever record that the submission is CLEARED to
    go to a human for deposition. `externally_confirmed` is written when the
    external body accepts — never by us. Hallmark cannot strike its own
    hallmark."""
    c = {"3_novelty_boundary": clause3_novelty_boundary(novelty)}
    c["6_mechanical"] = clause6_mechanical_promotion(c)
    ok = all(p for p, _ in c.values())
    if not ok:
        return "blocked_at_replicated", c
    if not human_gate_approved:
        return "awaiting_human_gate", c
    return "cleared_for_deposition", c


def _why(clause_results):
    return "; ".join(f"{k}: {d}" for k, (p, d) in clause_results.items() if not p)
