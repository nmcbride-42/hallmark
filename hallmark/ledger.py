"""Candidate ledger: states, transitions, and reproducibility bundles.

The state machine:

    proposal -> computed_candidate -> replicated -> externally_confirmed
                     |                    |               |
                     +--------------------+---------------+--> rejected (terminal)

Every transition is appended to `transitions` with the clause verdicts that
authorized it. Nothing mutates in place: the audit trail IS the record. A
candidate's current state is the last transition's `to_state`.

Enforced in code: a candidate id can be declared exactly once (so a rejected
candidate cannot be reopened under a different verifier), only the edges in
EDGES are legal, and `externally_confirmed` can only be written through
`record_external_acceptance()`, which requires an acceptance record from the
outside body. Hallmark's own contract code never calls it. Anyone with write
access to the SQLite file can still edit rows by hand; the ledger is an audit
log, not a tamper-proof store.
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from . import config

PROPOSAL = "proposal"
COMPUTED_CANDIDATE = "computed_candidate"
REPLICATED = "replicated"
EXTERNALLY_CONFIRMED = "externally_confirmed"
REJECTED = "rejected"

STATES = [PROPOSAL, COMPUTED_CANDIDATE, REPLICATED, EXTERNALLY_CONFIRMED, REJECTED]

# Legal edges. Anything else raises — a state machine that permits arbitrary
# jumps is not a state machine.
EDGES = {
    PROPOSAL: {COMPUTED_CANDIDATE, REJECTED},
    COMPUTED_CANDIDATE: {REPLICATED, REJECTED},
    REPLICATED: {EXTERNALLY_CONFIRMED, REJECTED},
    EXTERNALLY_CONFIRMED: set(),
    REJECTED: set(),
}


class IllegalTransition(RuntimeError):
    pass


def connect(path=None):
    conn = sqlite3.connect(str(path or config.db_path()), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init(conn)
    return conn


def _init(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS candidates (
        cid TEXT PRIMARY KEY,              -- stable id, e.g. 'tic-220481009'
        kind TEXT NOT NULL,                -- 'eclipsing_binary' | 'model_fit'
        target TEXT,                       -- catalogue-claim subject, if any
        dataset_key TEXT,                  -- frozen dataset it computes over
        dataset_sha256 TEXT,
        verifier_name TEXT,                -- PREDECLARED before compute
        verifier_spec_sha256 TEXT,         -- hash of the frozen verifier spec
        declared_at TEXT NOT NULL,
        suite TEXT,                        -- provenance only, never an input
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS transitions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cid TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        clauses TEXT NOT NULL,             -- json: {clause: [passed, detail]}
        reason TEXT,
        at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (cid) REFERENCES candidates(cid)
    );
    CREATE TABLE IF NOT EXISTS bundles (
        cid TEXT PRIMARY KEY,
        bundle TEXT NOT NULL,              -- json reproducibility bundle
        bundle_sha256 TEXT NOT NULL,
        at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_trans_cid ON transitions(cid);
    """)
    conn.commit()


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def declare(conn, cid, kind, dataset_key, dataset_sha256,
            verifier_name, verifier_spec_sha256, target=None, suite=None):
    """Open a candidate at `proposal`. The verifier is named and hashed HERE —
    before any result is computed. That ordering is contract clause 4, and it
    is why the verifier cannot be chosen to fit an answer.

    A candidate id is declared once. Re-declaring an existing id (for example
    a rejected candidate under a different verifier) raises IllegalTransition:
    a new attempt needs a new id, and the old one stays in the audit trail."""
    declared_at = now()
    try:
        conn.execute("""
            INSERT INTO candidates
            (cid, kind, target, dataset_key, dataset_sha256, verifier_name,
             verifier_spec_sha256, declared_at, suite)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (cid, kind, target, dataset_key, dataset_sha256, verifier_name,
              verifier_spec_sha256, declared_at, suite))
    except sqlite3.IntegrityError:
        conn.rollback()
        raise IllegalTransition(
            f"{cid}: already declared (state {state(conn, cid)!r}); "
            "a candidate cannot be re-declared, use a new id") from None
    clauses = {"verifier_name": verifier_name,
               "verifier_spec_sha256": verifier_spec_sha256,
               "dataset_sha256": dataset_sha256,
               "declared_at": declared_at}
    conn.execute("""
        INSERT INTO transitions (cid, from_state, to_state, clauses, reason)
        VALUES (?, NULL, ?, ?, 'declared')
    """, (cid, PROPOSAL, json.dumps(clauses, sort_keys=True)))
    conn.commit()


def state(conn, cid):
    row = conn.execute(
        "SELECT to_state FROM transitions WHERE cid=? ORDER BY id DESC LIMIT 1",
        (cid,)).fetchone()
    return row["to_state"] if row else None


def transition(conn, cid, to_state, clauses, reason=""):
    """Append a transition. Raises on an illegal edge.

    `externally_confirmed` is not reachable from here: use
    record_external_acceptance(), which requires the outside body's record."""
    if to_state == EXTERNALLY_CONFIRMED:
        raise IllegalTransition(
            f"{cid}: {EXTERNALLY_CONFIRMED} is written only by "
            "record_external_acceptance() with an external acceptance record")
    return _append(conn, cid, to_state, clauses, reason)


def record_external_acceptance(conn, cid, acceptance):
    """replicated -> externally_confirmed, on an outside body's acceptance.

    `acceptance` must name the external `body` (e.g. 'AAVSO VSX'), the
    `reference` it issued (e.g. a catalogue identifier or URL), and the
    `accepted_at` date. Hallmark's own pipeline never calls this; a person
    records it after the external body has accepted the submission."""
    required = ("body", "reference", "accepted_at")
    missing = [k for k in required
               if not isinstance((acceptance or {}).get(k), str)
               or not acceptance[k].strip()]
    if missing:
        raise IllegalTransition(
            f"{cid}: external acceptance record missing {missing}")
    return _append(conn, cid, EXTERNALLY_CONFIRMED,
                   {"external_acceptance": dict(acceptance)},
                   f"accepted by {acceptance['body']}")


def _append(conn, cid, to_state, clauses, reason):
    cur = state(conn, cid)
    if cur is None:
        raise IllegalTransition(f"{cid}: no prior state — declare() first")
    if to_state not in EDGES.get(cur, set()):
        raise IllegalTransition(f"{cid}: {cur} -> {to_state} is not a legal edge")
    conn.execute("""
        INSERT INTO transitions (cid, from_state, to_state, clauses, reason)
        VALUES (?,?,?,?,?)
    """, (cid, cur, to_state, json.dumps(clauses, sort_keys=True), reason))
    conn.commit()
    return to_state


def write_bundle(conn, cid, bundle: dict):
    """Contract clause 5 — data hash, code commit, parameters, outputs, and
    the falsification attempts that failed to kill it."""
    blob = json.dumps(bundle, sort_keys=True, indent=2)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    conn.execute("INSERT OR REPLACE INTO bundles (cid, bundle, bundle_sha256) "
                 "VALUES (?,?,?)", (cid, blob, digest))
    conn.commit()
    return digest


def history(conn, cid):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM transitions WHERE cid=? ORDER BY id", (cid,))]
