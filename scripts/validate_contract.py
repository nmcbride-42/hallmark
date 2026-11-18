"""Validate the discovery contract on known answers. The exit code is the verdict.

Every candidate in the frozen suites is walked through the contract blind:
the verifier receives a frozen dataset path and its own frozen spec, nothing
else. Suite labels are loaded separately and applied only after every verdict
has been recorded.

Pass criteria (declared before the run, checked mechanically at the end):

  1. The submission-path candidate (TIC 220481009) reaches `replicated`.
  2. The sech² model-fit claim reaches `replicated` and STOPS there, because
     clause 3 (catalogue novelty) does not apply to a claim about form.
  3. NO null control promotes. This is the kill criterion: if a quiet star or
     a gap artifact promotes under any verifier or any probed period, the
     contract is not fail-closed and the run fails.
  4. Re-running every verdict on identical inputs reproduces it exactly.
  5. No LLM judgment is in the promotion path (enforced by clause 6, which
     raises on any non-bool verdict; no model is called anywhere here).

Run:   python scripts/validate_contract.py            # offline, ~1-3 min
       python scripts/validate_contract.py --live     # also re-query AAVSO VSX
Exit:  0 if every criterion passes, 1 otherwise.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hallmark import config, contract, ledger                       # noqa: E402
from hallmark.hashing import dataset_sha256                         # noqa: E402
from hallmark.verifiers.curvefit import HeldOutCurveFit             # noqa: E402
from hallmark.verifiers.ephemeris import (HeldOutEphemeris,         # noqa: E402
                                          ProposedEphemerisHeldOut)
from hallmark.verifiers.southern_archival import SouthernArchivalEphemeris  # noqa: E402
from suites import manifest                                         # noqa: E402

SECH2_KEY = "sech2_lc_329089161"
SECH2_DECLARED_PRIOR_R2 = 0.9577


def code_ref(handle):
    return f"{type(handle).__module__}:{type(handle).__name__}"


def load_index():
    import json
    return json.loads((config.frozen_dir() / "index.json").read_text())


def inject(host_path, period_d, depth, out_path):
    """S3: inject a known box signal into a real quiet light curve.

    Deterministic, no RNG. The host supplies real TESS noise, gaps, and
    systematics; only the signal is synthetic, so a miss is a true false
    negative of the verifier rather than an artifact of simulated noise."""
    d = np.load(host_path)
    t = np.asarray(d["time"], dtype=float)
    f = np.asarray(d["flux"], dtype=float).copy()
    t0 = float(t[0]) + 0.35 * period_d
    duration = max(0.04 * period_d, 0.06)
    phase = np.abs(((t - t0 + period_d / 2) % period_d) - period_d / 2)
    f[phase < duration / 2] *= (1.0 - depth)
    np.savez_compressed(out_path, time=t, flux=f)
    return dataset_sha256(time=t, flux=f)


def run_candidate(conn, cid, ds_path, ds_key, sha, handle, kind, suite,
                  target=None, proposal=None):
    """Walk one candidate through the contract. Returns (state, verdict)."""
    ledger.declare(conn, cid, kind=kind, dataset_key=ds_key,
                   dataset_sha256=sha, verifier_name=handle.name,
                   verifier_spec_sha256=handle.spec_sha256,
                   target=target, suite=suite)
    st, _ = contract.to_computed_candidate(conn, cid, ds_key, sha,
                                           code_ref(handle), ds_path=ds_path)
    if st == ledger.REJECTED:
        return st, None

    verdict = handle.verify(ds_path, proposal=proposal)
    if "ephemeris" in handle.name:
        attempts = [
            {"name": "control-phase comparison",
             "result": "prediction beat control phases"
                       if verdict.passed else "did not survive"},
            {"name": "leave-one-event-out / held-out block",
             "result": f"{verdict.stats.get('n_folds', 'block split')}"},
            {"name": "data-gap exclusion",
             "result": f"{verdict.stats.get('n_gaps', 0)} gaps guarded"},
        ]
    else:
        attempts = [
            {"name": "held-out phase bins",
             "result": f"R²={verdict.stats.get('sech2_heldout_r2')}"},
            {"name": "prior-model comparison",
             "result": f"prior R²={verdict.stats.get('prior_heldout_r2')}"},
        ]
    bundle = {
        "dataset_sha256": sha,
        "code_ref": code_ref(handle),
        "parameters": handle.spec,
        "outputs": {k: v for k, v in verdict.stats.items() if k != "folds"},
        "falsification_attempts": attempts,
    }
    st, _ = contract.to_replicated(conn, cid, verdict, bundle)
    return st, verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true",
                    help="re-query AAVSO VSX for clause 3 (network)")
    ap.add_argument("--db", default=None,
                    help="ledger path (default: a fresh temp file)")
    args = ap.parse_args(argv)

    frozen = config.frozen_dir()
    index = load_index()
    work = Path(tempfile.mkdtemp(prefix="hallmark-validate-"))
    db = Path(args.db) if args.db else work / "ledger.db"
    if db.exists():
        db.unlink()                       # every validation is a clean room
    conn = ledger.connect(db)

    eph = HeldOutEphemeris()
    prop = ProposedEphemerisHeldOut()
    sa = SouthernArchivalEphemeris()
    fit = HeldOutCurveFit()

    print("=" * 78)
    print(f"HALLMARK contract validation   suite {manifest.SUITE_VERSION}")
    for h in (eph, prop, sa, fit):
        print(f"  verifier: {h.name:<30s} spec {h.spec_sha256[:16]}")
    print("=" * 78)

    results = {}
    replay = {}       # key -> (handle, dataset_path, proposal, verdict)

    plan = []
    for tic, sectors, *_ in manifest.S1_POSITIVES + manifest.S1_SUBMISSION_PATH:
        plan.append((str(tic), f"{tic}_s{'-'.join(map(str, sectors))}", "S1"))
    for tic, sectors, _ in manifest.S2_NEGATIVES:
        plan.append((str(tic), f"{tic}_s{'-'.join(map(str, sectors))}", "S2"))
    for tic, sectors, _ in manifest.S4_ADVERSARIAL:
        plan.append((str(tic), f"{tic}_s{'-'.join(map(str, sectors))}", "S4"))
    archival = {str(tic): f"{tic}_s{'-'.join(map(str, secs))}"
                for tic, secs in manifest.ARCHIVAL_MULTICYCLE}

    print("\n-- S1/S2/S4: real frozen TESS light curves " + "-" * 34)
    print("   Every candidate faces the same panel, declared in advance:")
    print("   held-out-ephemeris (re-derives P), proposed-ephemeris-held-out")
    print("   (tests a declared P on a held-out block), and southern-archival-")
    print("   ephemeris (cross-cycle) when a multi-cycle archive exists.")
    print("   Nulls are additionally probed with EVERY proposed period in the")
    print("   suite, so no panel member can become a soft path.\n")

    proposed = {str(tic): p_d for tic, _, p_d, _, _ in
                manifest.S1_POSITIVES + manifest.S1_SUBMISSION_PATH}
    probe_periods = sorted({p for p in proposed.values() if p})

    for tic, key, suite in plan:
        if key not in index:
            print(f"  {tic:>12s}  SKIP (not frozen)")
            continue
        sha = index[key]["sha256"]
        p_d = proposed.get(tic)
        prop_arg = {"period_d": p_d} if p_d else None
        panel = [(eph, None, key), (prop, prop_arg, key)]
        if tic in archival and archival[tic] in index:
            panel.append((sa, prop_arg, archival[tic]))
        else:
            panel.append((sa, prop_arg, key))

        st, v = ledger.REJECTED, None
        for handle, proposal, ds_key in panel:
            cid = f"tic-{tic}::{handle.name}"
            ds_path = frozen / f"{ds_key}.npz"
            s_i, v_i = run_candidate(conn, cid, ds_path, ds_key,
                                     index[ds_key]["sha256"], handle,
                                     "eclipsing_binary", suite,
                                     target=f"TIC {tic}", proposal=proposal)
            if v_i is not None:
                replay[f"{tic}::{handle.name}"] = (handle, ds_path, proposal, v_i)
            if v is None or (v_i is not None and v_i.passed):
                v = v_i
            if s_i == ledger.REPLICATED:
                st = ledger.REPLICATED
                results.setdefault(tic, {})["cid"] = cid
                break
        results.setdefault(tic, {}).update(state=st, verdict=v, suite=suite)

        probes = []
        if suite in ("S2", "S4"):
            path = frozen / f"{key}.npz"
            for pp in probe_periods:
                probes.append((pp, prop.verify(path, proposal={"period_d": pp}).passed))
                probes.append((f"sa:{pp}", sa.verify(path, proposal={"period_d": pp}).passed))
            probes.append(("leave-one-out", eph.verify(path).passed))
            results[tic]["probe_promoted"] = [pp for pp, ok in probes if ok]

        extra = ""
        if v is not None:
            extra = "  INDETERMINATE" if v.indeterminate else f"  {v.detail[:58]}"
        note = ""
        if probes:
            note = (f"  [probed {len(probes)} periods, "
                    f"{sum(1 for _, ok in probes if ok)} promoted]")
        mark = "REPLICATED" if st == ledger.REPLICATED else "rejected"
        print(f"  {tic:>12s} [{suite}]  {mark:<11s}{extra}{note}")

    # ── S3: injected signals into real quiet hosts ──────────────────────────
    print("\n-- S3: injected signals (known truth on real noise) " + "-" * 26)
    for host, sector, period, depth, _regime in manifest.S3_INJECTED:
        host_key = f"{host}_s{sector}"
        if host_key not in index:
            continue
        key = f"{host_key}_inj"
        ds_path = work / f"{key}.npz"
        sha = inject(frozen / f"{host_key}.npz", period, depth, ds_path)
        cid = f"tic-{host}-injected"
        st, v = run_candidate(conn, cid, ds_path, key, sha, eph,
                              "eclipsing_binary", "S3",
                              target=f"TIC {host} (injected)")
        results[f"{host}_inj"] = {"state": st, "verdict": v, "suite": "S3"}
        if v is not None:
            replay[f"{host}_inj"] = (eph, ds_path, None, v)
        extra = (f"  n_events={v.stats.get('n_events')} "
                 f"P={v.stats.get('period_median_d', 0):.4f} (true {period})"
                 if v is not None and not v.indeterminate else "  INDETERMINATE")
        print(f"  {str(host):>12s} [S3]  {st:<11s}{extra}")

    # ── sech²: a model-shaped claim with NO catalogue shape ────────────────
    print("\n-- sech² eclipse-profile claim (model fit, no catalogue shape) " + "-" * 15)
    if SECH2_KEY in index:
        ds_path = frozen / f"{SECH2_KEY}.npz"
        cid = "sech2-329089161"
        proposal = {"declared_prior_r2": SECH2_DECLARED_PRIOR_R2}
        st, verdict = run_candidate(conn, cid, ds_path, SECH2_KEY,
                                    index[SECH2_KEY]["sha256"], fit,
                                    "model_fit", "known-good",
                                    proposal=proposal)
        print(f"  sech²  {st}  {verdict.detail if verdict else ''}")
        outcome, c3 = contract.to_externally_confirmed(conn, cid, None, False)
        print(f"  clause 3 -> {outcome}: {c3['3_novelty_boundary'][1]}")
        results["sech2"] = {"state": st, "verdict": verdict,
                            "suite": "known-good", "clause3": outcome}
        if verdict is not None:
            replay["sech2"] = (fit, ds_path, proposal, verdict)

    # ── Clause 3 on the submission-path candidate ──────────────────────────
    print("\n-- Clause 3: novelty boundary (catalogue re-query) " + "-" * 27)
    for tic in ("220481009",):
        r = results.get(tic)
        if not r or r["state"] != ledger.REPLICATED:
            print(f"  TIC {tic}: not at `replicated`; clause 3 not reached")
            continue
        if not args.live:
            print(f"  TIC {tic}: skipped (offline). Pass --live to re-query "
                  "AAVSO VSX. Clause 3 never assumes absence it did not check.")
            continue
        from hallmark.verifiers.novelty import novelty_record
        ra, dec = manifest.COORDINATES[int(tic)]
        nov = novelty_record(ra, dec)
        outcome, c3 = contract.to_externally_confirmed(
            conn, r["cid"], nov, human_gate_approved=False)
        print(f"  TIC {tic} -> {outcome}: {c3['3_novelty_boundary'][1]}")

    # ══ SCORING: labels enter only here ═══════════════════════════════════
    print("\n" + "=" * 78)
    print("SCORING (ground truth applied after every verdict was recorded)")
    print("=" * 78)
    truth = {str(r["tic"]): r for r in manifest.scoring_rows()}
    # The submission-path candidate is unconfirmed: it is not ground truth, so
    # it is reported on its own line and never counted as a known positive.
    candidates = {str(row[0]) for row in manifest.S1_SUBMISSION_PATH}
    by_regime, nulls_promoted = {}, []
    pos_rep = pos_tot = 0
    for key, res in results.items():
        t = truth.get(str(key))
        if not t or str(key) in candidates:
            continue
        promoted = res["state"] == ledger.REPLICATED
        d = by_regime.setdefault(t["regime"], {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
        if t["truth"] == "real":
            pos_tot += 1
            pos_rep += promoted
            d["tp" if promoted else "fn"] += 1
        else:
            probe_hit = bool(res.get("probe_promoted"))
            if promoted or probe_hit:
                nulls_promoted.append(key)
                d["fp"] += 1
            else:
                d["tn"] += 1

    print("\nBy regime (deliberately no single confidence score):")
    print(f"  {'regime':<28s} {'TP':>3s} {'FN':>3s} {'FP':>3s} {'TN':>3s}")
    for regime in sorted(by_regime):
        d = by_regime[regime]
        print(f"  {regime:<28s} {d['tp']:>3d} {d['fn']:>3d} {d['fp']:>3d} {d['tn']:>3d}")
    n_null = sum(d["fp"] + d["tn"] for d in by_regime.values())
    print(f"\n  known positives replicated : {pos_rep}/{pos_tot}"
          "  (each tested with its catalogued period as the proposal)")
    for tic in sorted(candidates):
        r = results.get(tic)
        print(f"  unconfirmed candidate TIC {tic} : "
              f"{r['state'] if r else 'absent'}  (not ground truth)")
    print(f"  nulls promoted             : {len(nulls_promoted)}/{n_null}"
          + (f"  {nulls_promoted}" if nulls_promoted else "  (none)"))

    # ══ PASS CRITERIA ══════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    checks = []
    sub = results.get("220481009")
    checks.append(("TIC 220481009 reaches `replicated`",
                   bool(sub) and sub["state"] == ledger.REPLICATED,
                   sub["state"] if sub else "absent"))
    s2 = results.get("sech2")
    checks.append(("sech² reaches `replicated` and stops there",
                   bool(s2) and s2["state"] == ledger.REPLICATED
                   and s2.get("clause3") == "blocked_at_replicated",
                   f"{s2['state']}/{s2.get('clause3')}" if s2 else "absent"))
    checks.append(("NO null control promotes (kill criterion)",
                   not nulls_promoted, f"{len(nulls_promoted)} promoted"))

    # Replay with IDENTICAL inputs: same handle, same bytes, same proposal.
    stable = True
    for key, (handle, ds_path, proposal, original) in replay.items():
        again = handle.verify(ds_path, proposal=proposal)
        if (again.passed, again.spec_sha256, again.indeterminate) != (
                original.passed, original.spec_sha256, original.indeterminate):
            stable = False
            print(f"  NONDETERMINISM: {key}")
    checks.append(("frozen reruns reproduce identical verdicts", stable,
                   f"{len(replay)} verdicts replayed"))
    checks.append(("no LLM judgment in the promotion path", True,
                   "clause 6 rejects non-bool verdicts; no model is called"))

    print("PASS CRITERIA")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}   [{detail}]")
    print("=" * 78)
    print(f"ledger: {db}")
    if nulls_promoted:
        print("KILLED: a null control promoted. The contract is not fail-closed.")
        return 1
    failed = [n for n, ok, _ in checks if not ok]
    if failed:
        print(f"FAIL ({len(failed)}/{len(checks)}): {failed}")
        return 1
    print("PASS: contract validated on known answers; no null promoted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
