"""Verifiers on real frozen TESS data: positives replicate, nulls do not."""

import numpy as np
import pytest

from hallmark.verifiers.curvefit import HeldOutCurveFit
from hallmark.verifiers.ephemeris import (HeldOutEphemeris,
                                          ProposedEphemerisHeldOut, _fit_period)
from hallmark.verifiers.southern_archival import SouthernArchivalEphemeris
from suites import manifest

ARCHIVE = "220481009_s87-88-89-94-95-96.npz"


def _path(frozen, tic, sectors):
    return frozen / f"{tic}_s{'-'.join(map(str, sectors))}.npz"


# ── Period fitting: the P/k alias trap ─────────────────────────────────────

def test_fit_period_rejects_half_and_double_aliases():
    P = 3.0
    times = np.array([10.0 + n * P for n in range(8)])
    coverage = [(9.0, 10.0 + 8 * P)]
    fit = _fit_period(times, 0.4, 20.0, coverage=coverage)
    assert abs(fit["period_d"] - P) < 0.01


def test_two_events_are_underdetermined(frozen):
    """Fewer than 4 clean events -> INDETERMINATE, which fails closed."""
    v = HeldOutEphemeris().verify(_path(frozen, 220481009, [94, 95, 96]))
    assert v.passed is False and v.indeterminate is True


# ── Positives ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tic,sectors,period", [
    (128193913, [92], 1.8444),
    (14069011, [92], 2.0894),
])
def test_short_period_positives_replicate(frozen, tic, sectors, period):
    v = HeldOutEphemeris().verify(_path(frozen, tic, sectors))
    assert v.passed, v.detail
    assert abs(v.stats["period_median_d"] - period) / period < 0.01


def test_secondary_eclipse_on_a_control_phase_fails_closed(frozen):
    """An 8.13 d binary with a deep secondary: the basic held-out-block
    verifier sees the secondary at one of its control phases and refuses."""
    v = ProposedEphemerisHeldOut().verify(
        _path(frozen, 261335113, [95]), proposal={"period_d": 8.1306})
    assert v.passed is False
    assert v.stats["checks"]["beats_all_control_phases"] is False


def test_secondary_aware_verifier_replicates_only_via_replicated_secondary(frozen):
    """The southern-archival verifier excludes the secondary's phase from the
    controls ONLY because the secondary itself replicated in the held-out
    block, so nulls never get that exclusion."""
    v = SouthernArchivalEphemeris().verify(
        _path(frozen, 261335113, [95]), proposal={"period_d": 8.1306})
    assert v.passed, v.detail
    assert v.stats["secondary_detected"] is True


def test_southern_archival_replicates_multicycle_candidate(frozen):
    v = SouthernArchivalEphemeris().verify(frozen / ARCHIVE,
                                           proposal={"period_d": 52.144497})
    assert v.passed is True, v.detail
    assert not v.indeterminate
    assert v.stats["split_mode"] == "inter_cycle_gap"
    assert v.stats["depth_sigma"] > 100.0
    assert v.stats.get("secondary_detected") is True
    assert 0.43 < v.stats["secondary_phase"] < 0.46


def test_southern_archival_replicates_at_manifest_period(frozen):
    """52.1 d is the manifest's declared period, the one the validation run
    tests; the 6-decimal value above was refined on all six sectors."""
    v = SouthernArchivalEphemeris().verify(frozen / ARCHIVE,
                                           proposal={"period_d": 52.1})
    assert v.passed is True, v.detail
    assert v.stats["depth_sigma"] > 100.0


@pytest.mark.parametrize("period_d", [
    2 * 52.1,            # 2P of the manifest period
    2 * 52.144497,       # 2P of the refined period
    52.144497 / 2,       # P/2 of the refined period
])
def test_southern_archival_also_passes_period_aliases(frozen, period_d):
    """Documents a known LIMIT, not a feature: the cross-cycle verifier
    confirms that eclipses recur where an ephemeris predicts, and some
    aliases of the true period predict real eclipses too. A pass does not
    resolve P versus 2P or P/2. If a future verifier change breaks this test,
    update the README's worked example along with it."""
    v = SouthernArchivalEphemeris().verify(frozen / ARCHIVE,
                                           proposal={"period_d": period_d})
    assert v.passed is True, v.detail


def test_proposals_without_a_period_fail_closed(frozen):
    for h in (ProposedEphemerisHeldOut(), SouthernArchivalEphemeris()):
        v = h.verify(frozen / ARCHIVE, proposal=None)
        assert v.passed is False and v.indeterminate is True


def test_wrong_period_does_not_replicate(frozen):
    v = SouthernArchivalEphemeris().verify(frozen / ARCHIVE,
                                           proposal={"period_d": 37.0})
    assert v.passed is False


# ── Nulls: the kill criterion ──────────────────────────────────────────────

NULLS = list(manifest.S2_NEGATIVES) + list(manifest.S4_ADVERSARIAL)
PERIODS = sorted({float(r[2]) for r in
                  manifest.S1_POSITIVES + manifest.S1_SUBMISSION_PATH if r[2]})


@pytest.mark.parametrize("tic,sectors,regime", NULLS)
def test_no_null_promotes_under_any_probed_period(frozen, tic, sectors, regime):
    path = _path(frozen, tic, sectors)
    promoted = []
    for P in PERIODS:
        for h in (ProposedEphemerisHeldOut(), SouthernArchivalEphemeris()):
            v = h.verify(path, proposal={"period_d": P})
            assert isinstance(v.passed, bool)
            if v.passed:
                promoted.append((h.name, P))
    if HeldOutEphemeris().verify(path).passed:
        promoted.append(("held-out-ephemeris", None))
    assert promoted == [], f"{regime} null promoted: {promoted}"


# ── Model-shaped claim ─────────────────────────────────────────────────────

def test_sech2_curve_fit_passes_declared_claim(frozen):
    v = HeldOutCurveFit().verify(frozen / "sech2_lc_329089161.npz",
                                 proposal={"declared_prior_r2": 0.9577})
    assert v.passed, v.detail
    assert v.stats["sech2_heldout_r2"] > 0.95


def test_verifiers_are_deterministic(frozen):
    path = _path(frozen, 14069011, [92])
    a, b = HeldOutEphemeris().verify(path), HeldOutEphemeris().verify(path)
    assert (a.passed, a.detail, a.spec_sha256) == (b.passed, b.detail, b.spec_sha256)


# ── Network (skipped by default) ───────────────────────────────────────────

@pytest.mark.network
def test_vsx_cone_returns_a_complete_record():
    from hallmark.verifiers.novelty import novelty_record
    ra, dec = manifest.COORDINATES[220481009]
    rec = novelty_record(ra, dec)
    assert "error" in rec or isinstance(rec["hits"], int)
