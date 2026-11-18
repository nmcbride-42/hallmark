"""The gate must block what should be blocked AND pass what should pass.

A gate that only ever returns PARKED is indistinguishable from a broken import.
Negative controls are the point of this file.
"""

import pytest

from hallmark.feasibility import (
    Blocker,
    DatasetCapability,
    FeasibilityViolation,
    GateVerdict,
    HypothesisSpec,
    EXAMPLE_SPEC,
    Readout,
    RegimeBand,
    demo,
    gate,
)


# ── the known-bad case: the L1000 example ──────────────────────────────────────────────

def test_example_is_parked():
    v = demo()
    assert v.status == "PARKED"
    assert not v.feasible


def test_example_catches_all_four_blocker_classes():
    codes = {b.code for b in demo().blockers}
    assert "READOUT_ABSENT" in codes
    assert "REGIME_DISJOINT" in codes
    assert "UNDERPOWERED" in codes
    assert "STRATIFIER_CONTAMINATED" in codes


def test_example_underpower_counts_contamination():
    """4 WT units minus 2 contaminated = 2 effective, not 4."""
    ub = [b for b in demo().blockers if b.code == "UNDERPOWERED"][0]
    assert "TP53_WT=2" in ub.detail


def test_unblocking_spec_names_every_blocker():
    v = demo()
    spec = v.unblocking_spec()
    for b in v.blockers:
        assert b.unblock in spec


# ── the known-good case: a dataset that should clear ───────────────────────

GOOD = DatasetCapability(
    dataset_id="hypothetical/primary-fibroblast-senescence",
    measured=frozenset({"CDKN1A", "IL6", "CXCL8", "IL1B", "MMP3"}),
    units_per_arm={"TP53_WT": 6, "TP53_MUT": 7},
    regime_ranges={"dose": (500.0, 25_000.0, "uM"), "duration": (48.0, 336.0, "h")},
    contaminated_units={},
)


def test_good_dataset_is_feasible():
    v = gate(EXAMPLE_SPEC, GOOD)
    assert v.status == "FEASIBLE", v.report()
    assert v.feasible
    assert v.blockers == ()


def test_feasible_spec_says_so():
    assert "already clears the gate" in gate(EXAMPLE_SPEC, GOOD).unblocking_spec()


# ── one blocker at a time: each clause fires independently ─────────────────

def _tweak(**kw):
    base = dict(
        dataset_id=GOOD.dataset_id, measured=GOOD.measured,
        units_per_arm=dict(GOOD.units_per_arm),
        regime_ranges=dict(GOOD.regime_ranges),
        contaminated_units=dict(GOOD.contaminated_units),
    )
    base.update(kw)
    return DatasetCapability(**base)


def test_only_readout_missing():
    v = gate(EXAMPLE_SPEC, _tweak(measured=GOOD.measured - {"MMP3"}))
    assert [b.code for b in v.blockers] == ["READOUT_ABSENT"]


def test_only_underpowered():
    v = gate(EXAMPLE_SPEC, _tweak(units_per_arm={"TP53_WT": 4, "TP53_MUT": 7}))
    assert [b.code for b in v.blockers] == ["UNDERPOWERED"]


def test_only_regime_disjoint():
    v = gate(EXAMPLE_SPEC, _tweak(regime_ranges={
        "dose": (0.1, 10.0, "uM"), "duration": (48.0, 336.0, "h")}))
    assert [b.code for b in v.blockers] == ["REGIME_DISJOINT"]


def test_only_contaminated():
    v = gate(EXAMPLE_SPEC, _tweak(contaminated_units={"TP53_MUT": 1}))
    assert [b.code for b in v.blockers] == ["STRATIFIER_CONTAMINATED"]


def test_boundary_exactly_at_floor_passes():
    """min_units_per_arm=5 means 5 passes. Off-by-one here silently kills work."""
    v = gate(EXAMPLE_SPEC, _tweak(units_per_arm={"TP53_WT": 5, "TP53_MUT": 5}))
    assert v.feasible, v.report()


def test_regime_touching_at_the_edge_overlaps():
    v = gate(EXAMPLE_SPEC, _tweak(regime_ranges={
        "dose": (10.0, 1_000.0, "uM"), "duration": (48.0, 336.0, "h")}))
    assert v.feasible, v.report()


# ── fail-closed on ignorance ───────────────────────────────────────────────

def test_undeclared_regime_is_a_blocker_not_a_pass():
    v = gate(EXAMPLE_SPEC, _tweak(regime_ranges={"duration": (48.0, 336.0, "h")}))
    assert [b.code for b in v.blockers] == ["REGIME_UNDECLARED"]


def test_no_units_declared_is_a_blocker():
    v = gate(EXAMPLE_SPEC, _tweak(units_per_arm={}))
    assert "NO_UNITS" in {b.code for b in v.blockers}


def test_unit_mismatch_is_not_silently_compared():
    v = gate(EXAMPLE_SPEC, _tweak(regime_ranges={
        "dose": (0.5, 25.0, "mM"), "duration": (48.0, 336.0, "h")}))
    assert [b.code for b in v.blockers] == ["REGIME_UNIT_MISMATCH"]


def test_control_arm_entirely_unmeasured_is_flagged_separately():
    """Losing every control collapses the claim into its naive form."""
    v = gate(EXAMPLE_SPEC, _tweak(measured=frozenset({"CDKN1A"})))
    assert "CONTROL_ARM_UNMEASURABLE" in {b.code for b in v.blockers}


# ── no judgment may enter the path ─────────────────────────────────────────

def test_non_blocker_clause_output_raises():
    import hallmark.feasibility as F
    original = F.CLAUSES
    F.CLAUSES = (lambda s, c: "probably fine",)
    try:
        with pytest.raises(FeasibilityViolation):
            gate(EXAMPLE_SPEC, GOOD)
    finally:
        F.CLAUSES = original


def test_bad_readout_role_rejected():
    with pytest.raises(FeasibilityViolation):
        Readout("X", "sort-of-primary")
