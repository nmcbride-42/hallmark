"""The verifier feasibility gate: clause 4, checkable BEFORE you commit to it.

The contract requires a *predeclared* verifier (clause 4). Nothing in the
contract asks whether that verifier can resolve the claim at all. An earlier
experiment in a different domain (a pharmacology hypothesis tested against the
public LINCS L1000 dataset) spent a full predeclaration -> fetch -> analyse
cycle before discovering four independent blockers, every one of which was
knowable in about ten minutes:

    1. most predeclared readouts were not measured by the dataset
    2. the dataset's regime (uM, 6-24h) did not overlap the model's (mM, days)
    3. 4 units per arm against a predeclared floor of 5
    4. the stratifier was contaminated (SV40-LT / HPV-E6 lines carry a
       wild-type TP53 sequence but a non-functional p53)

This module is the gate that catches all four first. It is domain-agnostic:
nothing in it knows about astronomy or biology.

Design rules, inherited from the contract:

  * **Deterministic. No LLM judgment anywhere.** Every predicate is Python over
    declared facts. `gate()` raises on any non-bool verdict, same as `promote()`.
  * **Fail-closed on ignorance.** A fact that was never declared is a blocker,
    not a pass. "Unknown" and "no" are the same answer here.
  * **PARKED is not REJECTED.** A hypothesis that cannot clear the gate keeps
    its status and gains a named blocking condition. The generated unblocking
    spec is a real output: it describes the dataset that *would* settle the
    claim.

Usage is two declarations and a call. See `demo()` at the bottom, which
replays that experiment and must return PARKED with exactly those four
blockers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class FeasibilityViolation(RuntimeError):
    pass


# ── What the hypothesis needs ──────────────────────────────────────────────

@dataclass(frozen=True)
class Readout:
    """One measurable the claim is stated in terms of."""
    name: str
    role: str = "primary"          # "primary" | "control"
    substitutable: bool = False    # may a related measurable stand in?

    def __post_init__(self):
        if self.role not in ("primary", "control"):
            raise FeasibilityViolation(f"unknown readout role: {self.role!r}")


@dataclass(frozen=True)
class RegimeBand:
    """A range on some axis the model actually speaks about.

    `lo`/`hi` are in the axis's declared unit. A dataset satisfies the band if
    its own declared range overlaps at all — this is deliberately generous,
    because a partial overlap is a design problem, not a gate failure.
    """
    axis: str
    lo: float
    hi: float
    unit: str

    def overlaps(self, other_lo: float, other_hi: float) -> bool:
        return not (other_hi < self.lo or other_lo > self.hi)


@dataclass(frozen=True)
class HypothesisSpec:
    claim_id: str
    readouts: tuple[Readout, ...]
    stratifier: str
    min_units_per_arm: int
    regime: tuple[RegimeBand, ...] = ()

    def primary(self) -> tuple[Readout, ...]:
        return tuple(r for r in self.readouts if r.role == "primary")

    def controls(self) -> tuple[Readout, ...]:
        return tuple(r for r in self.readouts if r.role == "control")


# ── What a candidate dataset actually offers ───────────────────────────────

@dataclass(frozen=True)
class DatasetCapability:
    """Declared facts about a candidate verifier's data. Probed, not assumed.

    `units_per_arm` maps stratifier arm -> count of INDEPENDENT units. Replicate
    signatures on one cell line are one unit, not many; the caller collapses
    them before declaring.

    `contaminated_units` names units whose stratifier label is known unreliable.
    They are subtracted from the arm counts before the floor is applied.
    """
    dataset_id: str
    measured: frozenset[str]
    units_per_arm: dict[str, int]
    regime_ranges: dict[str, tuple[float, float, str]] = field(default_factory=dict)
    contaminated_units: dict[str, int] = field(default_factory=dict)
    access_note: str = ""

    def effective_units(self) -> dict[str, int]:
        return {
            arm: n - self.contaminated_units.get(arm, 0)
            for arm, n in self.units_per_arm.items()
        }


# ── Verdict ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Blocker:
    code: str
    detail: str
    unblock: str          # what a dataset would need to clear this


@dataclass(frozen=True)
class GateVerdict:
    status: str                      # "FEASIBLE" | "PARKED"
    claim_id: str
    dataset_id: str
    blockers: tuple[Blocker, ...]

    @property
    def feasible(self) -> bool:
        return self.status == "FEASIBLE"

    def unblocking_spec(self) -> str:
        """The dataset that would settle this claim. Real output, not an error."""
        if self.feasible:
            return f"{self.dataset_id} already clears the gate for {self.claim_id}."
        lines = [
            f"To settle {self.claim_id}, a dataset must satisfy all of:",
        ]
        for b in self.blockers:
            lines.append(f"  - {b.unblock}   [{b.code}]")
        return "\n".join(lines)

    def report(self) -> str:
        head = f"{self.status}: {self.claim_id} vs {self.dataset_id}"
        if self.feasible:
            return head
        body = "\n".join(f"  [{b.code}] {b.detail}" for b in self.blockers)
        return f"{head}\n{body}\n\n{self.unblocking_spec()}"


# ── The four clauses ───────────────────────────────────────────────────────

def _c1_readouts(spec: HypothesisSpec, cap: DatasetCapability) -> list[Blocker]:
    out = []
    for r in spec.readouts:
        if r.name in cap.measured or r.substitutable:
            continue
        out.append(Blocker(
            code="READOUT_ABSENT",
            detail=f"{r.role} readout {r.name!r} is not measured by {cap.dataset_id}",
            unblock=f"measures {r.name}",
        ))
    # A claim whose control arm is entirely unmeasured cannot be discriminated
    # from the naive version of itself — that is a separate, worse failure.
    ctrl = spec.controls()
    if ctrl and not any(c.name in cap.measured for c in ctrl):
        out.append(Blocker(
            code="CONTROL_ARM_UNMEASURABLE",
            detail=(f"no control readout measured "
                    f"({', '.join(c.name for c in ctrl)}); the claim cannot be "
                    f"distinguished from its naive form"),
            unblock=f"measures at least one of {', '.join(c.name for c in ctrl)}",
        ))
    return out


def _c2_regime(spec: HypothesisSpec, cap: DatasetCapability) -> list[Blocker]:
    out = []
    for band in spec.regime:
        got = cap.regime_ranges.get(band.axis)
        if got is None:
            out.append(Blocker(
                code="REGIME_UNDECLARED",
                detail=f"{cap.dataset_id} declares no range on axis {band.axis!r}",
                unblock=f"declares its {band.axis} range",
            ))
            continue
        lo, hi, unit = got
        if unit != band.unit:
            out.append(Blocker(
                code="REGIME_UNIT_MISMATCH",
                detail=(f"axis {band.axis!r}: dataset in {unit}, "
                        f"claim in {band.unit} — not comparable without conversion"),
                unblock=f"reports {band.axis} in {band.unit}",
            ))
            continue
        if not band.overlaps(lo, hi):
            out.append(Blocker(
                code="REGIME_DISJOINT",
                detail=(f"axis {band.axis!r}: dataset {lo}-{hi}{unit} does not "
                        f"overlap claim {band.lo}-{band.hi}{band.unit}"),
                unblock=f"covers {band.axis} {band.lo}-{band.hi}{band.unit}",
            ))
    return out


def _c3_power(spec: HypothesisSpec, cap: DatasetCapability) -> list[Blocker]:
    eff = cap.effective_units()
    if not eff:
        return [Blocker(
            code="NO_UNITS",
            detail=f"{cap.dataset_id} declares no units per arm",
            unblock=f"declares >={spec.min_units_per_arm} independent units per arm",
        )]
    short = {a: n for a, n in eff.items() if n < spec.min_units_per_arm}
    if not short:
        return []
    detail = ", ".join(f"{a}={n}" for a, n in sorted(short.items()))
    return [Blocker(
        code="UNDERPOWERED",
        detail=(f"effective units per arm ({detail}) below predeclared floor "
                f"of {spec.min_units_per_arm}"),
        unblock=f">={spec.min_units_per_arm} independent units in every arm",
    )]


def _c4_stratifier(spec: HypothesisSpec, cap: DatasetCapability) -> list[Blocker]:
    if not cap.contaminated_units:
        return []
    tot = sum(cap.contaminated_units.values())
    raw = sum(cap.units_per_arm.values()) or 1
    detail = ", ".join(f"{a}:{n}" for a, n in sorted(cap.contaminated_units.items()))
    return [Blocker(
        code="STRATIFIER_CONTAMINATED",
        detail=(f"{tot}/{raw} units carry an unreliable {spec.stratifier!r} label "
                f"({detail})"),
        unblock=f"a clean {spec.stratifier} label on every unit",
    )]


CLAUSES = (_c1_readouts, _c2_regime, _c3_power, _c4_stratifier)


def gate(spec: HypothesisSpec, cap: DatasetCapability) -> GateVerdict:
    """Run all four clauses. Conjunctive, fail-closed, deterministic.

    Every clause runs — we want the full blocker list, not the first failure,
    because the unblocking spec is assembled from all of them.
    """
    blockers: list[Blocker] = []
    for clause in CLAUSES:
        got = clause(spec, cap)
        if not isinstance(got, list) or any(not isinstance(b, Blocker) for b in got):
            raise FeasibilityViolation(
                f"clause {clause.__name__} returned {type(got).__name__}, "
                f"expected list[Blocker] — no judgment may enter this path"
            )
        blockers.extend(got)
    return GateVerdict(
        status="FEASIBLE" if not blockers else "PARKED",
        claim_id=spec.claim_id,
        dataset_id=cap.dataset_id,
        blockers=tuple(blockers),
    )


# ── Replay: the L1000 example must come back PARKED with four blockers ─────

EXAMPLE_SPEC = HypothesisSpec(
    claim_id="metformin-p53-gated-arrest-arm",
    readouts=(
        Readout("CDKN1A", "primary"),
        Readout("IL6", "control"),
        Readout("CXCL8", "control"),
        Readout("IL1B", "control"),
        Readout("MMP3", "control"),
    ),
    stratifier="p53 functional status",
    min_units_per_arm=5,
    regime=(
        RegimeBand("dose", 1_000.0, 20_000.0, "uM"),      # mM-scale, as used in senescence work
        RegimeBand("duration", 72.0, 240.0, "h"),          # days, not hours
    ),
)

EXAMPLE_L1000 = DatasetCapability(
    dataset_id="LINCS-L1000/metformin",
    measured=frozenset({"CDKN1A", "IL1B"}),                # IL6, CXCL8, MMP3 not on the landmark panel
    units_per_arm={"TP53_WT": 4, "TP53_MUT": 4},
    regime_ranges={"dose": (0.1, 10.0, "uM"), "duration": (6.0, 24.0, "h")},
    contaminated_units={"TP53_WT": 2},                     # HELA (HPV-E6), HA1E (SV40-LT)
    access_note="public LINCS L1000 signatures via iLINCS",
)


def demo() -> GateVerdict:
    return gate(EXAMPLE_SPEC, EXAMPLE_L1000)


if __name__ == "__main__":
    v = demo()
    print(v.report())
