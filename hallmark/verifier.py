"""Verification handles — the incorruptible half.

A handle is declared (and hashed) BEFORE a result is computed, then run
against the frozen dataset. It never sees the suite label, the object's
catalogue status, or any prior claim about it: `verify()` takes a dataset path
and the handle's own frozen spec, nothing else. That is what makes the
validation replay blind.

Verdict.passed is a plain bool. INDETERMINATE is NOT a pass — a verifier that
cannot decide fails closed.
"""

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class Verdict:
    passed: bool
    detail: str
    spec_sha256: str
    stats: dict = field(default_factory=dict)
    indeterminate: bool = False

    def __post_init__(self):
        if not isinstance(self.passed, bool):
            raise TypeError("Verdict.passed must be bool — the contract "
                            "forbids graded judgment in the promotion path")
        if self.indeterminate and self.passed:
            raise ValueError("indeterminate verdicts cannot pass (fail-closed)")


class VerificationHandle:
    """Base class. Subclasses set `name` and implement `_run`."""

    name = "abstract"

    def __init__(self, **spec):
        self.spec = dict(spec)

    @property
    def spec_sha256(self) -> str:
        """The frozen spec hash. Recorded at declaration; re-checked at
        replication. Any parameter change produces a different verifier, which
        invalidates prior calibration by construction."""
        blob = json.dumps({"name": self.name, "spec": self.spec},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def verify(self, dataset_path, proposal=None) -> Verdict:
        """Run against a frozen dataset. `proposal` carries only what the
        proposer committed to (e.g. a period); handles that re-derive
        everything themselves ignore it."""
        stats = self._run(dataset_path)
        return self._decide(stats)

    def _run(self, dataset_path) -> dict:
        raise NotImplementedError

    def _decide(self, stats) -> Verdict:
        raise NotImplementedError
