"""The four frozen suites: positive controls, null controls, and adversaries.

Versioned and frozen: any verifier change replays all four, and invalidates
prior calibration. Objects are real TESS targets, drawn from the output of an
earlier light-curve anomaly survey, except S3, which injects known signals into
real quiet light curves.

`label` is ground truth used ONLY for scoring after the fact. The verifier
never receives it — that is what "blind to their known answers" means, and it
is enforced structurally: `verify()` takes a frozen dataset path and declared
params, never a suite row.
"""

SUITE_VERSION = "2026-07-28.1"

# ── S1: known positives — real catalogued variables, spanning regime ─────────
# Chosen to span amplitude (0.04 -> 1.46) and period (1.7 -> 27 d) so the
# verifier's failure regimes surface instead of averaging out.
S1_POSITIVES = [
    # tic,        sectors,        period_d, depth,  regime
    (128193913, [92],            1.8444, 1.4564, "short/deep"),
    (14069011,  [92],            2.0894, 0.2715, "short/mid"),
    (213076699, [98],            1.7488, 0.0507, "short/shallow"),
    (261335113, [95],            8.1306, 0.2918, "mid/mid"),
    (293791446, [102],          27.0841, 0.2670, "long/mid"),
    (404927661, [95],           18.5357, 0.0417, "long/shallow"),
]

# ── The submission-path candidate ───────────────────────────────────────────
# A long-period eclipsing-binary candidate that walks the full path to the
# catalogue gate. Its single-cycle TESS data (sectors 94-96, ~76 d) cannot
# over-determine a ~52 d ephemeris; the archival multi-cycle dataset below
# (two TESS cycles, ~267 d) is what the southern-archival verifier tests.
# The sech2 model-fit claim also stops at `replicated`, because a curve fit
# has no catalogue-claim shape (contract clause mapping).
S1_SUBMISSION_PATH = [
    (220481009, [94, 95, 96], 52.1, None, "long/multi-sector"),
]

# Multi-cycle archival datasets: (tic, sectors). Frozen alongside the suites.
ARCHIVAL_MULTICYCLE = [
    (220481009, [87, 88, 89, 94, 95, 96]),
]

# J2000 coordinates (Gaia DR3) for the clause-3 catalogue re-query.
COORDINATES = {
    220481009: (76.073641, -58.322833),
}

# ── S2: known negatives — quiet stars, same sector/cadence as the positives ──
S2_NEGATIVES = [
    (6286534, [94], "quiet"),
    (6288424, [94], "quiet"),
    (6350228, [94], "quiet"),
    (6402381, [94], "quiet"),
]

# ── S3: injected signals — known truth on real noise. Tests false negatives. ─
# (host_tic, sector, injected_period_d, injected_depth, regime)
S3_INJECTED = [
    (6404965, 94, 2.5,  0.020, "short/shallow-injected"),
    (6406215, 94, 5.0,  0.010, "mid/very-shallow-injected"),
    (6406550, 94, 1.5,  0.050, "short/deep-injected"),
]

# ── S4: adversarial artifacts — the classes that mimic discoveries ──────────
# All six are real light curves that an earlier survey flagged and a later
# review classified as `gap-artifact`: apparent events produced by data gaps,
# momentum dumps, or scattered light. At least 6 of the 10 nulls are
# adversarial by design, because this is the class that mimics a discovery,
# and the reason a naive "found a dip" detector is not a verifier.
S4_ADVERSARIAL = [
    (615133,  [98], "gap-artifact"),
    (3990402, [97], "gap-artifact"),
    (4070682, [98], "gap-artifact"),
    (4124879, [92], "gap-artifact"),
    (4207747, [98], "gap-artifact"),
    (7219876, [97], "gap-artifact"),
]


def all_fetch_targets():
    """(tic, sectors) pairs needing a frozen light curve."""
    out = []
    for tic, sectors, *_ in S1_POSITIVES + S1_SUBMISSION_PATH:
        out.append((tic, sectors))
    for tic, sectors, _ in S2_NEGATIVES + S4_ADVERSARIAL:
        out.append((tic, sectors))
    for tic, sector, *_ in S3_INJECTED:
        out.append((tic, [sector]))
    out.extend((tic, sectors) for tic, sectors in ARCHIVAL_MULTICYCLE)
    return out


def scoring_rows():
    """Ground truth for post-hoc scoring. NEVER passed to a verifier."""
    rows = []
    for tic, _, p, d, regime in S1_POSITIVES:
        rows.append({"tic": tic, "suite": "S1", "truth": "real",
                     "regime": regime, "period_d": p, "depth": d})
    for tic, _, p, d, regime in S1_SUBMISSION_PATH:
        rows.append({"tic": tic, "suite": "S1", "truth": "real",
                     "regime": regime, "period_d": p, "depth": d})
    for tic, _, regime in S2_NEGATIVES:
        rows.append({"tic": tic, "suite": "S2", "truth": "null", "regime": regime})
    for tic, _, p, d, regime in S3_INJECTED:
        rows.append({"tic": f"{tic}_inj", "suite": "S3", "truth": "real",
                     "regime": regime, "period_d": p, "depth": d})
    for tic, _, regime in S4_ADVERSARIAL:
        rows.append({"tic": tic, "suite": "S4", "truth": "null", "regime": regime})
    return rows
