"""Held-out ephemeris recovery — the binding verifier for periodic variables.

The physical claim behind an eclipsing binary is not "there is a dip." It is
"eclipses recur on a fixed ephemeris." So the test is: fit (P, t0) with one
eclipse HELD OUT, predict when that eclipse should have happened, and check
whether it did — against control phases where nothing should be.

Why this shape rather than splitting the curve and re-running BLS:

- It works across the whole period range. A 27-day period in a 25-day sector
  cannot be re-fit on half the data; it can still be predicted from its own
  other eclipses.
- It is what defeats the adversarial class. A dip caused by a data gap,
  momentum dump, or scattered light is tied to the spacecraft's clock, not to
  a stellar orbit, so it does not recur where a fitted ephemeris says it must.

Determinism: fixed grids, no RNG, no LLM. Same bytes in, same verdict out.

Fail-closed: fewer than `min_events_for_fit` + 1 events leaves the ephemeris
underdetermined (any two points admit P = interval/k for every integer k), so
the verdict is INDETERMINATE, which does not pass.
"""

import numpy as np

from ..verifier import VerificationHandle, Verdict


def _robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) or 1e-9


def _detrend(t, f, window_d):
    """Running-median detrend. Removes stellar variability and slow systematics
    so that 'dip' means dip relative to the local continuum."""
    out = np.empty_like(f)
    # Bin edges make this O(n log n) rather than O(n^2).
    idx = np.searchsorted(t, t - window_d / 2), np.searchsorted(t, t + window_d / 2)
    for i in range(len(f)):
        lo, hi = idx[0][i], idx[1][i]
        out[i] = np.median(f[lo:hi]) if hi > lo else f[i]
    return f / np.where(out == 0, 1.0, out)


def _find_gaps(t, gap_min_d):
    """Return (start, end) times of data gaps longer than gap_min_d."""
    dt = np.diff(t)
    where = np.nonzero(dt > gap_min_d)[0]
    return [(float(t[i]), float(t[i + 1])) for i in where]


def _find_events(t, f, n_sigma, min_duration_d, gaps, gap_guard_d):
    """Contiguous runs significantly below the continuum, excluding anything
    touching a data gap. The gap guard is the mechanical replacement for what
    a vision model was doing by eye."""
    sigma = _robust_sigma(f)
    below = f < (1.0 - n_sigma * sigma)
    events = []
    i = 0
    while i < len(below):
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(below) and below[j + 1]:
            j += 1
        t0, t1 = t[i], t[j]
        dur = t1 - t0
        if dur >= min_duration_d:
            centre = 0.5 * (t0 + t1)
            depth = float(1.0 - np.min(f[i:j + 1]))
            near_gap = any(abs(centre - g0) < gap_guard_d or
                           abs(centre - g1) < gap_guard_d for g0, g1 in gaps)
            near_edge = (centre - t[0] < gap_guard_d) or (t[-1] - centre < gap_guard_d)
            if not near_gap and not near_edge:
                events.append({"t": float(centre), "depth": depth,
                               "duration_d": float(dur)})
        i = j + 1
    return events


def _coverage_windows(t, gap_min_d):
    """Contiguous observed windows — everything between gaps."""
    edges = np.nonzero(np.diff(t) > gap_min_d)[0]
    starts = np.concatenate(([0], edges + 1))
    ends = np.concatenate((edges, [len(t) - 1]))
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, ends) if b > a]


def _in_coverage(x, coverage, margin=0.0):
    return any(a + margin <= x <= b - margin for a, b in coverage)


def _fit_period(times, p_min, p_max, coverage=None, max_harmonic=10):
    """Find the ephemeris period for a set of eclipse times.

    Candidates are pairwise intervals divided by small integers — the periods
    that can actually explain the observed spacings.

    Selection is by CYCLE COMPLETENESS first, not phase dispersion, because
    dispersion cannot break the P/k degeneracy: if every eclipse falls on a
    multiple of P it also falls on a multiple of P/2, so both fit exactly and
    numerical noise decides. Completeness breaks it on physics — a half-period
    alias predicts an eclipse at every intervening phase, and those were
    observed and found empty. This is the "half/double period" adversarial
    class handled in the fitter rather than patched downstream.
    """
    times = np.sort(np.asarray(times, dtype=float))
    if len(times) < 2:
        return None
    cands = set()
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            d = times[j] - times[i]
            for k in range(1, max_harmonic + 1):
                p = d / k
                if p_min <= p <= p_max:
                    cands.add(round(p, 6))
    if not cands:
        return None

    def dispersion(p):
        ph = (times - times[0]) / p
        return float(np.sqrt(np.mean(((ph - np.round(ph)) * p) ** 2)))

    def completeness(p):
        """Fraction of epochs predicted inside observed coverage that have an
        actual detected event. A P/2 alias scores ~0.5; the true P scores ~1."""
        if not coverage:
            return 1.0
        t0 = times[0]
        lo = min(a for a, _ in coverage)
        hi = max(b for _, b in coverage)
        n_lo, n_hi = int(np.floor((lo - t0) / p)), int(np.ceil((hi - t0) / p))
        if n_hi - n_lo > 5000:
            return 0.0
        predicted, matched = 0, 0
        tol = max(0.02 * p, 0.02)
        for n in range(n_lo, n_hi + 1):
            pt = t0 + n * p
            if not _in_coverage(pt, coverage, margin=tol):
                continue
            predicted += 1
            if np.any(np.abs(times - pt) <= tol):
                matched += 1
        return matched / predicted if predicted else 0.0

    def explanation(p):
        """Fraction of OBSERVED events that land on a predicted epoch. A 2P
        alias scores ~0.5 — it predicts perfectly but leaves half the real
        eclipses unexplained."""
        t0 = times[0]
        tol = max(0.02 * p, 0.02)
        ph = (times - t0) / p
        return float(np.mean(np.abs((ph - np.round(ph)) * p) <= tol))

    # Completeness and explanation together pin P uniquely:
    #   P/2  -> completeness ~0.5 (predicts eclipses that were observed empty)
    #   2P   -> explanation  ~0.5 (leaves half the observed eclipses unexplained)
    #   P    -> both ~1.0
    # Scores are binned so near-ties don't flip on floating-point noise; ties
    # break to tighter dispersion, then to the SHORTEST period (the longest
    # alias always ties on completeness, so shortest is the safe direction).
    scored = [(round(min(completeness(p), explanation(p)), 2),
               -dispersion(p), -p) for p in cands]
    best_p = float(-max(scored)[2])

    span = best_p * 0.02
    grid = np.linspace(max(p_min, best_p - span), min(p_max, best_p + span), 401)
    best_p = float(min(grid, key=dispersion))
    return {"period_d": best_p, "t0": float(times[0]),
            "dispersion_d": dispersion(best_p),
            "completeness": completeness(best_p),
            "explanation": explanation(best_p)}


class HeldOutEphemeris(VerificationHandle):
    """Leave-one-eclipse-out ephemeris prediction with control phases."""

    name = "held-out-ephemeris"

    DEFAULTS = dict(
        detrend_window_d=0.75,
        n_sigma=4.0,
        min_duration_d=0.02,
        gap_min_d=0.15,
        gap_guard_d=0.25,
        p_min=0.4,
        p_max=60.0,
        min_events_for_fit=3,      # +1 held out => >=4 events required
        tolerance_frac=0.02,       # |residual| < 2% of period
        control_margin=2.0,        # predicted residual must beat controls by 2x
        n_controls=16,
    )

    def __init__(self, **spec):
        merged = dict(self.DEFAULTS)
        merged.update(spec)
        super().__init__(**merged)

    def _run(self, dataset_path):
        s = self.spec
        d = np.load(dataset_path)
        t = np.asarray(d["time"], dtype=float)
        f = np.asarray(d["flux"], dtype=float)
        order = np.argsort(t)
        t, f = t[order], f[order]

        fd = _detrend(t, f, s["detrend_window_d"])
        gaps = _find_gaps(t, s["gap_min_d"])
        coverage = _coverage_windows(t, s["gap_min_d"])
        events = _find_events(t, fd, s["n_sigma"], s["min_duration_d"],
                              gaps, s["gap_guard_d"])

        stats = {"n_cadences": int(len(t)),
                 "baseline_d": round(float(t[-1] - t[0]), 3),
                 "n_gaps": len(gaps),
                 "n_events": len(events),
                 "event_times": [round(e["t"], 4) for e in events[:40]],
                 "event_depths": [round(e["depth"], 5) for e in events[:40]]}

        need = s["min_events_for_fit"] + 1
        if len(events) < need:
            stats["indeterminate_reason"] = (
                f"{len(events)} clean event(s); need >={need} to over-determine "
                "an ephemeris (2 points admit P=interval/k for every k)")
            return stats

        # Eclipsing binaries show primaries and secondaries at different
        # phases. Fit on the deeper family when a split is evident, else all.
        depths = np.array([e["depth"] for e in events])
        times = np.array([e["t"] for e in events])
        families = {"all": times}
        if len(events) >= 6:
            cut = float(np.median(depths))
            deep = times[depths >= cut]
            if len(deep) >= need:
                families["primaries"] = deep

        best = None
        for fam_name, fam_times in families.items():
            folds = []
            for i in range(len(fam_times)):
                train = np.delete(fam_times, i)
                held = float(fam_times[i])
                fit = _fit_period(train, s["p_min"], s["p_max"],
                                  coverage=coverage)
                if not fit:
                    continue
                p, t0 = fit["period_d"], fit["t0"]
                n = round((held - t0) / p)
                pred = t0 + n * p
                resid = abs(held - pred)
                # Controls: the same statistic at phases where nothing is
                # predicted. If the "prediction" is no better than an arbitrary
                # phase, the ephemeris explains nothing.
                ctrl = [abs(held - (t0 + (n + k / s["n_controls"]) * p))
                        for k in range(1, s["n_controls"])]
                folds.append({"held_t": held, "period_d": p,
                              "residual_d": float(resid),
                              "control_median_d": float(np.median(ctrl))})
            if not folds:
                continue
            res = np.array([x["residual_d"] for x in folds])
            per = np.array([x["period_d"] for x in folds])
            ctl = np.array([x["control_median_d"] for x in folds])
            summary = {
                "family": fam_name,
                "n_folds": len(folds),
                "median_residual_d": float(np.median(res)),
                "median_control_d": float(np.median(ctl)),
                "period_median_d": float(np.median(per)),
                "period_spread_frac": float(
                    (np.max(per) - np.min(per)) / max(np.median(per), 1e-9)),
                "folds": folds[:12],
            }
            if best is None or summary["median_residual_d"] < best["median_residual_d"]:
                best = summary
        if best:
            stats.update(best)
        return stats

    def _decide(self, stats):
        s = self.spec
        h = self.spec_sha256
        if "median_residual_d" not in stats:
            return Verdict(False, stats.get("indeterminate_reason",
                                            "no ephemeris fit"),
                           h, stats, indeterminate=True)
        p = stats["period_median_d"]
        tol = s["tolerance_frac"] * p
        checks = {
            "residual_within_tolerance": stats["median_residual_d"] < tol,
            "beats_control_phases":
                stats["median_control_d"] >
                s["control_margin"] * max(stats["median_residual_d"], 1e-6),
            "period_stable_across_folds":
                stats["period_spread_frac"] < 0.05,
        }
        stats["checks"] = checks
        passed = all(checks.values())
        failed = [k for k, v in checks.items() if not v]
        detail = (f"P={p:.5f}d, {stats['n_folds']} folds, "
                  f"median residual {stats['median_residual_d']*24*60:.1f} min "
                  f"(tol {tol*24*60:.1f} min), controls "
                  f"{stats['median_control_d']*24*60:.1f} min"
                  + ("" if passed else f" — FAILED: {failed}"))
        return Verdict(bool(passed), detail, h, stats)


class ProposedEphemerisHeldOut(VerificationHandle):
    """Test a PROPOSED ephemeris against a held-out time block.

    This is the binding verifier for the long-period regime, and it is
    the honest shape of generate->verify: the proposer commits to (P, t0)
    before the verifier looks at held-out data, and the verifier's only job is
    to ask whether that commitment predicted what actually happened.

    `HeldOutEphemeris` requires the verifier to re-derive P from >=4 clean
    eclipses, which single-sector TESS supplies only for periods of a few
    days. A 52-day binary yields one or two eclipses per sector, so that
    verifier correctly refuses. This one does not need to re-derive anything:
    it fits only the epoch on the discovery block using the declared period,
    then measures whether the held-out block is fainter at the predicted phase
    than at control phases it never committed to.

    The proposal cannot game this. It never sees the held-out block, and a
    wrong period lands the prediction on ordinary out-of-eclipse flux, where
    the control phases sit.
    """

    name = "proposed-ephemeris-held-out"

    DEFAULTS = dict(
        detrend_window_d=0.75,
        gap_min_d=0.15,
        split_frac=0.6,          # discovery block = first 60% of the baseline
        duration_frac=0.02,      # eclipse window as a fraction of the period
        n_controls=24,
        min_depth_sigma=5.0,     # predicted-phase dip significance in block B
        control_margin=2.0,      # and it must beat the best control by 2x
    )

    def __init__(self, **spec):
        merged = dict(self.DEFAULTS)
        merged.update(spec)
        super().__init__(**merged)

    def verify(self, dataset_path, proposal=None):
        stats = self._run(dataset_path, proposal=proposal)
        return self._decide(stats)

    def _run(self, dataset_path, proposal=None):
        s = self.spec
        if not proposal or not proposal.get("period_d"):
            return {"indeterminate_reason": "no proposed period declared"}
        p = float(proposal["period_d"])

        d = np.load(dataset_path)
        t = np.asarray(d["time"], dtype=float)
        f = np.asarray(d["flux"], dtype=float)
        order = np.argsort(t)
        t, f = t[order], f[order]
        fd = _detrend(t, f, s["detrend_window_d"])

        t_split = t[0] + s["split_frac"] * (t[-1] - t[0])
        a, b = t <= t_split, t > t_split
        stats = {"period_proposed_d": p, "split_time": float(t_split),
                 "n_block_a": int(a.sum()), "n_block_b": int(b.sum()),
                 "baseline_d": round(float(t[-1] - t[0]), 3)}
        if a.sum() < 500 or b.sum() < 500:
            stats["indeterminate_reason"] = "a block is too sparse to test"
            return stats

        dur = max(s["duration_frac"] * p, 0.05)

        def depth_at(offset, mask):
            """Mean depth inside the eclipse window at a given phase offset."""
            ph = np.abs(((t[mask] - t0_fit - offset + p / 2) % p) - p / 2)
            sel = ph < dur / 2
            if sel.sum() < 5:
                return None
            return float(1.0 - np.mean(fd[mask][sel]))

        # Fit ONLY the epoch, on block A, using the declared period.
        grid = np.linspace(0, p, 400, endpoint=False)
        t0_fit = float(t[a][0])
        best, best_off = None, 0.0
        for off in grid:
            dv = depth_at(off, a)
            if dv is not None and (best is None or dv > best):
                best, best_off = dv, float(off)
        if best is None:
            stats["indeterminate_reason"] = "no epoch fit possible in block A"
            return stats
        stats["block_a_depth"] = best
        stats["epoch_offset_d"] = best_off

        # Held-out block: the prediction, and controls it never committed to.
        pred = depth_at(best_off, b)
        if pred is None:
            stats["indeterminate_reason"] = (
                "held-out block contains no cadence at the predicted phase")
            return stats
        controls = [depth_at(best_off + k * p / s["n_controls"], b)
                    for k in range(1, s["n_controls"])]
        controls = [c for c in controls if c is not None]
        if len(controls) < 4:
            stats["indeterminate_reason"] = "too few control phases in block B"
            return stats

        cmed = float(np.median(controls))
        csig = float(1.4826 * np.median(np.abs(np.array(controls) - cmed))) or 1e-9
        stats.update({
            "block_b_predicted_depth": pred,
            "block_b_control_median": cmed,
            "block_b_control_sigma": csig,
            "depth_sigma": float((pred - cmed) / csig),
            "best_control": float(np.max(controls)),
            "n_controls_used": len(controls),
        })
        return stats

    def _decide(self, stats):
        h = self.spec_sha256
        if "depth_sigma" not in stats:
            return Verdict(False, stats.get("indeterminate_reason", "no test"),
                           h, stats, indeterminate=True)
        s = self.spec
        checks = {
            "predicted_dip_significant":
                stats["depth_sigma"] >= s["min_depth_sigma"],
            "beats_all_control_phases":
                stats["block_b_predicted_depth"] >
                s["control_margin"] * max(stats["best_control"], 1e-9),
            "consistent_with_block_a":
                stats["block_b_predicted_depth"] > 0.3 * stats["block_a_depth"],
        }
        stats["checks"] = checks
        passed = all(checks.values())
        failed = [k for k, v in checks.items() if not v]
        return Verdict(
            bool(passed),
            f"P={stats['period_proposed_d']}d predicted a dip of "
            f"{stats['block_b_predicted_depth']*1e3:.2f}ppt in the held-out "
            f"block ({stats['depth_sigma']:.1f}σ over controls)"
            + ("" if passed else f" — FAILED: {failed}"),
            h, stats)
