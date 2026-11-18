"""Southern-sky archival cross-check and multi-cycle ephemeris verifier.

For southern candidates (Dec < -28 deg, e.g. TIC 220481009 at Dec -58.32 deg),
northern ground surveys such as ZTF have no sky coverage, and a single
3-sector TESS visit (~76 d) covers fewer than two orbital cycles of a
long-period eccentric binary (P near 52.1 d).

This module provides:
1. `query_southern_archival(tic, ra_deg=None, dec_deg=None)` — queries public
   southern-sky archival catalogues (MAST CAOM multi-cycle TESS sectors,
   ESA Gaia DR3 TAP astrometry/photometry/variability tables, ASAS-SN
   SkyPatrol V-band photometry, and AAVSO VSX cone check).
2. `fetch_and_freeze_southern_archival(tic, frozen_dir)` — fetches all
   available SPOC 2-minute TESS sectors across cycles from MAST and freezes
   the combined light curve with `dataset_sha256`.
3. `SouthernArchivalEphemeris(VerificationHandle)` — deterministic,
   fail-closed cross-cycle ephemeris verifier (`southern-archival-ephemeris`).
   Fits primary (and eccentric secondary, if present) eclipse epochs on
   Discovery Cycle A and tests zero-free-parameter phase predictions on
   Held-Out Cycle B against 24 control phases.

   Limit: with only a few eclipses per cycle, a period alias (2P, and some
   fractional ones such as P/2) can also predict real eclipses in Cycle B and
   pass. A pass confirms recurrence on an ephemeris compatible with the
   declared period; it does not single out the true period.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import config
from ..hashing import dataset_sha256
from ..verifier import VerificationHandle, Verdict
from .ephemeris import _detrend
from .novelty import vsx_cone

UA = "hallmark-southern-archival/1.0"
MAST_INVOKE = "https://mast.stsci.edu/api/v0/invoke"
GAIA_TAP_SYNC = "https://gea.esac.esa.int/tap-server/tap/sync"
ASASSN_PHOT = "https://asas-sn.osu.edu/photometry.json"


def _gaia_v_mag(g_mag: float, bp_rp: float) -> float:
    """Riello et al. (2021) Gaia DR3 G, (BP-RP) -> Johnson V transformation."""
    c = float(bp_rp)
    gv = -0.02704 + 0.01424 * c - 0.2156 * (c ** 2) + 0.01426 * (c ** 3)
    return round(float(g_mag) - gv, 3)


def query_mast_tess_sectors(tic: int | str, timeout: int = 20) -> dict:
    """Query MAST CAOM for all TESS SPOC 2-minute light-curve sectors."""
    req_data = {
        "service": "Mast.Caom.Filtered",
        "format": "json",
        "params": {
            "columns": "obs_id,target_name,sequence_number,s_ra,s_dec,t_min,t_max,t_exptime,dataURL",
            "filters": [
                {"paramName": "target_name", "values": [str(tic)]},
                {"paramName": "obs_collection", "values": ["TESS"]},
                {"paramName": "dataproduct_type", "values": ["timeseries"]},
                {"paramName": "provenance_name", "values": ["SPOC"]},
            ],
        },
    }
    req = urllib.request.Request(
        MAST_INVOKE,
        data=urllib.parse.urlencode({"request": json.dumps(req_data)}).encode(),
        headers={"User-Agent": UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rows = json.load(r).get("data", [])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}", "sectors": []}

    lc_rows = [
        r for r in rows
        if str(r.get("dataURL", "")).endswith("_lc.fits")
        or float(r.get("t_exptime") or 0) == 120.0
    ]
    lc_rows.sort(key=lambda x: int(x.get("sequence_number", 0)))
    sectors = sorted({int(r["sequence_number"]) for r in lc_rows if "sequence_number" in r})
    ra_deg = float(lc_rows[0]["s_ra"]) if lc_rows and lc_rows[0].get("s_ra") is not None else None
    dec_deg = float(lc_rows[0]["s_dec"]) if lc_rows and lc_rows[0].get("s_dec") is not None else None
    t_mins = [float(r["t_min"]) for r in lc_rows if r.get("t_min") is not None]
    t_maxs = [float(r["t_max"]) for r in lc_rows if r.get("t_max") is not None]
    span_d = round(max(t_maxs) - min(t_mins), 2) if t_mins and t_maxs else 0.0
    return {
        "tic": str(tic),
        "sectors": sectors,
        "n_sectors": len(sectors),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "mjd_min": round(min(t_mins), 3) if t_mins else None,
        "mjd_max": round(max(t_maxs), 3) if t_maxs else None,
        "multi_cycle_span_d": span_d,
    }


def query_gaia_dr3(ra_deg: float, dec_deg: float, radius_arcsec: float = 5.0,
                   timeout: int = 15) -> dict:
    """Query ESA Gaia DR3 TAP for astrometry, photometry, and EB catalogue status."""
    rad_deg = radius_arcsec / 3600.0
    adql = (
        "SELECT TOP 5 source_id, ra, dec, phot_g_mean_mag, bp_rp, ruwe, "
        "phot_variable_flag, non_single_star "
        "FROM gaiadr3.gaia_source "
        f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {float(ra_deg):.6f}, {float(dec_deg):.6f}, {rad_deg:.6f})) "
        "ORDER BY phot_g_mean_mag ASC"
    )
    params = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "json",
        "QUERY": adql,
    }).encode()
    req = urllib.request.Request(GAIA_TAP_SYNC, data=params, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}

    rows = payload.get("data", [])
    if not rows:
        return {"hits": 0}

    top = rows[0]
    source_id = int(top[0])
    g_mag = float(top[3]) if top[3] is not None else None
    bp_rp = float(top[4]) if top[4] is not None else None
    ruwe = float(top[5]) if top[5] is not None else None
    var_flag = str(top[6]) if top[6] is not None else "NOT_AVAILABLE"
    nss = int(top[7]) if top[7] is not None else 0
    v_mag = _gaia_v_mag(g_mag, bp_rp) if (g_mag is not None and bp_rp is not None) else None

    # Check gaiadr3.vari_eclipsing_binary for this source_id
    eb_hits = 0
    adql_eb = (
        f"SELECT source_id, frequency FROM gaiadr3.vari_eclipsing_binary "
        f"WHERE source_id = {source_id}"
    )
    params_eb = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "json",
        "QUERY": adql_eb,
    }).encode()
    try:
        req_eb = urllib.request.Request(GAIA_TAP_SYNC, data=params_eb, headers={"User-Agent": UA})
        with urllib.request.urlopen(req_eb, timeout=timeout) as r_eb:
            eb_rows = json.load(r_eb).get("data", [])
            eb_hits = len(eb_rows)
    except Exception:
        eb_hits = 0

    nss_orbit = None
    nss_accel = None
    if nss > 0:
        adql_orb = (
            "SELECT nss_solution_type, period, period_error, eccentricity, "
            "semi_amplitude_primary FROM gaiadr3.nss_two_body_orbit "
            f"WHERE source_id = {source_id}"
        )
        try:
            p_orb = urllib.parse.urlencode({
                "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql_orb
            }).encode()
            req_orb = urllib.request.Request(GAIA_TAP_SYNC, data=p_orb, headers={"User-Agent": UA})
            with urllib.request.urlopen(req_orb, timeout=timeout) as r_orb:
                orb_rows = json.load(r_orb).get("data", [])
                if orb_rows:
                    o = orb_rows[0]
                    nss_orbit = {
                        "solution_type": o[0],
                        "period_d": round(float(o[1]), 6) if o[1] is not None else None,
                        "period_error_d": round(float(o[2]), 6) if o[2] is not None else None,
                        "eccentricity": round(float(o[3]), 4) if o[3] is not None else None,
                        "semi_amplitude_k1_kms": round(float(o[4]), 3) if o[4] is not None else None,
                    }
        except Exception:
            nss_orbit = None

        adql_acc = (
            "SELECT nss_solution_type, significance "
            f"FROM gaiadr3.nss_acceleration_astro WHERE source_id = {source_id}"
        )
        try:
            p_acc = urllib.parse.urlencode({
                "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql_acc
            }).encode()
            req_acc = urllib.request.Request(GAIA_TAP_SYNC, data=p_acc, headers={"User-Agent": UA})
            with urllib.request.urlopen(req_acc, timeout=timeout) as r_acc:
                acc_rows = json.load(r_acc).get("data", [])
                if acc_rows:
                    a = acc_rows[0]
                    nss_accel = {
                        "solution_type": a[0],
                        "significance": round(float(a[1]), 2) if a[1] is not None else None,
                    }
        except Exception:
            nss_accel = None

    return {
        "hits": len(rows),
        "source_id": source_id,
        "g_mag": round(g_mag, 4) if g_mag is not None else None,
        "bp_rp": round(bp_rp, 4) if bp_rp is not None else None,
        "johnson_v_transformed": v_mag,
        "ruwe": round(ruwe, 3) if ruwe is not None else None,
        "ruwe_binary_excess": bool(ruwe is not None and ruwe > 1.4),
        "phot_variable_flag": var_flag,
        "non_single_star": nss,
        "nss_two_body_orbit": nss_orbit,
        "nss_acceleration_astro": nss_accel,
        "vari_eclipsing_binary_hits": eb_hits,
    }


def query_asassn_skypatrol(ra_deg: float, dec_deg: float,
                           radius_arcmin: float = 0.5, timeout: int = 15) -> dict:
    """Query ASAS-SN SkyPatrol photometry metadata for southern ground coverage."""
    url = (
        f"{ASASSN_PHOT}?ra={float(ra_deg):.5f}&dec={float(dec_deg):.5f}"
        f"&radius={float(radius_arcmin):.3f}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}

    results = data.get("results", []) if isinstance(data, dict) else []
    if not results:
        return {"hits": 0}
    top = results[0]
    return {
        "hits": len(results),
        "asas_sn_id": top.get("asas_sn_id"),
        "mean_vmag": top.get("mean_vmag"),
        "rms": top.get("rms"),
        "epochs": top.get("epochs"),
        "blend": top.get("blend"),
        "source": top.get("source"),
    }


def query_southern_archival(tic: int | str, ra_deg: float | None = None,
                            dec_deg: float | None = None,
                            timeout: int = 15) -> dict:
    """Run the full Southern-Sky Archival Cross-Check across MAST, Gaia DR3,
    ASAS-SN SkyPatrol, and AAVSO VSX."""
    mast = query_mast_tess_sectors(tic, timeout=timeout)
    ra = ra_deg if ra_deg is not None else mast.get("ra_deg")
    dec = dec_deg if dec_deg is not None else mast.get("dec_deg")

    gaia = query_gaia_dr3(ra, dec, timeout=timeout) if (ra is not None and dec is not None) else {}
    asassn = query_asassn_skypatrol(ra, dec, timeout=timeout) if (ra is not None and dec is not None) else {}
    vsx_hits, vsx_recs, vsx_err = (
        vsx_cone(ra, dec, radius_arcsec=60.0, timeout=timeout)
        if (ra is not None and dec is not None)
        else (None, [], "missing coordinates")
    )

    return {
        "tic": str(tic),
        "ra_deg": ra,
        "dec_deg": dec,
        "southern_sky": bool(dec is not None and dec < -28.0),
        "queried_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mast_tess": mast,
        "gaia_dr3": gaia,
        "asas_sn": asassn,
        "vsx": {
            "hits": vsx_hits,
            "names": [r.get("Name") for r in vsx_recs][:5],
            "error": vsx_err,
        },
    }


def fetch_and_freeze_southern_archival(tic: int | str,
                                       frozen_dir: Path | str | None = None,
                                       sectors: list[int] | None = None) -> tuple[Path, str, list[int]]:
    """Fetch all available SPOC 2-min sectors for `tic` from MAST, write the
    frozen `.npz` archive, and return `(path, sha256, sectors)`."""
    from ..mast import fetch_mast_direct

    if frozen_dir is None:
        frozen_dir = config.frozen_dir()
    frozen_dir = Path(frozen_dir)
    frozen_dir.mkdir(parents=True, exist_ok=True)

    if sectors is None:
        meta = query_mast_tess_sectors(tic)
        sectors = meta.get("sectors", [])
    if not sectors:
        raise RuntimeError(f"No TESS SPOC sectors found on MAST for TIC {tic}")

    sec_tag = "-".join(str(s) for s in sectors)
    out_path = frozen_dir / f"{tic}_s{sec_tag}.npz"
    if out_path.exists():
        d = np.load(out_path)
        t = np.asarray(d["time"], dtype=float)
        f = np.asarray(d["flux"], dtype=float)
        got = [int(x) for x in d["sectors"]] if "sectors" in d.files else list(sectors)
        return out_path, dataset_sha256(time=t, flux=f), got

    t, f, got = fetch_mast_direct(tic, sectors)
    if t is None:
        raise RuntimeError(f"No usable light-curve data downloaded for TIC {tic}")
    np.savez_compressed(
        out_path,
        time=t,
        flux=f,
        tic=int(tic),
        sectors=np.array(got, dtype=np.int32),
    )
    sha = dataset_sha256(time=t, flux=f)
    return out_path, sha, got


class SouthernArchivalEphemeris(VerificationHandle):
    """Cross-cycle southern-sky archival ephemeris verifier.

    Designed for long-period and eccentric eclipsing binaries in the southern
    hemisphere (Dec < -28 deg) where single-cycle TESS observations span
    fewer than two orbital periods and northern surveys (ZTF) have no coverage.

    Key physical invariants:
    1. Keplerian-bounded eclipse duration window: `dur = min(max(duration_frac * P, 0.05), max_duration_d)`
       so detached long-period binaries (whose eclipse duration scales as P^(-2/3),
       typically 6-12 hours for P ~ 50 d) do not have their dips diluted across
       multi-day windows.
    2. Inter-cycle split: when the light curve spans multiple TESS cycles separated
       by an unobserved gap >= `min_cycle_gap_d` (e.g. ~110 days between Sectors
       87-89 and Sectors 94-96), splits cleanly across the inter-cycle gap into
       Discovery Cycle A and Held-Out Cycle B; otherwise splits at `split_frac`.
    3. Eccentric secondary awareness without softening null controls: if Block A
       exhibits a secondary dip at phase offset >= 0.15*P from the primary, its
       phase is also predicted in Block B. The secondary phase window is excluded
       from the primary's out-of-eclipse control phases ONLY if the secondary dip
       itself replicates at >= `min_secondary_sigma` in Block B. Thus null controls
       receive zero phase exclusions.
    """

    name = "southern-archival-ephemeris"

    DEFAULTS = dict(
        detrend_window_d=1.5,
        gap_min_d=0.15,
        min_cycle_gap_d=20.0,
        split_frac=0.5,
        duration_frac=0.02,
        max_duration_d=0.42,
        n_controls=24,
        min_depth_sigma=5.0,
        min_secondary_sigma=5.0,
        control_margin=2.0,
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

        # Prefer splitting at the largest inter-cycle gap if >= min_cycle_gap_d
        dt = np.diff(t)
        max_gap_idx = int(np.argmax(dt)) if len(dt) else -1
        if max_gap_idx >= 0 and float(dt[max_gap_idx]) >= s["min_cycle_gap_d"]:
            t_split = 0.5 * float(t[max_gap_idx] + t[max_gap_idx + 1])
            split_mode = "inter_cycle_gap"
            inter_cycle_gap_d = round(float(dt[max_gap_idx]), 3)
        else:
            t_split = float(t[0] + s["split_frac"] * (t[-1] - t[0]))
            split_mode = "fractional"
            inter_cycle_gap_d = 0.0

        a, b = t <= t_split, t > t_split
        stats = {
            "period_proposed_d": p,
            "split_mode": split_mode,
            "split_time": round(t_split, 4),
            "inter_cycle_gap_d": inter_cycle_gap_d,
            "n_block_a": int(a.sum()),
            "n_block_b": int(b.sum()),
            "baseline_d": round(float(t[-1] - t[0]), 3),
        }
        if a.sum() < 500 or b.sum() < 500:
            stats["indeterminate_reason"] = "a block is too sparse to test"
            return stats

        dur = min(max(s["duration_frac"] * p, 0.05), s["max_duration_d"])
        stats["window_duration_d"] = round(float(dur), 4)
        t0_fit = float(t[a][0])

        def depth_at(offset, mask):
            ph = np.abs(((t[mask] - t0_fit - offset + p / 2) % p) - p / 2)
            sel = ph < dur / 2
            if sel.sum() < 5:
                return None
            return float(1.0 - np.mean(fd[mask][sel]))

        grid = np.linspace(0, p, 600, endpoint=False)
        best_a1, off1 = None, 0.0
        for off in grid:
            dv = depth_at(off, a)
            if dv is not None and (best_a1 is None or dv > best_a1):
                best_a1, off1 = dv, float(off)

        if best_a1 is None:
            stats["indeterminate_reason"] = "no primary epoch fit possible in block A"
            return stats

        stats["block_a_depth"] = best_a1
        stats["epoch_offset_d"] = off1

        pred_b1 = depth_at(off1, b)
        if pred_b1 is None:
            stats["indeterminate_reason"] = (
                "held-out block contains no cadence at the predicted primary phase"
            )
            return stats

        # Check for a potential eccentric secondary in Block A separated by >= 0.15*P
        best_a2, off2 = None, None
        for off in grid:
            if abs(((off - off1 + p / 2) % p) - p / 2) < 0.15 * p:
                continue
            dv = depth_at(off, a)
            if dv is not None and (best_a2 is None or dv > best_a2):
                best_a2, off2 = dv, float(off)

        # First compute unmasked control distribution in Block B
        raw_controls = []
        for k in range(1, s["n_controls"]):
            cand_off = (off1 + k * p / s["n_controls"]) % p
            c = depth_at(cand_off, b)
            if c is not None:
                raw_controls.append((cand_off, c))

        if len(raw_controls) < 4:
            stats["indeterminate_reason"] = "too few control phases in block B"
            return stats

        raw_vals = np.array([c for _, c in raw_controls], dtype=float)
        raw_med = float(np.median(raw_vals))
        raw_sig = float(1.4826 * np.median(np.abs(raw_vals - raw_med))) or 1e-9

        # Check whether the candidate secondary from Block A ALSO replicates in Block B
        secondary_replicated = False
        pred_b2 = depth_at(off2, b) if off2 is not None else None
        if (
            best_a2 is not None
            and pred_b2 is not None
            and best_a2 > 0.25 * best_a1
            and (pred_b2 - raw_med) / raw_sig >= s["min_secondary_sigma"]
            and pred_b2 > 0.3 * best_a2
        ):
            secondary_replicated = True
            sec_phase = float(((off2 - off1) % p) / p)
            stats.update({
                "secondary_detected": True,
                "secondary_phase": round(sec_phase, 4),
                "block_a_secondary_depth": best_a2,
                "block_b_secondary_depth": pred_b2,
                "secondary_depth_sigma": float((pred_b2 - raw_med) / raw_sig),
            })
        else:
            stats["secondary_detected"] = False

        # Exclude the secondary eclipse window from out-of-eclipse controls ONLY
        # when the secondary itself replicated in Block B.
        controls = []
        for cand_off, c in raw_controls:
            if secondary_replicated and off2 is not None:
                if abs(((cand_off - off2 + p / 2) % p) - p / 2) < 1.5 * dur:
                    continue
            controls.append(c)

        if len(controls) < 4:
            stats["indeterminate_reason"] = "too few out-of-eclipse control phases in block B"
            return stats

        cmed = float(np.median(controls))
        csig = float(1.4826 * np.median(np.abs(np.array(controls) - cmed))) or 1e-9
        stats.update({
            "block_b_predicted_depth": pred_b1,
            "block_b_control_median": cmed,
            "block_b_control_sigma": csig,
            "depth_sigma": float((pred_b1 - cmed) / csig),
            "best_control": float(np.max(controls)),
            "n_controls_used": len(controls),
        })
        return stats

    def _decide(self, stats):
        h = self.spec_sha256
        if "depth_sigma" not in stats:
            return Verdict(
                False,
                stats.get("indeterminate_reason", "no test"),
                h,
                stats,
                indeterminate=True,
            )
        s = self.spec
        checks = {
            "predicted_dip_significant": stats["depth_sigma"] >= s["min_depth_sigma"],
            "beats_all_control_phases": (
                stats["block_b_predicted_depth"]
                > s["control_margin"] * max(stats["best_control"], 1e-9)
            ),
            "consistent_with_block_a": (
                stats["block_b_predicted_depth"] > 0.3 * stats["block_a_depth"]
            ),
        }
        stats["checks"] = checks
        passed = all(checks.values())
        failed = [k for k, v in checks.items() if not v]
        sec_note = ""
        if stats.get("secondary_detected"):
            sec_note = (
                f", secondary at phase {stats['secondary_phase']:.4f} "
                f"({stats['block_b_secondary_depth']*1e3:.1f}ppt, "
                f"{stats['secondary_depth_sigma']:.1f}σ)"
            )
        return Verdict(
            bool(passed),
            f"P={stats['period_proposed_d']}d predicted a primary dip of "
            f"{stats['block_b_predicted_depth']*1e3:.2f}ppt in held-out cycle B "
            f"({stats['depth_sigma']:.1f}σ over controls{sec_note})"
            + ("" if passed else f" — FAILED: {failed}"),
            h,
            stats,
        )
