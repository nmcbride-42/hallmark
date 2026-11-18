"""Catalogue novelty boundary — contract clause 3.

"A declared novelty boundary: absent from WHICH catalogue, queried WHEN, with
what cone/tolerance. 'Novel' with no named catalogue is meaningless."

This is not a verifier in the Verdict sense — it produces the novelty record
that clause 3 consumes, and it is deliberately re-run immediately before any
deposition because catalogues move (A-Lab's death mode: novelty checked
against a stale representation instead of the world).
"""

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

VSX_LIST = "https://vsx.aavso.org/index.php?view=api.list&format=json"
UA = "hallmark-novelty/1.0 (variable-star novelty boundary check)"


def vsx_cone(ra_deg, dec_deg, radius_arcsec=60.0, timeout=30):
    """Cone search VSX. Returns (hits, records, error)."""
    radius_deg = radius_arcsec / 3600.0
    url = (f"{VSX_LIST}&ra={float(ra_deg):.6f}&dec={float(dec_deg):.6f}"
           f"&radius={radius_deg:.6f}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except Exception as e:
        return None, [], f"{type(e).__name__}: {str(e)[:80]}"
    objs = data.get("VSXObjects", {}) if data else {}
    recs = objs.get("VSXObject", []) if isinstance(objs, dict) else []
    if isinstance(recs, dict):
        recs = [recs]
    return len(recs), recs, None


def novelty_record(ra_deg, dec_deg, radius_arcsec=60.0):
    """Build the clause-3 record. A failed query returns None hits, which
    clause 3 treats as incomplete -> fail-closed. Never assume absence from a
    network error: 'we could not check' is not 'it is not there'."""
    hits, recs, err = vsx_cone(ra_deg, dec_deg, radius_arcsec)
    rec = {
        "catalogue": "AAVSO VSX",
        "queried_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cone_arcsec": radius_arcsec,
        "ra_deg": ra_deg, "dec_deg": dec_deg,
    }
    if err is not None:
        rec["error"] = err
        return rec           # no 'hits' key -> clause 3 reports incomplete
    rec["hits"] = hits
    rec["names"] = [r.get("Name") for r in recs][:5]
    rec["oids"] = [r.get("OID") for r in recs][:5]
    rec["discoverers"] = [r.get("Discoverer") for r in recs][:5]
    return rec
