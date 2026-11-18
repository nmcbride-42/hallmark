"""Fetch TESS SPOC 2-minute light curves from MAST.

Two paths, same normalization (each sector divided by its own median flux):

* `lightkurve` when it is installed (the path the shipped frozen suite was
  built with), and
* a dependency-free direct MAST download with a minimal FITS table reader,
  applying lightkurve's default quality bitmask.

The two paths are not guaranteed to produce byte-identical arrays, so a
dataset's sha256 is tied to the path that froze it. `data/frozen/index.json`
records the source for every shipped dataset.
"""

import json
import shutil
import urllib.parse
import urllib.request

import numpy as np

from . import config

UA = "hallmark/0.1 (TESS light-curve freezer)"
MAST_INVOKE = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

# lightkurve TessQualityFlags.DEFAULT_BITMASK
DEFAULT_QUALITY_BITMASK = 7919


def parse_tess_lc_fits(raw: bytes):
    """Minimal FITS binary-table reader for a TESS SPOC `_lc.fits` (HDU 1).

    Returns (TIME, PDCSAP_FLUX, QUALITY)."""
    pos = 0
    while True:                                  # skip the primary header
        block = raw[pos:pos + 2880].decode("ascii", errors="replace")
        pos += 2880
        if "END     " in block:
            break
    hdr_lines = []
    while True:
        block = raw[pos:pos + 2880].decode("ascii", errors="replace")
        pos += 2880
        for i in range(0, 2880, 80):
            hdr_lines.append(block[i:i + 80])
        if any(block[i:i + 8] == "END     " for i in range(0, 2880, 80)):
            break
    hdr = {}
    for card in hdr_lines:
        if "=" in card[:10]:
            hdr[card[:8].strip()] = card[10:].split("/")[0].strip().strip("' ")
    row_bytes, n_rows = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    tfields = int(hdr["TFIELDS"])
    fmt_map = {"D": (">f8", 8), "E": (">f4", 4), "J": (">i4", 4),
               "I": (">i2", 2), "B": ("u1", 1), "K": (">i8", 8)}
    cols, offset = {}, 0
    for c in range(1, tfields + 1):
        ttype, tform = hdr.get(f"TTYPE{c}", ""), hdr.get(f"TFORM{c}", "")
        rep = int("".join(ch for ch in tform if ch.isdigit()) or "1")
        code = "".join(ch for ch in tform if ch.isalpha())
        dt, sz = fmt_map.get(code, ("u1", 1))
        cols[ttype] = (offset, dt, sz * rep)
        offset += sz * rep
    arr = np.frombuffer(raw[pos:pos + row_bytes * n_rows],
                        dtype=np.uint8).reshape(n_rows, row_bytes)

    def get_col(name):
        off, dt, sz = cols[name]
        return arr[:, off:off + sz].copy().view(dt).ravel()

    return get_col("TIME"), get_col("PDCSAP_FLUX"), get_col("QUALITY")


def fetch_mast_direct(tic, sectors, timeout=60):
    """Direct MAST HTTPS fetch. Returns (time, flux, sectors_got)."""
    req_data = {
        "service": "Mast.Caom.Filtered",
        "format": "json",
        "params": {
            "columns": "obs_id,target_name,sequence_number,dataURL",
            "filters": [
                {"paramName": "target_name", "values": [str(tic)]},
                {"paramName": "obs_collection", "values": ["TESS"]},
                {"paramName": "sequence_number",
                 "values": [str(s) for s in sectors]},
            ],
        },
    }
    req = urllib.request.Request(
        MAST_INVOKE,
        data=urllib.parse.urlencode({"request": json.dumps(req_data)}).encode(),
        headers={"User-Agent": UA},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        rows = json.load(r).get("data", [])
    lc_rows = sorted(
        [r for r in rows if str(r.get("dataURL", "")).endswith("_lc.fits")],
        key=lambda x: int(x["sequence_number"]),
    )
    times, fluxes, got = [], [], []
    for row in lc_rows:
        sec = int(row["sequence_number"])
        if sec not in sectors:
            continue
        url = MAST_DOWNLOAD + row["dataURL"]
        with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": UA}),
                timeout=timeout) as r:
            raw = r.read()
        t_raw, f_raw, q_raw = parse_tess_lc_fits(raw)
        mask = (((q_raw & DEFAULT_QUALITY_BITMASK) == 0)
                & np.isfinite(t_raw) & np.isfinite(f_raw))
        t = t_raw[mask].astype(float)
        f = f_raw[mask].astype(float)
        med = np.median(f)
        if not np.isfinite(med) or med == 0:
            continue
        times.append(t)
        fluxes.append(f / med)
        got.append(sec)
    if not times:
        return None, None, []
    t_all = np.concatenate(times)
    f_all = np.concatenate(fluxes)
    order = np.argsort(t_all)
    return t_all[order], f_all[order], sorted(set(got))


def fetch(tic, sectors):
    """Download + normalize the requested sectors. Returns (time, flux, got).

    Uses lightkurve when available, else the direct MAST path."""
    try:
        import lightkurve as lk
    except ImportError:
        return fetch_mast_direct(tic, sectors)
    cache = config.cache_dir()
    sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC",
                              exptime=120)
    times, fluxes, got = [], [], []
    for i in range(len(sr)):
        try:
            sec = int(sr[i].mission[0].split()[-1])
        except Exception:
            sec = None
        if sec is not None and sec not in sectors:
            continue
        try:
            lc = sr[i].download(download_dir=str(cache)).remove_nans()
            f = np.asarray(lc.flux.value, dtype=float)
            med = np.median(f)
            if not np.isfinite(med) or med == 0:
                continue
            times.append(np.asarray(lc.time.value, dtype=float))
            fluxes.append(f / med)
            got.append(int(lc.sector))
        except Exception as e:
            print(f"    sector download failed: {type(e).__name__}: {str(e)[:60]}")
    shutil.rmtree(cache / "mastDownload", ignore_errors=True)
    if not times:
        return None, None, []
    t = np.concatenate(times)
    f = np.concatenate(fluxes)
    order = np.argsort(t)
    return t[order], f[order], sorted(set(got))
