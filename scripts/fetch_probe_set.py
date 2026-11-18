"""Download and hash-check the 60-curve probe set used to score detectors.

The probe set is 20 catalogued eclipsing binaries (members of the TESS
eclipsing-binary catalogue, Prsa et al. 2022, that an earlier survey's
detector missed), stratified by period, plus 40 control stars, one TESS SPOC
2-min sector each. `data/probe_set/index.json` pins the sector and sha256 of
every curve; `labels.json` holds the eb/ctl labels, which only the scoring
step reads.

About 60 MAST downloads (~13 MB once frozen). Each curve is re-hashed after
download. A mismatch is reported, not silently accepted: MAST reprocessing or
a different fetch path (lightkurve vs direct) can change the bytes, and a
result computed on different bytes is a different result.

Run:  python scripts/fetch_probe_set.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark import config                 # noqa: E402
from hallmark.hashing import dataset_sha256  # noqa: E402
from hallmark.mast import fetch              # noqa: E402


def main():
    probe = config.probe_dir()
    index = json.loads((probe / "index.json").read_text())
    ok = mismatch = failed = 0
    for n, (tic, meta) in enumerate(sorted(index.items()), 1):
        path = probe / f"{tic}.npz"
        tag = f"[{n}/{len(index)}] TIC {tic} s{meta['sector']}"
        if path.exists():
            d = np.load(path)
            h = dataset_sha256(time=d["time"], flux=d["flux"])
            print(f"{tag}: cached ({'hash ok' if h == meta['sha256'] else 'HASH MISMATCH'})")
            ok += h == meta["sha256"]
            mismatch += h != meta["sha256"]
            continue
        try:
            t, f, _ = fetch(int(tic), [meta["sector"]])
        except Exception as e:                       # noqa: BLE001
            print(f"{tag}: FETCH ERROR {type(e).__name__}: {str(e)[:70]}")
            failed += 1
            continue
        if t is None:
            print(f"{tag}: no usable data")
            failed += 1
            continue
        np.savez_compressed(path, time=t, flux=f)
        h = dataset_sha256(time=t, flux=f)
        if h == meta["sha256"]:
            ok += 1
            print(f"{tag}: frozen, hash ok")
        else:
            mismatch += 1
            print(f"{tag}: frozen, HASH MISMATCH (got {h[:12]}, "
                  f"pinned {meta['sha256'][:12]})")
    print(f"\n{ok} verified, {mismatch} hash mismatches, {failed} failed "
          f"of {len(index)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
