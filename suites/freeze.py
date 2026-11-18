"""Freeze suite light curves to hashed .npz files (contract clause 1).

"A frozen observational dataset. Pinned by hash and access date. No
live-drifting inputs."

Downloads TESS SPOC 2-min curves, normalizes each sector by its median, writes
`$HALLMARK_DATA_DIR/frozen/<tic>_s<sectors>.npz`, and records sha256 + access
date in `frozen/index.json`. Idempotent: anything already in the index and on
disk is never re-fetched.

The repository ships the frozen suite, so you only need this to rebuild it or
to freeze new targets. Injected-signal datasets (S3) are not fetched: they are
built deterministically at validation time from a frozen quiet host.

Run:  python -m suites.freeze
"""

import json
import sys
from datetime import datetime, timezone

from hallmark import config
from hallmark.hashing import dataset_sha256
from hallmark.mast import fetch
from suites.manifest import SUITE_VERSION, all_fetch_targets

import numpy as np


def main():
    frozen = config.frozen_dir()
    index_path = frozen / "index.json"
    frozen.mkdir(parents=True, exist_ok=True)
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    targets = all_fetch_targets()
    print(f"freeze: {len(targets)} targets, suite {SUITE_VERSION}\n")

    for n, (tic, sectors) in enumerate(targets, 1):
        key = f"{tic}_s{'-'.join(map(str, sectors))}"
        path = frozen / f"{key}.npz"
        if key in index and path.exists():
            print(f"[{n}/{len(targets)}] {key}: cached")
            continue
        print(f"[{n}/{len(targets)}] {key}: fetching...")
        try:
            t, f, got = fetch(tic, sectors)
        except Exception as e:
            print(f"    FETCH ERROR {type(e).__name__}: {str(e)[:80]}")
            continue
        if t is None or len(t) < 500:
            print("    no usable data, skipped")
            continue
        digest = dataset_sha256(time=t, flux=f)
        np.savez_compressed(path, time=t, flux=f, tic=tic, sectors=np.array(got))
        index[key] = {
            "tic": tic, "sectors_requested": sectors, "sectors_got": got,
            "n_cadences": int(len(t)),
            "baseline_days": round(float(t.max() - t.min()), 3),
            "sha256": digest,
            "accessed": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "MAST TESS SPOC 2-min",
        }
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
        print(f"    frozen: {len(t)} cadences, {index[key]['baseline_days']}d, "
              f"sha256 {digest[:12]}")

    print(f"\nfreeze complete: {len(index)} datasets in index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
