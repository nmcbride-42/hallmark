"""Rank the probe set with each absorbed detector and score the ranking.

Reports, per detector: catalogued EBs in the top-10 and top-20 of the 60-curve
probe set, with the exact hypergeometric p-value of doing at least that well
by chance (20 positives among 60). Labels are read only after scoring.

Needs the probe set on disk (scripts/fetch_probe_set.py). Point --probe-dir
elsewhere to score a different labelled set with the same layout.

Run:  python scripts/score_detectors.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark import config                                   # noqa: E402
from hallmark.detectors import (BASELINE_VERSION, TRANSFORMS,  # noqa: E402
                                combine_scores)
from hallmark.inventor import hypergeom_p_at_least, rank_hits  # noqa: E402


def load_curves(probe_dir):
    labels = json.loads((probe_dir / "labels.json").read_text())
    curves = {}
    for tic in labels:
        p = probe_dir / f"{tic}.npz"
        if p.exists():
            d = np.load(p)
            curves[tic] = (d["time"], d["flux"])
    return curves, labels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    probe_dir = args.probe_dir or config.probe_dir()
    curves, labels = load_curves(probe_dir)
    if len(curves) < len(labels):
        print(f"only {len(curves)}/{len(labels)} curves on disk; "
              "run scripts/fetch_probe_set.py first")
        if not curves:
            return 1
    ids = list(curves)
    n_pos = sum(1 for t in ids if labels[t] == "eb")
    print(f"detector library {BASELINE_VERSION}: {len(ids)} curves, "
          f"{n_pos} catalogued EBs\n")
    print(f"  {'detector':<26s} {'top10':>5s} {'p':>9s} {'top20':>5s} {'p':>9s}")

    def report(name, scores):
        h10, _ = rank_hits(scores, labels, 10)
        h20, _ = rank_hits(scores, labels, 20)
        print(f"  {name:<26s} {h10:>5d} {hypergeom_p_at_least(h10, len(ids), n_pos, 10):>9.2g}"
              f" {h20:>5d} {hypergeom_p_at_least(h20, len(ids), n_pos, 20):>9.2g}")

    per = {}
    for fn in TRANSFORMS:
        scores = {}
        for t in ids:
            try:
                scores[t] = float(fn(*curves[t]))
            except Exception:                                  # noqa: BLE001
                scores[t] = float("nan")
        per[fn.__name__] = scores
        report(fn.__name__, {k: (v if np.isfinite(v) else float("-inf"))
                             for k, v in scores.items()})
    report("combined (max z-score)", combine_scores(per))
    return 0


if __name__ == "__main__":
    sys.exit(main())
