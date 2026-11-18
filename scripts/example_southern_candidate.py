"""Worked example: one proposed ephemeris, one predeclared verifier, one verdict.

The candidate is TIC 220481009, a southern (Dec -58.3 deg) detached eclipsing
binary candidate with a period near 52.1 d. A single TESS visit (~76 d) holds at
most two primary eclipses, which cannot over-determine an ephemeris, so the
plain held-out verifier correctly returns INDETERMINATE. Ground surveys in the
north (e.g. ZTF) cannot see it.

TESS observed the field again about 110 days later. The southern-archival
verifier splits the frozen 6-sector archive at that inter-cycle gap, fits only
the epoch on cycle A using the declared period, and asks whether held-out
cycle B dims at the predicted phase and not at the control phases it never
committed to. It never re-fits the period on cycle B.

The default period is the suite manifest's 52.1 d, the value the validation
run uses. A 6-decimal value (52.144497 d) was later refined for the catalogue
submission on the combined multi-cycle data, so cycle B is NOT held out
from it, and the higher significance it gets here is partly circular. Pass it with --period to
see that, but do not read it as held-out evidence.

What a pass shows, and what it does not. The verifier asks whether dips recur
in cycle B where the declared ephemeris predicts them. With two primary
eclipses per visit, other periods can also line up with real eclipses: on the
shipped archive it also passes at 2P (104.2 d) and at some fractional aliases
(P/2 and P/5 of the refined period). The script prints these checks. So a
pass shows recurrence on an ephemeris compatible with the declared period, not
that the declared period is the true one; P versus 2P or P/2 is not resolved
by this verifier and needs further eclipse timings.

This shows the verifier's verdict. It is NOT a confirmed discovery: a
submission to the AAVSO VSX catalogue based on this analysis is awaiting
human moderation, and only that external body can confirm it.

Run:  python scripts/example_southern_candidate.py
      python scripts/example_southern_candidate.py --plot docs/figures/tic220481009_multicycle.png
      python scripts/example_southern_candidate.py --period 37.0    # a wrong period
      python scripts/example_southern_candidate.py --period 52.144497  # refined, not held out
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark import config                                         # noqa: E402
from hallmark.hashing import dataset_sha256                         # noqa: E402
from hallmark.verifiers.ephemeris import HeldOutEphemeris           # noqa: E402
from hallmark.verifiers.southern_archival import SouthernArchivalEphemeris  # noqa: E402

DATASET = "220481009_s87-88-89-94-95-96"
SINGLE_VISIT = "220481009_s94-95-96"
PERIOD_D = 52.1            # suite manifest value (suites/manifest.py)
REFINED_PERIOD_D = 52.144497   # refined on combined multi-cycle data: not held out


def plot(t, f, period, epoch, split_time, stats, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ph = ((t - epoch) / period) % 1.0
    ph_w = np.where(ph > 0.75, ph - 1.0, ph)
    a, b = t <= split_time, t > split_time
    sec = stats.get("secondary_phase")

    fig, axes = plt.subplots(1, 3 if sec else 2, figsize=(13.5 if sec else 9, 4.2))
    ax = axes[0]
    ax.plot(ph_w[a], f[a], ".", ms=1.5, color="#1f77b4", alpha=0.35, label="cycle A (fit epoch)")
    ax.plot(ph_w[b], f[b], ".", ms=1.5, color="#d62728", alpha=0.35, label="cycle B (held out)")
    ax.set_xlim(-0.15, 0.70)
    ax.set_xlabel("orbital phase")
    ax.set_ylabel("normalized TESS flux")
    ax.set_title(f"phase fold, {stats['baseline_d']:.0f} d span")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)

    def zoom(ax, dph, title, sigma):
        m = np.abs(dph) < 0.018
        ax.plot(dph[a & m] * period * 24, f[a & m], ".", ms=3, color="#1f77b4", alpha=0.65,
                label="cycle A")
        ax.plot(dph[b & m] * period * 24, f[b & m], ".", ms=3, color="#d62728", alpha=0.65,
                label=f"cycle B ({sigma:.0f}σ over controls)")
        ax.set_xlabel("hours from predicted minimum")
        ax.set_title(title)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.2)

    zoom(axes[1], ph_w, "primary eclipse", stats["depth_sigma"])
    if sec:
        dph = ((ph - sec + 0.5) % 1.0) - 0.5
        zoom(axes[2], dph, f"secondary eclipse (phase {sec:.4f})",
             stats["secondary_depth_sigma"])
    fig.suptitle(f"Held-out replication at P = {period} d "
                 "(verifier verdict, not a confirmed discovery)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--period", type=float, default=PERIOD_D)
    ap.add_argument("--plot", type=Path, default=None,
                    help="write a diagnostic figure (needs matplotlib)")
    args = ap.parse_args(argv)

    frozen = config.frozen_dir()
    index = json.loads((frozen / "index.json").read_text())

    single = HeldOutEphemeris().verify(frozen / f"{SINGLE_VISIT}.npz")
    print(f"single visit ({SINGLE_VISIT}), held-out-ephemeris:")
    print(f"  passed={single.passed} indeterminate={single.indeterminate}: {single.detail}\n")

    path = frozen / f"{DATASET}.npz"
    d = np.load(path)
    t, f = np.asarray(d["time"], float), np.asarray(d["flux"], float)
    sha = dataset_sha256(time=t, flux=f)
    pinned = index[DATASET]["sha256"]
    print(f"dataset {DATASET}: sha256 {sha[:16]} "
          f"({'matches pin' if sha == pinned else 'DOES NOT MATCH PIN'})")

    h = SouthernArchivalEphemeris()
    print(f"verifier {h.name} spec {h.spec_sha256[:16]}")
    v = h.verify(path, proposal={"period_d": args.period})
    print(f"\nproposal P = {args.period} d")
    print(f"  passed={v.passed} indeterminate={v.indeterminate}")
    print(f"  {v.detail}")
    for k in ("split_mode", "inter_cycle_gap_d", "n_block_a", "n_block_b",
              "n_controls_used", "checks"):
        if k in v.stats:
            print(f"  {k}: {v.stats[k]}")

    print("\nalias checks (same verifier, same data; a pass here is not a"
          " contradiction, it is the verifier's known limit):")
    for label, p_alias in (("2P", 2 * args.period), ("P/2", args.period / 2),
                           ("2P, refined", 2 * REFINED_PERIOD_D),
                           ("P/2, refined", REFINED_PERIOD_D / 2),
                           ("P/5, refined", REFINED_PERIOD_D / 5)):
        va = h.verify(path, proposal={"period_d": p_alias})
        sig = va.stats.get("depth_sigma")
        sig_s = f"{sig:.0f} sigma" if isinstance(sig, (int, float)) else "n/a"
        print(f"  {label:<13s} P = {p_alias:9.4f} d  passed={va.passed}  ({sig_s})")

    if args.plot and "epoch_offset_d" in v.stats:
        order = np.argsort(t)
        t, f = t[order], f[order]
        epoch = float(t[t <= v.stats["split_time"]][0]) + v.stats["epoch_offset_d"]
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(t, f, args.period, epoch, v.stats["split_time"], v.stats, args.plot)
        print(f"\nfigure: {args.plot}")
    return 0 if v.passed else 1


if __name__ == "__main__":
    sys.exit(main())
