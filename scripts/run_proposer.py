"""Proposer pre-test: an LLM writes detectors; a predeclared rule judges them.

    1. The success rule is fixed here, before any model call (see SuccessRule).
    2. The model gets the task in aggregate (no targets, no labels) and returns
       `score(time, flux)` transforms.
    3. Each transform runs in a restricted namespace over every probe curve.
    4. Labels are applied only after all scores exist, and the rule decides.

The baseline a proposal must beat is the current absorbed detector library
(`hallmark.detectors`), which already contains the best output of earlier
proposers. Beating it is meant to be hard.

Proposals come from one of:
  --proposals-file PATH   a JSON file of proposals (offline replay), e.g.
                          examples/first_round_proposals.json
  --provider anthropic    call the Anthropic API (needs `pip install anthropic`
                          and ANTHROPIC_API_KEY; model from HALLMARK_LLM_MODEL)

The sandbox is a guard against accidents, not a security boundary. Run
proposals from a model you do not control inside a container.

Run:  python scripts/run_proposer.py --proposals-file examples/first_round_proposals.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark import config                                         # noqa: E402
from hallmark.detectors import TRANSFORMS, combine_scores            # noqa: E402
from hallmark.inventor import (SuccessRule, build_prompt, evaluate,  # noqa: E402
                               extract_proposals)

# Predeclared. Do not edit after looking at a result.
RULE = SuccessRule(top_k=10, min_hits=7, must_beat_baseline=True)
N_PROPOSALS = 3
DEFAULT_MODEL = "claude-opus-5-5"


def propose_anthropic(prompt: str) -> str:
    """One Messages API call. The model's text is untrusted data: it is parsed
    and screened, never executed outside the restricted namespace."""
    import anthropic

    client = anthropic.Anthropic()
    model = os.environ.get("HALLMARK_LLM_MODEL", DEFAULT_MODEL)
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    return "".join(b.text for b in resp.content if b.type == "text")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--proposals-file", type=Path)
    src.add_argument("--provider", choices=["anthropic"])
    ap.add_argument("--probe-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the verdict JSON here")
    args = ap.parse_args(argv)

    probe_dir = args.probe_dir or config.probe_dir()
    labels = json.loads((probe_dir / "labels.json").read_text())
    curves = {}
    for tic in labels:
        p = probe_dir / f"{tic}.npz"
        if p.exists():
            d = np.load(p)
            curves[tic] = (d["time"], d["flux"])
    if not curves:
        print("probe set not on disk; run scripts/fetch_probe_set.py first")
        return 1
    print(f"probe set: {len(curves)} curves; rule: {RULE}")

    if args.proposals_file:
        proposals = extract_proposals(args.proposals_file.read_text())
    else:
        proposals = extract_proposals(propose_anthropic(build_prompt(N_PROPOSALS)))
    proposals = proposals[:N_PROPOSALS]
    if not proposals:
        print("no proposals parsed; INDETERMINATE (nothing to judge)")
        return 1

    print("scoring baseline (absorbed detector library)...")
    per = {}
    for fn in TRANSFORMS:
        vals = {}
        for t, (tt, ff) in curves.items():
            try:
                vals[t] = float(fn(tt, ff))
            except Exception:                                    # noqa: BLE001
                vals[t] = float("nan")
        per[fn.__name__] = vals
    baseline = combine_scores(per)

    print(f"scoring {len(proposals)} proposal(s)...")
    result = evaluate(proposals, curves, labels, baseline, RULE)

    b = result["baseline"]
    print(f"\nbaseline: {b['hits']}/{RULE.top_k} in top-{RULE.top_k} "
          f"(p={b['p']:.3g})")
    for r in result["proposals"]:
        if r.get("disqualified"):
            print(f"  {r['name']:<24s} DISQUALIFIED ({r['n_scored']} scored)")
            continue
        print(f"  {r['name']:<24s} {r['hits']}/{RULE.top_k} (p={r['p']:.3g})  "
              f"beats baseline: {r['beats_baseline']}  -> "
              f"{'SUCCESS' if r['success'] else 'no'}")
    verdict = ("PROPOSER ADDS RECALL: review and absorb the winner"
               if result["any_success"] else
               "NO PROPOSAL MET THE PREDECLARED RULE")
    result["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
