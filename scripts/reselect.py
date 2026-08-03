#!/usr/bin/env python3
"""Recompute the Pareto-knee selection from an existing ``sweeps.json``.

The sweep points are expensive to produce; the *selection rule* applied to them
is not.  This re-applies the rule in place, so a change to the rule does not
require re-running the sweep.

    python scripts/reselect.py                 # every model under results/
    python scripts/reselect.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sweep import DEGENERATE_LAMBDAS, pareto_knee, pareto_knee_choice  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.results).glob("*/sweeps.json"))
    if not files:
        print(f"no sweeps.json under {args.results}")
        return 1

    for f in files:
        payload = json.loads(f.read_text())
        model = payload.get("model", {}).get("key", f.parent.name)
        changed = False
        for name, key in (("lambda_sweep", "lambda"), ("alpha_sweep", "alpha")):
            block = payload.get(name)
            if not block:
                continue
            pts = block["points"]
            exclude = DEGENERATE_LAMBDAS if key == "lambda" else ()
            usable = [q for q in pts if not any(abs(q[key] - e) < 1e-9 for e in exclude)]
            knee = pareto_knee(usable)
            new = pareto_knee_choice(pts, key, exclude=exclude)
            old = block.get("chosen")
            block["chosen"] = new
            block["knee"] = {key: knee[key], "bias_avg": knee["bias_avg"],
                             "utility_avg": knee["utility_avg"]} if knee else None
            changed |= (old != new)
            knee_str = f"{knee[key]:g}" if knee else "undefined"
            print(f"{model:22s} {key:7s} knee={knee_str:>9s}  chosen {old} -> {new}")
        if changed and not args.dry_run:
            f.write_text(json.dumps(payload, indent=2, default=str))
            print(f"  updated {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
