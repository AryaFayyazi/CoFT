#!/usr/bin/env python3
"""Figures 3 and 4 -- validation sweeps over the fusion scale and the risk level.

Both sweeps follow the same protocol (Sec. 4.5): sweep the knob on a small
*validation* split, trace the ``(BiasAvg v, UtilityAvg ^)`` Pareto curve, and
select the **smallest** value within 2% of the knee.  Given the chosen alpha, the
per-position thresholds ``tau_t = 1 - q_t`` are computed offline on a disjoint
calibration set, with no test-time tuning.

The two branch logits do not depend on lambda or alpha, so one calibration pass
covers every point of both sweeps (see :mod:`coft.calibration`).

Also records, for each alpha, the *empirical* miscoverage and the normalised
certified-set size ``E|C_t| / V`` -- the diagnostic columns of Table 20 and the
quantities plotted in Fig. 7(c-d).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import base_parser, build_masker, results_dir, save_json, set_seed, setup
from run_ablation import bias_avg
from run_bias import evaluate_method, flatten, load_bias_datasets

from coft import evaluate as E
from coft.calibration import collect_calibration_scores
from coft.data.corpora import load_calibration_corpus  # noqa: F401  (pooled fallback)
from coft.data.tasks import load_arc_easy, load_piqa, load_strategyqa
from coft.registry import build_decoder
from coft.toxicity import ToxicityScorer

DEFAULT_LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0]
DEFAULT_ALPHAS = [0.02, 0.05, 0.10, 0.15, 0.20]


def pareto_knee_choice(points: List[Dict], key: str, tol: float = 0.02) -> float:
    """Smallest knob value whose BiasAvg is within ``tol`` (relative) of the best.

    This is the paper's "pick the smallest value within 2% of the knee" rule.
    """
    valid = [p for p in points if p["bias_avg"] == p["bias_avg"]]
    if not valid:
        return float("nan")
    best = min(p["bias_avg"] for p in valid)
    threshold = best * (1.0 + tol) if best > 0 else best + tol
    eligible = [p for p in valid if p["bias_avg"] <= threshold]
    return min(p[key] for p in eligible)


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--sweep", choices=["lambda", "alpha", "both"], default="both")
    ap.add_argument("--lambdas", type=float, nargs="*", default=DEFAULT_LAMBDAS)
    ap.add_argument("--alphas", type=float, nargs="*", default=DEFAULT_ALPHAS)
    ap.add_argument("--val-fraction", type=float, default=0.35,
                    help="fraction of the loaded bias items used as the validation split")
    args = ap.parse_args()

    cfg, lm = setup(args)
    out_dir = results_dir(cfg, args)
    progress = not args.no_progress
    masker = build_masker(lm, cfg)
    ccfg = cfg["conformal"]
    seed = cfg.get("seeds", [0])[0]
    set_seed(seed)
    bs = cfg["data"]["batch_size"]
    lim = cfg["data"]["utility"]

    # ---- validation data (a different seed from the Table-1 evaluation split) ----
    print("loading validation split ...")
    bias_data, cal_data = load_bias_datasets(cfg, seed=seed + 1000)
    for k in list(bias_data):
        n = max(2, int(len(bias_data[k]) * args.val_fraction))
        bias_data[k] = bias_data[k][:n]
        print(f"  {k}: {len(bias_data[k])} validation items")
    util_data = {
        "strategyqa": load_strategyqa(limit=max(8, lim["strategyqa"] // 3), seed=seed + 1000),
        "arc_easy": load_arc_easy(limit=max(8, lim["arc_easy"] // 3), seed=seed + 1000),
        "piqa": load_piqa(limit=max(8, lim["piqa"] // 3), seed=seed + 1000),
    }
    tox = ToxicityScorer(cfg["toxicity"]["model_id"]) if "bold" in bias_data else None

    # ---- one calibration pass per dataset covering every lambda at once ----
    # The branch logits do not depend on lambda or alpha, so a single pass over
    # each calibration slice serves every point of both sweeps.
    lams = sorted(set(args.lambdas) | {cfg["methods"]["coft"]["lam"]})
    n_ctx = ccfg.get("n_calibration_contexts", 400)
    bundles = {}
    for ds_name, corpus in cal_data.items():
        if ds_name not in bias_data or not corpus:
            continue
        corpus = corpus[:n_ctx]
        print(f"calibrating [{ds_name}] for lambdas={lams} on {len(corpus)} contexts ...")
        bundles[ds_name] = collect_calibration_scores(
            lm, corpus, lams=lams, score="dual", masker=masker,
            temperature=cfg["decoding"]["temperature"],
            batch_size=bs, bin_width=ccfg["bin_width"], max_position=ccfg["max_position"],
            max_continuation_tokens=ccfg["max_continuation_tokens"], progress=progress,
        )
    pooled_bundle = collect_calibration_scores(
        lm, [t for c in cal_data.values() for t in c][:n_ctx], lams=lams, score="dual",
        masker=masker, temperature=cfg["decoding"]["temperature"], batch_size=bs,
        bin_width=ccfg["bin_width"], max_position=ccfg["max_position"],
        max_continuation_tokens=ccfg["max_continuation_tokens"], progress=progress,
    )

    def evaluate_point(lam: float, alpha: float) -> Dict:
        def make_decoder(ds_name: str):
            th = bundles[ds_name].thresholds(lam, alpha)
            d = build_decoder("coft", lm, cfg, thresholds=th, masker=masker, overrides={"lam": lam})
            d.seed = seed
            return d

        pooled_th = pooled_bundle.thresholds(lam, alpha)
        dec = build_decoder(
            "coft", lm, cfg, thresholds=pooled_th, masker=masker, overrides={"lam": lam}
        )
        dec.seed = seed
        per_ds = evaluate_method(make_decoder, bias_data, cfg, tox, progress, seed)
        row = flatten(per_ds)
        accs = []
        for key in ("strategyqa", "arc_easy", "piqa"):
            rr = E.eval_choices(dec, util_data[key], key, bs, progress=progress)
            rr.pop("predictions", None)
            accs.append(rr["acc"])
        cov = [
            per_ds[d]["empirical_coverage"]
            for d in per_ds
            if "empirical_coverage" in per_ds[d] and per_ds[d]["empirical_coverage"] == per_ds[d]["empirical_coverage"]
        ]
        set_sizes = [per_ds[d]["mean_certified_set"] for d in per_ds if "mean_certified_set" in per_ds[d]]
        return {
            "lambda": lam,
            "alpha": alpha,
            "bias_avg": bias_avg(row),
            "utility_avg": sum(accs) / len(accs),
            "bias_row": row,
            "tau0": pooled_th.tau(0),
            "empirical_coverage": (sum(cov) / len(cov)) if cov else float("nan"),
            "empirical_miscoverage": (1.0 - sum(cov) / len(cov)) if cov else float("nan"),
            "normalized_set_size": (sum(set_sizes) / len(set_sizes) / lm.vocab_size) if set_sizes else float("nan"),
        }

    payload: Dict = {"model": cfg["model"], "seed": seed}

    if args.sweep in {"lambda", "both"}:
        alpha0 = ccfg["alpha"]
        print(f"\n=== lambda sweep (alpha fixed at {alpha0}) ===")
        pts = []
        for lam in args.lambdas:
            print(f"  lambda = {lam}")
            pts.append(evaluate_point(lam, alpha0))
        payload["lambda_sweep"] = {
            "alpha": alpha0,
            "points": pts,
            "chosen": pareto_knee_choice(pts, "lambda"),
        }
        print(f"  -> chosen lambda = {payload['lambda_sweep']['chosen']}")

    if args.sweep in {"alpha", "both"}:
        lam0 = cfg["methods"]["coft"]["lam"]
        print(f"\n=== alpha sweep (lambda fixed at {lam0}) ===")
        pts = []
        for alpha in args.alphas:
            print(f"  alpha = {alpha}")
            pts.append(evaluate_point(lam0, alpha))
        payload["alpha_sweep"] = {
            "lambda": lam0,
            "points": pts,
            "chosen": pareto_knee_choice(pts, "alpha"),
        }
        print(f"  -> chosen alpha = {payload['alpha_sweep']['chosen']}")

    save_json(out_dir / "sweeps.json", payload)

    for name in ("lambda_sweep", "alpha_sweep"):
        if name not in payload:
            continue
        knob = "lambda" if name == "lambda_sweep" else "alpha"
        print(f"\n--- {name} ---")
        header = f"{knob:>8}{'BiasAvg':>10}{'UtilityAvg':>12}"
        if knob == "alpha":
            header += f"{'miscov':>9}{'|C_t|/V':>10}"
        print(header)
        for p in payload[name]["points"]:
            line = f"{p[knob]:>8.2f}{p['bias_avg']:>10.3f}{p['utility_avg']:>12.1f}"
            if knob == "alpha":
                line += f"{p['empirical_miscoverage']:>9.3f}{p['normalized_set_size']:>10.4f}"
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
