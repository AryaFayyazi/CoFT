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
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (
    base_parser,
    build_masker,
    load_utility_datasets,
    results_dir,
    save_json,
    set_seed,
    setup,
)
from run_ablation import bias_avg
from run_bias import evaluate_method, flatten, load_bias_datasets

from coft import evaluate as E
from coft.calibration import collect_calibration_scores
from coft.data.corpora import load_calibration_corpus  # noqa: F401  (pooled fallback)
from coft.data.tasks import GSM8K_STOP
from coft.registry import build_decoder, merge_configs
from coft.toxicity import ToxicityScorer

# Resolution concentrated below 0.5: the measured trade-off turns there
# (bias keeps falling with lambda while GSM8K accuracy drops sharply past
# ~0.3), so that is where the knee lives and where the grid needs detail.
DEFAULT_LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
DEFAULT_ALPHAS = [0.02, 0.05, 0.10, 0.15, 0.20]


def pareto_knee(points: List[Dict]) -> Optional[Dict]:
    """The knee of the ``(BiasAvg v, UtilityAvg ^)`` curve, or ``None`` if undefined.

    The knee is the point of best trade-off, not the extreme: on the normalised
    Pareto frontier it is the point lying furthest *above* the chord joining the
    two end points.  Selecting on bias alone would always return the largest
    ``lambda`` on a monotone bias curve -- which is ``lambda = 1``, i.e. sampling
    purely from the masked branch, discarding the factual view entirely.

    Returns ``None`` when utility does not vary (no trade-off to balance), so the
    caller can fall back to "best bias at no utility cost".
    """
    valid = [
        p for p in points
        if p["bias_avg"] == p["bias_avg"] and p["utility_avg"] == p["utility_avg"]
    ]
    if len(valid) < 3:
        return None
    b = [p["bias_avg"] for p in valid]
    u = [p["utility_avg"] for p in valid]
    b_lo, b_hi = min(b), max(b)
    u_lo, u_hi = min(u), max(u)
    if b_hi - b_lo <= 1e-12 or u_hi - u_lo <= 1e-9:
        return None
    bn = [(x - b_lo) / (b_hi - b_lo) for x in b]      # 0 = least bias
    un = [(y - u_lo) / (u_hi - u_lo) for y in u]      # 1 = most utility

    # Pareto frontier: keep points nothing else beats on both objectives
    front = [
        i for i in range(len(valid))
        if not any(
            bn[j] <= bn[i] and un[j] >= un[i] and (bn[j] < bn[i] or un[j] > un[i])
            for j in range(len(valid))
        )
    ]
    if len(front) < 3:
        return None
    front.sort(key=lambda i: bn[i])
    x1, y1 = bn[front[0]], un[front[0]]
    x2, y2 = bn[front[-1]], un[front[-1]]
    if abs(x2 - x1) <= 1e-12:
        return None

    best_i, best_gain = None, 0.0
    for i in front[1:-1]:
        chord = y1 + (y2 - y1) * (bn[i] - x1) / (x2 - x1)
        gain = un[i] - chord            # positive == better than a linear trade-off
        if gain > best_gain:
            best_i, best_gain = i, gain
    return valid[best_i] if best_i is not None else None


#: ``lambda = 1`` is excluded from *selection* (it is still swept and plotted, as
#: Fig. 3 does).  At that value the fused distribution is exactly the masked one,
#: so the factual view is discarded altogether and the dual-branch score
#: ``min(pi_hat, pi^CF)`` degenerates to ``pi^CF`` -- Stage II stops being a
#: fusion and Stage III stops being dual-branch.  Whatever that configuration
#: scores, it is not the method the paper describes, so it must not be selected
#: as "COFT".  Every lambda* in the paper's Table 18 is interior (0.55-0.60).
DEGENERATE_LAMBDAS = (1.0,)


def pareto_knee_choice(
    points: List[Dict], key: str, tol: float = 0.02, exclude: Optional[Sequence[float]] = None
) -> float:
    """The paper's rule: "pick the smallest value within 2% of the knee" (Sec. 4.5).

    The knee anchors the choice; the tolerance then prefers the *smallest* knob
    setting that is no worse than it, since a smaller ``lambda`` intervenes less
    and a smaller ``alpha`` certifies more tightly.

    ``exclude`` drops degenerate operating points from the candidate set.  It
    defaults to :data:`DEGENERATE_LAMBDAS` when selecting ``lambda`` -- the safe
    behaviour, so a caller cannot accidentally select ``lambda = 1`` by omitting
    the argument.  Pass ``exclude=()`` to opt out explicitly.
    """
    if exclude is None:
        exclude = DEGENERATE_LAMBDAS if key == "lambda" else ()
    valid = [
        p for p in points
        if p["bias_avg"] == p["bias_avg"]
        and not any(abs(p[key] - e) < 1e-9 for e in exclude)
    ]
    if not valid:
        return float("nan")
    knee = pareto_knee(valid)
    anchor = knee["bias_avg"] if knee is not None else min(p["bias_avg"] for p in valid)
    threshold = anchor * (1.0 + tol) if anchor > 0 else anchor + tol
    eligible = [p for p in valid if p["bias_avg"] <= threshold]
    return min(p[key] for p in eligible)


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--sweep", choices=["lambda", "alpha", "both"], default="both")
    ap.add_argument("--lambdas", type=float, nargs="*", default=DEFAULT_LAMBDAS)
    ap.add_argument("--alphas", type=float, nargs="*", default=DEFAULT_ALPHAS)
    ap.add_argument("--val-fraction", type=float, default=0.5,
                    help="fraction of the loaded bias items used as the validation split")
    ap.add_argument("--min-val-items", type=int, default=150,
                    help="floor on validation items per utility task (the knee is selected on these)")
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
    # Utility items carry sensitive spans too (see load_utility_datasets): without
    # them UtilityAvg is constant in lambda and the Pareto knee has no trade-off
    # to balance, degenerating to "pick the largest lambda".  GSM8K is included
    # because it is the task where fusion visibly costs something.
    #
    # The validation split is NOT shrunk below `min_val_items`.  lambda and alpha
    # are chosen from this curve, and GSM8K is where the utility cost lives: at a
    # few dozen items its accuracy carries several points of Monte-Carlo error,
    # which is the same size as the effect being selected on, so the knee ends up
    # placed by noise.  A sweep is only worth running at a size that resolves the
    # thing it is selecting.
    val_cfg = merge_configs(cfg, {"data": {"utility": {
        k: max(args.min_val_items, lim.get(k) or args.min_val_items)
        for k in ("gsm8k", "strategyqa", "arc_easy", "piqa")
    }}})
    util_data = load_utility_datasets(val_cfg, seed + 1000)
    print(f"  validation utility sizes: "
          f"{ {k: len(v) for k, v in util_data.items()} }")
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
        rg = E.eval_generation(
            dec, util_data["gsm8k"], "gsm8k", bs,
            max_new_tokens=cfg["data"]["gsm8k_max_new_tokens"],
            greedy=True, progress=progress, seed=seed, stop_strings=GSM8K_STOP,
        )
        accs.append(rg["acc"])
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
            "chosen": pareto_knee_choice(pts, "lambda", exclude=DEGENERATE_LAMBDAS),
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
