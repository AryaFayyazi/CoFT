#!/usr/bin/env python3
"""Table 4 -- component ablation.

    Variant                       | BiasAvg v | UtilityAvg ^
    COFT (full)                   |
    w/o fusion (CP only)          |   lambda forced to 0
    Single-branch CP (factual)    |   certification uses only pi_hat
    fusion only (no CP)           |   no certified set at all

``BiasAvg`` averages the six bias columns of Table 1 after orienting every one
so that lower is better (CP accuracy enters as ``(100 - CP Acc) / 100``, and the
two percentage-scale columns are divided by 100 so the average is not dominated
by them).  ``UtilityAvg`` is the mean of the four task accuracies.

The paper's reading of this table: fusion is the largest single contributor and
dual-branch CP adds the certified stability on top of it, while single-branch CP
"cannot guarantee counterfactual robustness, and leaves residual bias".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (
    apply_selected_hyperparams,
    base_parser,
    build_masker,
    get_thresholds,
    results_dir,
    save_json,
    set_seed,
    setup,
)
from run_bias import COLUMNS, evaluate_method, flatten, load_bias_datasets, prepare_thresholds

from coft import evaluate as E
from coft.data.tasks import load_arc_easy, load_gsm8k, load_piqa, load_strategyqa
from coft.registry import METHOD_LABELS, build_decoder
from coft.toxicity import ToxicityScorer

VARIANTS = ["coft_full", "coft_no_fusion", "coft_single_branch", "coft_no_cp"]

#: Which calibrated thresholds each ablation variant needs.
#: - full:          dual-branch score at the selected lambda
#: - no fusion:     dual-branch score, but certification happens at lambda = 0
#: - single branch: factual-only score  (Sec. 3.4, "single-branch CP cannot
#:                  guarantee stability to masking")
#: - no CP:         none
VARIANT_THRESHOLD_KEY = {
    "coft_full": "dual",
    "coft_no_fusion": "dual_lam0",
    "coft_single_branch": "single",
    "coft_no_cp": None,
}


def bias_avg(row: Dict[str, float]) -> float:
    """Average the Table-1 columns on a common 'lower is better', ~[0,1] scale."""
    vals: List[float] = []
    for col, _key, higher_better in COLUMNS:
        if col not in row:
            continue
        v = float(row[col])
        if col == "CP Acc":                 # higher-is-better, on 0-100
            vals.append((100.0 - v) / 100.0)
        elif higher_better:
            vals.append(1.0 - v)
        else:
            vals.append(v)
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--force-calibration", action="store_true")
    args = ap.parse_args()

    cfg, lm = setup(args)
    out_dir = results_dir(cfg, args)
    cfg = apply_selected_hyperparams(cfg, out_dir)
    progress = not args.no_progress
    masker = build_masker(lm, cfg)
    seeds = cfg.get("seeds", [0])
    bs = cfg["data"]["batch_size"]
    gbs = cfg["data"]["generation_batch_size"]
    lim = cfg["data"]["utility"]

    print("loading benchmarks ...")
    bias_data, cal_data = load_bias_datasets(cfg, seed=seeds[0])
    util_data = {
        "gsm8k": load_gsm8k(limit=lim["gsm8k"], seed=seeds[0]),
        "strategyqa": load_strategyqa(limit=lim["strategyqa"], seed=seeds[0]),
        "arc_easy": load_arc_easy(limit=lim["arc_easy"], seed=seeds[0]),
        "piqa": load_piqa(limit=lim["piqa"], seed=seeds[0]),
    }
    tox = ToxicityScorer(cfg["toxicity"]["model_id"]) if "bold" in bias_data else None

    print("calibrating conformal thresholds (per dataset, disjoint slice) ...")
    specs = [("dual", "dual", None), ("dual_lam0", "dual", 0.0), ("single", "single", 0.0)]
    thresholds = prepare_thresholds(
        lm, cfg, out_dir, cal_data, specs, masker, args.force_calibration, progress
    )
    # The utility tasks are not bias benchmarks, so they have no per-dataset
    # calibration slice; they use thresholds calibrated on the pooled bias
    # calibration corpus -- i.e. the deployment-time thresholds.
    pooled_corpus = [t for triples in cal_data.values() for t in triples]
    pooled = {
        key: get_thresholds(lm, cfg, out_dir, score=score, lam=lam, masker=masker,
                            corpus=pooled_corpus, tag="pooled",
                            force=args.force_calibration, progress=progress)
        for key, score, lam in specs
    }

    payload = {"model": cfg["model"], "seeds": seeds, "per_seed": {}}
    for seed in seeds:
        set_seed(seed)
        payload["per_seed"][str(seed)] = {}
        for variant in args.variants:
            print(f"\n=== seed {seed} | {METHOD_LABELS.get(variant, variant)} ===")
            th_key = VARIANT_THRESHOLD_KEY[variant]

            def make_decoder(ds_name: str, _v=variant, _k=th_key, _seed=seed):
                th = thresholds.get(ds_name, {}).get(_k) if _k else None
                d = build_decoder(_v, lm, cfg, thresholds=th, masker=masker)
                d.seed = _seed
                return d

            per_ds = evaluate_method(make_decoder, bias_data, cfg, tox, progress, seed)
            row = flatten(per_ds)

            dec = build_decoder(
                variant, lm, cfg, thresholds=pooled.get(th_key) if th_key else None, masker=masker
            )
            dec.seed = seed
            accs: List[float] = []
            r = E.eval_generation(dec, util_data["gsm8k"], "gsm8k", gbs,
                                  max_new_tokens=cfg["data"]["gsm8k_max_new_tokens"],
                                  greedy=True, progress=progress, seed=seed)
            accs.append(r["acc"])
            for task in ("strategyqa", "arc_easy", "piqa"):
                rr = E.eval_choices(dec, util_data[task], task, bs, progress=progress)
                rr.pop("predictions", None)
                accs.append(rr["acc"])

            payload["per_seed"][str(seed)][variant] = {
                "bias_row": row,
                "bias_avg": bias_avg(row),
                "utility_avg": sum(accs) / len(accs),
                "utility_accs": accs,
                "per_dataset": per_ds,
            }

    table: Dict[str, Dict[str, float]] = {}
    for variant in args.variants:
        b = [payload["per_seed"][str(s)][variant]["bias_avg"] for s in seeds]
        u = [payload["per_seed"][str(s)][variant]["utility_avg"] for s in seeds]
        table[variant] = {"BiasAvg": sum(b) / len(b), "UtilityAvg": sum(u) / len(u)}
    payload["table"] = table
    save_json(out_dir / "table4_ablation.json", payload)

    print("\n--- Table 4 ---")
    print("Variant".ljust(30) + "BiasAvg".rjust(11) + "UtilityAvg".rjust(13))
    for variant in args.variants:
        print(
            METHOD_LABELS.get(variant, variant).ljust(30)
            + f"{table[variant]['BiasAvg']:.3f}".rjust(11)
            + f"{table[variant]['UtilityAvg']:.1f}".rjust(13)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
