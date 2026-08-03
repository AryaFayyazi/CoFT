#!/usr/bin/env python3
"""Table 2 -- utility preservation and language quality.

    Method | GSM8K | StrategyQA | ARC-easy | PIQA | PPL v | MAUVE ^

The claim under test is that COFT "matches vanilla on utility within +-0.2
points" while SDD/DExperts "incur 0.3-1.1 point drops", and that perplexity and
MAUVE stay indistinguishable from vanilla.

Note on perplexity: it is reported for the method's *corrected* next-token
distribution ``pi_hat`` (the quantity a decoding intervention actually changes),
which is the usual convention for decoding-time methods -- the certified set is
a sampling constraint, not a density.  Pass ``--ppl-certified`` to score under
the certified, renormalised policy instead.
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
    load_utility_datasets,
    results_dir,
    save_json,
    set_seed,
    setup,
)

from coft import evaluate as E
from coft.data.corpora import load_tldr, load_wikitext2
from coft.data.tasks import GSM8K_STOP
from coft.registry import METHOD_LABELS, build_decoder
from coft.toxicity import TokenToxicityTable, ToxicityScorer

METHODS = ["vanilla", "sdd", "dexperts", "dtcd", "coft"]
COLUMNS = ["GSM8K", "StrategyQA", "ARC-easy", "PIQA", "PPL", "MAUVE"]


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--tasks", nargs="*", default=["gsm8k", "strategyqa", "arc_easy", "piqa", "ppl", "mauve"])
    ap.add_argument("--ppl-certified", action="store_true",
                    help="score perplexity under the certified policy instead of pi_hat")
    ap.add_argument("--force-calibration", action="store_true")
    args = ap.parse_args()

    cfg, lm = setup(args)
    out_dir = results_dir(cfg, args)
    cfg = apply_selected_hyperparams(cfg, out_dir)
    progress = not args.no_progress
    masker = build_masker(lm, cfg)
    qlim = cfg["data"]["quality"]
    bs = cfg["data"]["batch_size"]
    gbs = cfg["data"]["generation_batch_size"]
    seeds = cfg.get("seeds", [0])

    print("loading utility benchmarks ...")
    wanted = tuple(t for t in ("gsm8k", "strategyqa", "arc_easy", "piqa") if t in args.tasks)
    data = load_utility_datasets(cfg, seeds[0], tasks=wanted)
    docs = load_wikitext2(limit=qlim["wikitext_docs"]) if "ppl" in args.tasks else []
    tldr = load_tldr(limit=qlim["mauve_samples"]) if "mauve" in args.tasks else []
    for k, v in data.items():
        print(f"  {k}: {len(v)} items")
    if docs:
        print(f"  wikitext-2: {len(docs)} documents")
    if tldr:
        print(f"  tldr: {len(tldr)} prompts")

    dual_th = single_th = None
    if "coft" in args.methods:
        dual_th = get_thresholds(lm, cfg, out_dir, "dual", masker=masker,
                                 force=args.force_calibration, progress=progress)
    token_tox = None
    if "dtcd" in args.methods:
        single_th = get_thresholds(lm, cfg, out_dir, "single", masker=masker,
                                   force=args.force_calibration, progress=progress)
        token_tox = TokenToxicityTable.build(
            lm, scorer=ToxicityScorer(cfg["toxicity"]["model_id"]),
            cache_dir=str(out_dir.parent / "cache"),
            toxicity_model=cfg["toxicity"]["model_id"], verbose=False,
        ).values

    payload = {"model": cfg["model"], "seeds": seeds, "columns": COLUMNS, "per_seed": {}}

    # Only MAUVE is stochastic here: GSM8K is decoded greedily, and the
    # multiple-choice tasks and perplexity are teacher-forced likelihoods.  The
    # deterministic columns are therefore computed once and reused across seeds.
    cached: Dict[str, Dict] = {}

    for si, seed in enumerate(seeds):
        set_seed(seed)
        payload["per_seed"][str(seed)] = {}
        for method in args.methods:
            print(f"\n=== seed {seed} | {METHOD_LABELS.get(method, method)} ===")
            th = dual_th if method == "coft" else (single_th if method == "dtcd" else None)
            dec = build_decoder(method, lm, cfg, thresholds=th, masker=masker, token_toxicity=token_tox)
            dec.seed = seed

            if si > 0 and method in cached:
                row = dict(cached[method]["row"])
                detail = dict(cached[method]["detail"])
                if tldr:
                    r = E.eval_mauve(dec, tldr, batch_size=gbs, seed=seed, progress=progress)
                    detail["mauve"] = r
                    row["MAUVE"] = r["mauve"]
                payload["per_seed"][str(seed)][method] = {"row": row, "detail": detail}
                continue

            row: Dict[str, float] = {}
            detail: Dict[str, Dict] = {}
            if "gsm8k" in data:
                r = E.eval_generation(
                    dec, data["gsm8k"], "gsm8k", gbs,
                    max_new_tokens=cfg["data"]["gsm8k_max_new_tokens"],
                    greedy=True, progress=progress, seed=seed,
                    stop_strings=GSM8K_STOP,
                )
                r.pop("generations", None)
                detail["gsm8k"] = r
                row["GSM8K"] = r["acc"]
            for key, col in (("strategyqa", "StrategyQA"), ("arc_easy", "ARC-easy"), ("piqa", "PIQA")):
                if key in data:
                    r = E.eval_choices(dec, data[key], key, bs, progress=progress)
                    r.pop("predictions", None)
                    detail[key] = r
                    row[col] = r["acc"]
            if docs:
                if args.ppl_certified:
                    r = E.eval_perplexity(dec, docs, batch_size=max(1, bs // 2), progress=progress)
                else:
                    with dec.without_support_restriction():
                        r = E.eval_perplexity(dec, docs, batch_size=max(1, bs // 2), progress=progress)
                detail["ppl"] = r
                row["PPL"] = r["ppl"]
            if tldr:
                r = E.eval_mauve(dec, tldr, batch_size=gbs, seed=seed, progress=progress)
                detail["mauve"] = r
                row["MAUVE"] = r["mauve"]

            payload["per_seed"][str(seed)][method] = {"row": row, "detail": detail}
            cached.setdefault(method, {"row": dict(row), "detail": dict(detail)})

    mean_rows: Dict[str, Dict[str, float]] = {}
    for method in args.methods:
        acc: Dict[str, List[float]] = {}
        for seed in seeds:
            for col, val in payload["per_seed"][str(seed)][method]["row"].items():
                if val == val:
                    acc.setdefault(col, []).append(val)
        mean_rows[method] = {c: sum(v) / len(v) for c, v in acc.items()}
    payload["table"] = mean_rows
    save_json(out_dir / "table2_utility.json", payload)

    print("\n--- Table 2 (mean over seeds) ---")
    cols = [c for c in COLUMNS if any(c in r for r in mean_rows.values())]
    print("Method".ljust(14) + "".join(c.rjust(12) for c in cols))
    for m in args.methods:
        cells = "".join(
            (f"{mean_rows[m][c]:.2f}" if c in mean_rows[m] else "n/a").rjust(12) for c in cols
        )
        print(METHOD_LABELS.get(m, m).ljust(14) + cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
