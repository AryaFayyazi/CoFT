#!/usr/bin/env python3
"""Table 1 -- bias mitigation across six benchmarks and five decoding methods.

Reproduces, per model:

    Method | SS v | CP Acc ^ | BBQ Bias v | BOLD Tox v | Utrecht DP v | COMPAS Gap v | Avg. Rank v

Calibration protocol (App. C.2)
-------------------------------
Every benchmark is split into a disjoint calibration slice (15% by default) and
an evaluation slice.  The conformal thresholds ``tau_t`` used on a benchmark are
computed *from that benchmark's own calibration slice* -- never from evaluation
data, and never from a different distribution.  This is what App. C.2 prescribes
and what keeps assumption (A2) plausible.

Usage
-----
    python scripts/run_bias.py --config configs/models/llama2-13b.yaml
    python scripts/run_bias.py --config configs/models/mistral-7b-instruct.yaml \\
        --override configs/smoke.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import base_parser, build_masker, get_thresholds, results_dir, save_json, set_seed, setup

from coft import evaluate as E
from coft.data.base import attach_terms, calibration_triples
from coft.data.bias import (
    DatasetUnavailable,
    load_bbq,
    load_bold,
    load_compas,
    load_crows,
    load_stereoset,
    load_utrecht,
)
from coft.data.corpora import split_calibration_eval
from coft.metrics import average_rank
from coft.registry import METHOD_LABELS, build_decoder
from coft.spans import SensitiveLexicon
from coft.toxicity import TokenToxicityTable, ToxicityScorer

METHODS = ["vanilla", "sdd", "dexperts", "dtcd", "coft"]

#: (display column, key in the per-dataset metric dict, higher_is_better)
COLUMNS = [
    ("SS", "ss_bias", False),
    ("CP Acc", "cp_acc", True),
    ("BBQ Bias", "bbq_bias", False),
    ("BOLD Tox", "toxicity", False),
    ("Utrecht DP", "dp_gap", False),
    ("COMPAS Gap", "dp_gap", False),
]
DATASETS = ["stereoset", "crows", "bbq", "bold", "utrecht", "compas"]
CAL_FRACTION = 0.15

#: Benchmarks whose score depends on the sampling seed.  StereoSet, CrowS-Pairs,
#: BBQ, Utrecht and COMPAS are all evaluated by *teacher-forced likelihood*, which
#: is a deterministic function of the model and the thresholds -- re-running them
#: per seed reproduces identical numbers.  Only BOLD free-generation is
#: stochastic, so only it is repeated across seeds.
STOCHASTIC_DATASETS = {"bold"}


def load_bias_datasets(cfg: Dict, seed: int, calibration_fraction: float = CAL_FRACTION):
    """Load every bias benchmark and split it into calibration / evaluation slices.

    Returns ``(eval_items, calibration_triples)`` keyed by dataset name.  The
    requested evaluation size is honoured: we over-load by ``1/(1 - frac)`` so
    that after removing the calibration slice the evaluation split still has the
    configured number of items.
    """
    lim = cfg["data"]["bias"]
    lex = SensitiveLexicon(cfg.get("masking", {}).get("categories"))
    use_ner = bool(cfg.get("masking", {}).get("use_ner", False))
    scale = 1.0 / max(1e-6, 1.0 - calibration_fraction)

    def _n(key):
        v = lim.get(key)
        return None if v is None else max(4, int(v * scale))

    loaders = {
        "stereoset": lambda: load_stereoset(limit=_n("stereoset"), seed=seed),
        "crows": lambda: load_crows(limit=_n("crows"), seed=seed),
        "bbq": lambda: load_bbq(condition="disambig", limit=_n("bbq"), seed=seed),
        "bold": lambda: load_bold(limit=_n("bold"), seed=seed),
        "utrecht": lambda: load_utrecht(limit=_n("utrecht"), seed=seed),
        "compas": lambda: load_compas(limit=_n("compas"), seed=seed),
    }

    eval_data: Dict[str, list] = {}
    cal_data: Dict[str, list] = {}
    for name, fn in loaders.items():
        try:
            items = fn()
        except DatasetUnavailable as exc:
            print(f"  [skip] {name}: {exc}")
            continue
        items = attach_terms(items, lex, use_ner=use_ner)
        cal_items, ev_items = split_calibration_eval(items, calibration_fraction, seed=seed)
        eval_data[name] = ev_items
        cal_data[name] = calibration_triples(cal_items)
        print(f"  {name}: {len(ev_items)} eval / {len(cal_data[name])} calibration items")
    return eval_data, cal_data


def evaluate_dataset(decoder, name: str, items, cfg: Dict, tox_scorer, progress: bool, seed: int) -> Dict:
    """Run one benchmark under one decoder and return its metric dict."""
    bs = cfg["data"]["batch_size"]
    gbs = cfg["data"]["generation_batch_size"]
    if name in {"stereoset", "crows"}:
        return E.eval_pairs(decoder, items, name, bs, progress=progress)
    if name == "bbq":
        r = E.eval_choices(decoder, items, "bbq", bs, progress=progress)
        r.pop("predictions", None)
        return r
    if name == "bold":
        r = E.eval_generation(
            decoder, items, "bold", gbs,
            max_new_tokens=cfg["data"]["bold_max_new_tokens"],
            toxicity_scorer=tox_scorer, progress=progress, seed=seed,
        )
        r.pop("generations", None)
        return r
    if name in {"utrecht", "compas"}:
        return E.eval_decisions(decoder, items, name, bs, progress=progress)
    raise ValueError(f"unknown bias dataset '{name}'")


def evaluate_method(
    make_decoder,
    eval_data: Dict[str, list],
    cfg: Dict,
    tox_scorer,
    progress: bool,
    seed: int,
    reuse: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """Evaluate one method across all benchmarks.

    ``make_decoder(dataset_name)`` returns the decoder to use on that benchmark,
    which is what lets each benchmark carry its own calibrated thresholds.

    ``reuse`` carries results from an earlier seed: deterministic benchmarks
    (see :data:`STOCHASTIC_DATASETS`) are copied instead of recomputed, which
    makes a three-seed run roughly three times cheaper without changing a single
    reported number.
    """
    if not callable(make_decoder):          # allow passing a plain decoder
        make_decoder = lambda _ds, _d=make_decoder: _d  # noqa: E731
    out: Dict[str, Dict] = {}
    for name, items in eval_data.items():
        if reuse and name not in STOCHASTIC_DATASETS and name in reuse:
            out[name] = reuse[name]
            continue
        out[name] = evaluate_dataset(make_decoder(name), name, items, cfg, tox_scorer, progress, seed)
    return out


def flatten(per_dataset: Dict[str, Dict]) -> Dict[str, float]:
    """Collapse the per-dataset metrics into the columns of Table 1."""
    row: Dict[str, float] = {}
    for (col, key, _), ds in zip(COLUMNS, DATASETS):
        if ds in per_dataset and key in per_dataset[ds]:
            v = per_dataset[ds][key]
            if v == v:
                row[col] = float(v)
    return row


def prepare_thresholds(lm, cfg, out_dir, cal_data, specs, masker, force, progress) -> Dict:
    """Calibrate ``tau_t`` per (dataset, spec).  ``specs`` is a list of ``(key, score, lam)``."""
    out: Dict[str, Dict[str, object]] = {}
    for ds_name, corpus in cal_data.items():
        if not corpus:
            continue
        out[ds_name] = {}
        for key, score, lam in specs:
            out[ds_name][key] = get_thresholds(
                lm, cfg, out_dir, score=score, lam=lam, masker=masker,
                corpus=corpus, tag=ds_name, force=force, progress=progress,
            )
    return out


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--force-calibration", action="store_true")
    args = ap.parse_args()

    cfg, lm = setup(args)
    out_dir = results_dir(cfg, args)
    progress = not args.no_progress
    masker = build_masker(lm, cfg)
    seeds = cfg.get("seeds", [0])

    print("loading bias benchmarks ...")
    eval_data, cal_data = load_bias_datasets(cfg, seed=seeds[0])
    if args.datasets:
        eval_data = {k: v for k, v in eval_data.items() if k in args.datasets}
        cal_data = {k: v for k, v in cal_data.items() if k in args.datasets}

    tox_scorer = None
    if "bold" in eval_data or "dtcd" in args.methods:
        print("loading toxicity classifier ...")
        tox_scorer = ToxicityScorer(cfg["toxicity"]["model_id"])

    specs = []
    if "coft" in args.methods:
        specs.append(("dual", "dual", None))
    if "dtcd" in args.methods:
        specs.append(("single", "single", 0.0))
    print("calibrating conformal thresholds (per dataset, disjoint slice) ...")
    thresholds = prepare_thresholds(
        lm, cfg, out_dir, cal_data, specs, masker, args.force_calibration, progress
    )

    token_tox = None
    if "dtcd" in args.methods:
        print("building token-level toxicity table for DT-CD ...")
        token_tox = TokenToxicityTable.build(
            lm, scorer=tox_scorer, cache_dir=str(out_dir.parent / "cache"),
            toxicity_model=cfg["toxicity"]["model_id"], verbose=False,
        ).values

    payload = {
        "model": cfg["model"],
        "seeds": seeds,
        "columns": [c[0] for c in COLUMNS],
        "datasets": list(eval_data),
        "calibration_fraction": CAL_FRACTION,
        "n_eval": {k: len(v) for k, v in eval_data.items()},
        "per_seed": {},
    }

    first_seed_results: Dict[str, Dict] = {}
    for si, seed in enumerate(seeds):
        set_seed(seed)
        payload["per_seed"][str(seed)] = {}
        for method in args.methods:
            print(f"\n=== seed {seed} | {METHOD_LABELS.get(method, method)} ===")
            if si > 0:
                print("  (reusing deterministic benchmarks from seed "
                      f"{seeds[0]}; re-running {sorted(STOCHASTIC_DATASETS & set(eval_data))})")

            # `_m` / `_seed` are bound as defaults so the closure captures this
            # iteration's values rather than the loop variables.
            def make_decoder(ds_name: str, _m=method, _seed=seed):
                key = {"coft": "dual", "dtcd": "single"}.get(_m)
                th = thresholds.get(ds_name, {}).get(key) if key else None
                if key and th is None:
                    raise RuntimeError(f"no {key} thresholds calibrated for '{ds_name}'")
                d = build_decoder(_m, lm, cfg, thresholds=th, masker=masker, token_toxicity=token_tox)
                d.seed = _seed
                return d

            per_ds = evaluate_method(
                make_decoder, eval_data, cfg, tox_scorer, progress, seed,
                reuse=first_seed_results.get(method),
            )
            first_seed_results.setdefault(method, per_ds)
            payload["per_seed"][str(seed)][method] = {
                "per_dataset": per_ds,
                "row": flatten(per_ds),
            }

    # mean over seeds; average rank is recomputed per seed then averaged (App. C.2)
    mean_rows: Dict[str, Dict[str, float]] = {}
    for method in args.methods:
        acc: Dict[str, List[float]] = {}
        for seed in seeds:
            for col, val in payload["per_seed"][str(seed)][method]["row"].items():
                acc.setdefault(col, []).append(val)
        mean_rows[method] = {c: sum(v) / len(v) for c, v in acc.items()}

    rank_cols = [(c, hb) for (c, _, hb) in COLUMNS if any(c in r for r in mean_rows.values())]
    per_seed_ranks: Dict[str, List[float]] = {m: [] for m in args.methods}
    for seed in seeds:
        rows = {m: payload["per_seed"][str(seed)][m]["row"] for m in args.methods}
        for m, v in average_rank(rows, rank_cols).items():
            per_seed_ranks[m].append(v)
    for m in args.methods:
        vals = [v for v in per_seed_ranks[m] if v == v]
        mean_rows[m]["Avg. Rank"] = sum(vals) / len(vals) if vals else float("nan")

    payload["table"] = mean_rows
    save_json(out_dir / "table1_bias.json", payload)

    print("\n--- Table 1 (mean over seeds) ---")
    cols = [c[0] for c in COLUMNS if any(c[0] in r for r in mean_rows.values())] + ["Avg. Rank"]
    print("Method".ljust(14) + "".join(c.rjust(13) for c in cols))
    for m in args.methods:
        cells = "".join(
            (f"{mean_rows[m][c]:.3f}" if c in mean_rows[m] else "n/a").rjust(13) for c in cols
        )
        print(METHOD_LABELS.get(m, m).ljust(14) + cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
