#!/usr/bin/env python3
"""Table 3 -- efficiency: tokens/second, overhead (%), peak memory (GB).

Protocol follows App. C.2: fixed host, batch size 4, max length 256, and the
mean over at least five repeated timing windows after a warm-up window.  Overhead
is measured against vanilla decoding on the *same* prompts and seed.

COFT's extra cost is one additional cached forward pass.  Because both branches
are packed into a single batch that shares one KV cache
(:meth:`coft.model.FrozenLM.prefill`), the cost is a batch-size increase rather
than a second sequential decode -- which is why the overhead is ~10% rather
than 100%.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch

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

from coft.data.base import attach_terms
from coft.data.bias import DatasetUnavailable, load_bold
from coft.registry import METHOD_LABELS, build_decoder
from coft.spans import SensitiveLexicon
from coft.toxicity import TokenToxicityTable, ToxicityScorer

METHODS = ["vanilla", "sdd", "dexperts", "dtcd", "coft"]


def make_prompts(cfg, n: int, target_tokens: int, lm) -> List:
    """Real BOLD prompts, padded with filler to a uniform prompt length."""
    lex = SensitiveLexicon()
    items = attach_terms(load_bold(limit=n * 4, seed=0), lex)[:n]
    tok = lm.tokenizer
    filler = (
        "The following passage is drawn from an encyclopedic article and is "
        "provided as background context for the continuation that follows. "
    )
    out = []
    for it in items:
        text = it.prompt
        while len(tok.encode(text, add_special_tokens=False)) < target_tokens:
            text = filler + text
        ids = tok.encode(text, add_special_tokens=False)[-target_tokens:]
        it.prompt = tok.decode(ids)
        out.append(it)
    return out


def time_method(decoder, items, max_new_tokens: int, n_warmup: int, n_windows: int) -> Dict[str, float]:
    prompts = [it.prompt for it in items]
    terms = [it.terms for it in items]

    # EOS stopping is disabled during timing: otherwise a method that happens to
    # emit EOS earlier runs fewer decode steps, and tokens/second stops being a
    # like-for-like measurement (prefill is then amortised over fewer tokens).
    for _ in range(n_warmup):
        decoder.generate(
            prompts, terms, max_new_tokens=min(16, max_new_tokens), greedy=True, stop_on_eos=False
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    rates: List[float] = []
    for w in range(n_windows):
        res = decoder.generate(
            prompts, terms, max_new_tokens=max_new_tokens, greedy=True, seed=w, stop_on_eos=False
        )
        assert sum(len(t) for t in res.tokens) == len(prompts) * max_new_tokens, (
            "timing window did not run the full step budget"
        )
        rates.append(res.tokens_per_second)

    peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else float("nan")
    return {
        "tokens_per_sec": statistics.mean(rates),
        "tokens_per_sec_std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "peak_mem_gb": peak,
        "windows": rates,
    }


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--force-calibration", action="store_true")
    args = ap.parse_args()

    cfg, lm = setup(args)
    out_dir = results_dir(cfg, args)
    cfg = apply_selected_hyperparams(cfg, out_dir)
    masker = build_masker(lm, cfg)
    ecfg = cfg["efficiency"]
    set_seed(0)

    try:
        items = make_prompts(cfg, ecfg["batch_size"], ecfg["prompt_tokens"], lm)
    except DatasetUnavailable:
        items = []
    if not items:
        print("no prompts available for the efficiency benchmark")
        return 1
    print(f"benchmarking with batch={len(items)}, max_new_tokens={ecfg['max_new_tokens']}, "
          f"windows={ecfg['n_windows']}")

    dual_th = single_th = token_tox = None
    if "coft" in args.methods:
        dual_th = get_thresholds(lm, cfg, out_dir, "dual", masker=masker,
                                 force=args.force_calibration, progress=not args.no_progress)
    if "dtcd" in args.methods:
        single_th = get_thresholds(lm, cfg, out_dir, "single", masker=masker,
                                   force=args.force_calibration, progress=not args.no_progress)
        token_tox = TokenToxicityTable.build(
            lm, scorer=ToxicityScorer(cfg["toxicity"]["model_id"]),
            cache_dir=str(out_dir.parent / "cache"),
            toxicity_model=cfg["toxicity"]["model_id"], verbose=False,
        ).values

    results: Dict[str, Dict] = {}
    for method in args.methods:
        th = dual_th if method == "coft" else (single_th if method == "dtcd" else None)
        dec = build_decoder(method, lm, cfg, thresholds=th, masker=masker, token_toxicity=token_tox)
        print(f"  timing {METHOD_LABELS.get(method, method)} ...", flush=True)
        results[method] = time_method(
            dec, items, ecfg["max_new_tokens"], ecfg["n_warmup"], ecfg["n_windows"]
        )

    base = results.get("vanilla", {}).get("tokens_per_sec")
    for method, r in results.items():
        r["overhead_pct"] = (
            100.0 * (base - r["tokens_per_sec"]) / base if base and method != "vanilla" else None
        )

    payload = {"model": cfg["model"], "config": ecfg, "table": results}
    save_json(out_dir / "table3_efficiency.json", payload)

    print("\n--- Table 3 ---")
    print("Method".ljust(14) + "tok/s".rjust(12) + "Overhead".rjust(12) + "Peak Mem".rjust(12))
    for method in args.methods:
        r = results[method]
        oh = "--" if r["overhead_pct"] is None else f"{r['overhead_pct']:.1f}%"
        print(
            METHOD_LABELS.get(method, method).ljust(14)
            + f"{r['tokens_per_sec']:.1f}".rjust(12)
            + oh.rjust(12)
            + f"{r['peak_mem_gb']:.1f}".rjust(12)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
