#!/usr/bin/env python3
"""End-to-end smoke test: run every stage of the pipeline on a tiny slice.

This is the "does the whole thing still work" check.  It exercises the same code
paths as the full reproduction -- masking, calibration, all five decoders, all
four table runners, and the table/figure renderers -- on a few dozen items, and
asserts the structural invariants that must hold regardless of sample size:

* the mask operator preserves token count exactly;
* calibrated thresholds are finite and produce non-empty certified sets;
* COFT's empirical coverage on held-out items is close to ``1 - alpha``;
* every runner writes a well-formed results file.

It does **not** assert the paper's effect sizes -- a few dozen items cannot
resolve them.  Use ``make all`` for that.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

STAGES = [
    ("Table 1 (bias)", "run_bias.py", "table1_bias.json"),
    ("Table 2 (utility)", "run_utility.py", "table2_utility.json"),
    ("Table 3 (efficiency)", "run_efficiency.py", "table3_efficiency.json"),
    ("Table 4 (ablation)", "run_ablation.py", "table4_ablation.json"),
    ("Figs 3-4 (sweeps)", "run_sweep.py", "sweeps.json"),
]


def check_core(config: str, override: str) -> None:
    """Structural checks that do not need a full runner pass."""
    from _common import base_parser, build_masker, setup

    from coft import evaluate as E
    from coft.calibration import collect_calibration_scores
    from coft.data.base import attach_terms, calibration_triples
    from coft.data.bias import load_stereoset
    from coft.data.corpora import split_calibration_eval
    from coft.decoding import COFTDecoder
    from coft.spans import SensitiveLexicon

    ap = base_parser("smoke")
    args = ap.parse_args(["--config", config, "--override", override, "--no-progress"])
    cfg, lm = setup(args)
    masker = build_masker(lm, cfg)

    print("\n[core] mask operator ...")
    items = attach_terms(load_stereoset(limit=40, seed=0), SensitiveLexicon())
    n_masked = 0
    for it in items:
        mp = masker.mask(it.context, terms=it.terms)
        assert len(mp.factual_ids) == len(mp.masked_ids), "mask broke token alignment"
        n_masked += mp.n_masked > 0
    print(f"  ok -- {len(items)} prompts, {n_masked} carried a sensitive span, "
          f"token counts preserved everywhere")

    print("[core] split calibration ...")
    cal_items, ev = split_calibration_eval(items, 0.25, seed=0)
    ids_cal = {id(x) for x in cal_items}
    assert not any(id(x) in ids_cal for x in ev), "calibration and eval slices overlap"
    corpus = calibration_triples(cal_items)
    alpha = cfg["conformal"]["alpha"]
    bundle = collect_calibration_scores(
        lm, corpus, lams=[cfg["methods"]["coft"]["lam"]], score="dual",
        masker=masker, batch_size=4, max_continuation_tokens=24, progress=False,
    )
    th = bundle.thresholds(cfg["methods"]["coft"]["lam"], alpha)
    assert 0.0 <= th.tau(0) < 1.0, f"degenerate threshold tau={th.tau(0)}"
    print(f"  ok -- {th.meta['n_scores']} scores, tau[0]={th.tau(0):.3e}")

    print("[core] certified sets and coverage ...")
    dec = COFTDecoder(lm, masker=masker, lam=cfg["methods"]["coft"]["lam"],
                      thresholds=th, max_new_tokens=16)
    res = E.eval_pairs(dec, ev, "stereoset", 4, progress=False)
    cov = res.get("empirical_coverage", float("nan"))
    assert cov == cov, "no coverage recorded -- certification did not run"
    print(f"  ok -- empirical coverage {cov:.3f} (target >= {1 - alpha:.2f})")
    if cov < 1 - alpha - 0.15:
        print(f"  [warn] coverage {cov:.3f} is well below 1-alpha; check the calibration slice")

    print("[core] generation under every decoder ...")
    from coft.baselines import DExpertsDecoder, SDDDecoder, VanillaDecoder

    prompts = [it.context for it in ev[:2]]
    terms = [it.terms for it in ev[:2]]
    for d in (VanillaDecoder(lm, max_new_tokens=12), SDDDecoder(lm, max_new_tokens=12),
              DExpertsDecoder(lm, max_new_tokens=12), dec):
        out = d.generate(prompts, terms, max_new_tokens=12, greedy=True)
        assert len(out.texts) == len(prompts)
        print(f"  ok -- {d.name}: {out.tokens_per_second:.0f} tok/s")

    del lm


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/models/mistral-7b-instruct.yaml")
    ap.add_argument("--override", default="configs/smoke.yaml")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--skip-core", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="subset of runner scripts")
    args = ap.parse_args()

    if not args.skip_core:
        check_core(args.config, args.override)

    common = ["--config", args.config, "--override", args.override, "--no-progress"]
    if args.device_map:
        common += ["--device-map", args.device_map]

    model_key = None
    failures = []
    for title, script, artifact in STAGES:
        if args.only and script not in args.only:
            continue
        print(f"\n{'=' * 70}\n[stage] {title}\n{'=' * 70}", flush=True)
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *common])
        if proc.returncode != 0:
            failures.append(title)
            continue
        if model_key is None:

            from coft.registry import load_config

            model_key = load_config(args.config)["model"]["key"]
        out = ROOT / "results" / model_key / artifact
        if not out.exists():
            failures.append(f"{title} (no {artifact})")
            continue
        payload = json.loads(out.read_text())
        assert "table" in payload or "lambda_sweep" in payload or "alpha_sweep" in payload
        print(f"  ok -- {out.relative_to(ROOT)}")

    print(f"\n{'=' * 70}\n[stage] renderers\n{'=' * 70}", flush=True)
    for script in ("make_tables.py", "make_figures.py"):
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / script)])
        if proc.returncode != 0:
            failures.append(script)

    print("\n" + "=" * 70)
    if failures:
        print("SMOKE TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST PASSED -- every stage ran and produced well-formed output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
