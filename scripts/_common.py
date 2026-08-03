"""Shared setup for the ``scripts/run_*.py`` entry points."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coft.calibration import collect_calibration_scores  # noqa: E402
from coft.conformal import ConformalThresholds  # noqa: E402
from coft.data.corpora import load_calibration_corpus  # noqa: E402
from coft.masking import Masker  # noqa: E402
from coft.model import FrozenLM, load_model  # noqa: E402
from coft.registry import load_config, merge_configs  # noqa: E402

__all__ = [
    "ROOT",
    "base_parser",
    "set_seed",
    "setup",
    "results_dir",
    "save_json",
    "load_json",
    "get_thresholds",
    "build_masker",
]


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config",
        default="configs/models/mistral-7b-instruct.yaml",
        help="model config (it includes configs/default.yaml)",
    )
    p.add_argument(
        "--override",
        default=None,
        help="extra YAML config merged on top, e.g. configs/smoke.yaml",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--device-map", default=None, help="override model device_map, e.g. '{\"\":0}'")
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def results_dir(cfg: Dict, args) -> Path:
    out = Path(args.output_dir or cfg.get("output_dir", "results")) / cfg["model"]["key"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote {path}")


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def build_masker(lm: FrozenLM, cfg: Dict) -> Masker:
    mcfg = cfg.get("masking") or {}
    return Masker(lm.tokenizer, sentinel=mcfg.get("sentinel"))


def setup(args, load_lm: bool = True):
    """Resolve the config and (optionally) load the frozen model."""
    cfg = load_config(args.config)
    if args.override:
        cfg = merge_configs(cfg, load_config(args.override))
    if args.seeds:
        cfg["seeds"] = list(args.seeds)
    if args.device_map:
        cfg["model"]["device_map"] = json.loads(args.device_map)

    lm = None
    if load_lm:
        print(f"loading {cfg['model']['model_id']} ...", flush=True)
        lm = load_model(cfg["model"])
        print(f"  loaded ({cfg['model']['label']}), vocab={lm.vocab_size}", flush=True)
    return cfg, lm


# --------------------------------------------------------------------------- #
# conformal thresholds
# --------------------------------------------------------------------------- #
def _cal_path(out_dir: Path, tag: str, score: str, lam: float, alpha: float) -> Path:
    return out_dir / "calibration" / f"{tag}_{score}_lam{lam:g}_alpha{alpha:g}.json"


def get_thresholds(
    lm: FrozenLM,
    cfg: Dict,
    out_dir: Path,
    score: str = "dual",
    lam: Optional[float] = None,
    alpha: Optional[float] = None,
    masker: Optional[Masker] = None,
    corpus=None,
    tag: str = "pooled",
    force: bool = False,
    progress: bool = True,
) -> ConformalThresholds:
    """Load cached split-conformal thresholds, computing them if necessary.

    ``corpus`` is the ``D_cal`` of Eq. 6.  Pass a *per-dataset* corpus (built from
    that benchmark's disjoint calibration slice) to follow App. C.2 -- "for each
    dataset and step index t, a disjoint calibration pool (10-15%) sets tau_t".
    When ``corpus`` is ``None`` a pooled corpus over the bias families is used.

    Thresholds depend on ``(tag, score, lambda, alpha)``; the calibration *scores*
    depend only on ``(tag, score, lambda)``, so re-running with a new ``alpha`` is
    cheap when the bundle is cached.
    """
    ccfg = cfg.get("conformal") or {}
    lam = (cfg.get("methods", {}).get("coft", {}).get("lam", 0.6)) if lam is None else lam
    if score == "single":
        lam = 0.0
    alpha = ccfg.get("alpha", 0.10) if alpha is None else alpha

    path = _cal_path(out_dir, tag, score, lam, alpha)
    if path.exists() and not force:
        print(f"  reusing calibration {path.name}")
        return ConformalThresholds.load(path)

    if corpus is None:
        corpus = load_calibration_corpus(
            n_contexts=ccfg.get("n_calibration_contexts", 400),
            seed=0,
            sources=tuple(ccfg.get("calibration_sources", ["stereoset", "crows", "bbq", "bold"])),
        )
    corpus = list(corpus)[: ccfg.get("n_calibration_contexts", 400)]
    print(f"  calibrating [{tag}/{score}] lam={lam} alpha={alpha} on {len(corpus)} contexts")
    bundle = collect_calibration_scores(
        lm,
        corpus,
        lams=[lam],
        score=score,
        masker=masker or build_masker(lm, cfg),
        temperature=cfg["decoding"].get("temperature", 1.0),
        batch_size=cfg["data"].get("batch_size", 8),
        bin_width=ccfg.get("bin_width", 8),
        max_position=ccfg.get("max_position", 256),
        max_continuation_tokens=ccfg.get("max_continuation_tokens", 48),
        progress=progress,
    )
    th = bundle.thresholds(lam, alpha)
    th.save(path)
    print(f"  saved {path.name}  (n_scores={th.meta.get('n_scores')}, tau[0]={th.tau(0):.3e})")
    return th
