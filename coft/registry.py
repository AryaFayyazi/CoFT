"""Config plumbing: build models, decoders and datasets from YAML.

Keeping construction in one place is what lets ``scripts/run_*.py`` stay thin and
what guarantees every method in a table is built with the same shared decoding
policy (nucleus ``p``, temperature, ``max_new_tokens``).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import yaml

from coft.baselines import DExpertsDecoder, DTCDDecoder, SDDDecoder, VanillaDecoder
from coft.conformal import ConformalThresholds
from coft.decoding import COFTDecoder
from coft.masking import Masker

__all__ = [
    "load_config",
    "merge_configs",
    "build_decoder",
    "DECODERS",
    "METHOD_LABELS",
    "repo_root",
]

DECODERS = {
    "vanilla": VanillaDecoder,
    "sdd": SDDDecoder,
    "dexperts": DExpertsDecoder,
    "dtcd": DTCDDecoder,
    "coft": COFTDecoder,
}

#: Display names used in the generated tables (matches Table 1 of the paper).
METHOD_LABELS = {
    "vanilla": "Vanilla",
    "sdd": "SDD",
    "dexperts": "DExperts",
    "dtcd": "DT-CD*",
    "coft": "COFT (ours)",
    "coft_full": "COFT (full)",
    "coft_no_fusion": "w/o fusion (CP only)",
    "coft_single_branch": "Single-branch CP (factual)",
    "coft_no_cp": "fusion only (no CP)",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str) -> Dict:
    """Load a YAML config, resolving an optional ``defaults:`` include list."""
    p = Path(path)
    if not p.is_absolute():
        cand = repo_root() / path
        p = cand if cand.exists() else p
    cfg = yaml.safe_load(p.read_text()) or {}
    includes = cfg.pop("defaults", []) or []
    merged: Dict = {}
    for inc in includes:
        merged = merge_configs(merged, load_config(inc))
    return merge_configs(merged, cfg)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursive dict merge (``override`` wins)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def build_decoder(
    method: str,
    lm,
    cfg: Dict,
    thresholds: Optional[ConformalThresholds] = None,
    masker: Optional[Masker] = None,
    token_toxicity=None,
    overrides: Optional[Dict] = None,
):
    """Instantiate a decoder for ``method`` under the shared decoding policy.

    ``method`` may be a base name (``vanilla``, ``coft``, ...) or one of the
    Table-4 ablation variants (``coft_no_fusion``, ``coft_single_branch``,
    ``coft_no_cp``).
    """
    dec_cfg = (cfg.get("decoding") or {})
    shared = dict(
        top_p=dec_cfg.get("top_p", 0.9),
        temperature=dec_cfg.get("temperature", 1.0),
        max_new_tokens=dec_cfg.get("max_new_tokens", 256),
        seed=dec_cfg.get("seed", 0),
    )
    method_cfg = dict((cfg.get("methods") or {}).get(method, {}) or {})
    method_cfg.update(overrides or {})

    if method in {"coft", "coft_full", "coft_no_fusion", "coft_single_branch", "coft_no_cp"}:
        use_fusion = method not in {"coft_no_fusion"}
        use_cp = method not in {"coft_no_cp"}
        base_cfg = dict((cfg.get("methods") or {}).get("coft", {}) or {})
        base_cfg.update(method_cfg)
        return COFTDecoder(
            lm,
            masker=masker or Masker(lm.tokenizer),
            lam=base_cfg.get("lam", 0.6),
            thresholds=thresholds if use_cp else None,
            use_fusion=use_fusion,
            use_cp=use_cp,
            **shared,
        )

    if method == "dtcd":
        if thresholds is None:
            raise ValueError("DT-CD needs single-branch conformal thresholds")
        return DTCDDecoder(
            lm,
            thresholds=thresholds,
            token_toxicity=token_toxicity,
            toxicity_threshold=method_cfg.get("toxicity_threshold", 0.5),
            **shared,
        )

    if method == "sdd":
        return SDDDecoder(lm, decay=method_cfg.get("decay", 50.0), **shared)

    if method == "dexperts":
        return DExpertsDecoder(
            lm,
            strength=method_cfg.get("strength", 1.0),
            top_k_filter=method_cfg.get("top_k_filter", 100),
            **shared,
        )

    if method == "vanilla":
        return VanillaDecoder(lm, **shared)

    raise ValueError(f"unknown method '{method}'; available: {sorted(set(DECODERS) | set(METHOD_LABELS))}")


def requires_dual_thresholds(method: str) -> bool:
    return method in {"coft", "coft_full", "coft_no_fusion"}


def requires_single_thresholds(method: str) -> bool:
    return method in {"dtcd", "coft_single_branch"}
