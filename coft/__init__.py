"""COFT -- Chain of Fair Thought.

Training-free counterfactual-conformal decoding for fair chain-of-thought
reasoning in large language models.

Reference
---------
Fayyazi, Kamal, Pedram. "COFT: Counterfactual-Conformal Decoding for Fair
Chain-of-Thought Reasoning in Large Language Models." ICML 2026.

The package mirrors the three stages of the method:

    Stage I   -- :mod:`coft.masking`    counterfactual (length-preserving) masking, Sec. 3.2
    Stage II  -- :mod:`coft.fusion`     counterfactual logit fusion,              Sec. 3.3
    Stage III -- :mod:`coft.conformal`  dual-branch split-conformal filtering,    Sec. 3.4

and the decoder that composes them (Algorithm 1) lives in :mod:`coft.decoding`.
"""

__version__ = "1.0.0"

from coft.conformal import (
    ConformalCalibrator,
    ConformalThresholds,
    ceiling_quantile,
    dual_branch_score,
)
from coft.fusion import fuse_logits, fused_distribution, geometric_mixture
from coft.masking import MaskedPrompt, Masker, resolve_sentinel

__all__ = [
    "__version__",
    "fuse_logits",
    "fused_distribution",
    "geometric_mixture",
    "Masker",
    "MaskedPrompt",
    "resolve_sentinel",
    "ConformalCalibrator",
    "ConformalThresholds",
    "ceiling_quantile",
    "dual_branch_score",
]
