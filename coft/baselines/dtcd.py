"""DT-CD -- Dual-Threshold Conformal Decoding.

The strongest baseline in the paper (Sec. 4.1): "single-branch conformal
acceptance based on toxicity and minimum probability, and the closest baseline
to our CP component without counterfactual reasoning."

A token is accepted iff it clears **both** thresholds:

1. *Minimum probability*, calibrated by split conformal prediction on the
   single-branch score ``s_t(v) = 1 - pi^F_t(v)`` -- the same finite-sample
   ceiling-corrected quantile machinery COFT uses, but with only the factual
   branch, so nothing about the masked world enters.
2. *Toxicity*, a fixed cap on a per-token toxicity prior.

Contrast with COFT: the accepted set here is
``{v : pi^F(v) >= tau} \\ {v : tox(v) > kappa}``, which can certify a token that
is probable only *because* of the protected span.  COFT's Eq. 7 instead demands
simultaneous support under the factual and masked views, which is what turns
certification into counterfactual stability (Sec. 3.4).
"""

from __future__ import annotations

import contextlib
from typing import Dict, Optional

import torch

from coft.conformal import ConformalThresholds
from coft.decoding import BaseDecoder, StepOutput

__all__ = ["DTCDDecoder"]


class DTCDDecoder(BaseDecoder):
    name = "dtcd"
    branch_names = ("factual",)

    def __init__(
        self,
        lm,
        thresholds: ConformalThresholds,
        token_toxicity: Optional[torch.Tensor] = None,
        toxicity_threshold: float = 0.5,
        **kw,
    ) -> None:
        super().__init__(lm, **kw)
        if thresholds.score != "single":
            raise ValueError(
                "DT-CD is a single-branch method; calibrate with score='single' "
                f"(got '{thresholds.score}')"
            )
        self.thresholds = thresholds
        self.toxicity_threshold = float(toxicity_threshold)
        self._tox = token_toxicity
        self._tox_mask: Optional[torch.Tensor] = None
        self.use_cp = True

    @contextlib.contextmanager
    def without_support_restriction(self):
        saved, self.use_cp = self.use_cp, False
        try:
            yield self
        finally:
            self.use_cp = saved

    def _toxic_mask(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        if self._tox is None:
            return None
        if self._tox_mask is None or self._tox_mask.device != like.device:
            vals = self._tox.to(like.device)
            v = like.shape[-1]
            if vals.shape[0] < v:
                vals = torch.cat([vals, torch.zeros(v - vals.shape[0], device=like.device)])
            self._tox_mask = (vals[:v] > self.toxicity_threshold).unsqueeze(0)
        return self._tox_mask

    def step_distribution(self, logits: Dict[str, torch.Tensor], t: int) -> StepOutput:
        probs = torch.softmax(logits["factual"].float() / self.temperature, dim=-1)
        if not self.use_cp:
            return StepOutput(probs=probs)

        # threshold 1: single-branch conformal minimum probability
        mask = self.thresholds.candidate_mask(probs, None, t)

        # threshold 2: token-level toxicity cap
        toxic = self._toxic_mask(probs)
        if toxic is not None:
            mask = mask & ~toxic

        empty = mask.sum(dim=-1) == 0
        return StepOutput(
            probs=probs,
            certified_mask=mask,
            tau=self.thresholds.tau(t),
            empty_set=empty,
        )
