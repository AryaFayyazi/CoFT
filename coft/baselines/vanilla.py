"""Vanilla decoding -- no mitigation (paper Sec. 4.1, the bias lower bound)."""

from __future__ import annotations

from typing import Dict

import torch

from coft.decoding import BaseDecoder, StepOutput

__all__ = ["VanillaDecoder"]


class VanillaDecoder(BaseDecoder):
    """Plain nucleus sampling from the unmodified model.

    Present so that every other method can be read as a delta against it, and so
    that the shared decoding policy (``p = 0.9``, ``T = 1.0``, 256 max tokens) is
    identical across the table.
    """

    name = "vanilla"
    branch_names = ("factual",)

    def step_distribution(self, logits: Dict[str, torch.Tensor], t: int) -> StepOutput:
        probs = torch.softmax(logits["factual"].float() / self.temperature, dim=-1)
        return StepOutput(probs=probs)
