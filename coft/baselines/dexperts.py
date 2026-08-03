"""DExperts / GeDi-style expert reweighting toward neutral labels.

Liu et al. (2021), "DExperts"; Krause et al. (2021), "GeDi".

Both methods shift the base model's logits by the *contrast* between an expert
(desired behaviour) and an anti-expert (undesired behaviour):

    z_tilde = z_base + strength * (z_expert - z_antiexpert)

Two ways to obtain the experts are supported.

``prompted`` (default)
    GeDi's generative-discriminator trick: the same frozen LM is conditioned on
    a positive and a negative control prompt.  This keeps the baseline inside
    COFT's frozen-weights, no-extra-checkpoint threat model (Sec. 4.1) while
    reproducing the method's mechanism.

``models``
    Genuine separate expert / anti-expert checkpoints, as in the original
    DExperts.  Provide ``expert_model`` / ``antiexpert_model`` in the config;
    they must share the base tokenizer.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Sequence

import torch

from coft.decoding import BaseDecoder, StepOutput

__all__ = ["DExpertsDecoder", "DEFAULT_EXPERT_PROMPT", "DEFAULT_ANTIEXPERT_PROMPT"]

DEFAULT_EXPERT_PROMPT = (
    "The following text is respectful, unbiased and treats all demographic "
    "groups fairly:\n"
)
DEFAULT_ANTIEXPERT_PROMPT = (
    "The following text is toxic, offensive and full of demographic "
    "stereotypes:\n"
)


class DExpertsDecoder(BaseDecoder):
    name = "dexperts"
    branch_names = ("factual", "expert", "antiexpert")

    def __init__(
        self,
        lm,
        strength: float = 1.0,
        expert_prompt: str = DEFAULT_EXPERT_PROMPT,
        antiexpert_prompt: str = DEFAULT_ANTIEXPERT_PROMPT,
        top_k_filter: Optional[int] = 100,
        **kw,
    ) -> None:
        super().__init__(lm, **kw)
        self.strength = float(strength)
        self.expert_prompt = expert_prompt
        self.antiexpert_prompt = antiexpert_prompt
        # DExperts restricts the ensemble to the base model's top-k support so
        # that the anti-expert cannot resurrect implausible tokens.
        self.top_k_filter = top_k_filter

    @contextlib.contextmanager
    def without_support_restriction(self):
        """Lift the top-k truncation so perplexity measures the density, not the support."""
        saved, self.top_k_filter = self.top_k_filter, None
        try:
            yield self
        finally:
            self.top_k_filter = saved

    def branches(self, prompts: Sequence[str], terms) -> Dict[str, List[List[int]]]:
        tok = self.lm.tokenizer
        return {
            "factual": [tok.encode(p, add_special_tokens=True) for p in prompts],
            "expert": [tok.encode(self.expert_prompt + p, add_special_tokens=True) for p in prompts],
            "antiexpert": [
                tok.encode(self.antiexpert_prompt + p, add_special_tokens=True) for p in prompts
            ],
        }

    def step_distribution(self, logits: Dict[str, torch.Tensor], t: int) -> StepOutput:
        z_base = logits["factual"].float()
        z_exp = logits["expert"].float()
        z_anti = logits["antiexpert"].float()

        z = z_base + self.strength * (z_exp - z_anti)

        if self.top_k_filter:
            k = min(self.top_k_filter, z_base.shape[-1])
            kth = z_base.topk(k, dim=-1).values[..., -1:]
            z = z.masked_fill(z_base < kth, float("-inf"))

        return StepOutput(probs=torch.softmax(z / self.temperature, dim=-1))
