"""Self-Debiased Decoding (SDD) -- anti-prompt logit subtraction.

Schick, Udupa & Schuetze (2021), "Self-Diagnosis and Self-Debiasing".

The model is run twice: once on the plain input ``x`` and once on a
*self-debiasing* input ``sdb(x)`` that explicitly asks for the undesired
behaviour.  Tokens made *more* likely by the biased framing are damped:

    Delta(w, x) = p(w | x) - p(w | sdb(x))
    alpha(Delta) = 1                if Delta >= 0
                 = exp(decay * Delta)  otherwise
    p_tilde(w | x)  proportional to  alpha(Delta(w, x)) * p(w | x)

This is a faithful reimplementation of the original scaling rule, run under the
shared decoding policy so it is directly comparable to COFT.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from coft.decoding import BaseDecoder, StepOutput

__all__ = ["SDDDecoder", "DEFAULT_SDB_PROMPT"]

#: Covers the attribute families spanned by the paper's bias suite (gender,
#: race, religion, age, ...), in the imperative style of the original work.
DEFAULT_SDB_PROMPT = (
    "The following text is biased, discriminatory and relies on harmful "
    "stereotypes about gender, race, religion, age and disability:\n"
)


class SDDDecoder(BaseDecoder):
    name = "sdd"
    branch_names = ("factual", "sdb")

    def __init__(self, lm, sdb_prompt: str = DEFAULT_SDB_PROMPT, decay: float = 50.0, **kw) -> None:
        super().__init__(lm, **kw)
        self.sdb_prompt = sdb_prompt
        self.decay = float(decay)

    def branches(self, prompts: Sequence[str], terms) -> Dict[str, List[List[int]]]:
        tok = self.lm.tokenizer
        factual = [tok.encode(p, add_special_tokens=True) for p in prompts]
        biased = [tok.encode(self.sdb_prompt + p, add_special_tokens=True) for p in prompts]
        return {"factual": factual, "sdb": biased}

    def step_distribution(self, logits: Dict[str, torch.Tensor], t: int) -> StepOutput:
        p_norm = torch.softmax(logits["factual"].float() / self.temperature, dim=-1)
        p_bias = torch.softmax(logits["sdb"].float() / self.temperature, dim=-1)

        delta = p_norm - p_bias
        alpha = torch.where(delta >= 0, torch.ones_like(delta), torch.exp(self.decay * delta))
        scaled = alpha * p_norm
        total = scaled.sum(dim=-1, keepdim=True)
        probs = torch.where(total > 0, scaled / total.clamp_min(1e-12), p_norm)
        return StepOutput(probs=probs)
