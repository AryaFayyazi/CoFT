"""Offline split calibration (paper Sec. 3.4, Eq. 6; App. B.2, B.5).

For every calibration context ``i`` and step ``t`` we record the nonconformity
score of the **observed** next token ``v*_t``,

    s_t(v*_t) = 1 - min{ pi_hat_t(v*_t), pi^CF_t(v*_t) },

then take the ceiling-corrected ``(1 - alpha)`` quantile within each position
bin.  Nothing here touches the evaluation split: ``D_cal`` is disjoint by
construction (App. C.2, "a disjoint calibration pool (10-15%); no test leakage").

Sweeping cheaply
----------------
The two branch *logits* do not depend on ``lambda`` or ``alpha``.  A single pass
over ``D_cal`` therefore suffices for every point of both ablation sweeps
(Figs. 3 and 4): scores are accumulated for all requested ``lambda`` at once, and
``alpha`` is applied afterwards when the quantile is taken.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from coft.conformal import (
    DEFAULT_BIN_WIDTH,
    DEFAULT_MAX_POSITION,
    ConformalCalibrator,
    ConformalThresholds,
)
from coft.fusion import fuse_logits
from coft.masking import Masker

__all__ = ["collect_calibration_scores", "calibrate", "CalibrationBundle"]


class CalibrationBundle:
    """Raw per-``lambda`` calibration scores, from which any ``alpha`` can be read off."""

    def __init__(
        self,
        calibrators: Dict[float, ConformalCalibrator],
        score: str,
        n_contexts: int,
        meta: Optional[Dict] = None,
    ) -> None:
        self.calibrators = calibrators
        self.score = score
        self.n_contexts = n_contexts
        self.meta = meta or {}

    @property
    def lams(self) -> List[float]:
        return sorted(self.calibrators)

    def thresholds(self, lam: float, alpha: float) -> ConformalThresholds:
        """Finalize thresholds for one ``(lambda, alpha)`` pair."""
        if lam not in self.calibrators:
            raise KeyError(f"no calibration scores for lambda={lam}; collected {self.lams}")
        src = self.calibrators[lam]
        cal = ConformalCalibrator(
            alpha=alpha, bin_width=src.bin_width, max_position=src.max_position, score=src.score
        )
        cal._scores = [list(b) for b in src._scores]
        return cal.finalize(meta={"lambda": lam, "n_contexts": self.n_contexts, **self.meta})


@torch.no_grad()
def collect_calibration_scores(
    lm,
    corpus: Sequence[Tuple[str, str, Sequence[str]]],
    lams: Sequence[float] = (0.6,),
    score: str = "dual",
    masker: Optional[Masker] = None,
    temperature: float = 1.0,
    batch_size: int = 8,
    bin_width: int = DEFAULT_BIN_WIDTH,
    max_position: int = DEFAULT_MAX_POSITION,
    max_continuation_tokens: int = 64,
    progress: bool = True,
) -> CalibrationBundle:
    """One pass over ``D_cal`` accumulating scores for every requested ``lambda``.

    ``corpus`` holds ``(prompt, reference continuation, sensitive terms)`` triples.
    """
    masker = masker or Masker(lm.tokenizer)
    lams = list(dict.fromkeys(float(x) for x in lams))
    calibrators = {
        lam: ConformalCalibrator(
            alpha=0.10, bin_width=bin_width, max_position=max_position, score=score
        )
        for lam in lams
    }

    tok = lm.tokenizer
    batches = [list(corpus[i : i + batch_size]) for i in range(0, len(corpus), batch_size)]
    for batch in tqdm(batches, desc=f"calibrate[{score}]", disable=not progress):
        prompts = [p for p, _, _ in batch]
        conts = [
            tok.encode(c, add_special_tokens=False)[:max_continuation_tokens] for _, c, _ in batch
        ]
        keep = [i for i, c in enumerate(conts) if len(c) > 0]
        if not keep:
            continue
        prompts = [prompts[i] for i in keep]
        conts = [conts[i] for i in keep]
        terms = [list(batch[i][2]) for i in keep]

        masked = [masker.mask(p, terms=t) for p, t in zip(prompts, terms)]
        branches = {
            "factual": [m.factual_ids for m in masked],
            "masked": [m.masked_ids for m in masked],
        }
        per_branch = lm.branch_logits_teacher_forced_batch(branches, conts)

        dev = per_branch["factual"].device
        B = len(conts)
        L = max(len(c) for c in conts)
        lengths = torch.tensor([len(c) for c in conts], device=dev)
        padded = torch.zeros((B, L), dtype=torch.long, device=dev)
        for i, c in enumerate(conts):
            padded[i, : len(c)] = torch.tensor(c, dtype=torch.long, device=dev)

        for t in range(L):
            live = lengths > t
            if not bool(live.any()):
                break
            zf = per_branch["factual"][:, t, :]
            zcf = per_branch["masked"][:, t, :]
            pi_cf = torch.softmax(zcf / temperature, dim=-1)
            tok_t = padded[:, t].view(-1, 1)
            p_cf = pi_cf.gather(-1, tok_t).squeeze(-1)

            for lam in lams:
                pi_hat = torch.softmax(fuse_logits(zf, zcf, lam) / temperature, dim=-1)
                p_hat = pi_hat.gather(-1, tok_t).squeeze(-1)
                if score == "dual":
                    s = 1.0 - torch.minimum(p_hat, p_cf)
                else:
                    s = 1.0 - p_hat
                vals = s[live].tolist()
                calibrators[lam].add_batch([t] * len(vals), vals)

    return CalibrationBundle(
        calibrators,
        score=score,
        n_contexts=len(corpus),
        meta={"temperature": temperature, "max_continuation_tokens": max_continuation_tokens},
    )


def calibrate(
    lm,
    corpus,
    lam: float = 0.6,
    alpha: float = 0.10,
    score: str = "dual",
    **kw,
) -> ConformalThresholds:
    """Convenience wrapper: collect scores for one ``lambda`` and finalize at ``alpha``."""
    bundle = collect_calibration_scores(lm, corpus, lams=[lam], score=score, **kw)
    return bundle.thresholds(lam, alpha)
