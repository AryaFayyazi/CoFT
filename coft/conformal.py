"""Stage III -- Dual-branch split-conformal filtering (paper Sec. 3.4, App. B.5).

Nonconformity score (Eq. 5)::

    s_t(v) = 1 - min{ pi_hat_t(v), pi^CF_t(v) }

so ``s_t(v)`` is small only when ``v`` is probable in *both* worlds -- the fused
(debiased) view and the masked view.

Split calibration (Eq. 6) computes, offline and on a disjoint calibration set,
the ceiling-corrected empirical ``(1 - alpha)`` quantile ``q_t`` of the scores of
the *true* next tokens.  At test time the certified candidate set is (Eq. 7)::

    C_t = { v : s_t(v) <= q_t } = { v : min(pi_hat_t(v), pi^CF_t(v)) >= tau_t },
    tau_t = 1 - q_t

Theorem 1 (App. B.8) then gives, under exchangeability (A2),

    P[ v*_t in C_t under p  AND  v*_t in C_t under p~ ] >= 1 - alpha.

Position binning
----------------
Open-ended generations have variable length, so late-step calibration data is
sparse.  Following Sec. 3.4 we share a threshold across *position bins* of width
8 up to a maximum ``T`` (default 256) and tie every step beyond ``T`` to the last
bin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch

__all__ = [
    "dual_branch_score",
    "single_branch_score",
    "ceiling_quantile",
    "ConformalCalibrator",
    "ConformalThresholds",
    "DEFAULT_BIN_WIDTH",
    "DEFAULT_MAX_POSITION",
]

DEFAULT_BIN_WIDTH = 8
DEFAULT_MAX_POSITION = 256


# --------------------------------------------------------------------------- #
# scores
# --------------------------------------------------------------------------- #
def dual_branch_score(fused_probs: torch.Tensor, masked_probs: torch.Tensor) -> torch.Tensor:
    """Eq. 5, evaluated for *every* token in the vocabulary.

    Returns a tensor shaped like the inputs holding ``1 - min(pi_hat, pi^CF)``.
    """
    return 1.0 - torch.minimum(fused_probs.float(), masked_probs.float())


def single_branch_score(fused_probs: torch.Tensor, *_ignored) -> torch.Tensor:
    """Factual-only ablation score ``1 - pi_hat_t(v)``.

    Used by the "Single-branch CP (factual)" row of Table 4 and by the DT-CD
    baseline; the paper notes such a score "cannot guarantee stability to
    masking" (Sec. 3.4).
    """
    return 1.0 - fused_probs.float()


_SCORES = {"dual": dual_branch_score, "single": single_branch_score}


# --------------------------------------------------------------------------- #
# quantile
# --------------------------------------------------------------------------- #
def ceiling_quantile(scores: Sequence[float], alpha: float) -> float:
    """Finite-sample split-conformal quantile (Eq. 6, App. B.5).

    ``q`` is the ``ceil((1 - alpha)(n + 1))``-th smallest of the ``n``
    calibration scores.  When that rank exceeds ``n`` the quantile is
    ``+infinity``; because our scores live in ``[0, 1]`` we return ``1.0``, which
    admits the whole vocabulary -- the conservative, guarantee-preserving
    behaviour for tiny calibration sets.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    n = len(scores)
    if n == 0:
        return 1.0
    rank = math.ceil((1.0 - alpha) * (n + 1))
    if rank > n:
        return 1.0
    ordered = sorted(scores)
    return float(ordered[rank - 1])


# --------------------------------------------------------------------------- #
# calibrated thresholds
# --------------------------------------------------------------------------- #
@dataclass
class ConformalThresholds:
    """Per-position-bin thresholds ``tau_t = 1 - q_t`` produced by split calibration."""

    alpha: float
    q_by_bin: List[float]
    counts_by_bin: List[int]
    bin_width: int = DEFAULT_BIN_WIDTH
    max_position: int = DEFAULT_MAX_POSITION
    score: str = "dual"
    meta: Dict = field(default_factory=dict)

    # -- lookup ------------------------------------------------------------ #
    def bin_index(self, step: int) -> int:
        """Map decode step ``t`` to its bin, tying everything past ``T`` to the last bin."""
        if step < 0:
            raise ValueError("decode step must be non-negative")
        idx = min(step, self.max_position - 1) // self.bin_width
        return min(idx, len(self.q_by_bin) - 1)

    def q(self, step: int) -> float:
        return self.q_by_bin[self.bin_index(step)]

    def tau(self, step: int) -> float:
        return 1.0 - self.q(step)

    # -- the certified set -------------------------------------------------- #
    def candidate_mask(
        self,
        fused_probs: torch.Tensor,
        masked_probs: Optional[torch.Tensor],
        step: int,
    ) -> torch.Tensor:
        """Boolean mask of ``C_t`` (Eq. 7) over the vocabulary."""
        score_fn = _SCORES[self.score]
        if self.score == "dual":
            if masked_probs is None:
                raise ValueError("dual-branch certification requires the masked distribution")
            scores = score_fn(fused_probs, masked_probs)
        else:
            scores = score_fn(fused_probs)
        return scores <= self.q(step)

    # -- (de)serialisation --------------------------------------------------- #
    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "q_by_bin": self.q_by_bin,
            "counts_by_bin": self.counts_by_bin,
            "tau_by_bin": [1.0 - q for q in self.q_by_bin],
            "bin_width": self.bin_width,
            "max_position": self.max_position,
            "score": self.score,
            "meta": self.meta,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: Dict) -> "ConformalThresholds":
        return cls(
            alpha=d["alpha"],
            q_by_bin=list(d["q_by_bin"]),
            counts_by_bin=list(d["counts_by_bin"]),
            bin_width=d.get("bin_width", DEFAULT_BIN_WIDTH),
            max_position=d.get("max_position", DEFAULT_MAX_POSITION),
            score=d.get("score", "dual"),
            meta=d.get("meta", {}),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConformalThresholds":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def permissive(cls, alpha: float = 0.1, **kw) -> "ConformalThresholds":
        """Thresholds that certify everything -- the ``no-CP`` ablation of Table 4."""
        n_bins = max(1, DEFAULT_MAX_POSITION // DEFAULT_BIN_WIDTH)
        return cls(alpha=alpha, q_by_bin=[1.0] * n_bins, counts_by_bin=[0] * n_bins, **kw)


# --------------------------------------------------------------------------- #
# calibrator
# --------------------------------------------------------------------------- #
class ConformalCalibrator:
    """Accumulates true-next-token scores and emits :class:`ConformalThresholds`.

    Usage mirrors Eq. 6: for every calibration context ``i`` and step ``t`` we
    record ``s_t(v*_t)`` where ``v*_t`` is the *observed* next token, then take
    the ceiling-corrected ``(1 - alpha)`` quantile within each position bin.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        bin_width: int = DEFAULT_BIN_WIDTH,
        max_position: int = DEFAULT_MAX_POSITION,
        score: str = "dual",
    ) -> None:
        if score not in _SCORES:
            raise ValueError(f"unknown score '{score}', expected one of {sorted(_SCORES)}")
        self.alpha = alpha
        self.bin_width = bin_width
        self.max_position = max_position
        self.score = score
        self.n_bins = max(1, max_position // bin_width)
        self._scores: List[List[float]] = [[] for _ in range(self.n_bins)]

    # ------------------------------------------------------------------ #
    def _bin(self, step: int) -> int:
        return min(min(step, self.max_position - 1) // self.bin_width, self.n_bins - 1)

    def add(self, step: int, score: float) -> None:
        """Record one calibration score at decode step ``t``."""
        self._scores[self._bin(step)].append(float(score))

    def add_batch(self, steps: Sequence[int], scores: Sequence[float]) -> None:
        for s, sc in zip(steps, scores):
            self.add(s, sc)

    def add_from_probs(
        self,
        step: int,
        fused_probs: torch.Tensor,
        masked_probs: Optional[torch.Tensor],
        true_token: int,
    ) -> float:
        """Compute and record ``s_t(v*_t)`` from the two branch distributions."""
        pf = float(fused_probs[true_token])
        if self.score == "dual":
            if masked_probs is None:
                raise ValueError("dual-branch calibration requires the masked distribution")
            value = 1.0 - min(pf, float(masked_probs[true_token]))
        else:
            value = 1.0 - pf
        self.add(step, value)
        return value

    # ------------------------------------------------------------------ #
    @property
    def n_scores(self) -> int:
        return sum(len(b) for b in self._scores)

    def finalize(self, meta: Optional[Dict] = None) -> ConformalThresholds:
        """Compute per-bin quantiles.

        Bins that never received a score inherit the nearest populated bin to
        their left (and, failing that, to their right).  An entirely empty
        calibration set yields the permissive ``q = 1`` thresholds, which keeps
        the coverage statement true (trivially) instead of silently filtering
        with an unjustified threshold.
        """
        q_by_bin: List[float] = []
        counts: List[int] = []
        for bin_scores in self._scores:
            counts.append(len(bin_scores))
            q_by_bin.append(ceiling_quantile(bin_scores, self.alpha) if bin_scores else float("nan"))

        # forward fill, then backward fill
        last = None
        for i, q in enumerate(q_by_bin):
            if math.isnan(q):
                q_by_bin[i] = last if last is not None else float("nan")
            else:
                last = q
        nxt = None
        for i in range(len(q_by_bin) - 1, -1, -1):
            if math.isnan(q_by_bin[i]):
                q_by_bin[i] = nxt if nxt is not None else 1.0
            else:
                nxt = q_by_bin[i]

        return ConformalThresholds(
            alpha=self.alpha,
            q_by_bin=[float(q) for q in q_by_bin],
            counts_by_bin=counts,
            bin_width=self.bin_width,
            max_position=self.max_position,
            score=self.score,
            meta={"n_scores": self.n_scores, **(meta or {})},
        )
