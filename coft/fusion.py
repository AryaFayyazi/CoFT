"""Stage II -- Counterfactual logit fusion (paper Sec. 3.3, Eq. 4).

The fusion rule is a convex interpolation in *logit* space,

    z_hat_t = z^F_t - lambda * Delta_t = (1 - lambda) z^F_t + lambda z^CF_t,
    Delta_t = z^F_t - z^CF_t,          lambda in [0, 1]                       (Eq. 4)

followed by a softmax.  By Lemma 1 (App. B.4) this is exactly the *normalised
geometric mixture* of the two next-token distributions,

    pi_hat_t(v)  =  (pi^F_t(v))^{1-lambda} (pi^CF_t(v))^{lambda} / Z_t(lambda),

and by Lemma 2 (App. B.4) the pairwise log-odds interpolate linearly in lambda
(Proposition 1, App. B.7).

All computation is carried out in float32 regardless of the model dtype: the
conformal score of Eq. 5 thresholds *probabilities* at values around 1e-2, and
bf16 has only ~3 decimal digits of mantissa, which would make the certified set
depend on rounding noise.
"""

from __future__ import annotations

import torch

__all__ = [
    "fuse_logits",
    "fused_distribution",
    "geometric_mixture",
    "attribute_sensitivity",
    "log_odds",
]


def attribute_sensitivity(factual_logits: torch.Tensor, masked_logits: torch.Tensor) -> torch.Tensor:
    """Per-token attribute sensitivity ``Delta_t = z^F_t - z^CF_t`` (Sec. 3.3)."""
    return factual_logits.float() - masked_logits.float()


def fuse_logits(
    factual_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Counterfactual logit fusion, Eq. 4.

    Parameters
    ----------
    factual_logits, masked_logits:
        Logit tensors ``z^F_t`` and ``z^CF_t`` of identical shape ``(..., V)``.
        They must be indexed by the *same* tokenizer/vocabulary (assumption A1).
    lam:
        Fusion scale ``lambda in [0, 1]``.  ``lam = 0`` recovers the factual
        branch; ``lam = 1`` recovers the masked branch.

    Returns
    -------
    torch.Tensor
        The fused logits ``z_hat_t`` in float32.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"fusion scale lambda must lie in [0, 1], got {lam}")
    if factual_logits.shape != masked_logits.shape:
        raise ValueError(
            "factual and masked logits must have the same shape "
            f"(got {tuple(factual_logits.shape)} vs {tuple(masked_logits.shape)}); "
            "this usually means the mask operator was not length-preserving"
        )
    zf = factual_logits.float()
    zcf = masked_logits.float()
    return (1.0 - lam) * zf + lam * zcf


def fused_distribution(
    factual_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    lam: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return ``pi_hat_t = softmax(z_hat_t / temperature)`` (Eq. 4)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    fused = fuse_logits(factual_logits, masked_logits, lam) / temperature
    return torch.softmax(fused, dim=-1)


def geometric_mixture(
    factual_probs: torch.Tensor,
    masked_probs: torch.Tensor,
    lam: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalised geometric mixture of Lemma 1 (App. B.4), computed in log space.

    This is an *independent* implementation of the same object produced by
    :func:`fused_distribution`; :mod:`tests.test_theory` asserts that the two
    agree, which is the empirical check of Lemma 1.
    """
    log_pf = torch.log(factual_probs.float().clamp_min(eps))
    log_pcf = torch.log(masked_probs.float().clamp_min(eps))
    return torch.softmax((1.0 - lam) * log_pf + lam * log_pcf, dim=-1)


def log_odds(probs: torch.Tensor, u: int, v: int, eps: float = 1e-12) -> torch.Tensor:
    """``log pi(u) / pi(v)`` -- the quantity that interpolates linearly (Lemma 2)."""
    p = probs.float().clamp_min(eps)
    return torch.log(p[..., u]) - torch.log(p[..., v])
