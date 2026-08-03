"""Empirical checks of the propositions and theorems in Sec. 3.6 / App. B.

Each test names the result it exercises.  They are pure-tensor tests -- no model
is loaded -- so they run in seconds and are the fastest way to detect a
regression in the mathematical core.
"""

from __future__ import annotations

import math

import pytest
import torch

from coft.conformal import ConformalCalibrator, ceiling_quantile, dual_branch_score
from coft.fusion import fused_distribution, geometric_mixture, log_odds

V = 512


def _random_logits(seed: int = 0, batch: int = 4, vocab: int = V, scale: float = 4.0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, vocab, generator=g) * scale,
        torch.randn(batch, vocab, generator=g) * scale,
    )


# --------------------------------------------------------------------------- #
# Lemma 1 (App. B.4): fusion == normalised geometric mixture
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 0.6, 0.9, 1.0])
def test_lemma1_geometric_mixture(lam):
    zf, zcf = _random_logits(1)
    pi_f, pi_cf = torch.softmax(zf, -1), torch.softmax(zcf, -1)

    from_logits = fused_distribution(zf, zcf, lam)
    from_probs = geometric_mixture(pi_f, pi_cf, lam)

    assert torch.allclose(from_logits, from_probs, atol=1e-5), "Lemma 1 violated"
    assert torch.allclose(from_logits.sum(-1), torch.ones(from_logits.shape[0]), atol=1e-5)


# --------------------------------------------------------------------------- #
# Lemma 2 / Proposition 1 (App. B.7): log-odds interpolate linearly in lambda
# --------------------------------------------------------------------------- #
def test_proposition1_log_odds_interpolation():
    zf, zcf = _random_logits(2, batch=1)
    pi_f, pi_cf = torch.softmax(zf, -1), torch.softmax(zcf, -1)
    u, v = 3, 17
    lo_f = log_odds(pi_f, u, v)
    lo_cf = log_odds(pi_cf, u, v)

    for lam in (0.0, 0.1, 0.37, 0.6, 0.85, 1.0):
        pi_hat = fused_distribution(zf, zcf, lam)
        expected = (1 - lam) * lo_f + lam * lo_cf
        assert torch.allclose(log_odds(pi_hat, u, v), expected, atol=1e-4), (
            f"log-odds not linear in lambda at lam={lam}"
        )


# --------------------------------------------------------------------------- #
# Theorem 2 (App. B.11): KL(pi_hat || pi^CF) is non-increasing in lambda
# --------------------------------------------------------------------------- #
def test_theorem2_monotone_kl_decay():
    zf, zcf = _random_logits(3, batch=6)
    pi_cf = torch.softmax(zcf, -1)

    prev = None
    for lam in [i / 20 for i in range(21)]:
        pi_hat = fused_distribution(zf, zcf, lam)
        kl = (pi_hat * (torch.log(pi_hat + 1e-30) - torch.log(pi_cf + 1e-30))).sum(-1)
        if prev is not None:
            assert bool((kl <= prev + 1e-5).all()), f"KL increased at lambda={lam}"
        prev = kl
    assert bool((prev < 1e-5).all()), "KL should vanish at lambda = 1"


def test_theorem2_fixed_point():
    """pi_hat == pi^F for some lambda in (0,1] iff pi^F == pi^CF."""
    zf, _ = _random_logits(4, batch=2)
    same = fused_distribution(zf, zf.clone(), 0.7)
    assert torch.allclose(same, torch.softmax(zf, -1), atol=1e-5)

    zf2, zcf2 = _random_logits(5, batch=2)
    moved = fused_distribution(zf2, zcf2, 0.7)
    assert not torch.allclose(moved, torch.softmax(zf2, -1), atol=1e-3)


def test_softmax_translation_invariance():
    """Eq. 12: softmax(z) == softmax(z + c) -- relied on by the fixed-point proof."""
    zf, zcf = _random_logits(6, batch=2)
    c = 3.14159
    assert torch.allclose(
        fused_distribution(zf, zcf, 0.4), fused_distribution(zf + c, zcf + c, 0.4), atol=1e-5
    )


# --------------------------------------------------------------------------- #
# Eq. 6 / App. B.5: ceiling-corrected quantile
# --------------------------------------------------------------------------- #
def test_ceiling_quantile_rank():
    scores = [i / 100 for i in range(100)]  # n = 100
    alpha = 0.1
    q = ceiling_quantile(scores, alpha)
    rank = math.ceil((1 - alpha) * (len(scores) + 1))     # = 91
    assert q == pytest.approx(sorted(scores)[rank - 1])


def test_ceiling_quantile_small_sample_is_conservative():
    """When ceil((1-a)(n+1)) > n the quantile is +inf; on [0,1] scores that is 1.0."""
    assert ceiling_quantile([0.2, 0.3], 0.1) == 1.0
    assert ceiling_quantile([], 0.1) == 1.0


# --------------------------------------------------------------------------- #
# Theorem 1 (App. B.8): dual-branch marginal coverage >= 1 - alpha
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_theorem1_marginal_coverage(alpha):
    """Exchangeable synthetic contexts: empirical coverage must reach 1 - alpha.

    Calibration and test scores are drawn i.i.d. from the same generator, which
    is exactly assumption (A2); the split-conformal guarantee should then hold up
    to Monte-Carlo noise.
    """
    torch.manual_seed(0)
    n_cal, n_test, vocab = 3000, 3000, 256

    def draw(n):
        zf = torch.randn(n, vocab) * 3
        zcf = zf + torch.randn(n, vocab) * 1.5
        pi_hat = fused_distribution(zf, zcf, 0.6)
        pi_cf = torch.softmax(zcf, -1)
        true = torch.multinomial(pi_hat, 1).squeeze(-1)
        return pi_hat, pi_cf, true

    pi_hat, pi_cf, true = draw(n_cal)
    cal = ConformalCalibrator(alpha=alpha, score="dual")
    idx = true.view(-1, 1)
    scores = 1.0 - torch.minimum(pi_hat.gather(-1, idx), pi_cf.gather(-1, idx)).squeeze(-1)
    cal.add_batch([0] * n_cal, scores.tolist())
    th = cal.finalize()

    pi_hat_t, pi_cf_t, true_t = draw(n_test)
    mask = th.candidate_mask(pi_hat_t, pi_cf_t, 0)
    covered = mask.gather(-1, true_t.view(-1, 1)).squeeze(-1).float().mean().item()

    assert covered >= 1 - alpha - 0.02, f"coverage {covered:.3f} below 1-alpha={1-alpha:.2f}"
    # not absurdly conservative either
    assert covered <= 1 - alpha + 0.06


def test_dual_branch_score_definition():
    """Eq. 5: s_t(v) = 1 - min(pi_hat(v), pi^CF(v))."""
    a = torch.tensor([[0.4, 0.1, 0.5]])
    b = torch.tensor([[0.2, 0.7, 0.1]])
    assert torch.allclose(dual_branch_score(a, b), torch.tensor([[0.8, 0.9, 0.9]]), atol=1e-6)


# --------------------------------------------------------------------------- #
# Theorem 3 (App. B.12): C_t = U_t ∩ V_t and |C_t| <= min(|U_t|, |V_t|)
# --------------------------------------------------------------------------- #
def test_theorem3_set_size_bound():
    torch.manual_seed(1)
    pi_hat = torch.softmax(torch.randn(8, V) * 3, -1)
    pi_cf = torch.softmax(torch.randn(8, V) * 3, -1)
    tau = 1.0 / V

    U = pi_hat >= tau
    W = pi_cf >= tau
    C = torch.minimum(pi_hat, pi_cf) >= tau

    assert torch.equal(C, U & W), "C_t is not the intersection U_t ∩ V_t"
    assert bool((C.sum(-1) <= torch.minimum(U.sum(-1), W.sum(-1))).all())


# --------------------------------------------------------------------------- #
# Lemma 3 / Corollary 1 (App. B.6, B.9): TV under restriction to a common support
# --------------------------------------------------------------------------- #
def test_corollary1_tv_bound_under_restriction():
    torch.manual_seed(2)
    for _ in range(20):
        pi_hat = torch.softmax(torch.randn(V) * 2, -1)
        pi_cf = torch.softmax(torch.randn(V) * 2, -1)
        tau = 1.0 / (4 * V)
        A = torch.minimum(pi_hat, pi_cf) >= tau
        if A.sum() < 2:
            continue

        p_a = (pi_hat * A) / (pi_hat * A).sum()
        q_a = (pi_cf * A) / (pi_cf * A).sum()
        tv_restricted = 0.5 * (p_a - q_a).abs().sum()

        tv_full = 0.5 * (pi_hat - pi_cf).abs().sum()
        denom = min(float((pi_hat * A).sum()), float((pi_cf * A).sum()))
        assert tv_restricted <= tv_full / denom + 1e-6, "Lemma 3 bound violated"

        kl = (pi_hat * (torch.log(pi_hat + 1e-30) - torch.log(pi_cf + 1e-30))).sum()
        pinsker = math.sqrt(max(0.0, 0.5 * float(kl)))
        assert tv_restricted <= pinsker / max(denom, 1e-12) + 1e-6, "Corollary 1 bound violated"


# --------------------------------------------------------------------------- #
# Theorem 4 (App. B.13): union-bound composition
# --------------------------------------------------------------------------- #
def test_theorem4_union_bound():
    alpha, T = 0.1, 5
    torch.manual_seed(3)
    per_step_cov = torch.full((10000, T), 0.0)
    for t in range(T):
        per_step_cov[:, t] = (torch.rand(10000) < 1 - alpha).float()
    joint = per_step_cov.prod(-1).mean().item()
    assert joint >= 1 - T * alpha - 0.02
