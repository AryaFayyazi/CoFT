"""Split-conformal plumbing: binning, serialisation, ablation scores, Eq. 7/8."""

from __future__ import annotations

import math

import pytest
import torch

from coft.conformal import (
    ConformalCalibrator,
    ConformalThresholds,
    single_branch_score,
)
from coft.decoding import StepOutput, top_p_filter


def _make_thresholds(alpha=0.1, n=500, score="dual"):
    torch.manual_seed(0)
    cal = ConformalCalibrator(alpha=alpha, score=score)
    for t in range(0, 64):
        cal.add_batch([t] * n, torch.rand(n).tolist())
    return cal.finalize()


def test_position_bins_and_tying():
    th = _make_thresholds()
    assert th.bin_index(0) == 0
    assert th.bin_index(7) == 0
    assert th.bin_index(8) == 1
    # every step beyond max_position is tied to the last bin (Sec. 3.4)
    assert th.bin_index(10_000) == th.bin_index(th.max_position - 1)
    assert th.tau(0) == 1.0 - th.q(0)


def test_empty_bins_are_filled_not_dropped():
    cal = ConformalCalibrator(alpha=0.1)
    cal.add_batch([0] * 200, torch.rand(200).tolist())   # only bin 0 populated
    th = cal.finalize()
    assert all(q == q for q in th.q_by_bin), "NaN threshold leaked through"
    assert th.q(200) == th.q(0)


def test_no_calibration_data_is_permissive():
    th = ConformalCalibrator(alpha=0.1).finalize()
    assert all(q == 1.0 for q in th.q_by_bin)
    probs = torch.softmax(torch.randn(1, 64), -1)
    mask = th.candidate_mask(probs, probs, 0)
    assert bool(mask.all()), "an uncalibrated threshold must not filter anything"


def test_roundtrip_serialisation(tmp_path):
    th = _make_thresholds()
    p = tmp_path / "cal.json"
    th.save(p)
    back = ConformalThresholds.load(p)
    assert back.alpha == th.alpha
    assert back.q_by_bin == th.q_by_bin
    assert back.score == th.score
    assert back.tau(13) == th.tau(13)


def test_candidate_mask_matches_definition():
    """Eq. 7: C_t = {v : min(pi_hat, pi^CF) >= tau_t}."""
    th = _make_thresholds()
    torch.manual_seed(1)
    pi_hat = torch.softmax(torch.randn(3, 128) * 2, -1)
    pi_cf = torch.softmax(torch.randn(3, 128) * 2, -1)
    mask = th.candidate_mask(pi_hat, pi_cf, 5)
    expected = torch.minimum(pi_hat, pi_cf) >= th.tau(5)
    assert torch.equal(mask, expected)


def test_single_branch_score_ignores_masked_branch():
    th = _make_thresholds(score="single")
    pi_hat = torch.softmax(torch.randn(2, 64), -1)
    m1 = th.candidate_mask(pi_hat, None, 0)
    assert torch.equal(m1, single_branch_score(pi_hat) <= th.q(0))


def test_effective_probs_renormalises_on_the_certified_set():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    mask = torch.tensor([[True, True, False, False]])
    out = StepOutput(probs=probs, certified_mask=mask, tau=0.1)
    eff = out.effective_probs()
    assert torch.allclose(eff, torch.tensor([[0.625, 0.375, 0.0, 0.0]]), atol=1e-6)
    assert float(eff.sum()) == 1.0


def test_empty_set_falls_back_to_unrestricted():
    """Eq. 8: when C_t is empty the policy is argmax over the *unrestricted* pi_hat."""
    probs = torch.tensor([[0.5, 0.3, 0.2]])
    mask = torch.zeros_like(probs, dtype=torch.bool)
    out = StepOutput(probs=probs, certified_mask=mask, tau=0.9, empty_set=torch.tensor([True]))
    eff = out.effective_probs()
    assert torch.allclose(eff, probs)
    assert int(eff.argmax()) == 0


def test_uncertified_token_logprob_is_floored_at_tau():
    """The evaluation convention documented in coft.decoding."""
    probs = torch.tensor([[0.6, 0.3, 0.1]])
    mask = torch.tensor([[True, True, False]])
    tau = 0.2
    out = StepOutput(probs=probs, certified_mask=mask, tau=tau)

    lp_in = float(out.token_logprob(torch.tensor([0])))
    lp_out = float(out.token_logprob(torch.tensor([2])))
    mass = 0.9
    assert lp_in == pytest.approx(math.log(0.6 / mass), abs=1e-5)
    # min(p, tau) = min(0.1, 0.2) = 0.1
    assert lp_out == pytest.approx(math.log(0.1 / mass), abs=1e-5)
    assert lp_out < lp_in
    assert lp_out > float("-inf")


def test_top_p_filter_keeps_nucleus():
    probs = torch.tensor([[0.5, 0.25, 0.15, 0.07, 0.03]])
    out = top_p_filter(probs, 0.9)
    assert out[0, 4] == 0.0
    assert torch.allclose(out.sum(-1), torch.ones(1), atol=1e-6)
    assert out[0, 0] > out[0, 1] > out[0, 2]


def test_top_p_one_is_identity():
    probs = torch.softmax(torch.randn(2, 32), -1)
    assert torch.allclose(top_p_filter(probs, 1.0), probs)
