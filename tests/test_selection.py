"""The Pareto-knee selection rule of Sec. 4.5."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_sweep import DEGENERATE_LAMBDAS, pareto_knee, pareto_knee_choice  # noqa: E402


def _pts(lams, bias, util):
    return [{"lambda": l, "bias_avg": b, "utility_avg": u} for l, b, u in zip(lams, bias, util)]


def test_knee_is_the_trade_off_point_not_the_extreme():
    """A monotone bias curve must not select its endpoint.

    Selecting on bias alone always returns the largest lambda, which is lambda=1:
    the fused distribution is then exactly the masked one.
    """
    pts = _pts(
        [0.0, 0.2, 0.4, 0.6, 0.7, 0.8],
        [0.228, 0.207, 0.195, 0.190, 0.182, 0.172],
        [65.2, 63.7, 62.6, 63.4, 63.1, 61.6],
    )
    knee = pareto_knee(pts)
    assert knee is not None
    assert knee["lambda"] == 0.7
    assert pareto_knee_choice(pts, "lambda") == 0.7


def test_degenerate_lambda_is_never_selected():
    """lambda = 1 discards the factual branch; it is swept but not selectable."""
    pts = _pts(
        [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0],
        [0.242, 0.244, 0.235, 0.219, 0.214, 0.202, 0.144],
        [58.6, 56.1, 55.4, 55.7, 55.3, 54.7, 55.0],
    )
    # without the guard the rule collapses onto the endpoint
    assert pareto_knee_choice(pts, "lambda") == 0.8
    assert pareto_knee_choice(pts, "lambda", exclude=()) == 1.0
    assert DEGENERATE_LAMBDAS == (1.0,)


def test_flat_utility_has_no_knee():
    pts = _pts([0.0, 0.5, 1.0], [0.3, 0.2, 0.1], [70.0, 70.0, 70.0])
    assert pareto_knee(pts) is None


def test_tolerance_prefers_the_smaller_setting():
    """Within 2% of the knee, the smaller (less interventionist) value wins."""
    pts = _pts(
        [0.2, 0.3, 0.4, 0.6],
        [0.300, 0.204, 0.200, 0.150],
        [70.0, 66.0, 60.0, 50.0],
    )
    knee = pareto_knee(pts)
    assert knee is not None and knee["lambda"] == 0.3
    # 0.4 has lower bias but 0.3 is within 2% of it (0.204 <= 0.200 * 1.02) and smaller
    assert pareto_knee_choice(pts, "lambda") == 0.3


def test_too_few_points_is_undefined():
    assert pareto_knee(_pts([0.0, 1.0], [0.3, 0.1], [70.0, 60.0])) is None
