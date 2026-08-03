"""Metric definitions -- the scales that make the tables reproducible."""

from __future__ import annotations

import math

import pytest

from coft.metrics import (
    average_rank,
    bbq_metrics,
    choice_accuracy,
    crows_metrics,
    extract_final_number,
    gsm8k_exact_match,
    parity_gap,
    perplexity_from_logprobs,
    stereoset_metrics,
    toxicity_metrics,
)


def test_stereoset_parity_is_zero_bias():
    stereo = [1.0, 0.0, 1.0, 0.0]
    anti = [0.0, 1.0, 0.0, 1.0]
    m = stereoset_metrics(stereo, anti)
    assert m["ss_raw"] == 50.0
    assert m["ss_bias"] == 0.0


def test_stereoset_full_stereotype_preference():
    m = stereoset_metrics([1.0] * 10, [0.0] * 10)
    assert m["ss_raw"] == 100.0
    assert m["ss_bias"] == 1.0


def test_stereoset_anti_preference_clamps_at_zero():
    """Preferring anti-stereotypes is not rewarded below zero."""
    m = stereoset_metrics([0.0] * 10, [1.0] * 10)
    assert m["ss_bias"] == 0.0


def test_stereoset_icat():
    m = stereoset_metrics([1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [-5.0] * 4)
    assert m["lms"] == 100.0
    assert m["icat"] == 100.0     # ss = 50 -> perfect ICAT


def test_ties_score_as_no_preference():
    """Exact ties are abstentions, not anti-stereotype wins.

    As lambda -> 1 both CrowS branches mask to the same prompt, so their scores
    coincide; a strict `>` would report perfect parity for what is an abstention.
    """
    tied = crows_metrics([1.0] * 10, [1.0] * 10)
    assert tied["cp_stereo"] == 50.0
    assert tied["cp_acc"] == 50.0
    assert tied["cp_parity_gap"] == 0.0

    ss = stereoset_metrics([1.0] * 10, [1.0] * 10)
    assert ss["ss_raw"] == 50.0 and ss["ss_bias"] == 0.0

    half = crows_metrics([1.0, 1.0], [0.0, 1.0])   # one win, one tie
    assert half["cp_stereo"] == 75.0


def test_crows_definition():
    m = crows_metrics([1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0])
    assert m["cp_stereo"] == 50.0
    assert m["cp_acc"] == 50.0
    assert m["cp_parity_gap"] == 0.0


def test_bbq_bias_counts_stereotype_aligned_errors():
    # item 0: picks the stereotyped identity, wrong  -> biased
    # item 1: picks the gold answer                  -> not biased
    # item 2: picks UNKNOWN                          -> not biased
    preds = [0, 1, 2]
    labels = [1, 1, 1]
    targets = [0, 0, 0]
    unknown = [2, 2, 2]
    m = bbq_metrics(preds, labels, targets, unknown)
    assert m["bbq_bias"] == 1 / 3
    assert m["bbq_acc"] == 100 * 1 / 3


def test_bbq_bias_zero_when_always_correct():
    m = bbq_metrics([1, 1], [1, 1], [0, 0], [2, 2])
    assert m["bbq_bias"] == 0.0


def test_bbq_headline_uses_ambiguous_items_only():
    """The bias column must not be contaminated by disambiguated items.

    On a disambiguated item the protected span *is* the evidence, so masking it
    lowers accuracy and scatters errors -- that is semantic drift, not bias.
    """
    preds = [0, 2, 1, 1]
    labels = [2, 2, 1, 1]          # ambiguous gold is UNKNOWN(2)
    conds = ["ambig", "ambig", "disambig", "disambig"]
    m = bbq_metrics(preds, labels, [0] * 4, [2] * 4, conditions=conds)

    assert m["bbq_bias"] == 0.5          # 1 of the 2 ambiguous picks the target
    assert m["n_ambig"] == 2 and m["n_disambig"] == 2
    assert m["bbq_acc_disambig"] == 100.0
    assert m["bbq_bias_disambig"] == 0.0


def test_bbq_falls_back_to_all_items_without_conditions():
    m = bbq_metrics([0, 2], [2, 2], [0, 0], [2, 2])
    assert m["bbq_bias"] == 0.5
    assert m["n_ambig"] == 2


def test_parity_gap():
    m = parity_gap({"A": [1.0, 1.0, 0.0], "B": [0.0, 0.0, 0.0]})
    assert m["dp_gap"] == 2 / 3
    assert m["n_groups"] == 2


def test_parity_gap_ignores_undersupported_groups():
    """A rare group must not dominate a max-over-groups statistic through noise."""
    rates = {
        "A": [1.0] * 40,
        "B": [0.9] * 40,
        "rare": [0.0, 0.0],          # 2 samples -- pure noise
    }
    m = parity_gap(rates, min_support=20)
    assert m["dp_gap"] == pytest.approx(0.1)      # A vs B, not A vs rare
    assert m["n_groups"] == 2
    assert "rare" in m["group_rates"]             # still reported, just not used
    assert m["group_counts"]["rare"] == 2


def test_parity_gap_falls_back_when_nothing_has_support():
    m = parity_gap({"A": [1.0], "B": [0.0]}, min_support=20)
    assert m["dp_gap"] == 1.0


def test_parity_gap_single_group_is_nan():
    assert math.isnan(parity_gap({"A": [1.0]})["dp_gap"])


def test_toxicity_metrics():
    m = toxicity_metrics([0.1, 0.9, 0.2, 0.8])
    assert m["toxicity"] == 0.5
    assert m["toxicity_rate"] == 0.5


def test_extract_final_number():
    assert extract_final_number("so 3 + 4 = 7. The answer is 7.") == "7"
    assert extract_final_number("blah 12 then 48") == "48"
    assert extract_final_number("The answer is $1,250.") == "1250"
    assert extract_final_number("no digits here") is None


def test_extract_uses_the_first_answer_marker():
    """Few-shot bleed: the model invents a further Question/Answer block.

    Taking the last marker would score the answer to the model's own
    hallucinated question, which gets worse the longer it is allowed to run.
    """
    text = "48 + 24 = 72. The answer is 72.\n\nQuestion: cost of gas?\nAnswer: The answer is 52."
    assert extract_final_number(text) == "72"


def test_gsm8k_exact_match():
    m = gsm8k_exact_match(["The answer is 72.", "The answer is 5."], ["72", "6"])
    assert m["acc"] == 50.0


def test_choice_accuracy():
    assert choice_accuracy([0, 1, 2], [0, 1, 1])["acc"] == 100 * 2 / 3


def test_perplexity():
    assert perplexity_from_logprobs(-10.0, 10) == math.exp(1.0)
    assert math.isnan(perplexity_from_logprobs(-1.0, 0))


def test_average_rank_orders_and_handles_ties():
    per_method = {
        "best": {"bias": 0.1, "acc": 90.0},
        "mid": {"bias": 0.2, "acc": 80.0},
        "worst": {"bias": 0.3, "acc": 70.0},
    }
    r = average_rank(per_method, [("bias", False), ("acc", True)])
    assert r["best"] == 1.0
    assert r["mid"] == 2.0
    assert r["worst"] == 3.0

    tied = {"a": {"x": 1.0}, "b": {"x": 1.0}}
    rt = average_rank(tied, [("x", False)])
    assert rt["a"] == rt["b"] == 1.5
