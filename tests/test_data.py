"""Dataset arrangement invariants -- the shapes the method depends on."""

from __future__ import annotations

import pytest

from coft.data.base import PairItem, calibration_triples
from coft.data.bias import DatasetUnavailable, _minimal_pair_split, load_crows


def test_minimal_pair_split_isolates_the_modified_span():
    a = "most black kids do not have the best education"
    b = "most white kids do not have the best education"
    prefix, mid_a, mid_b, suffix = _minimal_pair_split(a, b)
    assert prefix == ["most"]
    assert mid_a == ["black"] and mid_b == ["white"]
    assert suffix == ["kids", "do", "not", "have", "the", "best", "education"]


def test_minimal_pair_split_handles_multiword_and_length_changes():
    a = "the Middle Eastern man arrived early"
    b = "the Irish man arrived early"
    prefix, mid_a, mid_b, suffix = _minimal_pair_split(a, b)
    assert prefix == ["the"]
    assert mid_a == ["Middle", "Eastern"] and mid_b == ["Irish"]
    assert suffix == ["man", "arrived", "early"]


def test_minimal_pair_split_with_no_shared_suffix():
    prefix, mid_a, mid_b, suffix = _minimal_pair_split("he was poor", "he was rich")
    assert prefix == ["he", "was"]
    assert suffix == []
    assert mid_a == ["poor"] and mid_b == ["rich"]


def test_pair_item_defaults_to_a_shared_context():
    it = PairItem(context="Many people live in Ethiopia.", stereo=" A", anti=" B", terms=["Ethiopia"])
    assert not it.split_contexts
    assert it.ctx_stereo == it.ctx_anti == it.context
    assert it.spans_stereo == it.spans_anti == ["Ethiopia"]


def test_pair_item_split_contexts():
    it = PairItem(
        context="x a black", stereo=" tail", anti=" tail",
        stereo_context="x a black", anti_context="x a white",
        stereo_terms=["black"], anti_terms=["white"],
    )
    assert it.split_contexts
    assert it.ctx_stereo != it.ctx_anti
    assert it.spans_stereo == ["black"] and it.spans_anti == ["white"]


def test_calibration_triples_use_the_anti_branch_prompt():
    it = PairItem(
        context="x a black", stereo=" tail", anti=" tail",
        stereo_context="x a black", anti_context="x a white",
        stereo_terms=["black"], anti_terms=["white"],
    )
    (prompt, cont, terms), = calibration_triples([it])
    assert prompt == "x a white"
    assert terms == ["white"]
    assert cont.strip() == "tail"


@pytest.mark.slow
def test_crows_puts_the_protected_span_in_the_prompt():
    """The protected span must be maskable, i.e. in the prompt and not the continuation.

    Both COFT branches receive the continuation verbatim (Sec. 3.1), so a
    protected span sitting there would be invisible to the masked probe.
    """
    try:
        items = load_crows(limit=200, seed=0)
    except DatasetUnavailable:
        pytest.skip("CrowS-Pairs not downloaded (run scripts/fetch_data.py)")

    split = [it for it in items if it.split_contexts]
    assert len(split) > 0.7 * len(items), "most pairs should use the modified-as-prompt arrangement"
    for it in split:
        # the modified span differs between branches and lives in the prompt
        assert it.ctx_stereo != it.ctx_anti
        assert it.stereo == it.anti            # the scored remainder is shared
        assert it.meta["modified_stereo"] in it.ctx_stereo
        assert it.meta["modified_anti"] in it.ctx_anti
