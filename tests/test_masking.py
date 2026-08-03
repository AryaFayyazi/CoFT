"""Stage I invariants: length preservation, idempotence, order preservation (Eq. 3)."""

from __future__ import annotations

import os

import pytest

from coft.masking import Masker, resolve_sentinel
from coft.spans import SensitiveLexicon, detect_spans

TOKENIZER_ID = os.environ.get("COFT_TEST_TOKENIZER", "mistralai/Mistral-7B-Instruct-v0.2")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            TOKENIZER_ID, cache_dir=os.environ.get("HF_HUB_CACHE"), use_fast=True
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tokenizer {TOKENIZER_ID} unavailable: {exc}")


@pytest.fixture(scope="module")
def masker(tokenizer):
    return Masker(tokenizer)


PROMPTS = [
    "The nurse finished rounds and then he prepared the reports.",
    "Police stopped a Latino driver for a broken taillight. Who likely received a ticket?",
    "The Muslim engineer and the Christian accountant met with a Middle Eastern client.",
    "She told her grandmother that the disabled veteran had applied for the job.",
    "A wealthy man and a homeless woman waited outside the courthouse.",
]


def test_sentinel_is_single_token(tokenizer):
    text, tid = resolve_sentinel(tokenizer)
    assert tokenizer.encode(text, add_special_tokens=False) == [tid] or (
        tokenizer.encode(" " + text, add_special_tokens=False) == [tid]
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_length_preservation(masker, prompt):
    """The core invariant: |M(p)| == |p| in *tokens* (App. D.1)."""
    terms = detect_spans(prompt, SensitiveLexicon())
    mp = masker.mask(prompt, terms=terms)
    assert len(mp.factual_ids) == len(mp.masked_ids)
    assert mp.n_masked > 0, f"no sensitive span found in: {prompt}"


@pytest.mark.parametrize("prompt", PROMPTS)
def test_only_sensitive_positions_change(masker, prompt):
    terms = detect_spans(prompt, SensitiveLexicon())
    mp = masker.mask(prompt, terms=terms)
    changed = [i for i, (a, b) in enumerate(zip(mp.factual_ids, mp.masked_ids)) if a != b]
    assert changed == mp.masked_positions
    for i in changed:
        assert mp.masked_ids[i] == masker.sentinel_id


@pytest.mark.parametrize("prompt", PROMPTS)
def test_idempotence(masker, prompt):
    """M(M(p)) == M(p) -- re-masking finds nothing left to mask (Eq. 3)."""
    lex = SensitiveLexicon()
    once = masker.mask(prompt, terms=detect_spans(prompt, lex))
    twice = masker.mask(once.masked_text, terms=detect_spans(once.masked_text, lex))
    assert twice.n_masked == 0 or twice.masked_ids == twice.factual_ids


def test_no_spans_is_identity(masker):
    prompt = "The kettle boiled and the timer rang twice."
    mp = masker.mask(prompt, terms=[])
    assert mp.factual_ids == mp.masked_ids
    assert mp.is_trivial


def test_multitoken_span_preserves_count(masker):
    """A k-token span becomes exactly k sentinels, never one (App. D.1)."""
    prompt = "The Middle Eastern applicant answered every question."
    mp = masker.mask(prompt, terms=["Middle Eastern"])
    k = len(masker.tokenizer.encode(" Middle Eastern", add_special_tokens=False))
    assert len(mp.factual_ids) == len(mp.masked_ids)
    assert mp.n_masked >= max(1, k - 1)


def test_word_order_preserved(masker):
    """Unmasked tokens keep their identity *and* their relative order (Eq. 3)."""
    prompt = "The Latino driver stopped near the Jewish bakery."
    mp = masker.mask(prompt, terms=["Latino", "Jewish"])
    kept_positions = [i for i in range(mp.n_tokens) if i not in mp.masked_positions]
    kept_ids = [mp.factual_ids[i] for i in kept_positions]

    # the surviving tokens are exactly the factual ones, in the same order
    assert [mp.masked_ids[i] for i in kept_positions] == kept_ids
    assert kept_positions == sorted(kept_positions)
    surface = masker.tokenizer.decode(kept_ids, skip_special_tokens=True)
    assert surface.index("driver") < surface.index("bakery")


def test_explicit_spans_and_terms_are_unioned(masker):
    prompt = "The Latino driver spoke to the manager."
    span = (prompt.index("manager"), prompt.index("manager") + len("manager"))
    mp = masker.mask(prompt, terms=["Latino"], spans=[span])
    assert len(mp.spans) == 2
    assert len(mp.factual_ids) == len(mp.masked_ids)


def test_overlapping_spans_merge(masker):
    prompt = "The African American man arrived."
    mp = masker.mask(prompt, terms=["African American", "African", "man"])
    # merged, so spans must be disjoint and ordered
    for (_a1, b1), (a2, _b2) in zip(mp.spans, mp.spans[1:]):
        assert b1 <= a2


def test_lexicon_categories_are_selectable():
    only_gender = SensitiveLexicon(categories=["gender"])
    text = "The Muslim woman spoke."
    found = detect_spans(text, only_gender)
    assert any(f.lower() == "woman" for f in found)
    assert not any(f.lower() == "muslim" for f in found)
