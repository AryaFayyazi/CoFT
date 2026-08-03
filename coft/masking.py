"""Stage I -- Counterfactual masking (paper Sec. 3.2 and App. D.1).

The mask operator ``M`` deterministically replaces every sensitive span
``s in S`` with a neutral sentinel while preserving word order and, crucially,
*token count*:

    M(M(p)) = M(p)      and      len(M(p)) ~= len(p)                     (Eq. 3)

Why token count matters
-----------------------
COFT fuses the two branches at the *same* decode step ``t`` (Eq. 1/Eq. 4).  If a
``k``-token sensitive span collapsed to a single sentinel, every position after
the edit point would shift by ``k - 1`` and ``z^F_t`` / ``z^CF_t`` would no
longer describe the same autoregressive index -- the paired comparison, and with
it the split-conformal score of Eq. 5, would be meaningless (App. D.1, "Why
Sentinel Masking (and Why Length-Preserving)?").

We therefore perform the substitution *in token space*: a span that tokenises to
``k`` tokens is replaced by exactly ``k`` copies of a single tokenizer-stable
sentinel token.  This makes length preservation exact rather than approximate,
which is stronger than the ``~=`` of Eq. 3.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = ["MaskedPrompt", "Masker", "resolve_sentinel", "SENTINEL_CANDIDATES"]

# Ordered by preference.  The first entry that encodes to a *single* token under
# the model's tokenizer is used.  Footnote 1 of the paper: "any tokenizer-stable,
# semantics-light sentinel is acceptable; its role is structural neutrality, not
# cloze semantics."
SENTINEL_CANDIDATES: Tuple[str, ...] = (
    "[MASK]",
    "<mask>",
    "▁_",
    "_",
    "#",
    "*",
    "X",
)


def resolve_sentinel(tokenizer, preferred: Optional[str] = None) -> Tuple[str, int]:
    """Pick a sentinel that is a *single* token under ``tokenizer``.

    Returns the ``(text, token_id)`` pair.  Raises :class:`ValueError` if no
    candidate is tokenizer-stable, which would break length preservation.
    """
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(c for c in SENTINEL_CANDIDATES if c != preferred)

    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return cand, ids[0]
        # SentencePiece models often prepend a word-boundary marker; retry the
        # variant that the tokenizer would produce mid-sentence.
        ids_sp = tokenizer.encode(" " + cand, add_special_tokens=False)
        if len(ids_sp) == 1:
            return cand, ids_sp[0]

    # Last resort: any existing single-token "unused"/special token.
    for attr in ("mask_token", "unk_token"):
        tok = getattr(tokenizer, attr, None)
        if tok:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if len(ids) == 1:
                return tok, ids[0]

    raise ValueError(
        "could not resolve a single-token sentinel for this tokenizer; "
        "pass `sentinel=` explicitly with a token-stable string"
    )


@dataclass
class MaskedPrompt:
    """A factual prompt paired with its length-matched masked counterfactual.

    Attributes
    ----------
    factual_text, masked_text:
        Human-readable renderings (the masked one is a *decode* of
        ``masked_ids`` so it faithfully shows what the model sees).
    factual_ids, masked_ids:
        Token id lists of **identical length** -- this is the invariant the
        whole method rests on.
    masked_positions:
        Indices into ``factual_ids`` that were overwritten by the sentinel.
    spans:
        The character spans of ``factual_text`` that were treated as sensitive.
    """

    factual_text: str
    masked_text: str
    factual_ids: List[int]
    masked_ids: List[int]
    masked_positions: List[int] = field(default_factory=list)
    spans: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.factual_ids) != len(self.masked_ids):
            raise AssertionError(
                "mask operator broke token alignment: "
                f"{len(self.factual_ids)} factual vs {len(self.masked_ids)} masked tokens"
            )

    @property
    def n_tokens(self) -> int:
        return len(self.factual_ids)

    @property
    def n_masked(self) -> int:
        return len(self.masked_positions)

    @property
    def is_trivial(self) -> bool:
        """True when no sensitive span was found (branches are then identical)."""
        return self.n_masked == 0


class Masker:
    """The deterministic mask operator ``M`` of Sec. 3.2.

    Parameters
    ----------
    tokenizer:
        A HuggingFace tokenizer.  A *fast* tokenizer is strongly preferred: it
        exposes character offsets, which lets us map a character span onto an
        exact token range.  Slow tokenizers fall back to a subsequence search.
    sentinel:
        Optional explicit sentinel string.  Defaults to the first stable entry
        of :data:`SENTINEL_CANDIDATES`.
    add_special_tokens:
        Whether the prompt encoding includes BOS/EOS.  Kept identical between
        branches so positions line up.
    """

    def __init__(
        self,
        tokenizer,
        sentinel: Optional[str] = None,
        add_special_tokens: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.sentinel_text, self.sentinel_id = resolve_sentinel(tokenizer, sentinel)
        self.add_special_tokens = add_special_tokens
        self._has_offsets = bool(getattr(tokenizer, "is_fast", False))

    # ------------------------------------------------------------------ #
    # span resolution
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def find_spans(self, text: str, terms: Iterable[str]) -> List[Tuple[int, int]]:
        """Locate character spans of ``terms`` in ``text`` (whole-word, case-insensitive).

        Overlapping matches are merged so that ``M`` stays idempotent and
        order-preserving (App. D.2, "Detected spans are unioned with user lists;
        overlapping spans are merged").
        """
        spans: List[Tuple[int, int]] = []
        for term in terms:
            term = term.strip()
            if not term:
                continue
            # \b does not fire next to non-word characters, so guard manually.
            pattern = re.compile(
                r"(?<![\w-])" + re.escape(term) + r"(?![\w-])",
                flags=re.IGNORECASE,
            )
            for m in pattern.finditer(text):
                spans.append((m.start(), m.end()))
        return self._merge_spans(spans)

    @staticmethod
    def _merge_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not spans:
            return []
        ordered = sorted(spans)
        merged = [list(ordered[0])]
        for start, end in ordered[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(a, b) for a, b in merged]

    # ------------------------------------------------------------------ #
    # the operator itself
    # ------------------------------------------------------------------ #
    def mask(
        self,
        text: str,
        terms: Optional[Iterable[str]] = None,
        spans: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> MaskedPrompt:
        """Apply ``M`` to ``text``.

        Either ``terms`` (strings to look up) or ``spans`` (explicit character
        intervals) must be given; when both are supplied their union is masked,
        matching the user-list / NER union of App. D.2.
        """
        text = self._normalise(text)
        all_spans: List[Tuple[int, int]] = list(spans or [])
        if terms:
            all_spans.extend(self.find_spans(text, terms))
        all_spans = self._merge_spans(all_spans)

        enc = self.tokenizer(
            text,
            add_special_tokens=self.add_special_tokens,
            return_offsets_mapping=self._has_offsets,
        )
        factual_ids: List[int] = list(enc["input_ids"])
        masked_ids = list(factual_ids)

        if not all_spans:
            return MaskedPrompt(text, text, factual_ids, masked_ids, [], [])

        if self._has_offsets:
            token_positions = self._positions_from_offsets(enc["offset_mapping"], all_spans)
        else:  # pragma: no cover - exercised only with slow tokenizers
            token_positions = self._positions_by_search(text, all_spans, factual_ids)

        for pos in token_positions:
            masked_ids[pos] = self.sentinel_id

        masked_text = self.tokenizer.decode(masked_ids, skip_special_tokens=True)
        return MaskedPrompt(
            factual_text=text,
            masked_text=masked_text,
            factual_ids=factual_ids,
            masked_ids=masked_ids,
            masked_positions=sorted(token_positions),
            spans=all_spans,
        )

    # ------------------------------------------------------------------ #
    def _positions_from_offsets(
        self,
        offsets: Sequence[Tuple[int, int]],
        spans: Sequence[Tuple[int, int]],
    ) -> List[int]:
        """Token indices whose character extent overlaps any sensitive span."""
        hits: List[int] = []
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= tok_start:  # special tokens carry an empty offset
                continue
            for span_start, span_end in spans:
                # strict overlap test
                if tok_start < span_end and span_start < tok_end:
                    hits.append(idx)
                    break
        return hits

    def _positions_by_search(
        self,
        text: str,
        spans: Sequence[Tuple[int, int]],
        factual_ids: Sequence[int],
    ) -> List[int]:
        """Offset-free fallback: locate each span's token ids as a subsequence."""
        hits: List[int] = []
        used = set()
        for span_start, span_end in spans:
            surface = text[span_start:span_end]
            for variant in (" " + surface, surface):
                span_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if not span_ids:
                    continue
                k = len(span_ids)
                for start in range(len(factual_ids) - k + 1):
                    if start in used:
                        continue
                    if list(factual_ids[start : start + k]) == span_ids:
                        hits.extend(range(start, start + k))
                        used.update(range(start, start + k))
                        break
                else:
                    continue
                break
        return sorted(set(hits))

    # ------------------------------------------------------------------ #
    def mask_text_only(self, text: str, terms: Optional[Iterable[str]] = None) -> str:
        """Convenience: the masked *string* (used for logging and for baselines)."""
        return self.mask(text, terms=terms).masked_text
