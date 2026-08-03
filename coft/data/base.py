"""Item schemas and dataset plumbing shared by every benchmark loader."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TypeVar

__all__ = [
    "PairItem",
    "ChoiceItem",
    "GenItem",
    "DecisionItem",
    "subsample",
    "attach_terms",
    "hf_kwargs",
    "raw_data_dir",
]

T = TypeVar("T")


def hf_kwargs() -> Dict:
    """Cache settings shared by every ``datasets`` call."""
    kw: Dict = {}
    cache = os.environ.get("HF_DATASETS_CACHE")
    if cache:
        kw["cache_dir"] = cache
    return kw


def raw_data_dir() -> Path:
    """Where third-party files that we cannot redistribute are expected."""
    root = os.environ.get("COFT_DATA_DIR")
    path = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# item schemas
# --------------------------------------------------------------------------- #
@dataclass
class PairItem:
    """A minimal pair scored by likelihood under the evaluated decoder.

    Two shapes are supported, because the two benchmarks are built differently:

    *Shared context, differing continuations* (StereoSet).  ``context`` holds the
    prompt -- which is where the protected target sits, so Stage I masks it --
    and ``stereo`` / ``anti`` are the competing continuations.

    *Differing contexts, shared continuation* (CrowS-Pairs).  The pair differs
    only in the protected span itself, so the standard methodology of Nangia et
    al. conditions on the **modified** tokens and scores the **unmodified**
    remainder.  ``stereo_context`` / ``anti_context`` then carry the two
    identity-bearing prompts and ``stereo == anti`` is the shared suffix.  This
    is also the only arrangement under which COFT can act on CrowS at all: the
    protected span has to be in the prompt for the masked branch to be blind to
    it (both branches receive the continuation verbatim, Sec. 3.1).
    """

    context: str
    stereo: str
    anti: str
    terms: List[str] = field(default_factory=list)
    bias_type: str = "unknown"
    unrelated: Optional[str] = None
    stereo_context: Optional[str] = None
    anti_context: Optional[str] = None
    stereo_terms: List[str] = field(default_factory=list)
    anti_terms: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def ctx_stereo(self) -> str:
        return self.context if self.stereo_context is None else self.stereo_context

    @property
    def ctx_anti(self) -> str:
        return self.context if self.anti_context is None else self.anti_context

    @property
    def spans_stereo(self) -> List[str]:
        return self.stereo_terms or self.terms

    @property
    def spans_anti(self) -> List[str]:
        return self.anti_terms or self.terms

    @property
    def split_contexts(self) -> bool:
        """True when the two branches have different prompts (CrowS-style)."""
        return self.stereo_context is not None or self.anti_context is not None


@dataclass
class ChoiceItem:
    """A multiple-choice item.

    ``target_idx`` marks the stereotype-aligned option (BBQ) and is ``None`` for
    ordinary utility benchmarks.  ``unknown_idx`` marks BBQ's
    "not enough information" option.
    """

    context: str
    question: str
    choices: List[str]
    label: int
    terms: List[str] = field(default_factory=list)
    target_idx: Optional[int] = None
    unknown_idx: Optional[int] = None
    condition: str = "n/a"
    group: Optional[str] = None
    meta: Dict = field(default_factory=dict)

    def prompt(self, style: str = "cloze") -> str:
        """Render the item as a prompt.

        ``cloze`` (default) is the lm-eval-harness convention: the options are
        *not* listed, and each candidate answer is scored as a continuation of
        ``"... Answer:"``.  This keeps the comparison on content tokens -- which
        is what Stage III certifies -- rather than on option letters.

        ``mcq`` lists lettered options and is scored on the letter; kept for
        studies that need a single-token decision.
        """
        body = f"{self.context}\n{self.question}".strip() if self.context else self.question
        if style == "cloze":
            return f"{body}\nAnswer:"
        if style == "mcq":
            opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(self.choices))
            return f"{body}\n{opts}\nAnswer:"
        raise ValueError(f"unknown prompt style {style}")

    def continuation(self, index: int, style: str = "cloze") -> str:
        if style == "cloze":
            return " " + str(self.choices[index]).strip()
        return f" {chr(65 + index)}"


@dataclass
class GenItem:
    """A free-generation prompt (BOLD continuations, GSM8K questions)."""

    prompt: str
    terms: List[str] = field(default_factory=list)
    group: Optional[str] = None
    answer: Optional[str] = None
    meta: Dict = field(default_factory=dict)


@dataclass
class DecisionItem:
    """A binary decision framed in natural language, used for parity gaps."""

    prompt: str
    group: str
    terms: List[str] = field(default_factory=list)
    positive: str = " Yes"
    negative: str = " No"
    label: Optional[int] = None
    meta: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def subsample(items: Sequence[T], limit: Optional[int], seed: int = 0) -> List[T]:
    """Deterministic subsample; ``limit=None`` keeps everything."""
    items = list(items)
    if limit is None or limit >= len(items):
        return items
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(items)), limit))
    return [items[i] for i in idx]


def calibration_triples(items: Sequence, max_words: int = 40) -> List[tuple]:
    """Turn benchmark items into ``(prompt, reference continuation, terms)`` triples.

    This is what ``D_cal`` is made of (Eq. 6): the reference continuation supplies
    the *observed* next tokens ``v*_t`` whose nonconformity scores are quantiled.
    Calibrating from the same benchmark family the method is deployed on is what
    App. C.2 prescribes ("for each dataset ... a disjoint calibration pool"), and
    it is what keeps assumption (A2) plausible.

    The reference is always the *unbiased* branch of the item -- the
    anti-stereotypical sentence, the gold answer, the factual Wikipedia
    continuation -- so calibration never rewards the behaviour we are filtering.
    """
    out: List[tuple] = []
    for it in items:
        if isinstance(it, PairItem):
            # calibrate on the anti-stereotypical branch, under its own prompt
            prompt, cont, terms = it.ctx_anti, it.anti, it.spans_anti
        elif isinstance(it, ChoiceItem):
            prompt, cont, terms = it.prompt(), it.continuation(it.label), it.terms
        elif isinstance(it, DecisionItem):
            gold = it.negative if it.label in (0, None) else it.positive
            prompt, cont, terms = it.prompt, gold, it.terms
        elif isinstance(it, GenItem):
            ref = (it.meta or {}).get("wikipedia") or it.answer or ""
            prompt, cont, terms = it.prompt, (" " + str(ref).strip() if ref else ""), it.terms
        else:  # pragma: no cover
            continue
        cont = " ".join(str(cont).split()[:max_words])
        if prompt is not None and cont.strip():
            out.append((prompt, " " + cont if not cont.startswith(" ") else cont, list(terms)))
    return out


def attach_terms(items: Sequence, lexicon=None, use_ner: bool = False) -> List:
    """Fill in each item's sensitive spans by unioning dataset terms with detection.

    Mirrors App. D.2: user/dataset-provided spans take precedence and are then
    unioned with detector output.
    """
    from coft.spans import SensitiveLexicon, detect_spans

    lexicon = lexicon or SensitiveLexicon()
    for it in items:
        if isinstance(it, PairItem) and it.split_contexts:
            # each branch carries its own prompt, so detect on each separately
            it.stereo_terms = detect_spans(
                it.ctx_stereo, lexicon, user_terms=tuple(it.spans_stereo), use_ner=use_ner
            )
            it.anti_terms = detect_spans(
                it.ctx_anti, lexicon, user_terms=tuple(it.spans_anti), use_ner=use_ner
            )
            it.terms = it.stereo_terms
            continue

        text = getattr(it, "prompt", None)
        if not isinstance(text, str):
            text = " ".join(
                str(x)
                for x in (
                    getattr(it, "context", ""),
                    getattr(it, "question", ""),
                    " ".join(getattr(it, "choices", []) or []),
                )
                if x
            )
        it.terms = detect_spans(text, lexicon, user_terms=tuple(it.terms), use_ner=use_ner)
    return list(items)
