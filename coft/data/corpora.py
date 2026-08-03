"""Corpora for language quality (Table 2) and for split calibration (Eq. 6).

* ``wikitext2``  -- perplexity.
* ``tldr``       -- the OpenAI Summaries subset used for MAUVE.
* ``calibration``-- the disjoint pool ``D_cal`` from which the conformal
  thresholds are computed.

Calibration pool
----------------
Assumption (A2)/(A4) of App. B.2 requires calibration and test contexts to be
exchangeable under the deployed policy.  The pool built here is drawn from the
*same* bias-benchmark families as evaluation but from a **disjoint slice**
(``calibration_fraction`` of each dataset, taken before any evaluation
subsample), which is what "a disjoint calibration pool (10-15%)" in App. C.2
prescribes.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from coft.data.base import hf_kwargs

__all__ = [
    "load_wikitext2",
    "load_tldr",
    "load_calibration_corpus",
    "split_calibration_eval",
]


def load_wikitext2(
    split: str = "test", limit: Optional[int] = None, min_chars: int = 200
) -> List[str]:
    """Wikitext-2 (raw) documents for perplexity (Merity et al., 2016)."""
    from datasets import load_dataset

    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, **hf_kwargs())
    except Exception:  # pragma: no cover - older mirror name
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, **hf_kwargs())

    def _is_article_header(line: str) -> bool:
        s = line.strip()
        return s.startswith("= ") and not s.startswith("= =") and s.endswith(" =")

    chunks: List[str] = []
    cur: List[str] = []
    for row in ds:
        line = row["text"]
        if _is_article_header(line) and cur:
            joined = "".join(cur).strip()
            if len(joined) >= min_chars:
                chunks.append(joined)
            cur = []
        cur.append(line)
    joined = "".join(cur).strip()
    if len(joined) >= min_chars:
        chunks.append(joined)
    if limit:
        chunks = chunks[:limit]
    return chunks


def load_tldr(split: str = "test", limit: Optional[int] = 500) -> List[Dict[str, str]]:
    """OpenAI TL;DR summaries subset, used as the MAUVE reference (Table 2)."""
    from datasets import load_dataset

    ds = load_dataset("CarperAI/openai_summarize_tldr", split=split, **hf_kwargs())
    rows = [{"prompt": r["prompt"], "reference": r["label"]} for r in ds]
    if limit:
        rows = rows[:limit]
    return rows


def load_calibration_corpus(
    n_contexts: int = 400,
    seed: int = 0,
    max_continuation_words: int = 40,
    sources: Sequence[str] = ("stereoset", "crows", "bbq", "bold"),
) -> List[Tuple[str, str, List[str]]]:
    """Build ``D_cal``: ``(prompt, reference continuation, sensitive terms)`` triples.

    The reference continuation supplies the *observed* next tokens ``v*_t`` that
    Eq. 6 scores.  Only bias-benchmark families are used, so the calibration
    contexts match the deployment distribution; the slices taken here are
    disjoint from the evaluation slices used by ``scripts/run_bias.py`` (both
    derive from the same seeded ordering, with calibration taking the head and
    evaluation the tail -- see ``coft.data.corpora.split_calibration_eval``).
    """
    from coft.data.bias import DatasetUnavailable, load_bbq, load_bold, load_crows, load_stereoset

    rng = random.Random(seed)
    pool: List[Tuple[str, str, List[str]]] = []

    if "stereoset" in sources:
        for it in load_stereoset(limit=None, seed=seed):
            # the anti-stereotypical continuation is the reference we calibrate on
            pool.append((it.context, it.anti, list(it.terms)))
    if "crows" in sources:
        try:
            for it in load_crows(limit=None, seed=seed):
                pool.append((it.context, it.anti, list(it.terms)))
        except DatasetUnavailable:
            pass
    if "bbq" in sources:
        try:
            for it in load_bbq(condition="ambig", limit=None, seed=seed):
                pool.append((it.prompt(), it.continuation(it.label), list(it.terms)))
        except DatasetUnavailable:
            pass
    if "bold" in sources:
        for it in load_bold(limit=None, seed=seed):
            wiki = (it.meta or {}).get("wikipedia")
            if wiki:
                pool.append((it.prompt, " " + str(wiki), list(it.terms)))

    pool = [
        (p, " ".join(c.split()[:max_continuation_words]), t)
        for (p, c, t) in pool
        if p and c and c.strip()
    ]
    rng.shuffle(pool)
    return pool[:n_contexts]


def split_calibration_eval(items: Sequence, calibration_fraction: float = 0.15, seed: int = 0):
    """Deterministically split a benchmark into disjoint calibration / evaluation halves."""
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_cal = max(1, int(len(items) * calibration_fraction))
    cal_idx = set(idx[:n_cal])
    cal = [items[i] for i in sorted(cal_idx)]
    ev = [items[i] for i in range(len(items)) if i not in cal_idx]
    return cal, ev
