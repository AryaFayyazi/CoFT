"""Evaluation engine: run a decoder over a benchmark and return paper metrics.

One function per item shape.  All of them route through the decoder's own
teacher-forced scorer or generator, so the *method* -- not just the base model --
determines every number reported.  That is what makes a decoding-time
intervention visible on likelihood-based benchmarks such as StereoSet and
CrowS-Pairs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import torch
from tqdm.auto import tqdm

from coft import metrics as M
from coft.data.base import ChoiceItem, DecisionItem, GenItem, PairItem

__all__ = [
    "eval_pairs",
    "eval_choices",
    "eval_generation",
    "eval_decisions",
    "eval_perplexity",
    "eval_mauve",
]


def _chunks(seq: Sequence, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _truncate(text: str, stop_strings: Optional[Sequence[str]]) -> str:
    """Cut ``text`` at the earliest of ``stop_strings`` (if any occurs).

    Fails safe: if cutting would throw away everything of substance -- no digits
    left -- the untruncated text is kept, so a model that emits the delimiter
    before its answer is not scored as having answered nothing.
    """
    if not stop_strings:
        return text
    cut = min((text.find(s) for s in stop_strings if text.find(s) >= 0), default=-1)
    if cut < 0:
        return text
    head = text[:cut]
    return head if any(ch.isdigit() for ch in head) else text


def _score_key(res: Dict[str, float], normalize: bool) -> float:
    return res["mean_logprob"] if normalize else res["logprob"]


# --------------------------------------------------------------------------- #
# StereoSet / CrowS-Pairs
# --------------------------------------------------------------------------- #
def eval_pairs(
    decoder,
    items: Sequence[PairItem],
    kind: str = "stereoset",
    batch_size: int = 8,
    normalize: bool = True,
    progress: bool = True,
) -> Dict[str, float]:
    """Compare stereotypical vs anti-stereotypical continuations by likelihood."""
    stereo_scores: List[float] = []
    anti_scores: List[float] = []
    unrelated_scores: List[float] = []
    coverage: List[float] = []

    bar = tqdm(list(_chunks(list(items), batch_size)), desc=f"{decoder.name}/{kind}", disable=not progress)
    for batch in bar:
        # Each branch is scored under its *own* prompt: StereoSet shares one
        # context, CrowS-Pairs conditions each branch on its own modified span.
        s = decoder.score_batch(
            [it.ctx_stereo for it in batch],
            [it.stereo for it in batch],
            [it.spans_stereo for it in batch],
        )
        a = decoder.score_batch(
            [it.ctx_anti for it in batch],
            [it.anti for it in batch],
            [it.spans_anti for it in batch],
        )
        stereo_scores.extend(_score_key(r, normalize) for r in s)
        anti_scores.extend(_score_key(r, normalize) for r in a)
        coverage.extend(r["coverage"] for r in s if r["coverage"] == r["coverage"])
        if kind == "stereoset" and all(it.unrelated for it in batch):
            u = decoder.score_batch(
                [it.ctx_stereo for it in batch],
                [it.unrelated for it in batch],
                [it.spans_stereo for it in batch],
            )
            unrelated_scores.extend(_score_key(r, normalize) for r in u)

    if kind == "stereoset":
        out = M.stereoset_metrics(
            stereo_scores, anti_scores, unrelated_scores if len(unrelated_scores) == len(stereo_scores) else None
        )
    else:
        out = M.crows_metrics(stereo_scores, anti_scores)
    if coverage:
        out["empirical_coverage"] = sum(coverage) / len(coverage)
    return out


# --------------------------------------------------------------------------- #
# multiple choice (BBQ, ARC-easy, PIQA, StrategyQA)
# --------------------------------------------------------------------------- #
def eval_choices(
    decoder,
    items: Sequence[ChoiceItem],
    kind: str = "bbq",
    batch_size: int = 8,
    normalize: bool = True,
    progress: bool = True,
    style: str = "cloze",
) -> Dict[str, float]:
    """Pick the option with the highest likelihood under the decoder's distribution."""
    preds: List[int] = []
    coverage: List[float] = []

    bar = tqdm(list(_chunks(list(items), batch_size)), desc=f"{decoder.name}/{kind}", disable=not progress)
    for batch in bar:
        prompts = [it.prompt(style) for it in batch]
        terms = [it.terms for it in batch]
        n_choices = max(len(it.choices) for it in batch)
        scores = [[float("-inf")] * n_choices for _ in batch]
        for c in range(n_choices):
            idx = [i for i, it in enumerate(batch) if c < len(it.choices)]
            if not idx:
                continue
            sub_prompts = [prompts[i] for i in idx]
            sub_terms = [terms[i] for i in idx]
            cands = [batch[i].continuation(c, style) for i in idx]
            res = decoder.score_batch(sub_prompts, cands, sub_terms)
            for j, i in enumerate(idx):
                scores[i][c] = _score_key(res[j], normalize)
                if res[j]["coverage"] == res[j]["coverage"]:
                    coverage.append(res[j]["coverage"])
        preds.extend(int(max(range(n_choices), key=lambda c: row[c])) for row in scores)

    labels = [it.label for it in items]
    if kind == "bbq":
        out = M.bbq_metrics(
            preds, labels,
            [it.target_idx for it in items],
            [it.unknown_idx for it in items],
            conditions=[it.condition for it in items],
        )
    else:
        out = M.choice_accuracy(preds, labels)
    if coverage:
        out["empirical_coverage"] = sum(coverage) / len(coverage)
    out["predictions"] = preds
    return out


# --------------------------------------------------------------------------- #
# free generation (BOLD, GSM8K)
# --------------------------------------------------------------------------- #
def eval_generation(
    decoder,
    items: Sequence[GenItem],
    kind: str = "bold",
    batch_size: int = 8,
    max_new_tokens: int = 64,
    greedy: bool = False,
    toxicity_scorer=None,
    progress: bool = True,
    seed: int = 0,
    stop_strings: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Generate continuations and score them (toxicity for BOLD, EM for GSM8K).

    ``stop_strings`` truncates each continuation at the first delimiter that
    appears.  Few-shot prompts otherwise induce the model to keep going and
    invent a further exemplar, which pollutes both exact-match extraction and
    toxicity scoring with text that is not an answer to the question asked.
    """
    texts: List[str] = []
    empty_steps = 0
    total_steps = 0
    set_sizes: List[float] = []

    bar = tqdm(list(_chunks(list(items), batch_size)), desc=f"{decoder.name}/{kind}", disable=not progress)
    for bi, batch in enumerate(bar):
        res = decoder.generate(
            [it.prompt for it in batch],
            [it.terms for it in batch],
            max_new_tokens=max_new_tokens,
            greedy=greedy,
            seed=seed + bi,
        )
        texts.extend(_truncate(t, stop_strings) for t in res.texts)
        empty_steps += res.n_empty_sets
        total_steps += res.n_steps
        if res.n_certified_steps:
            set_sizes.append(res.mean_set_size)

    if kind == "gsm8k":
        out = M.gsm8k_exact_match(texts, [it.answer or "" for it in items])
    else:
        if toxicity_scorer is None:
            raise ValueError("BOLD evaluation requires a toxicity scorer")
        scores = toxicity_scorer.score(texts)
        out = M.toxicity_metrics(scores)
        by_group: Dict[str, List[float]] = defaultdict(list)
        for it, s in zip(items, scores):
            by_group[(it.group or "all").split("/")[0]].append(s)
        out["toxicity_by_domain"] = {g: sum(v) / len(v) for g, v in by_group.items()}

    out["empty_set_rate"] = empty_steps / max(1, total_steps)
    if set_sizes:
        out["mean_certified_set"] = sum(set_sizes) / len(set_sizes)
    out["generations"] = texts[:20]
    return out


# --------------------------------------------------------------------------- #
# binary decisions (Utrecht, COMPAS)
# --------------------------------------------------------------------------- #
def eval_decisions(
    decoder,
    items: Sequence[DecisionItem],
    kind: str = "compas",
    batch_size: int = 8,
    progress: bool = True,
) -> Dict[str, float]:
    """Demographic-parity gap of the model's ``Yes`` rate across protected groups.

    ``P(Yes)`` is the two-way softmax of the decoder's log-likelihoods for the
    ``Yes`` and ``No`` continuations, so the decoding intervention -- fusion and
    certification alike -- shifts it.
    """
    by_group: Dict[str, List[float]] = defaultdict(list)
    coverage: List[float] = []

    bar = tqdm(list(_chunks(list(items), batch_size)), desc=f"{decoder.name}/{kind}", disable=not progress)
    for batch in bar:
        prompts = [it.prompt for it in batch]
        terms = [it.terms for it in batch]
        yes = decoder.score_batch(prompts, [it.positive for it in batch], terms)
        no = decoder.score_batch(prompts, [it.negative for it in batch], terms)
        for it, y, n in zip(batch, yes, no):
            pair = torch.tensor([y["logprob"], n["logprob"]])
            p_yes = float(torch.softmax(pair, dim=-1)[0])
            by_group[it.group].append(p_yes)
            if y["coverage"] == y["coverage"]:
                coverage.append(y["coverage"])

    out = M.parity_gap(by_group)
    if coverage:
        out["empirical_coverage"] = sum(coverage) / len(coverage)
    return out


# --------------------------------------------------------------------------- #
# language quality
# --------------------------------------------------------------------------- #
def eval_perplexity(
    decoder,
    documents: Sequence[str],
    max_tokens: int = 512,
    stride_prompt_tokens: int = 8,
    batch_size: int = 4,
    progress: bool = True,
) -> Dict[str, float]:
    """Perplexity of ``documents`` under the decoder's next-token distribution.

    The first ``stride_prompt_tokens`` tokens of each document seed the context
    (they are not scored) and the remainder is teacher-forced.
    """
    tok = decoder.lm.tokenizer
    prompts: List[str] = []
    conts: List[List[int]] = []
    for doc in documents:
        ids = tok.encode(doc, add_special_tokens=False)[:max_tokens]
        if len(ids) <= stride_prompt_tokens + 1:
            continue
        prompts.append(tok.decode(ids[:stride_prompt_tokens]))
        conts.append(ids[stride_prompt_tokens:])

    total_lp, total_tok = 0.0, 0
    idx = list(range(len(prompts)))
    bar = tqdm(list(_chunks(idx, batch_size)), desc=f"{decoder.name}/ppl", disable=not progress)
    for chunk in bar:
        res = decoder.score_batch(
            [prompts[i] for i in chunk],
            ["" for _ in chunk],
            [() for _ in chunk],
            continuation_ids=[conts[i] for i in chunk],
        )
        for r in res:
            total_lp += r["logprob"]
            total_tok += r["n_tokens"]
    return {
        "ppl": M.perplexity_from_logprobs(total_lp, total_tok),
        "n_tokens": total_tok,
        "n_docs": len(prompts),
    }


def eval_mauve(
    decoder,
    rows: Sequence[Dict[str, str]],
    max_new_tokens: int = 128,
    batch_size: int = 8,
    seed: int = 0,
    progress: bool = True,
    featurize_model_name: str = "gpt2-large",
) -> Dict[str, float]:
    """MAUVE between the decoder's summaries and the human references."""
    gens: List[str] = []
    bar = tqdm(list(_chunks(list(rows), batch_size)), desc=f"{decoder.name}/mauve", disable=not progress)
    for bi, batch in enumerate(bar):
        res = decoder.generate(
            [r["prompt"] for r in batch],
            [() for _ in batch],
            max_new_tokens=max_new_tokens,
            seed=seed + bi,
        )
        gens.extend(res.texts)
    refs = [r["reference"] for r in rows]
    return {
        "mauve": M.mauve_score(gens, refs, featurize_model_name=featurize_model_name),
        "n": len(gens),
    }
