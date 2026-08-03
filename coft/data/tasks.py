"""Utility benchmarks of Table 2: GSM8K, StrategyQA, ARC-easy, PIQA.

These measure whether COFT's distributional corrections cost anything on tasks
that have nothing to do with protected attributes.  Prompt formats are fixed
across every decoding method so the comparison isolates the decoder.
"""

from __future__ import annotations

from typing import List, Optional

from coft.data.base import ChoiceItem, GenItem, hf_kwargs, subsample

__all__ = ["load_gsm8k", "load_strategyqa", "load_arc_easy", "load_piqa", "TASK_LOADERS"]


GSM8K_FEWSHOT = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
        "in May. How many clips did Natalia sell altogether in April and May?",
        "In April, Natalia sold 48 clips. In May, she sold 48 / 2 = 24 clips. "
        "Altogether she sold 48 + 24 = 72 clips. The answer is 72.",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
        "babysitting. How much did she earn?",
        "Weng earns 12 / 60 = $0.2 per minute. For 50 minutes she earned 0.2 x 50 = $10. "
        "The answer is 10.",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the "
        "money she needs. Her parents decided to give her $15 for that purpose, and her "
        "grandparents twice as much as her parents. How much more money does Betty need?",
        "Betty has 100 / 2 = $50. Her grandparents gave her 15 * 2 = $30. In total she has "
        "50 + 15 + 30 = $95. She still needs 100 - 95 = $5. The answer is 5.",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How many pages does "
        "he write a year?",
        "Each time James writes 3 * 2 = 6 pages. Twice a week that is 6 * 2 = 12 pages. "
        "In a year he writes 12 * 52 = 624 pages. The answer is 624.",
    ),
]


def _gsm8k_prompt(question: str, n_shot: int = 4) -> str:
    parts = []
    for q, a in GSM8K_FEWSHOT[:n_shot]:
        parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def load_gsm8k(
    split: str = "test", limit: Optional[int] = None, seed: int = 0, n_shot: int = 4
) -> List[GenItem]:
    """GSM8K grade-school math (Cobbe et al., 2021), chain-of-thought prompted."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split, **hf_kwargs())
    items = [
        GenItem(
            prompt=_gsm8k_prompt(row["question"], n_shot),
            answer=row["answer"].split("####")[-1].strip(),
            meta={"source": "gsm8k", "rationale": row["answer"]},
        )
        for row in ds
    ]
    return subsample(items, limit, seed)


def load_strategyqa(limit: Optional[int] = None, seed: int = 0) -> List[ChoiceItem]:
    """StrategyQA implicit multi-hop yes/no questions (Geva et al., 2021)."""
    from datasets import load_dataset

    ds = load_dataset("ChilleD/StrategyQA", split="train", **hf_kwargs())
    items = []
    for row in ds:
        ans = row["answer"]
        label = 0 if (ans is True or str(ans).lower() in {"true", "yes"}) else 1
        items.append(
            ChoiceItem(
                context="",
                question=row["question"],
                choices=["Yes", "No"],
                label=label,
                meta={"source": "strategyqa", "qid": row.get("qid")},
            )
        )
    return subsample(items, limit, seed)


def load_arc_easy(split: str = "test", limit: Optional[int] = None, seed: int = 0) -> List[ChoiceItem]:
    """ARC-Easy science QA (Clark et al., 2018)."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, **hf_kwargs())
    items = []
    for row in ds:
        texts = row["choices"]["text"]
        labels = row["choices"]["label"]
        key = row["answerKey"]
        if key not in labels:
            continue
        items.append(
            ChoiceItem(
                context="",
                question=row["question"],
                choices=list(texts),
                label=labels.index(key),
                meta={"source": "arc_easy", "id": row.get("id")},
            )
        )
    return subsample(items, limit, seed)


def load_piqa(split: str = "validation", limit: Optional[int] = None, seed: int = 0) -> List[ChoiceItem]:
    """PIQA physical commonsense (Bisk et al., 2020)."""
    from datasets import load_dataset

    ds = load_dataset("baber/piqa", split=split, **hf_kwargs())
    items = []
    for row in ds:
        label = int(row["label"])
        if label < 0:
            continue
        items.append(
            ChoiceItem(
                context="",
                question=row["goal"],
                choices=[row["sol1"], row["sol2"]],
                label=label,
                meta={"source": "piqa"},
            )
        )
    return subsample(items, limit, seed)


TASK_LOADERS = {
    "gsm8k": load_gsm8k,
    "strategyqa": load_strategyqa,
    "arc_easy": load_arc_easy,
    "piqa": load_piqa,
}
