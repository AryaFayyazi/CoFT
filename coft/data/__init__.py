"""Datasets used by the paper, normalised into four evaluation shapes.

+------------------+------------------------------------------------------------+
| shape            | benchmarks                                                 |
+==================+============================================================+
| :class:`PairItem`   | StereoSet, CrowS-Pairs -- minimal pairs compared by       |
|                     | likelihood under the *method's own* distribution          |
| :class:`ChoiceItem` | BBQ, ARC-easy, PIQA, StrategyQA -- pick one option        |
| :class:`GenItem`    | BOLD -- free continuation, scored for toxicity            |
| :class:`DecisionItem`| Utrecht, COMPAS -- binary decision, scored for parity    |
+------------------+------------------------------------------------------------+

GSM8K uses :class:`GenItem` with an exact-match answer check.

Every loader is deterministic given ``(split, limit, seed)`` and attaches the
sensitive spans ``S`` that Stage I should mask.
"""

from coft.data.base import (
    ChoiceItem,
    DecisionItem,
    GenItem,
    PairItem,
    attach_terms,
    subsample,
)
from coft.data.bias import (
    BIAS_LOADERS,
    load_bbq,
    load_bold,
    load_compas,
    load_crows,
    load_stereoset,
    load_utrecht,
)
from coft.data.corpora import load_calibration_corpus, load_tldr, load_wikitext2
from coft.data.tasks import (
    TASK_LOADERS,
    load_arc_easy,
    load_gsm8k,
    load_piqa,
    load_strategyqa,
)

__all__ = [
    "PairItem",
    "ChoiceItem",
    "GenItem",
    "DecisionItem",
    "attach_terms",
    "subsample",
    "BIAS_LOADERS",
    "TASK_LOADERS",
    "load_stereoset",
    "load_crows",
    "load_bbq",
    "load_bold",
    "load_utrecht",
    "load_compas",
    "load_gsm8k",
    "load_strategyqa",
    "load_arc_easy",
    "load_piqa",
    "load_wikitext2",
    "load_tldr",
    "load_calibration_corpus",
]
