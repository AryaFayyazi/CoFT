"""Toxicity scoring: sequence-level (BOLD metric) and token-level (DT-CD filter).

Sequence level
--------------
:class:`ToxicityScorer` wraps a public toxicity classifier and scores generated
continuations.  It is used for the BOLD toxicity column of Table 1 and for the
"toxic fallbacks" diagnostic of App. D.2.

Token level
-----------
The DT-CD baseline accepts a token only if it clears *both* a minimum-probability
conformal threshold and a toxicity threshold, so it needs a score for every entry
of the vocabulary.  :class:`TokenToxicityTable` builds that vector once by
running the classifier over each token's surface string and caches it to disk.
Deriving the table from a classifier -- rather than shipping a hand-written slur
list -- keeps the repository free of a hard-coded pejorative lexicon and makes
the filter reproducible from public artefacts.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
from typing import List, Optional, Sequence

import torch

__all__ = ["ToxicityScorer", "TokenToxicityTable", "DEFAULT_TOXICITY_MODEL"]

DEFAULT_TOXICITY_MODEL = "s-nlp/roberta_toxicity_classifier"


@functools.lru_cache(maxsize=16)
def _repo_has_safetensors(repo_id: str):
    """True/False if the Hub repo's file list is known, ``None`` if we cannot tell."""
    try:
        from huggingface_hub import list_repo_files

        return any(f.endswith(".safetensors") for f in list_repo_files(repo_id))
    except Exception:
        return None


class ToxicityScorer:
    """Classifier-based toxicity in ``[0, 1]`` for a batch of strings."""

    def __init__(
        self,
        model_id: str = DEFAULT_TOXICITY_MODEL,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 256,
        cache_dir: Optional[str] = None,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cache_dir = cache_dir or os.environ.get("HF_HUB_CACHE")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        # Several public toxicity checkpoints ship only a .bin.  Asking for
        # safetensors on such a repo makes transformers spawn a background thread
        # that tries to open a conversion PR on the Hub and then prints an
        # alarming (but entirely harmless) traceback into the run logs.  Probing
        # the file list first avoids triggering it at all.
        kw = dict(cache_dir=cache_dir)
        has_st = _repo_has_safetensors(model_id)
        if has_st is not None:
            kw["use_safetensors"] = has_st
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, **kw)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self.toxic_index = self._resolve_toxic_index()
        self.multi_label = self._is_multi_label()

    def _resolve_toxic_index(self) -> int:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            if str(label).lower() in {"toxic", "toxicity", "label_1", "hate", "offensive"}:
                return int(idx)
        return int(self.model.config.num_labels - 1)

    def _is_multi_label(self) -> bool:
        """Multi-label heads (toxic / obscene / insult / ...) need a sigmoid, not a softmax.

        Applying a softmax across independent sigmoid labels produces garbage --
        e.g. ``unitary/toxic-bert`` would score "the weather is nice" at 0.43.
        """
        cfg = self.model.config
        if getattr(cfg, "problem_type", None) == "multi_label_classification":
            return True
        labels = {str(v).lower() for v in (getattr(cfg, "id2label", {}) or {}).values()}
        siblings = {"severe_toxic", "severe_toxicity", "obscene", "insult", "identity_hate", "threat"}
        return bool(labels & siblings)

    @torch.no_grad()
    def score(self, texts: Sequence[str]) -> List[float]:
        out: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t if t.strip() else " " for t in texts[i : i + self.batch_size]]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.device)
            logits = self.model(**enc).logits.float()
            if logits.shape[-1] == 1:                      # single-logit regression head
                probs = torch.sigmoid(logits).squeeze(-1)
            elif self.multi_label:                          # independent per-label sigmoids
                probs = torch.sigmoid(logits)[:, self.toxic_index]
            else:                                           # mutually exclusive classes
                probs = torch.softmax(logits, dim=-1)[:, self.toxic_index]
            out.extend(probs.tolist())
        return out

    def mean(self, texts: Sequence[str]) -> float:
        scores = self.score(texts)
        return float(sum(scores) / len(scores)) if scores else 0.0

    def rate(self, texts: Sequence[str], threshold: float = 0.5) -> float:
        """Fraction of texts above ``threshold`` -- the "toxicity rate (%)" of Table 16."""
        scores = self.score(texts)
        return float(sum(s > threshold for s in scores) / len(scores)) if scores else 0.0


class TokenToxicityTable:
    """A per-vocabulary-entry toxicity prior, cached to disk."""

    def __init__(self, values: torch.Tensor, model_id: str, vocab_size: int) -> None:
        self.values = values
        self.model_id = model_id
        self.vocab_size = vocab_size

    @staticmethod
    def _cache_path(cache_dir: Path, lm_name: str, tox_model: str, vocab_size: int) -> Path:
        key = hashlib.sha1(f"{lm_name}|{tox_model}|{vocab_size}".encode()).hexdigest()[:16]
        return cache_dir / f"token_toxicity_{key}.pt"

    @classmethod
    def build(
        cls,
        lm,
        scorer: Optional[ToxicityScorer] = None,
        cache_dir: str = "results/cache",
        toxicity_model: str = DEFAULT_TOXICITY_MODEL,
        chunk: int = 1024,
        verbose: bool = True,
    ) -> "TokenToxicityTable":
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        vocab_size = lm.vocab_size
        path = cls._cache_path(cache_root, str(lm.name), toxicity_model, vocab_size)
        if path.exists():
            values = torch.load(path, map_location="cpu")
            return cls(values, toxicity_model, vocab_size)

        scorer = scorer or ToxicityScorer(toxicity_model)
        tok = lm.tokenizer
        values = torch.zeros(vocab_size, dtype=torch.float32)
        ids = list(range(vocab_size))
        for i in range(0, vocab_size, chunk):
            batch_ids = ids[i : i + chunk]
            surfaces = tok.convert_ids_to_tokens(batch_ids)
            texts = [
                (s or "").replace("▁", " ").replace("Ġ", " ").strip() or " " for s in surfaces
            ]
            values[i : i + len(batch_ids)] = torch.tensor(scorer.score(texts), dtype=torch.float32)
            if verbose:
                print(f"  token-toxicity {min(i + chunk, vocab_size)}/{vocab_size}", flush=True)
        torch.save(values, path)
        return cls(values, toxicity_model, vocab_size)

    def to(self, device) -> "TokenToxicityTable":
        self.values = self.values.to(device)
        return self

    def mask_above(self, threshold: float) -> torch.Tensor:
        """Boolean mask of tokens whose toxicity *exceeds* ``threshold`` (to be rejected)."""
        return self.values > threshold
