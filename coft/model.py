"""Frozen causal-LM wrapper and the multi-branch, KV-cached forward pass.

COFT evaluates the same frozen model ``f_theta`` twice per step (Eq. 1): once on
the factual prompt ``p`` and once on the masked prompt ``p~``, *conditioned on
the same generated prefix* ``w_<t``.  Baselines need their own extra branches
(a self-debiasing prompt for SDD, expert / anti-expert prompts for
DExperts/GeDi-style steering), so the runner here is generic: a decoder declares
a dict of named branches and receives a dict of named next-token logits.

All branches are packed into a *single* padded batch and share one KV cache
object, so the extra cost is one additional cached forward pass -- the
``<= 11%`` overhead reported in Sec. 4.4, rather than a doubled latency.

Left padding + explicit ``position_ids`` are used throughout: branches may have
different prompt lengths (SDD's anti-prompt is longer than ``p``), and without
explicit position ids a left-padded batch would be given wrong positions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

__all__ = ["FrozenLM", "BranchBatch", "load_model"]


def _default_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


@dataclass
class BranchBatch:
    """Live decoding state for a set of named branches sharing one KV cache."""

    names: List[str]
    n_prompts: int
    attention_mask: torch.Tensor           # (n_rows, cur_len)
    next_position: torch.Tensor            # (n_rows, 1) position id for the next token
    past_key_values: object = None
    last_logits: Optional[torch.Tensor] = None   # (n_rows, V) float32

    @property
    def n_rows(self) -> int:
        return len(self.names) * self.n_prompts

    def logits_for(self, branch: str) -> torch.Tensor:
        """Rows of ``last_logits`` belonging to ``branch``; shape ``(n_prompts, V)``."""
        i = self.names.index(branch)
        return self.last_logits[i * self.n_prompts : (i + 1) * self.n_prompts]


class FrozenLM:
    """A frozen causal LM plus the machinery COFT needs on top of it.

    The model is loaded in eval mode with gradients disabled -- COFT is
    inference-only and never touches the weights (Sec. 3).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (next(model.parameters()).device.type)
        self.name = name or getattr(model.config, "_name_or_path", "model")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = "auto",
        cache_dir: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        **kw,
    ) -> "FrozenLM":
        """Load a frozen causal LM.

        ``tokenizer_id`` lets the tokenizer come from a different repository than
        the weights.  Some community mirrors ship only the SentencePiece
        ``tokenizer.model`` and no ``tokenizer.json``; recent ``transformers``
        cannot always convert those on the fly, and Stage I wants a *fast*
        tokenizer because character offsets are what map a sensitive span onto
        an exact token range.  Pointing at a mirror that ships the same
        tokenizer in fast form fixes this without changing a single weight --
        the loader verifies the vocabulary sizes agree.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = cache_dir or os.environ.get("HF_HUB_CACHE")
        tok = AutoTokenizer.from_pretrained(
            tokenizer_id or model_id, cache_dir=cache_dir, use_fast=True
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        load_kw = dict(cache_dir=cache_dir, **kw)
        load_kw["dtype"] = dtype or _default_dtype()
        if device_map is not None:
            load_kw["device_map"] = device_map
        if attn_implementation:
            load_kw["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)

        if tokenizer_id and tokenizer_id != model_id:
            n_model = int(model.config.vocab_size)
            n_tok = len(tok)
            if n_tok != n_model:
                raise ValueError(
                    f"tokenizer '{tokenizer_id}' has {n_tok} tokens but model "
                    f"'{model_id}' expects {n_model}; they are not interchangeable"
                )
        return cls(model, tok, name=model_id)

    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    def _dev(self) -> torch.device:
        return next(self.model.parameters()).device

    # ------------------------------------------------------------------ #
    # padding / packing
    # ------------------------------------------------------------------ #
    def _pack(self, sequences: Sequence[Sequence[int]]):
        """Left-pad ``sequences`` into ``(input_ids, attention_mask, position_ids)``."""
        pad_id = self.tokenizer.pad_token_id or 0
        max_len = max(len(s) for s in sequences)
        ids, masks = [], []
        for s in sequences:
            pad = max_len - len(s)
            ids.append([pad_id] * pad + list(s))
            masks.append([0] * pad + [1] * len(s))
        dev = self._dev()
        input_ids = torch.tensor(ids, dtype=torch.long, device=dev)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=dev)
        # With left padding, position 0 must land on the first *real* token.
        position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        return input_ids, attention_mask, position_ids

    # ------------------------------------------------------------------ #
    # prefill / step
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def prefill(self, branches: Dict[str, List[List[int]]]) -> BranchBatch:
        """Run the prompt(s) for every branch and prime the shared KV cache.

        ``branches`` maps a branch name to a list of ``n_prompts`` token-id
        sequences.  Every branch must supply the same number of prompts; their
        *lengths* may differ (left padding handles that).
        """
        names = list(branches.keys())
        n_prompts = len(branches[names[0]])
        for n in names:
            if len(branches[n]) != n_prompts:
                raise ValueError(f"branch '{n}' has {len(branches[n])} prompts, expected {n_prompts}")

        flat: List[Sequence[int]] = []
        for n in names:
            flat.extend(branches[n])
        input_ids, attention_mask, position_ids = self._pack(flat)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        state = BranchBatch(
            names=names,
            n_prompts=n_prompts,
            attention_mask=attention_mask,
            next_position=position_ids[:, -1:] + 1,
            past_key_values=out.past_key_values,
            last_logits=out.logits[:, -1, :].float(),
        )
        return state

    @torch.no_grad()
    def step(self, state: BranchBatch, tokens: torch.Tensor) -> BranchBatch:
        """Append ``tokens`` (shape ``(n_prompts,)``) to *every* branch and advance.

        The same selected token is appended to both branches, as required by
        Sec. 3 ("At each step, the same selected token is appended to both
        branches").
        """
        if tokens.ndim != 1 or tokens.shape[0] != state.n_prompts:
            raise ValueError(f"expected {state.n_prompts} tokens, got {tuple(tokens.shape)}")
        dev = self._dev()
        rep = tokens.to(dev).repeat(len(state.names)).unsqueeze(-1)   # (n_rows, 1)
        attn = torch.cat(
            [state.attention_mask, torch.ones_like(rep, dtype=state.attention_mask.dtype)], dim=-1
        )
        out = self.model(
            input_ids=rep,
            attention_mask=attn,
            position_ids=state.next_position,
            past_key_values=state.past_key_values,
            use_cache=True,
        )
        state.attention_mask = attn
        state.next_position = state.next_position + 1
        state.past_key_values = out.past_key_values
        state.last_logits = out.logits[:, -1, :].float()
        return state

    # ------------------------------------------------------------------ #
    # teacher forcing
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def branch_logits_teacher_forced(
        self,
        branches: Dict[str, List[int]],
        continuation: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Logits at every continuation step for each branch, in one forward pass.

        For branch ``b`` with prompt ``x_b`` and shared continuation
        ``c_1..c_L``, returns a ``(L, V)`` tensor whose row ``t`` holds the
        next-token logits given ``x_b, c_1..c_t`` -- i.e. the distribution from
        which ``c_{t+1}`` would be drawn, plus a final row predicting past the
        end.  Row ``0`` predicts ``c_1``.

        This is the workhorse for split calibration (Eq. 6), for likelihood-based
        bias metrics, and for perplexity: all of them need the *method's* own
        per-step distribution rather than the raw model's.
        """
        names = list(branches.keys())
        seqs = [list(branches[n]) + list(continuation) for n in names]
        prompt_lens = [len(branches[n]) for n in names]
        input_ids, attention_mask, position_ids = self._pack(seqs)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = out.logits.float()                       # (n_branches, padded_len, V)
        padded_len = logits.shape[1]
        L = len(continuation)

        result: Dict[str, torch.Tensor] = {}
        for i, n in enumerate(names):
            # The row that predicts continuation[0] sits at the last prompt token.
            offset = padded_len - L - 1
            result[n] = logits[i, offset : offset + L, :]
            assert result[n].shape[0] == L, (n, prompt_lens[i], padded_len, L)
        return result


    @torch.no_grad()
    def branch_logits_teacher_forced_batch(
        self,
        branches: Dict[str, List[List[int]]],
        continuations: List[List[int]],
    ) -> Dict[str, torch.Tensor]:
        """Batched teacher forcing with *column-aligned* continuations.

        Prompts are padded on the **left** to a common length ``P`` and
        continuations on the **right** to a common length ``L``.  Continuation
        step ``t`` therefore lands on the same column for every item, which lets
        the caller evaluate one decode step across the whole batch in a single
        vectorised call to ``step_distribution``.

        Returns ``{branch: (B, L, V)}`` where row ``(i, t)`` holds the logits
        that predict ``continuations[i][t]``.  Positions past ``len(continuations[i])``
        are present but meaningless; the caller masks them with the true lengths.
        """
        names = list(branches.keys())
        n = len(continuations)
        pad_id = self.tokenizer.pad_token_id or 0
        P = max(len(p) for b in names for p in branches[b])
        L = max(len(c) for c in continuations)

        rows, masks = [], []
        for b in names:
            for i, prompt in enumerate(branches[b]):
                cont = continuations[i]
                left = P - len(prompt)
                right = L - len(cont)
                rows.append([pad_id] * left + list(prompt) + list(cont) + [pad_id] * right)
                masks.append([0] * left + [1] * (len(prompt) + len(cont)) + [0] * right)

        dev = self._dev()
        input_ids = torch.tensor(rows, dtype=torch.long, device=dev)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=dev)
        position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        # column P-1 predicts continuation token 0, column P+L-2 predicts token L-1
        sliced = out.logits[:, P - 1 : P + L - 1, :].float()
        return {name: sliced[i * n : (i + 1) * n] for i, name in enumerate(names)}


def load_model(cfg: Dict) -> FrozenLM:
    """Instantiate a :class:`FrozenLM` from a model config block."""
    dtype = cfg.get("dtype")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return FrozenLM.from_pretrained(
        cfg["model_id"],
        dtype=dtype_map.get(dtype) if dtype else None,
        device_map=cfg.get("device_map", "auto"),
        cache_dir=cfg.get("cache_dir"),
        attn_implementation=cfg.get("attn_implementation"),
        tokenizer_id=cfg.get("tokenizer_id"),
    )
