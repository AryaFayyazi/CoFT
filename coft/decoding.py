"""Decoders: the common interface plus COFT itself (paper Algorithm 1).

Every decoding method in this repository -- COFT and all baselines -- implements
the same two operations:

``branches(prompts, terms)``
    Declare the named token sequences the frozen LM must run in parallel.
    Vanilla declares one (``factual``); COFT declares two (``factual``,
    ``masked``); SDD and the expert-steering baseline declare their own.

``step_distribution(logits, t)``
    Turn this step's per-branch logits into (i) the distribution actually
    sampled from and (ii) an optional certified mask.

Given those, :meth:`BaseDecoder.generate` (free-running) and
:meth:`BaseDecoder.score` (teacher-forced) are shared, which guarantees that
every method is evaluated under exactly the same decoding policy -- the "fair
decoding" requirement of App. C.2.

Evaluation convention for certified likelihoods
-----------------------------------------------
Soundness (Prop. 2) says COFT never *emits* a token outside ``C_t``, so the
policy assigns zero probability to an uncertified continuation.  Likelihood-based
metrics (StereoSet, CrowS-Pairs, perplexity) need a finite, monotone score, so an
uncertified token is floored at the certification boundary ``tau_t`` -- the
largest probability it could have had while still failing certification -- and
the whole thing is renormalised by the certified mass.  On the empty-set event
the argmax fallback applies and no restriction is imposed, matching App. B.14.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from coft.conformal import ConformalThresholds
from coft.fusion import fuse_logits
from coft.masking import Masker

__all__ = [
    "StepOutput",
    "GenerationResult",
    "BaseDecoder",
    "COFTDecoder",
    "sample_from_probs",
    "top_p_filter",
]


# --------------------------------------------------------------------------- #
# sampling helpers
# --------------------------------------------------------------------------- #
def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out the nucleus tail and renormalise.  ``probs`` is ``(B, V)``."""
    if top_p is None or top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = sorted_probs.cumsum(dim=-1)
    # keep everything up to and including the token that crosses top_p
    keep = cum - sorted_probs < top_p
    keep[..., 0] = True
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, sorted_probs * keep)
    total = filtered.sum(dim=-1, keepdim=True)
    return torch.where(total > 0, filtered / total.clamp_min(1e-12), probs)


def sample_from_probs(
    probs: torch.Tensor,
    top_p: float = 0.9,
    greedy: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw one token per row from ``probs`` under the shared decoding policy."""
    if greedy:
        return probs.argmax(dim=-1)
    filtered = top_p_filter(probs, top_p)
    filtered = filtered.clamp_min(0)
    total = filtered.sum(dim=-1, keepdim=True)
    filtered = torch.where(total > 0, filtered / total.clamp_min(1e-12), torch.ones_like(filtered) / filtered.shape[-1])
    return torch.multinomial(filtered, num_samples=1, generator=generator).squeeze(-1)


# --------------------------------------------------------------------------- #
# step / generation containers
# --------------------------------------------------------------------------- #
@dataclass
class StepOutput:
    """What a decoder produces at one decode step."""

    probs: torch.Tensor                          # (B, V) distribution to sample from
    certified_mask: Optional[torch.Tensor] = None  # (B, V) bool, None if no certification
    tau: float = 0.0                             # certification boundary at this step
    empty_set: Optional[torch.Tensor] = None     # (B,) bool -- argmax fallback fired

    def effective_probs(self) -> torch.Tensor:
        """The policy's actual next-token distribution, Eq. 8.

        On rows with a non-empty certified set: ``pi_hat(.| C_t)``.
        On rows that fell back: the unrestricted ``pi_hat`` (the argmax of which
        is what Eq. 8 emits).
        """
        if self.certified_mask is None:
            return self.probs
        restricted = self.probs * self.certified_mask
        mass = restricted.sum(dim=-1, keepdim=True)
        ok = mass.squeeze(-1) > 0
        out = torch.where(
            ok.unsqueeze(-1),
            restricted / mass.clamp_min(1e-12),
            self.probs,
        )
        return out

    def token_logprob(self, tokens: torch.Tensor) -> torch.Tensor:
        """Certification-aware log-probability of ``tokens`` (see module docstring)."""
        idx = tokens.view(-1, 1)
        p = self.probs.gather(-1, idx).squeeze(-1)
        if self.certified_mask is None:
            return torch.log(p.clamp_min(1e-30))

        restricted = self.probs * self.certified_mask
        mass = restricted.sum(dim=-1)                    # (B,)
        in_set = self.certified_mask.gather(-1, idx).squeeze(-1)
        # uncertified tokens are floored at the certification boundary tau
        numer = torch.where(in_set, p, torch.minimum(p, torch.full_like(p, self.tau)))
        empty = mass <= 0
        logp = torch.log(numer.clamp_min(1e-30)) - torch.log(mass.clamp_min(1e-30))
        return torch.where(empty, torch.log(p.clamp_min(1e-30)), logp)


@dataclass
class GenerationResult:
    texts: List[str]
    tokens: List[List[int]]
    n_steps: int = 0
    n_empty_sets: int = 0
    n_certified_steps: int = 0
    set_size_sum: float = 0.0
    wall_time: float = 0.0
    extra: Dict = field(default_factory=dict)

    @property
    def empty_set_rate(self) -> float:
        return self.n_empty_sets / max(1, self.n_steps)

    @property
    def mean_set_size(self) -> float:
        return self.set_size_sum / max(1, self.n_certified_steps)

    @property
    def tokens_per_second(self) -> float:
        total = sum(len(t) for t in self.tokens)
        return total / self.wall_time if self.wall_time > 0 else float("nan")


# --------------------------------------------------------------------------- #
# base decoder
# --------------------------------------------------------------------------- #
class BaseDecoder:
    """Shared generation / scoring loop.  Subclasses define the branches and the step rule."""

    name = "base"
    #: branch names this decoder needs, in a stable order
    branch_names: Tuple[str, ...] = ("factual",)

    def __init__(
        self,
        lm,
        top_p: float = 0.9,
        temperature: float = 1.0,
        max_new_tokens: int = 256,
        seed: int = 0,
    ) -> None:
        self.lm = lm
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    # -- to be provided by subclasses -------------------------------------- #
    def branches(self, prompts: Sequence[str], terms: Sequence[Sequence[str]]) -> Dict[str, List[List[int]]]:
        ids = [self.lm.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
        return {"factual": ids}

    def step_distribution(self, logits: Dict[str, torch.Tensor], t: int) -> StepOutput:
        raise NotImplementedError

    @contextlib.contextmanager
    def without_support_restriction(self):
        """Temporarily drop any hard restriction on the next-token *support*.

        Several methods truncate the support as part of their sampling policy:
        COFT and DT-CD to the certified set, DExperts to the base model's top-k.
        Those are constraints on what may be *emitted*, not statements about the
        density, so scoring a corpus through them assigns near-zero probability
        to every out-of-support gold token and inflates perplexity by orders of
        magnitude (DExperts: 13 -> 50 on Wikitext-2).

        Perplexity is therefore reported with these restrictions lifted, i.e. for
        the corrected distribution each method induces -- the usual convention
        for decoding-time interventions.  Methods that only *reweight* (SDD,
        Vanilla) are unaffected either way.
        """
        yield self

    # -- shared loops -------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str],
        terms: Optional[Sequence[Sequence[str]]] = None,
        max_new_tokens: Optional[int] = None,
        greedy: bool = False,
        stop_on_eos: bool = True,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """Free-running generation under this decoder's policy."""
        terms = terms if terms is not None else [() for _ in prompts]
        max_new_tokens = max_new_tokens or self.max_new_tokens
        branches = self.branches(prompts, terms)
        n = len(prompts)

        dev = self.lm._dev()
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(self.seed if seed is None else seed))

        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        state = self.lm.prefill(branches)
        produced: List[List[int]] = [[] for _ in range(n)]
        finished = torch.zeros(n, dtype=torch.bool, device=dev)
        eos = self.lm.eos_token_id

        res = GenerationResult(texts=[], tokens=produced)

        # Book-keeping counters live on-device and are read back once, after the
        # loop.  Touching a CUDA tensor from Python forces a synchronisation, so
        # doing it per item per step would add O(batch) stalls to every decode
        # step -- and those stalls would land squarely in the latency that
        # Table 3 is supposed to measure.  The loop below performs exactly one
        # device-to-host transfer per step.
        zeros = lambda dt: torch.zeros((), dtype=dt, device=dev)  # noqa: E731
        c_steps, c_cert_steps, c_empty = zeros(torch.long), zeros(torch.long), zeros(torch.long)
        c_set_size = zeros(torch.float)

        for t in range(max_new_tokens):
            logits = {b: state.logits_for(b) for b in state.names}
            out = self.step_distribution(logits, t)

            probs = out.effective_probs()
            tokens = sample_from_probs(probs, top_p=self.top_p, greedy=greedy, generator=gen)
            tokens = torch.where(finished, torch.full_like(tokens, eos), tokens)

            live = ~finished
            c_steps += live.sum()
            if out.certified_mask is not None:
                c_cert_steps += live.sum()
                c_set_size += (out.certified_mask.sum(-1) * live).sum()
                if out.empty_set is not None:
                    c_empty += (out.empty_set & live).sum()

            # the single sync: tokens and liveness together
            tok_list, live_list = torch.stack([tokens, live.long()]).tolist()
            for i in range(n):
                if live_list[i]:
                    produced[i].append(tok_list[i])

            if stop_on_eos:
                finished = finished | (tokens == eos)
                # derived from the values just transferred, so no extra sync
                if all((not lv) or tk == eos for lv, tk in zip(live_list, tok_list)):
                    break
            state = self.lm.step(state, tokens)

        if dev.type == "cuda":
            torch.cuda.synchronize()
        res.wall_time = time.perf_counter() - t0
        res.n_steps = int(c_steps)
        res.n_certified_steps = int(c_cert_steps)
        res.set_size_sum = float(c_set_size)
        res.n_empty_sets = int(c_empty)
        res.texts = [
            self.lm.tokenizer.decode(
                [tk for tk in seq if tk != eos], skip_special_tokens=True
            )
            for seq in produced
        ]
        return res

    @torch.no_grad()
    def score(
        self,
        prompt: str,
        continuation: str,
        terms: Sequence[str] = (),
        continuation_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Teacher-forced log-likelihood of ``continuation`` under this decoder.

        Returns the total and per-token log-probability, plus diagnostics about
        how often the observed token was certified -- which is exactly the
        empirical coverage of Theorem 1 on this sequence.
        """
        branches_batched = self.branches([prompt], [terms])
        branches = {k: v[0] for k, v in branches_batched.items()}
        if continuation_ids is None:
            continuation_ids = self.lm.tokenizer.encode(continuation, add_special_tokens=False)
        if not continuation_ids:
            return {"logprob": 0.0, "n_tokens": 0, "mean_logprob": 0.0, "covered": 0, "n_empty": 0}

        per_branch = self.lm.branch_logits_teacher_forced(branches, continuation_ids)

        total, covered, n_empty = 0.0, 0, 0
        for t, tok in enumerate(continuation_ids):
            logits_t = {b: per_branch[b][t : t + 1] for b in per_branch}
            out = self.step_distribution(logits_t, t)
            tok_t = torch.tensor([tok], device=out.probs.device)
            total += float(out.token_logprob(tok_t)[0])
            if out.certified_mask is not None:
                covered += int(out.certified_mask[0, tok])
                if out.empty_set is not None:
                    n_empty += int(out.empty_set[0])
        L = len(continuation_ids)
        return {
            "logprob": total,
            "n_tokens": L,
            "mean_logprob": total / L,
            "covered": covered,
            "coverage": covered / L,
            "n_empty": n_empty,
        }

    @torch.no_grad()
    def score_batch(
        self,
        prompts: Sequence[str],
        continuations: Sequence[str],
        terms: Optional[Sequence[Sequence[str]]] = None,
        continuation_ids: Optional[Sequence[List[int]]] = None,
    ) -> List[Dict[str, float]]:
        """Teacher-forced scoring of a whole batch -- the workhorse of evaluation.

        Semantically identical to calling :meth:`score` once per item, but the
        batch shares one forward pass and one ``step_distribution`` call per
        decode step.
        """
        terms = list(terms) if terms is not None else [() for _ in prompts]
        if continuation_ids is None:
            continuation_ids = [
                self.lm.tokenizer.encode(c, add_special_tokens=False) or [self.lm.eos_token_id]
                for c in continuations
            ]
        continuation_ids = [c if c else [self.lm.eos_token_id] for c in continuation_ids]

        branches = self.branches(list(prompts), terms)
        per_branch = self.lm.branch_logits_teacher_forced_batch(branches, list(continuation_ids))

        B = len(prompts)
        L = max(len(c) for c in continuation_ids)
        dev = per_branch[next(iter(per_branch))].device
        lengths = torch.tensor([len(c) for c in continuation_ids], device=dev)

        padded = torch.full((B, L), 0, dtype=torch.long, device=dev)
        for i, c in enumerate(continuation_ids):
            padded[i, : len(c)] = torch.tensor(c, dtype=torch.long, device=dev)

        totals = torch.zeros(B, device=dev)
        covered = torch.zeros(B, device=dev)
        empties = torch.zeros(B, device=dev)
        has_mask = False

        for t in range(L):
            live = (lengths > t).float()
            logits_t = {b: per_branch[b][:, t, :] for b in per_branch}
            out = self.step_distribution(logits_t, t)
            tok_t = padded[:, t]
            totals += out.token_logprob(tok_t) * live
            if out.certified_mask is not None:
                has_mask = True
                covered += out.certified_mask.gather(-1, tok_t.view(-1, 1)).squeeze(-1).float() * live
                if out.empty_set is not None:
                    empties += out.empty_set.float() * live

        results: List[Dict[str, float]] = []
        for i in range(B):
            n_tok = int(lengths[i])
            results.append(
                {
                    "logprob": float(totals[i]),
                    "n_tokens": n_tok,
                    "mean_logprob": float(totals[i]) / max(1, n_tok),
                    "covered": int(covered[i]),
                    "coverage": (float(covered[i]) / n_tok) if (has_mask and n_tok) else float("nan"),
                    "n_empty": int(empties[i]),
                }
            )
        return results


# --------------------------------------------------------------------------- #
# COFT
# --------------------------------------------------------------------------- #
class COFTDecoder(BaseDecoder):
    """COFT -- Chain of Fair Thought (Algorithm 1).

    One step:

    1. ``p~ <- M(p)``                                        (Stage I,   Sec. 3.2)
    2. ``z^F, z^CF <- f_theta(w_<t; p), f_theta(w_<t; p~)``  (Eq. 1)
    3. ``z_hat <- (1-lambda) z^F + lambda z^CF``             (Stage II,  Eq. 4)
    4. ``C_t <- {v : min(pi_hat(v), pi^CF(v)) >= tau_t}``    (Stage III, Eq. 7)
    5. sample from ``pi_hat(.| C_t)``, or ``argmax pi_hat`` if ``C_t`` is empty (Eq. 8)

    Ablation switches (Table 4)
    ---------------------------
    ``use_fusion=False``   -> ``lambda`` is forced to 0 ("w/o fusion (CP only)")
    ``use_cp=False``       -> no certification at all ("fusion only (no CP)")
    ``score='single'``     -> factual-only certification ("Single-branch CP")
    """

    name = "coft"
    branch_names = ("factual", "masked")

    def __init__(
        self,
        lm,
        masker: Optional[Masker] = None,
        lam: float = 0.6,
        thresholds: Optional[ConformalThresholds] = None,
        use_fusion: bool = True,
        use_cp: bool = True,
        **kw,
    ) -> None:
        super().__init__(lm, **kw)
        self.masker = masker or Masker(lm.tokenizer)
        self.lam = float(lam) if use_fusion else 0.0
        self.use_fusion = use_fusion
        self.use_cp = use_cp
        self.thresholds = thresholds
        if use_cp and thresholds is None:
            raise ValueError(
                "COFT with certification enabled needs calibrated thresholds; "
                "run scripts/calibrate.py first or pass use_cp=False"
            )
        self._mask_cache: Dict[str, object] = {}

    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def without_support_restriction(self):
        saved, self.use_cp = self.use_cp, False
        try:
            yield self
        finally:
            self.use_cp = saved

    def masked_prompt(self, prompt: str, terms: Sequence[str]):
        key = (prompt, tuple(terms))
        cached = self._mask_cache.get(key)
        if cached is None:
            cached = self.masker.mask(prompt, terms=terms)
            self._mask_cache[key] = cached
        return cached

    def branches(self, prompts, terms) -> Dict[str, List[List[int]]]:
        factual, masked = [], []
        for p, tm in zip(prompts, terms):
            mp = self.masked_prompt(p, tm)
            factual.append(mp.factual_ids)
            masked.append(mp.masked_ids)
        return {"factual": factual, "masked": masked}

    # ------------------------------------------------------------------ #
    def step_distribution(self, logits: Dict[str, torch.Tensor], t: int) -> StepOutput:
        zf, zcf = logits["factual"], logits["masked"]

        # Stage II -- fusion (Eq. 4).  Temperature is folded in here so that every
        # method shares the same nucleus/temperature policy (App. C.2).
        fused = fuse_logits(zf, zcf, self.lam) / self.temperature
        pi_hat = torch.softmax(fused, dim=-1)

        if not self.use_cp:
            return StepOutput(probs=pi_hat)

        # Stage III -- dual-branch certification (Eq. 5/7).
        pi_cf = torch.softmax(zcf.float() / self.temperature, dim=-1)
        mask = self.thresholds.candidate_mask(pi_hat, pi_cf, t)
        empty = mask.sum(dim=-1) == 0
        return StepOutput(
            probs=pi_hat,
            certified_mask=mask,
            tau=self.thresholds.tau(t),
            empty_set=empty,
        )

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def branch_distributions(
        self,
        prompt: str,
        continuation_ids: Sequence[int],
        terms: Sequence[str] = (),
    ) -> Iterable[Tuple[int, torch.Tensor, torch.Tensor]]:
        """Yield ``(t, pi_hat_t, pi^CF_t)`` along a teacher-forced continuation.

        Used by split calibration (Eq. 6), which needs both branch
        distributions at the *observed* next token.
        """
        b = {k: v[0] for k, v in self.branches([prompt], [terms]).items()}
        per_branch = self.lm.branch_logits_teacher_forced(b, list(continuation_ids))
        for t in range(len(continuation_ids)):
            zf = per_branch["factual"][t : t + 1]
            zcf = per_branch["masked"][t : t + 1]
            pi_hat = torch.softmax(fuse_logits(zf, zcf, self.lam) / self.temperature, dim=-1)[0]
            pi_cf = torch.softmax(zcf.float() / self.temperature, dim=-1)[0]
            yield t, pi_hat, pi_cf
