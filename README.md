# COFT — Chain of Fair Thought

Reference implementation of

> **COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models**
> Arya Fayyazi, Mehdi Kamal, Massoud Pedram — *ICML 2026*

COFT is a **training-free decoding method** that applies token-level fairness control at decode
time, with distribution-free *marginal* validity guarantees (under exchangeability) for any frozen
causal language model. No retraining, no gradients, no auxiliary classifiers, no weight access.

---

## The method in three stages

At every decoding step `t`, COFT runs the frozen model twice on the **same generated prefix**
`w_<t` — once on the factual prompt `p`, once on a masked counterfactual `p̃ = M(p)` — and then:

| Stage | Paper | Code | What it does |
|---|---|---|---|
| **I. Counterfactual masking** | §3.2, App. D.1 | [`coft/masking.py`](coft/masking.py) | Replaces each sensitive span with a tokenizer-stable sentinel, **preserving token count exactly** so the two branches stay index-aligned |
| **II. Logit fusion** | §3.3, Eq. 4 | [`coft/fusion.py`](coft/fusion.py) | `ẑ_t = (1−λ)·z^F_t + λ·z^CF_t` — a convex interpolation in logit space, equivalently a normalised geometric mixture of the two next-token distributions |
| **III. Dual-branch split-CP** | §3.4, Eq. 5–7 | [`coft/conformal.py`](coft/conformal.py) | Certifies `C_t = {v : min(π̂_t(v), π^CF_t(v)) ≥ τ_t}` using an offline, ceiling-corrected `(1−α)` quantile |

Sampling is then restricted to `C_t` (Eq. 8), with a deterministic `argmax π̂_t` fallback on the
rare empty-set event. The whole loop is [`coft/decoding.py`](coft/decoding.py) (`COFTDecoder`),
which mirrors Algorithm 1 line for line.

**Why length-preserving masking matters.** A `k`-token span becomes exactly `k` sentinels, never
one. If a span collapsed, every position after the edit would shift and `z^F_t` / `z^CF_t` would
stop describing the same autoregressive index — the paired comparison, and with it the conformal
score, would be meaningless. The substitution is therefore done in *token* space, which makes
length preservation exact rather than approximate.

---

## Install

```bash
git clone https://github.com/AryaFayyazi/CoFT.git
cd CoFT
pip install -e ".[mauve]"
python -m spacy download en_core_web_sm   # optional NER span route (App. D.2)
python scripts/fetch_data.py              # downloads the benchmarks
```

`fetch_data.py` pulls CrowS-Pairs, BBQ and COMPAS from their canonical GitHub releases, the
Utrecht recruitment set from Kaggle via `kagglehub`, and warms the Hub-hosted datasets. Nothing
third-party is redistributed in this repository.

Models are loaded straight from the Hub. The two main-text checkpoints are
`NousResearch/Llama-2-13b-hf` (an ungated mirror of `meta-llama/Llama-2-13b-hf`) and
`mistralai/Mistral-7B-Instruct-v0.2`; swap in the gated originals by editing
[`configs/models/`](configs/models).

---

## Quick start

```python
from coft.model import FrozenLM
from coft.masking import Masker
from coft.decoding import COFTDecoder
from coft.calibration import calibrate
from coft.spans import SensitiveLexicon, detect_spans

lm     = FrozenLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
masker = Masker(lm.tokenizer)
lex    = SensitiveLexicon()

# Stage III is calibrated offline on a disjoint pool of (prompt, reference, spans) triples.
thresholds = calibrate(lm, my_calibration_corpus, lam=0.6, alpha=0.10, masker=masker)

coft = COFTDecoder(lm, masker=masker, lam=0.6, thresholds=thresholds)

prompt = "Police stopped a Latino driver for a broken taillight. Who likely received a ticket?"
out = coft.generate([prompt], [detect_spans(prompt, lex)], max_new_tokens=32, greedy=True)
print(out.texts[0])
```

Vanilla decoding continues this prompt with identity-marked options
(`"A) A white man B) A black man C) A Latino man"`); COFT continues it with
`"A. A driver with a broken taillight …"` — the masked branch cannot see the protected span, so
the fused distribution never concentrates on it and the certified set excludes it.

### Worked examples

`examples/worked_examples.py` reproduces the step-by-step tables of App. D.3 on a
real model: for each example it prints the factual, masked and fused
distributions over the top candidates, the certification decision for each, and
what vanilla decoding and COFT actually generate.

```bash
python examples/worked_examples.py
```

---

## Reproducing the paper

```bash
make setup                                     # install + fetch data
make test                                      # fast unit tests, no model needed
make smoke                                     # tiny end-to-end run of every stage
make all MODEL=configs/models/llama2-13b.yaml  # Tables 1-4 + Figures 3-4
```

Individual tables:

| Target | Script | Output |
|---|---|---|
| Table 1 — bias | `scripts/run_bias.py` | `results/<model>/table1_bias.json` |
| Table 2 — utility & quality | `scripts/run_utility.py` | `results/<model>/table2_utility.json` |
| Table 3 — efficiency | `scripts/run_efficiency.py` | `results/<model>/table3_efficiency.json` |
| Table 4 — ablations | `scripts/run_ablation.py` | `results/<model>/table4_ablation.json` |
| Figures 3–4 — λ and α sweeps | `scripts/run_sweep.py` | `results/<model>/sweeps.json` |
| Render everything | `scripts/make_tables.py`, `scripts/make_figures.py` | `results/TABLES.md`, `results/<model>/fig*.pdf` |

Add `--override configs/main.yaml` for the sample sizes used to produce the numbers below, or
`--override configs/smoke.yaml` for a fast sanity run. All decoding hyper-parameters live in
[`configs/default.yaml`](configs/default.yaml) and are shared by *every* method, which is the
"fair decoding" requirement of App. C.2: nucleus `p = 0.9`, temperature `1.0`, `T = 256`.

---

## Experimental protocol

**Calibration is per benchmark and disjoint.** Following App. C.2 ("for each dataset and step
index `t`, a disjoint calibration pool (10–15%) sets `τ_t`; no test leakage"), each benchmark is
split into a 15% calibration slice and an evaluation slice. Thresholds used on a benchmark come
from that benchmark's own calibration slice. The reference continuation is always the *unbiased*
branch of the item — the anti-stereotypical sentence, the gold answer, the factual Wikipedia
continuation — so calibration never rewards the behaviour being filtered.

**Hyper-parameters are selected, not assumed.** §4.5 picks `λ` and `α` on a validation split by
the Pareto-knee rule — the smallest value within 2% of the knee — and the reported tables then use
what was picked. `scripts/run_all.sh` therefore runs the sweep *first*, and every table runner
adopts the selection from `results/<model>/sweeps.json`.

**Position binning.** Thresholds are shared across bins of width 8 up to `T = 256`, with all later
steps tied to the last bin (§3.4).

**Seeds.** Results are averaged over three seeds. StereoSet, CrowS-Pairs, BBQ, Utrecht, COMPAS,
the multiple-choice tasks and perplexity are teacher-forced likelihoods and therefore
*deterministic* given the model and thresholds; only BOLD free-generation and MAUVE are stochastic.
The runners exploit this and recompute only the stochastic parts per seed — identical numbers, ~3×
less compute.

**Efficiency.** Timed with EOS stopping disabled so every method runs the identical step budget;
otherwise a method that happens to emit EOS earlier amortises prefill over fewer tokens and the
tokens/second comparison stops being like-for-like.

---

## Metric definitions

Bias benchmarks are reported on several scales in the literature, so the exact definitions used
here are pinned down in [`coft/metrics.py`](coft/metrics.py):

| Column | Definition |
|---|---|
| **SS** ↓ | `2 · max(0, p_stereo − 0.5)`, where `p_stereo` is the fraction of StereoSet triples whose stereotypical sentence outscores the anti-stereotypical one. `0` = parity. The classical `ss`, `lms` and `icat` are reported alongside. |
| **CP Acc** ↑ | `100 − CP-STEREO`, exactly as in §4.1, where `CP-STEREO` is the standard CrowS-Pairs metric. `cp_parity_gap = \|CP-STEREO − 50\|` is also recorded, since that is the quantity that is zero for an unbiased model. Exact ties count as 0.5: as `λ → 1` both branches mask to the same prompt and the two scores coincide, and a strict `>` would score every such abstention as an anti-stereotype win. |
| **BBQ Bias** ↓ | Fraction of items where the model picks the stereotype-aligned identity answer *and* that answer is wrong, computed on **ambiguous** contexts. On a *disambiguated* item the protected attribute is the answer-relevant evidence, so masking it destroys the information needed to answer — accuracy collapses and the scattered errors register as "bias" while actually measuring semantic drift (App. D.2). Ambiguous contexts have no such evidence (gold is always UNKNOWN), so the metric isolates the model's prior. Disambiguated accuracy is reported as `bbq_acc_disambig` so the cost of masking evidence stays visible. |
| **BOLD Tox** ↓ | Mean toxicity of generated continuations under `s-nlp/roberta_toxicity_classifier`. |
| **Utrecht DP / COMPAS Gap** ↓ | Maximum pairwise demographic-parity gap of `P(Yes)` across protected groups, where `P(Yes)` is the two-way softmax of the decoder's log-likelihoods for the `Yes` / `No` continuations. Groups with fewer than 20 items are excluded from the max (and still reported): a max-over-groups statistic is otherwise set by sampling noise in a rare category. |

**Likelihood under a certified policy.** Soundness (Prop. 2) means COFT never *emits* a token
outside `C_t`, so the policy assigns it zero probability. Likelihood-based metrics need a finite,
monotone score, so an uncertified token is floored at the certification boundary `τ_t` — the
largest probability it could have had while still failing certification — and renormalised by the
certified mass. On the empty-set event the argmax fallback applies and no restriction is imposed
(App. B.14). This convention is what lets Stage III show up on StereoSet and CrowS-Pairs at all;
without it, certification would be invisible to every likelihood-based benchmark.

**Perplexity** is reported for the method's corrected distribution `π̂` — the density a decoding
intervention actually changes — since the certified set is a sampling constraint, not a density.
`--ppl-certified` scores under the certified policy instead.

---

## Reproduced results

Two models, the paper's main-text pair: **Mistral-7B-Instruct-v0.2** and **LLaMA-2-13B**.
600–800 evaluation items per bias benchmark, three seeds, hyper-parameters selected on a
disjoint validation split by the Pareto-knee rule (Sec. 4.5) and then used, not assumed:
λ\*=0.7, α\*=0.05 (Mistral, knee-anchored) and λ\*=0.8, α\*=0.02 (LLaMA, no knee — see below).
Full tables in [`results/TABLES.md`](results/TABLES.md), claim-by-claim verdicts in
[`results/CLAIMS.txt`](results/CLAIMS.txt), raw JSON under `results/<model>/`.

### What reproduces

| Claim | Paper | Reproduced |
|---|---|---|
| Bias reduction vs. Vanilla | 30–55% (median 38%) | **median 53%** (Mistral), 40% (LLaMA) |
| Theorem 1 coverage | ≥ 1−α | **0.982** mean over 15 (dataset, seed) cells, worst 0.949 |
| Miscoverage tracks α | conservative | 0.006 / 0.018 / 0.040 / 0.063 / 0.136 at α = 0.02…0.20 |
| Decoding overhead | ≤ 11% | **+0.3%** (Mistral), **+2.5%** (LLaMA) |
| Memory overhead | ≤ 0.8 GB | +0.1 GB / +0.6 GB |
| Perplexity, MAUVE | unchanged | Δ PPL 0.000, Δ MAUVE ≤ 0.005 |
| Fusion contributes more than CP alone | 0.149 < 0.171 | 0.168 < 0.220 |
| Bias decreases monotonically in λ | yes | 0.228 → 0.131 (Mistral) |

The central mechanism holds: masking + fusion attenuates measured bias by roughly half, and the
split-conformal guarantee is tight rather than vacuous.

### What does not reproduce

| Claim | Paper | Reproduced |
|---|---|---|
| COFT average rank | **1.0** (best on every column) | **2.61** (Mistral), 2.67 (LLaMA) — DExperts 2.03, SDD 2.11 rank better |
| CrowS-Pairs accuracy gain | +2.2 to +2.4 | **−3.0** (Mistral), −0.5 (LLaMA) |
| Utility within ±0.2 | ±0.2 | **−12.7 GSM8K** with masking active; **+0.7** with it inactive |
| Full COFT best ablation variant | 0.129 < {0.149, 0.158, 0.171} | **ties** fusion-only (0.169 vs 0.168) |

Three of these have identifiable causes rather than being noise.

**Utility.** The cost is entirely attributable to *what gets masked*, not to fusion or
certification. The span detector fires on ordinary task prompts, and in a GSM8K word problem
"her friends" is a coreference device, not a protected attribute — masking it destroys the
referent. Running the same evaluation with masking inactive isolates this exactly:

| | GSM8K | StrategyQA | ARC-easy | PIQA | max \|Δ\| |
|---|---|---|---|---|---|
| COFT, spans active (Mistral) | 36.00 | 61.00 | 72.00 | 78.50 | **12.67** |
| COFT, masking inactive (Mistral) | 49.33 | 64.25 | 72.75 | 78.75 | **0.67** |
| Vanilla (Mistral) | 48.67 | 64.00 | 72.75 | 79.00 | — |

So the paper's ±0.2 is reachable, but only in the configuration where Stage I has nothing to act
on — which is the configuration in which "utility preserved" is true by construction. This is the
coreference strain the paper itself documents (App. D.2), and the remedy it prescribes is span
whitelisting. Both configurations are reported (`--no-spans`) rather than picking the flattering
one.

**CrowS-Pairs.** Fusion cannot move a *ranking* metric. It is a monotone contraction toward the
masked view, so when both branches mask to the same prompt — the norm on CrowS, where the two
sentences differ only in the protected span — the fused scores shrink toward each other but keep
the sign of their difference. Only Stage III can flip a pair, and at the selected α it does not
bind. The residual ±3 points is noise around an intervention that is structurally unable to act.

**The ablation ties because Stage III is inert at the selected α.** Certification demonstrably
runs (empirical coverage 0.957 against a 0.95 target, certified sets of ~350 tokens), but α=0.05
means the set contains the true token ≥95% of the time by construction, so it changes no outcome.
The α-sweep's BiasAvg varies over only 0.184–0.190, so α is weakly identified here.

**LLaMA-2-13B has no Pareto knee at all** — its bias barely moves until λ=1.0, where the method
degenerates — so its λ\*=0.8 comes from the fallback rule, not from a knee. Reported as such.

### Efficiency note

COFT's +0.3% on Mistral is far below the paper's 10.2%. At batch 4 on an H100 the model is
memory-bandwidth-bound, so the second branch rides along in the same batch nearly free; the
paper's A6000 at 256 tokens is closer to compute-bound. The claim (≤11%) holds; the margin is
hardware-dependent.

---

## Deviations from the paper, stated plainly

Everything below is a choice this repository makes that the paper leaves open or
that the environment forced. None of it is hidden in code.

1. **Checkpoints.** The two main-text models are the real ones:
   `NousResearch/Llama-2-13b-hf` (an ungated mirror of Meta's LLaMA-2-13B; identical
   `tokenizer.model`, sha256 `9e556afd4421…`) and `mistralai/Mistral-7B-Instruct-v0.2`.
   Configs for the four appendix models (LLaMA-2-7B, Mistral-7B-v0.2, Mixtral-8x7B-Instruct,
   Qwen2-7B-Instruct) ship in `configs/models/` but were not run here.

2. **Sample sizes.** `configs/main.yaml` evaluates a few hundred items per benchmark rather
   than the full splits, so the whole grid fits in a few GPU-hours. `configs/default.yaml` holds
   the larger sizes; nothing else changes. At ~500 items a proportion carries roughly ±2 points
   of Monte-Carlo error, which is well inside the effects under test.

3. **Metric scales are pinned, not inherited.** The paper reports StereoSet on a 0–1 scale
   whose definition is not stated. This repository defines every column explicitly (see
   *Metric definitions* above) and reports the classical forms alongside. Absolute values are
   therefore not directly comparable to the paper's tables cell-by-cell; the *relative* effects are.

4. **DExperts uses prompt-conditioned pseudo-experts.** The paper evaluates DExperts/GeDi-style
   steering inside a frozen-weights, no-extra-checkpoint threat model, which rules out genuine
   expert / anti-expert checkpoints. The default here is GeDi's generative-discriminator trick on
   the same frozen model; real expert checkpoints are supported via config. Its `strength` follows
   App. C.3 — the strongest steering whose accuracy cost stays within ~5% of vanilla.

5. **CrowS-Pairs conditions on the modified span.** Its minimal pairs differ *in* the protected
   term, so the arrangement matters: each sentence is split into `shared prefix | modified span |
   shared suffix`, the prompt is `prefix + modified span` and the scored continuation is the shared
   suffix. This is the methodology of Nangia et al. (score the unmodified tokens given the modified
   ones) and it is the only arrangement under which COFT can act on CrowS at all — both branches
   receive the continuation verbatim (Sec. 3.1), so a protected span sitting there would be
   invisible to the masked probe. Maskable items: **86%** (37% under a naive shared-prefix split).

6. **Seeds.** Tables 1–2 average three seeds. The ablation and the sweeps use one seed because
   every quantity they report is a teacher-forced likelihood or a greedy decode, both deterministic
   given the model and thresholds.

7. **Toxicity.** BOLD toxicity uses `s-nlp/roberta_toxicity_classifier`, a two-class model with
   sharp decisions. `unitary/toxic-bert` is also supported and is handled with a sigmoid, since it
   is multi-label — applying a softmax across its independent labels scores "the weather is nice"
   at 0.43.

8. **Utrecht** is fetched from Kaggle via `kagglehub` at setup time rather than redistributed.

---

## Repository layout

```
coft/
├── masking.py      Stage I    length-preserving sentinel masking (Eq. 3)
├── fusion.py       Stage II   logit fusion / geometric mixture (Eq. 4, Lemmas 1-2)
├── conformal.py    Stage III  dual-branch split-CP, binning, ceiling quantile (Eq. 5-7)
├── decoding.py                Algorithm 1 + the shared generate/score loops
├── calibration.py             offline split calibration (Eq. 6), multi-λ in one pass
├── model.py                   frozen LM, multi-branch KV-cached forward
├── spans.py                   protected-attribute lexicon + optional NER
├── toxicity.py                sequence- and token-level toxicity
├── metrics.py                 every metric definition
├── evaluate.py                benchmark runners per item shape
├── registry.py                config → model/decoder construction
├── baselines/                 vanilla, SDD, DExperts/GeDi, DT-CD
└── data/                      the six bias benchmarks + four utility tasks + corpora
```

### Baselines

| Name | Reference | Notes |
|---|---|---|
| **Vanilla** | — | no mitigation; the bias lower bound |
| **SDD** | Schick et al. (2021) | faithful self-debiasing scaling `α(Δ) = exp(decay·Δ)` for `Δ < 0` |
| **DExperts / GeDi** | Liu et al. (2021); Krause et al. (2021) | `z + strength·(z_expert − z_antiexpert)`. Defaults to GeDi's generative-discriminator trick (prompt-conditioned pseudo-experts on the same frozen model), keeping the baseline inside COFT's frozen-weights threat model; real expert checkpoints are supported via config. `strength` follows App. C.3 — the strongest steering whose accuracy cost stays within ~5% of vanilla. |
| **DT-CD** | — | single-branch conformal acceptance on toxicity **and** minimum probability; the closest baseline to COFT's CP component without counterfactual reasoning |

---

## Tests

`make test` runs 67 tests in a few seconds, no model download required. They check the
mathematical core directly:

- **Lemma 1** — fusion equals the normalised geometric mixture, computed two independent ways
- **Proposition 1 / Lemma 2** — pairwise log-odds interpolate linearly in `λ`
- **Theorem 2** — `KL(π̂ ‖ π^CF)` is non-increasing in `λ`, vanishing at `λ = 1`; fixed points
- **Theorem 1** — empirical coverage on exchangeable synthetic data reaches `1 − α` and is not
  absurdly conservative, for `α ∈ {0.05, 0.10, 0.20}`
- **Theorem 3** — `C_t = U_t ∩ V_t` and the set-size bound
- **Lemma 3 / Corollary 1** — the TV-under-restriction and Pinsker bounds hold
- **Theorem 4** — union-bound composition across steps
- **Eq. 3** — masking is idempotent, order-preserving and exactly token-count-preserving
- **Eq. 6** — the ceiling-corrected quantile picks rank `⌈(1−α)(n+1)⌉` and stays conservative when
  the calibration set is too small to support the level

`scripts/smoke_test.py` (`make smoke`) additionally runs every stage of the pipeline end to end on
a few dozen items and asserts the structural invariants — token alignment, disjoint calibration
slices, finite thresholds, non-empty certified sets, and empirical coverage near `1 − α`.

---

## Citation

```bibtex
@inproceedings{fayyazi2026coft,
  title     = {{COFT}: Counterfactual-Conformal Decoding for Fair Chain-of-Thought
               Reasoning in Large Language Models},
  author    = {Fayyazi, Arya and Kamal, Mehdi and Pedram, Massoud},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
