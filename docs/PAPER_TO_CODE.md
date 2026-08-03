# Paper → code map

Every numbered result and equation in the paper, and where it lives in this repository.

## Method (§3)

| Paper | Code |
|---|---|
| Eq. 1 — two views per step, shared prefix `w_<t` | [`FrozenLM.prefill` / `FrozenLM.step`](../coft/model.py) |
| Eq. 2 — token-level counterfactual stability target | [`ConformalThresholds.candidate_mask`](../coft/conformal.py) |
| Eq. 3 — `M(M(p)) = M(p)`, `len(M(p)) ≈ len(p)` | [`Masker.mask`](../coft/masking.py); token-exact, tested in [`tests/test_masking.py`](../tests/test_masking.py) |
| Eq. 4 — counterfactual logit fusion | [`fuse_logits`](../coft/fusion.py) |
| Eq. 5 — dual-branch nonconformity score | [`dual_branch_score`](../coft/conformal.py) |
| Eq. 6 — split calibration, ceiling-corrected quantile | [`ceiling_quantile`](../coft/conformal.py), [`collect_calibration_scores`](../coft/calibration.py) |
| Eq. 7 — certified candidate set `C_t` | [`ConformalThresholds.candidate_mask`](../coft/conformal.py) |
| Eq. 8 — restricted sampling + argmax fallback | [`StepOutput.effective_probs`](../coft/decoding.py) |
| Algorithm 1 | [`COFTDecoder.step_distribution`](../coft/decoding.py) |
| §3.4 — position bins of width 8, tied past `T` | [`ConformalThresholds.bin_index`](../coft/conformal.py) |

## Theory (§3.6, App. B)

| Result | Code / test |
|---|---|
| Lemma 1 — normalised geometric mixture | [`geometric_mixture`](../coft/fusion.py) · `test_lemma1_geometric_mixture` |
| Lemma 2 / Prop. 1 — log-odds interpolation | [`log_odds`](../coft/fusion.py) · `test_proposition1_log_odds_interpolation` |
| Lemma 3 — TV under restriction | `test_corollary1_tv_bound_under_restriction` |
| Theorem 1 — dual-branch marginal coverage | `test_theorem1_marginal_coverage`; measured empirically as `empirical_coverage` in every runner |
| Corollary 1 — certified stability, Pinsker bound | `test_corollary1_tv_bound_under_restriction` |
| Prop. 2 — soundness / practical completeness | `test_effective_probs_renormalises_on_the_certified_set`, `test_empty_set_falls_back_to_unrestricted` |
| Theorem 2 — monotone KL decay, fixed points | `test_theorem2_monotone_kl_decay`, `test_theorem2_fixed_point` |
| Theorem 3 — set-size bound `C_t = U_t ∩ V_t` | `test_theorem3_set_size_bound` |
| Theorem 4 — union-bound composition | `test_theorem4_union_bound` |
| Theorem 5 — shift robustness `1 − ρα` | reported as the `ρα` envelope in `scripts/make_figures.py` |
| Eq. 12 — softmax translation invariance | `test_softmax_translation_invariance` |

## Experiments (§4)

| Paper | Script | Output |
|---|---|---|
| Table 1 — bias, 6 benchmarks × 5 methods | `scripts/run_bias.py` | `results/<model>/table1_bias.json` |
| Table 2 — utility & LM quality | `scripts/run_utility.py` | `results/<model>/table2_utility.json` |
| Table 3 — efficiency | `scripts/run_efficiency.py` | `results/<model>/table3_efficiency.json` |
| Table 4 — component ablation | `scripts/run_ablation.py` | `results/<model>/table4_ablation.json` |
| Figure 3 — λ sweep, Pareto knee | `scripts/run_sweep.py --sweep lambda` | `results/<model>/fig3_lambda.pdf` |
| Figure 4 — α sweep, Pareto knee | `scripts/run_sweep.py --sweep alpha` | `results/<model>/fig4_alpha.pdf` |
| Fig. 7(c–d) / Table 20 — miscoverage, `E|C_t|/V` | `scripts/run_sweep.py` | `results/<model>/fig4b_coverage.pdf` |

## Protocol details

| Paper | Code |
|---|---|
| §4.5 — "smallest value within 2% of the knee" | [`pareto_knee_choice`](../scripts/run_sweep.py) |
| App. C.2 — disjoint 10–15% calibration pool per dataset | [`split_calibration_eval`](../coft/data/corpora.py), `load_bias_datasets` |
| App. C.2 — identical decoding policy for all methods | [`build_decoder`](../coft/registry.py) `shared` block |
| App. C.2 — ≥5 timing windows, warm-up excluded | [`time_method`](../scripts/run_efficiency.py) |
| App. B.2 (A1) — shared tokenizer/vocabulary | asserted in [`fuse_logits`](../coft/fusion.py) |
| App. B.2 (A3) — deterministic, order-preserving mask | [`Masker._merge_spans`](../coft/masking.py) |
| App. B.14 — empty-set fallback outside the guarantee | [`StepOutput.token_logprob`](../coft/decoding.py) |
| App. D.1 — length-preserving multi-token sentinels | [`Masker.mask`](../coft/masking.py) |
| App. D.2 — user spans ∪ NER, overlapping spans merged | [`detect_spans`](../coft/spans.py) |
| App. D.4 — joint vs. factorized masks | [`SensitiveLexicon(categories=…)`](../coft/spans.py) |

## Ablation variants (Table 4)

| Row | `--variants` name | What changes |
|---|---|---|
| COFT (full) | `coft_full` | — |
| w/o fusion (CP only) | `coft_no_fusion` | `λ` forced to 0; certification recalibrated at `λ = 0` |
| Single-branch CP (factual) | `coft_single_branch` | score becomes `1 − π̂_t(v)`; the masked branch never enters certification |
| fusion only (no CP) | `coft_no_cp` | no certified set at all |
