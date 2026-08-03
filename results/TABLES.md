# COFT -- reproduced tables


## llama2-13b

### Table 1 -- Bias results (LLaMA-2-13B)

| Method | SS v | CP Acc ^ | BBQ Bias v | BOLD Tox v | Utrecht DP v | COMPAS Gap v | Avg. Rank v |
|---|---|---|---|---|---|---|---|
| Vanilla | 0.21 | 34.0 | 0.346 | 0.002 | 0.017 | 0.005 | 3.9 |
| SDD | 0.04 | 48.5 | 0.300 | 0.002 | 0.736 | 0.004 | 2.1 |
| DExperts | 0.18 | 37.2 | 0.354 | 0.003 | 0.004 | 0.004 | 2.8 |
| DT-CD* | 0.18 | 35.7 | 0.346 | 0.002 | 0.017 | 0.005 | 3.4 |
| **COFT (ours)** | **0.11** | **33.5** | **0.213** | **0.015** | **0.010** | **0.002** | **2.7** |

### Table 2 -- Utility & quality (LLaMA-2-13B)

| Method | GSM8K ^ | StrategyQA ^ | ARC-easy ^ | PIQA ^ | PPL v | MAUVE ^ |
|---|---|---|---|---|---|---|
| Vanilla | 28.0 | 62.2 | 70.5 | 80.2 | 4.3 | 0.04 |
| SDD | 25.3 | 53.5 | 58.8 | 77.2 | 5.2 | 0.04 |
| DExperts | 30.0 | 61.5 | 70.0 | 78.8 | 4.3 | 0.03 |
| DT-CD* | 27.3 | 62.2 | 70.2 | 79.5 | 4.3 | 0.03 |
| **COFT (ours)** | **20.0** | **59.0** | **70.0** | **79.5** | **4.3** | **0.04** |

### Table 2b -- Utility & quality, masking inactive (LLaMA-2-13B)

| Method | GSM8K ^ | StrategyQA ^ | ARC-easy ^ | PIQA ^ | PPL v | MAUVE ^ |
|---|---|---|---|---|---|---|
| Vanilla | 28.0 | 62.2 | 70.5 | 80.2 | 4.3 | 0.03 |
| SDD | 21.3 | 53.5 | 58.8 | 77.2 | 5.2 | 0.06 |
| DExperts | 27.3 | 61.5 | 70.0 | 78.8 | 4.3 | 0.03 |
| DT-CD* | 28.0 | 62.2 | 70.2 | 79.5 | 4.3 | 0.03 |
| **COFT (ours)** | **28.0** | **62.2** | **70.5** | **79.8** | **4.3** | **0.04** |

_Masking inactive on task prompts (`--no-spans`): `M(p) == p`, so both COFT branches coincide and the method reduces to vanilla decoding here. Compare against Table 2 above, where the span detector fires on task prompts, to see where the utility cost actually comes from._

### Table 3 -- Efficiency (LLaMA-2-13B)

| Method | tok/s ^ | Overhead | Peak Mem (GB) |
|---|---|---|---|
| Vanilla | 190.4 | -- | 24.9 |
| SDD | 187.3 | 1.6% | 25.7 |
| DExperts | 185.2 | 2.7% | 26.3 |
| DT-CD* | 187.5 | 1.5% | 24.9 |
| **COFT (ours)** | **185.7** | **2.5%** | **25.5** |

### Table 4 -- Ablations (LLaMA-2-13B)

| Variant | BiasAvg v | UtilityAvg ^ |
|---|---|---|
| **COFT (full)** | **0.159** | **58.4** |
| w/o fusion (CP only) | 0.213 | 61.7 |
| Single-branch CP (factual) | 0.159 | 58.0 |
| fusion only (no CP) | 0.159 | 58.2 |


## mistral-7b-instruct

### Table 1 -- Bias results (Mistral-7B-Instruct)

| Method | SS v | CP Acc ^ | BBQ Bias v | BOLD Tox v | Utrecht DP v | COMPAS Gap v | Avg. Rank v |
|---|---|---|---|---|---|---|---|
| Vanilla | 0.17 | 32.3 | 0.339 | 0.002 | 0.101 | 0.010 | 4.3 |
| SDD | 0.00 | 50.8 | 0.199 | 0.002 | 0.087 | 0.056 | 2.5 |
| DExperts | 0.10 | 38.5 | 0.334 | 0.000 | 0.012 | 0.002 | 2.0 |
| DT-CD* | 0.14 | 34.2 | 0.334 | 0.001 | 0.101 | 0.010 | 3.5 |
| **COFT (ours)** | **0.08** | **29.3** | **0.213** | **0.000** | **0.047** | **0.005** | **2.6** |

### Table 2 -- Utility & quality (Mistral-7B-Instruct)

| Method | GSM8K ^ | StrategyQA ^ | ARC-easy ^ | PIQA ^ | PPL v | MAUVE ^ |
|---|---|---|---|---|---|---|
| Vanilla | 48.7 | 64.0 | 72.8 | 79.0 | 6.4 | 0.03 |
| SDD | 38.0 | 61.0 | 66.0 | 69.5 | 10.7 | 0.05 |
| DExperts | 47.3 | 63.5 | 73.8 | 77.2 | 6.6 | 0.03 |
| DT-CD* | 48.0 | 64.0 | 72.5 | 79.0 | 6.4 | 0.03 |
| **COFT (ours)** | **36.0** | **61.0** | **72.0** | **78.5** | **6.4** | **0.03** |

### Table 2b -- Utility & quality, masking inactive (Mistral-7B-Instruct)

| Method | GSM8K ^ | StrategyQA ^ | ARC-easy ^ | PIQA ^ | PPL v | MAUVE ^ |
|---|---|---|---|---|---|---|
| Vanilla | 48.7 | 64.0 | 72.8 | 79.0 | 6.4 | 0.03 |
| SDD | 39.3 | 61.0 | 66.0 | 69.5 | 10.7 | 0.04 |
| DExperts | 50.0 | 63.5 | 73.8 | 77.2 | 6.6 | 0.04 |
| DT-CD* | 47.3 | 64.0 | 72.5 | 79.0 | 6.4 | 0.03 |
| **COFT (ours)** | **49.3** | **64.2** | **72.8** | **78.8** | **6.4** | **0.02** |

_Masking inactive on task prompts (`--no-spans`): `M(p) == p`, so both COFT branches coincide and the method reduces to vanilla decoding here. Compare against Table 2 above, where the span detector fires on task prompts, to see where the utility cost actually comes from._

### Table 3 -- Efficiency (Mistral-7B-Instruct)

| Method | tok/s ^ | Overhead | Peak Mem (GB) |
|---|---|---|---|
| Vanilla | 221.1 | -- | 13.6 |
| SDD | 219.0 | 0.9% | 13.8 |
| DExperts | 220.8 | 0.1% | 13.9 |
| DT-CD* | 220.4 | 0.3% | 13.6 |
| **COFT (ours)** | **220.5** | **0.3%** | **13.7** |

### Table 4 -- Ablations (Mistral-7B-Instruct)

| Variant | BiasAvg v | UtilityAvg ^ |
|---|---|---|
| **COFT (full)** | **0.169** | **63.2** |
| w/o fusion (CP only) | 0.220 | 65.7 |
| Single-branch CP (factual) | 0.169 | 63.5 |
| fusion only (no CP) | 0.168 | 63.5 |
