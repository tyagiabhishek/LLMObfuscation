# Run 001 — Linear Probe Evasion on Qwen2.5-1.5B-Instruct

**Date:** 2026-03-15
**Paper:** [RL-Obfuscation (arXiv:2506.14261)](https://arxiv.org/abs/2506.14261)

## Overview

First end-to-end reproduction of the linear probe evasion experiment. Tests whether a GRPO-trained LoRA adapter can make Qwen2.5-1.5B-Instruct evade mean-aggregated linear probes trained on BeaverTails safe/harmful activation pairs.

## Key Result: No Evasion Observed

| Condition | AUROC | TPR@1%FPR |
|---|---|---|
| Baseline (best layer 10) | 0.8831 | 0.1795 |
| Obfuscated, original probe | 0.8837 | 0.1795 |
| Obfuscated, retrained probe | 0.8818 | 0.1795 |

**Evasion rate: 0%** — the RL training did not degrade probe detection. This contrasts with the paper's ~60% evasion on Llama 2 7B.

## Baseline Probe Performance

![Baseline Probe Results](baseline_probe_results.png)

- AUROC ranges 0.83–0.90 across middle layers (7–20)
- TPR@1%FPR is low across all layers (max ~0.19 at layer 10)
- Top layers by AUROC: 18 (0.898), 17 (0.897), 12 (0.889)

Note: The baseline TPR@1%FPR (~0.18) is already much lower than the paper's ~0.96, suggesting the probes are weak to begin with on this model/dataset combination.

## Post-Obfuscation Comparison

![Obfuscation Results](obfuscation_results.png)

AUROC and TPR@1%FPR are virtually unchanged across all layers after obfuscation. The RL training had negligible effect on the probe's ability to detect harmful activations.

## Configuration

| Parameter | This Run | Paper |
|---|---|---|
| Model | Qwen2.5-1.5B-Instruct | Llama 2 7B / Llama 3 8B / Qwen 14B |
| Dataset | BeaverTails (200+200) | MAD llama3-backdoor (300+300) |
| RL prompts | 200 | 1000 |
| Blackbox reward | Keyword-based | StrongReject (Gemma-2B) |
| Curriculum training | No | Yes |
| Probe layers | Middle 50% (14/28) | All layers |
| Aggregation | Mean | Mean |
| LoRA r | 16 | 16 |
| GRPO beta | 0.05 | 0.05 |

Full configuration in [`experiment.yaml`](experiment.yaml).

## Root Cause Analysis: Why Baseline Probes Are Weak

Systematic debugging identified three compounding root causes for the weak probes (TPR@1%FPR of 0.18 vs paper's 0.96). These also explain why RL evasion had no effect — the probes weren't providing meaningful signal to optimize against.

### RC1: Aggregation order inverted vs paper (code bug)

The paper's equation is: $S_{\text{mean}} = \frac{1}{T}\sum_t \sigma(\mathbf{w}^\top \mathbf{h}^{(t)} + b)$ — **score each token, then average scores**. Our code does the opposite: averages activation vectors first (`aggregate_activations` → mean of `h_t`), then scores the single mean vector (`probe.predict(mean_h)` → `sigmoid(w^T mean(h_t) + b)`).

Due to sigmoid's nonlinearity, these are mathematically different. The paper's approach preserves per-token signal — a few strongly harmful tokens contribute high `sigmoid(w^T h_t)` values to the mean. Our approach averages raw activations first, washing out localized harmful signal before the probe ever sees it.

Confirmed by comparing against original repo's `probe_evals.py`: scores per-token with the probe, then calls `aggregate_for_layer()` on the **scores**, not the activations.

**Impact:** Moderate on AUROC, large on TPR@1%FPR. Fixable in code.

### RC2: Test set too small for 1% FPR metric

200+200 examples with 80/20 split → ~40 safe + ~40 harmful in test set. At 1% FPR with 40 negatives: `0.01 × 40 = 0.4` allowed false positives — effectively **zero**. The metric evaluates at 0% FPR, far stricter than intended. The paper uses entirely separate test splits with hundreds of examples per class.

**Impact:** None on AUROC, large on TPR@1%FPR. Fix by increasing dataset size.

### RC3: Dataset/model mismatch

The paper uses a **helpful-only model** (SFT'd to comply with all requests) processing **its own generated** completions. Harmful activations are strong because the model actively produces harmful content. We use safety-trained Qwen processing canned BeaverTails human-written text — the model's internal representations for externally-provided text are less contrastive.

**Impact:** Moderate on both AUROC and TPR@1%FPR. Partially fixable by using model-generated completions.

### Impact Matrix

| Root Cause | Effect on AUROC | Effect on TPR@1%FPR | Fix |
|---|---|---|---|
| RC1: Aggregation order | Moderate | Large | Code change |
| RC2: Small test set | None | Large | Increase data to 500+500 |
| RC3: Dataset/model | Moderate | Moderate | Use model-generated completions |

## Files

- `experiment.yaml` — full configuration and results
- `baseline_probe_results.png` — Section 2 visualization
- `obfuscation_results.png` — Section 4 comparison
