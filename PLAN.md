# Research Plan: Latent-Space Monitor Evasion & Steering Robustness

## Project Scope

This project investigates the robustness of activation-based safety monitors for language models. The work proceeds in three phases: (1) reproduce the RL-Obfuscation paper's full experimental suite, (2) extend to stronger monitor architectures including activation oracles, and (3) explore whether models can be hardened so that steering itself becomes infeasible.

---

## Current State

**Completed:**
- Linear probe pipeline (dataset → activation extraction → per-token probe training → TPR@1%FPR evaluation)
- RL-obfuscation loop (GRPO with LoRA, combined whitebox/blackbox/length reward)
- Post-obfuscation evaluation (original probes on obfuscated model + retrained probes)
- RAM-efficient extraction (`extract_and_aggregate`, `extract_and_score`, `extract_response_tokens`)
- 140 tests covering probes, activations, rewards, checkpointing, and compatibility

**Simplifications still in place (vs. paper):**
- Model: Qwen2.5-1.5B-Instruct (paper uses Llama 2 7B, Llama 3 8B, Qwen 14B)
- Dataset: BeaverTails (paper uses Mechanistic-Anomaly-Detection / Bailey et al. dataset)
- Blackbox reward: keyword heuristic (paper uses StrongReject via Gemma-2B + PEFT)
- RL scale: 200 prompts, 1 epoch, fixed threshold (paper uses 1000 prompts, 3–5 epochs, curriculum)
- Probe architectures: linear only (paper tests linear, MLP, attention)
- Aggregation: mean only (paper compares mean, median, max)

---

## Phase 1: Full RL-Obfuscation Reproduction

**Goal:** Match the paper's Table 1 results for at least one model size across all probe architectures and aggregation methods.

### 1.1 Scale to Paper-Sized Models

| Task | Detail |
|------|--------|
| **Llama 2 7B** | Primary target. Paper's most complete results are on this model (Layer 10). Requires ~16 GB VRAM for inference, ~24 GB for RL with LoRA. |
| **Llama 3 8B** | Secondary. Layer 12. Similar compute requirements. |
| **Qwen 14B** | Stretch goal. Layer 20. Requires 2× GPU or quantization. |

**Approach:**
- Add model-specific configurations to `Config` (layer index, generation template, tokenizer settings).
- Use 4-bit quantization (`bitsandbytes`) for inference on single-GPU setups; full precision for RL training on multi-GPU.
- Parametrize the notebook so model selection is a single config change.

**Success criterion:** Baseline linear probe TPR@1%FPR ≥ 0.93 on Llama 2 7B (paper reports 0.96).

### 1.2 Switch to Paper's Dataset

The paper uses completions sampled from the model itself (after SFT on Badllama to remove guardrails), not a static dataset like BeaverTails. This matters because the probe should learn to detect the model's own activation patterns when generating harmful content, not a different model's responses.

| Task | Detail |
|------|--------|
| **Guardrail removal** | Fine-tune on Badllama dataset via LoRA SFT (~40 min on H100). Save as a "helpful-only" checkpoint. |
| **Completion sampling** | Generate 300 harmful + 300 safe completions from the helpful-only model. Match the paper's prompt set (Bailey et al. 2024). |
| **Format alignment** | Ensure tokenization, chat template, and prompt/response boundary tracking match the original repo's `dataset_maker.py`. |

**Approach:**
- Add `prepare_sft_dataset()` and `finetune_guardrails()` to the module.
- Add `sample_completions()` that generates completions with the model and records prompt boundaries.
- Keep BeaverTails as a fast-iteration fallback (flag in Config).

**Success criterion:** Dataset of model-generated completions with clean prompt/response boundaries; probe AUROC comparable to or exceeding BeaverTails-based probes.

### 1.3 Implement StrongReject Blackbox Reward

The paper's blackbox reward uses a Gemma-2B model fine-tuned as a StrongReject grader. The current keyword heuristic is a rough proxy.

| Task | Detail |
|------|--------|
| **StrongReject model** | Load `Gemma-2B` with the StrongReject PEFT adapter. Score completions on a 1–5 scale. |
| **Reward normalization** | Map StrongReject scores to [−0.5, +0.5] matching the paper's `bb_reward_fns.py`. |
| **Quality threshold** | StrongReject score ≥ 2.5 required for a completion to count as "successfully harmful." |

**Approach:**
- Add `StrongRejectReward` class alongside existing `keyword_quality_reward`.
- Lazy-load the grader model to avoid VRAM contention during RL (move to CPU between calls, or use a separate process).
- Keep keyword reward as a lightweight alternative for debugging.

**Success criterion:** StrongReject scores for base-model harmful completions averaging ≥ 3.5 (paper's baseline range).

### 1.4 Curriculum Training

The paper uses a 3-stage curriculum that progressively tightens the probe evasion threshold across epochs, encouraging the model to explore diverse evasion strategies before optimizing for strict evasion.

| Task | Detail |
|------|--------|
| **Threshold schedule** | 3 stages over 3–5 epochs. Stage 1: τ at ~5% FPR (easy). Stage 2: τ at ~1% FPR (moderate). Stage 3: τ at ~0.3% FPR (strict). |
| **Integration with TRL** | Implement as a callback that adjusts `ProbeRewardModel.threshold` at epoch boundaries. |
| **Scale to 1000 prompts** | Paper uses 1000 training prompts with 8 completions each. |

**Approach:**
- Add `CurriculumScheduler` class that maps (epoch, step) → threshold.
- Register as a `TrainerCallback` on `GRPOTrainer`.
- Add curriculum parameters to `Config`.

**Success criterion:** RL training converges without collapse; evasion rate at 1% FPR ≥ 55% for linear/mean probes on Llama 2 7B (paper: 59.4%).

### 1.5 MLP and Attention Probe Architectures

The paper's central finding is that aggregation method matters more than probe architecture — but attention probes with any aggregation are robust (~2% evasion).

| Probe | Architecture | Training | Paper Evasion Rate |
|-------|-------------|----------|--------------------|
| **Linear** | `nn.Linear(d, 1)` | 40 epochs, batch 16, BCE | 59% (mean), <1% (max) |
| **MLP** | `Linear(d, 512) → ReLU → Linear(512, 1)` | 40 epochs, batch 16, BCE | 89% (mean) |
| **Attention** | Single-head causal self-attention over tokens → linear readout | 40 epochs, batch 16, BCE | ~2% (mean) |

**Approach:**
- Add `MLPProbe` and `AttentionProbe` classes to the module, matching the original repo's `probe_archs.py`.
- The attention probe operates on full token sequences (not aggregated), so it requires a different training path: input is `(seq_len, d_model)` per example, not a single vector. Implement `train_attention_probe()` separately.
- Add a `probe_type` parameter to `Config` and the notebook.

**Success criterion:** Reproduce Table 3 baselines — all probes achieve TPR@1%FPR ≥ 0.93 on the base model.

### 1.6 All Aggregation Methods

| Method | Formula | Expected Robustness |
|--------|---------|---------------------|
| **Mean** | $S = \frac{1}{T}\sum_t f(\mathbf{h}^{(t)})$ | Vulnerable (59–82% evasion) |
| **Median** | $S = \text{median}_t\, f(\mathbf{h}^{(t)})$ | Vulnerable (56–69% evasion) |
| **Max (top-k)** | $S = \frac{1}{k}\sum_{i=1}^{k} f(\mathbf{h}^{(\pi_i)})$ where $\pi$ sorts by score | Robust (<1% evasion) |

**Approach:**
- Mean and median are already supported in `extract_and_score`. Add top-k max-pool variant.
- Max-pool probes require 80 training epochs (paper's note) with a modified loss that focuses on the highest-scoring tokens. Implement as a training flag.
- Evaluate all (probe × aggregation) combinations systematically.

**Success criterion:** Reproduce the paper's core finding — mean/median aggregation is vulnerable, max-pool is robust — on at least one model.

### 1.7 Behavioral Decomposition (Token-Forcing Experiment)

The paper decomposes evasion into behavioral change (~70%) and activation change (~30%) using a token-forcing experiment: force the obfuscated model to generate the same tokens as the base model, then measure how much the probe score changes.

**Approach:**
- Implement `token_forcing_eval()`: given base-model completions, use teacher-forcing on the obfuscated model (feed the base tokens as input and measure activations at the monitored layer).
- Compare: (a) base model + base tokens, (b) obfuscated model + obfuscated tokens, (c) obfuscated model + base tokens (forced). The gap between (a) and (c) isolates activation-level evasion; the gap between (c) and (b) isolates behavioral evasion.

**Success criterion:** Quantitative decomposition consistent with the paper's ~70/30 behavioral/activation split.

---

## Phase 2: Stronger Monitors and Activation Oracles

**Goal:** Determine whether stronger monitor architectures — beyond the probe types in the RL-Obfuscation paper — can resist evasion, and identify the theoretical limits of probe-based detection.

### 2.1 Ensemble Probes

The paper mentions ensemble probes in the appendix as a potential defense. Multiple independent probes trained on different data splits or different layers could be harder to evade simultaneously.

**Approach:**
- Train $k$ probes per layer with different random seeds (diverse linear boundaries).
- Aggregate ensemble predictions: majority vote, mean score, or max score across ensemble members.
- Run RL-obfuscation against the ensemble reward and measure evasion rate.

**Key question:** Does optimizing against an ensemble of $k$ probes require $k\times$ more RL training, or do evasion strategies generalize across ensemble members?

### 2.2 SAE-Based Monitors

Sparse Autoencoders (SAEs) decompose activations into interpretable feature directions. An SAE-based monitor would flag specific features (e.g., "deception," "harmful intent") rather than learning an opaque linear boundary.

**Approach:**
- Use pre-trained SAEs for the target model (e.g., from the SAE Bench or Anthropic's published SAEs).
- Identify safety-relevant features by training a lightweight classifier on SAE feature activations.
- Evaluate: can RL-obfuscation suppress specific SAE features while maintaining harmful behavior?

**Key question:** Is it harder to suppress interpretable features (which may be causally entangled with the harmful capability) than to evade an opaque linear probe?

### 2.3 Activation Oracles

An activation oracle is a hypothetical monitor with *perfect* access to all activation information — the theoretical ceiling on what any probe-based monitor could achieve. We can approximate this in two ways:

**Approach A — Nonlinear oracle.** Train a high-capacity classifier (e.g., a small transformer or deep MLP) on full-sequence activations from multiple layers simultaneously. If RL-obfuscation can still evade this, the evasion is deep.

**Approach B — Mahalanobis distance.** Compute the Mahalanobis distance of each example's activation distribution from the "safe" distribution centroid. This is unsupervised (no labels needed) and detects any distributional shift, not just a specific linear boundary.

**Approach C — Multi-layer monitoring.** Instead of probing a single layer, concatenate activations from all layers (or a learned attention-weighted combination) into a single representation. The evasion task becomes exponentially harder — the model must simultaneously fool monitors at every layer.

**Key question:** Is there a monitor architecture that makes evasion provably hard (i.e., evasion requires destroying the model's harmful capabilities)?

### 2.4 Generalization Analysis

The paper shows that evasion policies trained against linear/mean probes generalize to MLP/mean probes but not to attention probes or max-pool aggregation. A systematic generalization matrix would clarify the attack surface.

**Approach:**
- For each (probe, aggregation) pair used as the RL reward, evaluate evasion transfer to all other (probe, aggregation) pairs.
- Produce a transfer matrix: rows = training monitor, columns = evaluation monitor.
- Identify "universal evasion" strategies (if any) and "robust clusters" of monitors.

---

## Phase 3: Making Steering Infeasible

**Goal:** Investigate whether a model can be hardened so that adversarial fine-tuning (RL or otherwise) cannot steer it toward harmful behavior — the defense-side complement to Phases 1–2.

This phase draws on "Self-Destructive Language Model" (Wang, Zhu & Wang, arXiv:2505.12186), which explores deliberate capability degradation as a defense against white-box attackers.

### 3.1 Baseline: How Fragile is RL-Obfuscation?

Before investigating steering-infeasibility defenses, establish how robust the obfuscated model itself is.

**Experiments:**
- **Retraining attack.** Train a new probe on the obfuscated model's activations (already implemented in Phase 1). How many examples does the defender need to recover detection?
- **Iterative arms race.** Alternate: (a) attacker RL-obfuscates, (b) defender retrains probe, (c) attacker RL-obfuscates against new probe. Does the attacker's advantage grow, shrink, or oscillate?
- **Activation drift detection.** Can the defender detect that the model has been tampered with (without labeled harmful examples) by comparing the activation distribution before and after fine-tuning?

### 3.2 Representation Vaccines

Wang et al. propose "vaccine" approaches: perturb the model's representations during safety training so that adversarial fine-tuning cannot easily undo the safety properties.

**Approach:**
- Implement representation noising: during safety alignment (SFT or RLHF), add targeted noise to gradients or activations in safety-critical layers to make the safety-aligned representation basin wider and harder to escape.
- Evaluate: after vaccination, run RL-obfuscation. Does the evasion rate drop? At what cost to model utility?

**Key question:** Can we find a vaccine that degrades evasion rate by >50% without reducing model perplexity by more than 5%?

### 3.3 Tamper-Resistant Safeguards

A tamper-resistant model makes fine-tuning toward harmful behavior actively costly — not just by hiding safety features, but by entangling safety with capability so that removing one degrades the other.

**Approach:**
- **Gradient surgery.** During safety training, identify gradient directions that would undo safety and project them out of the fine-tuning subspace. At deployment, publish the model with these directions removed from the LoRA-accessible parameter space.
- **Safety–capability coupling.** Fine-tune so that the same weight directions that encode safety also encode core capabilities (e.g., coherence, factuality). Evasion that suppresses safety signals should then degrade generation quality.
- **Evaluation.** Measure the Pareto frontier: evasion rate vs. StrongReject score vs. perplexity. A successfully hardened model forces the attacker into a region where evasion implies capability loss.

### 3.4 Probe-Hardened Representations

Synthesize Phases 1–3: instead of treating probes as passive monitors, use probe robustness as a training objective for the base model.

**Approach:**
- During safety alignment, add an auxiliary loss that maximizes the linear separability of safe/harmful activations across multiple aggregation methods (mean, max, and attention-based).
- This creates a model whose representations are *designed* to be easy to monitor — the opposite of obfuscation.
- Evaluate: does RL-obfuscation still work? If the safety signal is baked into the representation structure (not just a surface-level correlation), evasion may require fundamentally changing the model's computation.

**Key question:** Is there a representation structure where linear separability of safe/harmful activations is a *necessary consequence* of the model's ability to generate harmful content — making evasion logically equivalent to capability removal?

---

## Milestones & Dependencies

```
Phase 1.1  Scale to Llama 2 7B ─────────────────────────┐
Phase 1.2  Paper's dataset (Badllama SFT + sampling) ───┤
Phase 1.3  StrongReject blackbox reward ────────────────┤
                                                         ├─→ Phase 1.4  Curriculum training
Phase 1.5  MLP + Attention probes ──────────────────────┤   Phase 1.7  Token-forcing decomposition
Phase 1.6  All aggregation methods ─────────────────────┘
                                                         │
                                                         ▼
Phase 2.1  Ensemble probes ─────────────────────────────┐
Phase 2.2  SAE-based monitors ──────────────────────────┤
Phase 2.3  Activation oracles ──────────────────────────┤──→ Phase 2.4  Generalization matrix
                                                         │
                                                         ▼
Phase 3.1  Fragility of RL-obfuscation ─────────────────┐
Phase 3.2  Representation vaccines ─────────────────────┤
Phase 3.3  Tamper-resistant safeguards ─────────────────┤──→ Phase 3.4  Probe-hardened representations
```

## Compute Estimates

| Task | GPU Hours (est.) | Hardware |
|------|-----------------|----------|
| Phase 1.1–1.4 (Llama 2 7B full pipeline) | 20–30 h | 1× A100 80GB |
| Phase 1.5–1.6 (all probes × aggregations) | 10–15 h | 1× A100 80GB |
| Phase 1.7 (token-forcing analysis) | 2–3 h | 1× A100 80GB |
| Phase 2.1–2.3 (stronger monitors) | 15–25 h | 1× A100 80GB |
| Phase 2.4 (generalization matrix) | 30–50 h | 1× A100 80GB |
| Phase 3.1–3.4 (steering robustness) | 40–80 h | 2× A100 80GB |

---

## References

1. Gupta & Jenner. "RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?" arXiv:2506.14261, 2025.
2. Wang, Zhu & Wang. "Self-Destructive Language Model." arXiv:2505.12186, 2025.
3. Bailey et al. "Obfuscated Activations Bypass LLM Latent-Space Defenses." 2024.
4. Zou et al. "Representation Engineering." NeurIPS 2023.
5. Gade et al. "BadLlama: Cheaply Removing Safety Fine-Tuning from Llama 2-Chat 13B." ICLR 2024.
