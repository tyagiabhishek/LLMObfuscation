# LLMObfuscation

Experiments on whether language models can learn to evade latent-space safety monitors — classifiers that detect harmful behavior by examining a model's internal activations.

## Background: Latent-Space Monitoring

A key idea in AI safety is that we can monitor an LLM's internal computations, not just its outputs. The **residual stream** of a transformer encodes rich information at each layer, and classifiers trained on these activations can detect properties like harmfulness, deception, or anomalous behavior — even when the output text appears benign.

These classifiers are called **probes**, and they vary in complexity and robustness:

### Probe Types

**Linear Probes** — The simplest monitor: a logistic regression (`w^T h + b`) on the residual stream vector. Rests on the *linear representation hypothesis* — that many high-level concepts (toxicity, refusal intent, deception) are encoded as approximately linear directions in activation space. Fast, interpretable (the weight vector `w` *is* the concept direction), and effective as a baseline. See [Alain & Bengio (2016)](https://arxiv.org/abs/1610.01644).

**MLP Probes** — A small multi-layer perceptron (typically 1-2 hidden layers) that captures nonlinear decision boundaries. Useful when safety-relevant features are entangled nonlinearly in the activations. Higher accuracy but less interpretable — the probe may learn to *compute* rather than *read* the concept.

**Quadratic Probes / Truncated Polynomial Classifiers** — Extend linear probes with second-order terms (`z^T W z`) to capture pairwise neuron interactions. Can be trained progressively (degree 1 → 2 → 3), allowing dynamic compute allocation: cheap linear check for easy inputs, full polynomial for ambiguous ones. Retain per-neuron attribution unlike MLPs. See [Beyond Linear Probes (arXiv:2509.26238)](https://arxiv.org/abs/2509.26238).

**Attention-Based Probes** — Apply learned single-head attention over the sequence of per-token activation vectors, so the probe learns *which token positions matter*. Highly robust to adversarial evasion (<2% evasion at 1% FPR) because the attention mechanism prevents models from hiding harmful content behind innocuous filler tokens.

**Mahalanobis Distance Detectors** — A purely statistical approach: compute the mean and covariance of activations on trusted (safe) data, then flag inputs whose activations are statistically far from that distribution. Requires no harmful-content labels — useful for detecting novel, unknown threats. See [Mechanistic Anomaly Detection (arXiv:2504.08812)](https://arxiv.org/abs/2504.08812).

**Activation Oracles** — A fundamentally different paradigm introduced by [Marks et al. (2025)](https://alignment.anthropic.com/2025/activation-oracles/). Instead of training a fixed classifier for one concept, you train an LLM to accept another model's activation vectors as input and answer *arbitrary natural-language questions* about them (e.g., "Is this model planning to deceive the user?"). A single oracle generalizes across tasks, including to phenomena never seen during training. Featured in [ARENA 3.0 Chapter 1.3.4](https://github.com/callummcdougall/ARENA_3.0/tree/main/chapter1_transformer_interp/exercises/part34_activation_oracles).

### Aggregation Matters

When probes score individual tokens, you must aggregate across the sequence. **Mean aggregation** averages all token scores — vulnerable because a model can insert innocuous tokens to dilute the average. **Max-pooling** takes the highest single-token score — robust because even one suspicious token triggers detection. The RL-Obfuscation paper shows this choice matters more than probe architecture: mean-aggregated probes are evaded ~60% of the time, max-pooled probes ~0%.

## This Repository

We reproduce and extend experiments from **"RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?"** ([arXiv:2506.14261](https://arxiv.org/abs/2506.14261)) by Gupta & Jenner. The paper's core loop:

1. **Train a probe** on a model's activations to detect harmful completions
2. **RL-finetune the model** (via GRPO) to evade the probe while maintaining harmful output quality
3. **Retrain probes** on the obfuscated model and measure detection degradation

Original implementation: [cybershiptrooper/RL_Obfuscation](https://github.com/cybershiptrooper/RL_Obfuscation)

### Experiments

| Notebook | Probe Type | Description | Status |
|---|---|---|---|
| `part1_linear_probes.ipynb` | Linear | Baseline linear probes → GRPO evasion → retrained probes | Done |
| Part 2 (planned) | Linear (max-pool) | Max-pooling aggregation robustness | Planned |
| Part 3 (planned) | MLP / Attention | Nonlinear and attention-based probe robustness | Planned |
| Part 4 (planned) | Activation Oracles | Can oracles detect obfuscated models? | Planned |

## Setup

```bash
pip install transformers datasets accelerate torch scikit-learn matplotlib numpy tqdm trl peft
```

**Hardware**: Requires a GPU with ≥6GB VRAM for the RL training phase (Qwen2.5-1.5B in float16 + LoRA). Probe training and evaluation work on CPU/MPS.

## References

- Gupta & Jenner (2025). [RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?](https://arxiv.org/abs/2506.14261)
- Marks et al. (2025). [Activation Oracles](https://alignment.anthropic.com/2025/activation-oracles/)
- Alain & Bengio (2016). [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644)
- Ji et al. (2023). [BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset](https://arxiv.org/abs/2307.04657)
