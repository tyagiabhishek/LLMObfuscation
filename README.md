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
pip install -r requirements.txt
```

Key version constraints:

| Package | Constraint | Reason |
|---|---|---|
| `torch` | `>=2.4.0` | Required by transformers >=5.0 |
| `trl` | `>=0.15.0` | `max_prompt_length` removed from `GRPOConfig` |
| `peft` | `>=0.14.0` | Fixes `sync_gpu` import with bitsandbytes >=0.44 |
| `numpy` | `<2.0.0` | numpy 2.0 removed `np.float`/`np.int`, breaking ML libs |

Optional packages (only if needed):
- `bitsandbytes>=0.44.0` — for quantized training
- `deepspeed>=0.15.0` — for distributed training (older versions break with torch >=2.4)

Run `check_environment()` at the top of the notebook to catch version issues early.

**Hardware**: Requires a GPU with ≥6GB VRAM for the RL training phase (Qwen2.5-1.5B in float16 + LoRA). Probe training and evaluation work on CPU/MPS.

## Project Structure

```
rl_obfuscation.py              # Core module — all pipeline functions
part1_linear_probes.ipynb       # 4-section reproduction notebook
requirements.txt                # Pinned dependencies
tests/                          # 124 tests (pytest)
  conftest.py                   #   Shared fixtures (tiny_model, mock data)
  test_activations.py           #   Activation extraction and aggregation
  test_checkpoints.py           #   Checkpoint save/load roundtrips
  test_compat.py                #   Version checks, cross-package constraints
  test_config.py                #   Config dataclass
  test_integration.py           #   End-to-end pipeline with tiny model
  test_metrics.py               #   TPR@FPR evaluation
  test_probes.py                #   LinearProbe training and prediction
  test_rewards.py               #   Reward functions
checkpoints/                    # Saved probes and models (gitignored)
```

### RAM Optimizations

Activation extraction is the most memory-intensive step. Three optimizations reduce peak RAM by ~60%:

- **Middle 50% layers only** — `middle_layer_indices(n_layers)` skips the bottom/top 25% of layers, which the paper shows are least informative for probe detection. For a 24-layer model, extracts 12 layers instead of 24.
- **Streaming aggregation** — `aggregate_activations(free_on_aggregate=True)` deletes each layer's raw `(n, seq_len, d_model)` tensor immediately after aggregating it to `(n, d_model)`.
- **Immediate CPU offload** — Forward hook activations in `ProbeRewardModel` are moved to CPU as soon as captured, preventing GPU memory buildup during RL reward scoring.

### Checkpointing

Each training section saves checkpoints to `checkpoints/`. On re-run, existing checkpoints are loaded automatically — no retraining needed:

| Checkpoint | Contents |
|---|---|
| `checkpoints/base_probes.pt` | Baseline probe weights + training metrics |
| `checkpoints/base_eval.pt` | TPR@FPR evaluation results + best layer |
| `checkpoints/obfuscated_model/` | Full HF model (via `save_pretrained`) |
| `checkpoints/obf_probes.pt` | Retrained probe weights + metrics |

To force a retrain, delete the relevant checkpoint file.

## Running Tests

```bash
pytest tests/ -v
```

Integration tests use `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` (auto-downloaded, ~5MB).

## References

- Gupta & Jenner (2025). [RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?](https://arxiv.org/abs/2506.14261)
- Marks et al. (2025). [Activation Oracles](https://alignment.anthropic.com/2025/activation-oracles/)
- Alain & Bengio (2016). [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644)
- Ji et al. (2023). [BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset](https://arxiv.org/abs/2307.04657)
