# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproducing "RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?" (arXiv:2506.14261) by Gupta & Jenner. Original code: https://github.com/cybershiptrooper/RL_Obfuscation

## Repository Structure

```
rl_obfuscation.py              # Core module — all functions extracted from notebook
part1_linear_probes.ipynb       # 4-section reproduction notebook
requirements.txt                # Pinned dependencies with version constraints
tests/                          # 124 tests (pytest)
  conftest.py                   # Shared fixtures (tiny_model, mock data)
  test_activations.py           # extract_activations, aggregate_activations, middle_layer_indices
  test_checkpoints.py           # save/load probes and eval results
  test_compat.py                # Version checks, cross-package constraints, trl API
  test_config.py                # Config dataclass validation
  test_integration.py           # End-to-end pipeline with tiny model
  test_metrics.py               # TPR@FPR evaluation
  test_probes.py                # LinearProbe training and prediction
  test_rewards.py               # Reward functions
checkpoints/                    # Runtime checkpoint dir (gitignored)
```

## Notebook Structure

`part1_linear_probes.ipynb` — 4-section reproduction notebook:
1. **Dataset Preparation**: BeaverTails safe/harmful pairs + RL prompt set
2. **Initial Probe Training**: Extract middle-50% layer activations → train per-layer linear probes → evaluate TPR@1%FPR
3. **RL-Obfuscation**: GRPO via TRL with LoRA, combined reward (probe evasion + quality + length)
4. **Post-Obfuscation Evaluation**: Original probes on obfuscated model + retrained probes

Each training section has checkpoint save/load — if a checkpoint exists in `checkpoints/`, training is skipped. Delete the checkpoint file to force a retrain.

## Key Technical Details

- Model: `Qwen/Qwen2.5-1.5B-Instruct` (paper uses Llama 2 7B / Llama 3 8B / Qwen 14B)
- Dataset: `PKU-Alignment/BeaverTails` (paper uses Mechanistic-Anomaly-Detection datasets)
- Activations: PyTorch forward hooks on `model.model.layers`, response tokens only, **middle 50% layers only** (RAM optimization via `middle_layer_indices()`)
- Probes: `nn.Linear(d_model, 1)` with BCE, 40 epochs, batch 16 (matches paper)
- RL: GRPO with β=0.05, lr=1e-5, 8 generations/prompt, LoRA r=16
- Reward: λ_wb=0.8 (threshold), λ_bb=1.0 (keyword quality proxy), λ_length=2.0
- Eval metric: TPR at 1% FPR

## Module API (`rl_obfuscation.py`)

Key functions grouped by section:

- **Environment**: `check_environment()` — validates packages, detects cross-package incompatibilities (transformers/torch, bitsandbytes/peft, deepspeed/torch)
- **Data**: `prepare_dataset()`, `prepare_rl_prompts()`
- **Activations**: `middle_layer_indices()`, `extract_activations(layer_indices=...)`, `aggregate_activations(free_on_aggregate=...)`
- **Probes**: `LinearProbe`, `train_probe()`, `compute_tpr_at_fpr()`
- **Rewards**: `ProbeRewardModel`, `keyword_quality_reward()`, `length_reward()`, `make_combined_reward_fn()`
- **Checkpoints**: `save_probes()`, `load_probes()`, `save_eval_results()`, `load_eval_results()`

## RAM Optimizations

- `middle_layer_indices(n_layers)` — extracts only middle 50% of layers (~50% RAM reduction)
- `extract_activations(layer_indices=...)` — hooks only selected layers
- `aggregate_activations(free_on_aggregate=True)` — deletes raw tensors per-layer during aggregation
- `ProbeRewardModel.score_completions()` — moves hook activations to CPU immediately

## Simplifications vs Paper

- Smaller model (1.5B vs 7-14B)
- Keyword-based blackbox reward instead of StrongReject Gemma-2B classifier
- 200 prompts instead of 1000
- No curriculum training (progressive threshold tightening)

## Dependencies

```
pip install -r requirements.txt
```

Key version constraints: `torch>=2.4.0`, `trl>=0.15.0` (max_prompt_length removed), `peft>=0.14.0` (bitsandbytes sync_gpu fix), `numpy<2.0.0`.

Optional packages (install only if needed): `bitsandbytes>=0.44.0`, `deepspeed>=0.15.0`.

## Running Tests

```bash
pytest tests/ -v
```

Integration tests use `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` (auto-downloaded).

## Paper Reference Values

- Probe layers: L10 (Llama2), L12 (Llama3), L20 (Qwen14B)
- Baseline TPR@1%FPR: ~96% (linear, mean aggregation)
- Post-obfuscation evasion: ~60% (mean), ~0% (max-pool)
- GRPO config: temperature=1.0, KL β=0.05, lr=1e-5, 8 completions/prompt
