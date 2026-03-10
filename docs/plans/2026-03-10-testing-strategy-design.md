# Testing Strategy for RL-Obfuscation Notebook

## Goal
Refactor notebook code into a testable module, add comprehensive unit and integration tests to catch bugs (numpy compat, transformers API, edge cases).

## Module: `rl_obfuscation.py`
All functions/classes extracted from notebook:
- `Config` dataclass
- `prepare_dataset()`, `prepare_rl_prompts()`
- `extract_activations()`, `aggregate_activations()`
- `LinearProbe`, `train_probe()`
- `compute_tpr_at_fpr()`
- `ProbeRewardModel`, `length_reward()`, `keyword_quality_reward()`

## Test Files

| File | Scope | Tests |
|------|-------|-------|
| `conftest.py` | Fixtures | Tiny Qwen2 model, tokenizer, mock tensors |
| `test_config.py` | Unit | Config defaults, types, seed reproducibility |
| `test_probes.py` | Unit | LinearProbe shapes/gradients, train_probe on separable data |
| `test_activations.py` | Unit+Integration | aggregate_activations (mock), extract_activations (tiny model) |
| `test_metrics.py` | Unit | compute_tpr_at_fpr known scenarios |
| `test_rewards.py` | Unit | keyword/length reward correctness |
| `test_integration.py` | Integration | End-to-end pipeline with tiny model |
| `test_compat.py` | Compat | numpy/torch/sklearn/transformers API checks |

## Model Strategy
- Unit tests: mock tensors only, no downloads
- Integration tests: `hf-internal-testing/tiny-random-Qwen2ForCausalLM`, marked `@pytest.mark.integration`

## Notebook Changes
Replace inline definitions with `from rl_obfuscation import ...`, keep markdown/visualization/orchestration.
