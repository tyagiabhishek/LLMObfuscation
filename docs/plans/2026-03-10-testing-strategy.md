# RL-Obfuscation Testing Strategy — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract notebook code into a testable module, add comprehensive unit and integration tests.

**Architecture:** Single `rl_obfuscation.py` module with all functions/classes. Tests in `tests/` using pytest. Unit tests use mock tensors. Integration tests use `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` (2.43M params). Notebook imports from the module.

**Tech Stack:** pytest, torch, numpy, sklearn, transformers (4.50.0.dev0), trl, peft

**Environment:** conda `fastai_env`. Missing: trl, peft, pytest.

---

### Task 1: Install missing dependencies

**Files:** None

**Step 1: Install packages**

Run:
```bash
conda run -n fastai_env pip install pytest trl peft
```

**Step 2: Verify**

Run:
```bash
conda run -n fastai_env python -c "import pytest, trl, peft; print('OK')"
```

---

### Task 2: Create `rl_obfuscation.py` module

**Files:**
- Create: `rl_obfuscation.py`

Extract all functions/classes from the notebook. Key changes during extraction:
- `device` becomes a module-level variable (same logic as notebook)
- `combined_reward_fn` becomes a factory function `make_combined_reward_fn(probe_reward, tokenizer, config)` that returns a closure — needed since the notebook version relies on globals
- All other functions keep their signatures

Functions/classes to extract:
- `Config` dataclass
- `prepare_dataset(tokenizer, n_safe, n_harmful, max_length, seed)`
- `prepare_rl_prompts(tokenizer, n_prompts, seed, skip_first)`
- `extract_activations(model, input_ids, attention_mask, batch_size)`
- `aggregate_activations(activations, attention_mask, prompt_lengths, method)`
- `LinearProbe(nn.Module)`
- `train_probe(acts, labels, d_model, n_epochs, batch_size, lr, train_frac, seed, device)`
- `compute_tpr_at_fpr(scores, labels, target_fpr)`
- `ProbeRewardModel`
- `length_reward(completions, tokenizer, target, scale)`
- `keyword_quality_reward(completions)`
- `make_combined_reward_fn(probe_reward, tokenizer, config)`

**Step 1: Write the module**

**Step 2: Verify it imports cleanly**

Run: `conda run -n fastai_env python -c "from rl_obfuscation import *; print('OK')"`

**Step 3: Commit**

---

### Task 3: Create `tests/conftest.py` with fixtures

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pyproject.toml` (pytest config + markers)

Fixtures:
- `@pytest.fixture(scope="session") tiny_model` — loads `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`
- `@pytest.fixture(scope="session") tiny_tokenizer` — tokenizer for same model
- `@pytest.fixture mock_activations` — dict of random tensors mimicking extract_activations output
- `@pytest.fixture mock_labels` — balanced 0/1 labels
- `@pytest.fixture mock_attention_mask` — realistic attention mask
- `@pytest.fixture mock_prompt_lengths` — realistic prompt lengths

Markers:
- `integration` — requires model download, slow

---

### Task 4: Create `tests/test_compat.py`

**Files:**
- Create: `tests/test_compat.py`

Tests:
- `test_numpy_version` — numpy >= 1.22 (default_rng support)
- `test_torch_version` — torch >= 2.0
- `test_sklearn_roc_curve_api` — roc_curve/roc_auc_score work with float32 arrays
- `test_numpy_shuffle_on_list` — default_rng.shuffle works on Python lists
- `test_transformers_auto_classes` — AutoModelForCausalLM, AutoTokenizer importable
- `test_torch_device_detection` — device detection logic works

---

### Task 5: Create `tests/test_config.py`

**Files:**
- Create: `tests/test_config.py`

Tests:
- `test_config_defaults` — all fields have expected types
- `test_config_override` — can override fields
- `test_config_seed_type` — seed is int

---

### Task 6: Create `tests/test_metrics.py`

**Files:**
- Create: `tests/test_metrics.py`

Tests:
- `test_perfect_separation` — all positive scores above all negative → TPR=1.0
- `test_random_scores` — random scores give TPR near 0 at 1% FPR
- `test_all_same_label` — handles degenerate case (single class)
- `test_threshold_returned` — threshold is a valid float
- `test_fpr_tpr_curves_returned` — curves are numpy arrays

---

### Task 7: Create `tests/test_probes.py`

**Files:**
- Create: `tests/test_probes.py`

Tests:
- `test_linear_probe_output_shape` — forward returns (batch,) shape
- `test_linear_probe_predict_range` — predict returns values in [0,1]
- `test_linear_probe_gradient_flows` — backward pass works
- `test_train_probe_returns_probe_and_metrics` — correct return types
- `test_train_probe_separable_data` — achieves >0.9 AUROC on linearly separable data
- `test_train_probe_metrics_keys` — metrics dict has expected keys

---

### Task 8: Create `tests/test_activations.py`

**Files:**
- Create: `tests/test_activations.py`

Tests (unit, mock tensors):
- `test_aggregate_mean_simple` — known input → known output
- `test_aggregate_max_simple` — known input → known output
- `test_aggregate_respects_response_mask` — prompt tokens excluded
- `test_aggregate_handles_padding` — padding tokens excluded
- `test_aggregate_empty_response` — response length 0 → doesn't crash

Tests (integration, tiny model):
- `test_extract_activations_shapes` — correct (n, seq_len, d_model) per layer
- `test_extract_activations_all_layers` — returns dict with all layer indices
- `test_extract_activations_hooks_cleaned` — no lingering hooks after extraction
- `test_extract_activations_deterministic` — same input → same output

---

### Task 9: Create `tests/test_rewards.py`

**Files:**
- Create: `tests/test_rewards.py`

Tests:
- `test_keyword_detects_refusal` — "I cannot help" → negative reward
- `test_keyword_rewards_substantive` — long non-refusal → positive reward
- `test_keyword_short_text_penalized` — <20 words → negative reward
- `test_keyword_reward_range` — all rewards in [-0.5, 0.5]
- `test_length_reward_at_target` — exact target length → reward ~1.0
- `test_length_reward_far_from_target` — very short/long → reward ~0.0
- `test_length_reward_symmetric` — equidistant above/below target → same reward
- `test_length_reward_range` — all rewards in [0, 1]

---

### Task 10: Create `tests/test_integration.py`

**Files:**
- Create: `tests/test_integration.py`

All tests marked `@pytest.mark.integration`.

Tests:
- `test_extract_aggregate_pipeline` — extract → aggregate → correct shapes
- `test_train_probe_on_real_activations` — train probe on tiny model activations, get valid metrics
- `test_probe_reward_model_scores` — ProbeRewardModel produces scores in [0,1]
- `test_end_to_end_pipeline` — extract → aggregate → train → evaluate TPR@FPR

---

### Task 11: Update notebook to import from module

**Files:**
- Modify: `part1_linear_probes.ipynb`

Replace function/class definitions with:
```python
from rl_obfuscation import (
    Config, prepare_dataset, prepare_rl_prompts,
    extract_activations, aggregate_activations,
    LinearProbe, train_probe, compute_tpr_at_fpr,
    ProbeRewardModel, length_reward, keyword_quality_reward,
    make_combined_reward_fn,
)
```

Keep all markdown, visualization, and orchestration code in the notebook.

---

### Task 12: Run all tests, fix bugs, commit

Run: `cd /Users/tyagi/Documents/MechInterp/LLMObfuscation && conda run -n fastai_env python -m pytest tests/ -v`

Fix any failures. Commit.
