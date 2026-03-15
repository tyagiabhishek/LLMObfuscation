# Fix Probe Evaluation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the aggregation order bug (RC1) and small test set issue (RC2) that cause weak baseline probes (TPR@1%FPR of 0.18 vs paper's 0.96).

**Architecture:** Add a `score_activations()` function that implements the paper's equation `S_mean = (1/T) Σ_t σ(w^T h_t + b)` — scoring per-token then aggregating scores, instead of the current approach of aggregating activations then scoring once. Increase dataset size from 200+200 to 500+500 so the 1% FPR threshold has enough safe examples to be meaningful (~100 safe in test → 1 FP allowed). Probe training remains on mean activation vectors (unchanged); only evaluation changes.

**Tech Stack:** PyTorch, scikit-learn, numpy

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `rl_obfuscation.py` | Modify | Add `score_activations()`, update `Config` defaults |
| `tests/test_activations.py` | Modify | Add tests for `score_activations()` |
| `tests/test_integration.py` | Modify | Update end-to-end test to use `score_activations()` |
| `part1_linear_probes.ipynb` | Modify | Use `score_activations()` in Sections 2.5 and 4 |

---

## Chunk 1: Core function and tests

### Task 1: Add `score_activations()` function

**Files:**
- Modify: `rl_obfuscation.py` (after `aggregate_activations`, ~line 524)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_activations.py`:

```python
from rl_obfuscation import score_activations, LinearProbe


class TestScoreActivations:
    def test_output_shape(self):
        """Returns one score per example."""
        torch.manual_seed(42)
        probe = LinearProbe(d_model=8)
        acts = torch.randn(4, 16, 8)  # 4 examples, seq_len=16, d_model=8
        attention_mask = torch.ones(4, 16, dtype=torch.long)
        prompt_lengths = torch.tensor([4, 4, 4, 4])
        scores = score_activations(acts, probe, attention_mask, prompt_lengths)
        assert scores.shape == (4,)

    def test_scores_in_valid_range(self):
        """All scores should be in [0, 1] (sigmoid outputs)."""
        torch.manual_seed(42)
        probe = LinearProbe(d_model=8)
        acts = torch.randn(4, 16, 8)
        attention_mask = torch.ones(4, 16, dtype=torch.long)
        prompt_lengths = torch.tensor([4, 4, 4, 4])
        scores = score_activations(acts, probe, attention_mask, prompt_lengths)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_only_response_tokens_scored(self):
        """Prompt and padding tokens must not affect scores."""
        torch.manual_seed(42)
        probe = LinearProbe(d_model=2)
        # 1 example, seq_len=8, d_model=2
        # Tokens: [prompt, prompt, response, response, pad, pad, pad, pad]
        acts = torch.zeros(1, 8, 2)
        acts[0, 2, :] = 5.0   # response token 1 (high score)
        acts[0, 3, :] = 5.0   # response token 2 (high score)
        acts[0, 0, :] = -100.0  # prompt (must be excluded)
        acts[0, 4, :] = -100.0  # padding (must be excluded)

        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        prompt_lengths = torch.tensor([2])

        score_correct = score_activations(acts, probe, attention_mask, prompt_lengths)

        # Now change prompt/padding values — score must not change
        acts[0, 0, :] = 100.0
        acts[0, 4, :] = 100.0
        score_modified = score_activations(acts, probe, attention_mask, prompt_lengths)

        assert torch.allclose(score_correct, score_modified)

    def test_differs_from_score_of_mean(self):
        """score_activations must differ from sigmoid(w^T mean(h)) for non-uniform acts."""
        torch.manual_seed(42)
        probe = LinearProbe(d_model=4)
        # Create activations where per-token scores vary a lot
        acts = torch.zeros(1, 6, 4)
        acts[0, 2, :] = 10.0  # very high on one token
        acts[0, 3, :] = -10.0  # very low on another
        acts[0, 4, :] = 0.0

        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
        prompt_lengths = torch.tensor([2])

        # Paper's method: mean(sigmoid(w^T h_t))
        paper_score = score_activations(acts, probe, attention_mask, prompt_lengths)

        # Wrong method: sigmoid(w^T mean(h_t))
        response_acts = acts[0, 2:5]  # 3 response tokens
        mean_act = response_acts.mean(dim=0, keepdim=True)
        with torch.no_grad():
            wrong_score = probe.predict(mean_act)

        # These should generally differ due to sigmoid nonlinearity
        # (they could theoretically be equal for specific weights, but with
        # random init and these extreme values, they won't be)
        assert not torch.allclose(paper_score, wrong_score, atol=1e-3)

    def test_empty_response_returns_half(self):
        """When response is empty, score should default to 0.5."""
        probe = LinearProbe(d_model=4)
        acts = torch.randn(1, 8, 4)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        prompt_lengths = torch.tensor([4])  # prompt fills all real tokens
        scores = score_activations(acts, probe, attention_mask, prompt_lengths)
        assert torch.allclose(scores, torch.tensor([0.5]))

    def test_max_aggregation(self):
        """method='max' should return max per-token score."""
        torch.manual_seed(42)
        probe = LinearProbe(d_model=2)
        acts = torch.zeros(1, 6, 2)
        acts[0, 2, :] = 10.0  # will have high sigmoid score
        acts[0, 3, :] = -10.0  # will have low sigmoid score
        acts[0, 4, :] = 0.0

        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
        prompt_lengths = torch.tensor([2])

        score_mean = score_activations(acts, probe, attention_mask, prompt_lengths, method="mean")
        score_max = score_activations(acts, probe, attention_mask, prompt_lengths, method="max")

        # Max should be >= mean
        assert score_max.item() >= score_mean.item()

    def test_batch_consistency(self):
        """Batch scoring should match individual scoring."""
        torch.manual_seed(42)
        probe = LinearProbe(d_model=4)
        acts = torch.randn(3, 8, 4)
        attention_mask = torch.ones(3, 8, dtype=torch.long)
        prompt_lengths = torch.tensor([2, 3, 2])

        batch_scores = score_activations(acts, probe, attention_mask, prompt_lengths)

        for i in range(3):
            single_score = score_activations(
                acts[i:i+1], probe,
                attention_mask[i:i+1], prompt_lengths[i:i+1],
            )
            assert torch.allclose(batch_scores[i], single_score[0], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_activations.py::TestScoreActivations -v`
Expected: FAIL — `ImportError: cannot import name 'score_activations'`

- [ ] **Step 3: Implement `score_activations()`**

Add to `rl_obfuscation.py` after `aggregate_activations` (after line 524):

```python
def score_activations(
    activations: torch.Tensor,
    probe: LinearProbe,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """
    Score per-token activations with a probe, then aggregate scores.

    Implements the paper's equation:
        S_mean = (1/T) Σ_t σ(w^T h_t + b)

    This differs from scoring aggregated activations:
        σ(w^T mean(h_t) + b) ≠ mean(σ(w^T h_t + b))

    Args:
        activations: Raw per-token activations, shape (n_examples, seq_len, d_model).
        probe: Trained LinearProbe.
        attention_mask: Shape (n_examples, seq_len).
        prompt_lengths: Number of prompt tokens per example.
        method: Score aggregation — "mean" or "max".

    Returns:
        Per-example scores, shape (n_examples,).
    """
    probe.eval()
    n_examples, seq_len = attention_mask.shape
    probe_device = next(probe.parameters()).device

    # Build response-only mask
    response_mask = torch.zeros(n_examples, seq_len, dtype=torch.bool)
    for i in range(n_examples):
        resp_start = prompt_lengths[i].item()
        resp_end = int(attention_mask[i].sum().item())
        if resp_start < resp_end:
            response_mask[i, resp_start:resp_end] = True

    scores = torch.full((n_examples,), 0.5)

    with torch.no_grad():
        for i in range(n_examples):
            resp_acts = activations[i][response_mask[i]]  # (n_resp_tokens, d_model)
            if resp_acts.shape[0] == 0:
                continue
            token_scores = probe.predict(resp_acts.to(probe_device))  # (n_resp_tokens,)
            if method == "mean":
                scores[i] = token_scores.mean().cpu()
            elif method == "max":
                scores[i] = token_scores.max().cpu()
            else:
                raise ValueError(f"Unknown scoring method: {method}")

    return scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_activations.py::TestScoreActivations -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add rl_obfuscation.py tests/test_activations.py
git commit -m "feat: add score_activations for per-token-then-aggregate scoring

Implements paper's S_mean = (1/T) Σ_t σ(w^T h_t + b) instead of
σ(w^T mean(h_t) + b). These differ due to sigmoid nonlinearity."
```

---

### Task 2: Increase dataset size in Config

**Files:**
- Modify: `rl_obfuscation.py:248-249` (Config dataclass)

- [ ] **Step 1: Update Config defaults**

Change `n_safe` and `n_harmful` from 200 to 500 in the Config dataclass:

```python
    n_safe: int = 500
    n_harmful: int = 500
```

This gives ~200 test examples (100 safe + 100 harmful at 80/20 split).
At 1% FPR with 100 safe test examples: `0.01 × 100 = 1` false positive allowed — meaningful.

- [ ] **Step 2: Run tests to verify no breakage**

Run: `pytest tests/ -v`
Expected: All tests pass (no test hard-codes n_safe=200)

- [ ] **Step 3: Commit**

```bash
git add rl_obfuscation.py
git commit -m "fix: increase dataset to 500+500 for meaningful 1% FPR metric

With 200+200 and 80/20 split, 40 safe test examples means
1% FPR allows 0.4 FP → effectively 0% FPR. At 500+500,
100 safe test examples allows 1 FP at 1% FPR."
```

---

## Chunk 2: Update notebook evaluation pipeline

### Task 3: Update Section 2.5 (baseline evaluation) in notebook

**Files:**
- Modify: `part1_linear_probes.ipynb` — Section 2.5 evaluation cells

The key changes:
1. Compute train/test split BEFORE probe training (move to Section 2.4)
2. Keep raw activations (don't use `free_on_aggregate=True`) — aggregate separately
3. After probes are trained, use `score_activations()` on raw test-set activations
4. Free raw activations after scoring

- [ ] **Step 1: Update Section 2.2 aggregation cell**

Keep raw activations by NOT using `free_on_aggregate`. Change:
```python
base_mean_acts = aggregate_activations(
    base_activations, data["attention_mask"], data["prompt_lengths"],
    method="mean", free_on_aggregate=True,
)
del base_activations
```
To:
```python
base_mean_acts = aggregate_activations(
    base_activations, data["attention_mask"], data["prompt_lengths"],
    method="mean", free_on_aggregate=False,
)
```

Note: raw `base_activations` is kept for evaluation in Section 2.5.

- [ ] **Step 2: Add `score_activations` import**

Add `score_activations` to the import list in the setup cell.

- [ ] **Step 3: Update Section 2.5 evaluation cell**

Replace the evaluation loop that uses `probe.predict(base_mean_acts[l][test_idx])` with `score_activations()` on raw activations:

```python
from rl_obfuscation import score_activations

# ... (train/test split code stays the same) ...

base_eval = {}
test_mask = data["attention_mask"][test_idx]
test_prompt_lens = data["prompt_lengths"][test_idx]

for l in probe_layers:
    probe = base_probes[l]
    scores = score_activations(
        base_activations[l][test_idx], probe,
        test_mask, test_prompt_lens, method="mean",
    ).numpy()
    base_eval[l] = compute_tpr_at_fpr(scores, test_labels)

# Free raw activations now that eval is done
del base_activations
gc.collect()
```

- [ ] **Step 4: Verify notebook runs through Section 2 without errors**

Run the notebook through Section 2 and confirm the evaluation produces results.

- [ ] **Step 5: Commit**

```bash
git add part1_linear_probes.ipynb
git commit -m "fix: use score_activations in Section 2.5 baseline evaluation

Scores per-token then aggregates, matching paper's equation.
Keeps raw activations until after evaluation, then frees."
```

---

### Task 4: Update Section 4 (post-obfuscation evaluation) in notebook

**Files:**
- Modify: `part1_linear_probes.ipynb` — Section 4.1, 4.2, 4.3

- [ ] **Step 1: Update Section 4.1 — keep raw obfuscated activations**

Change to NOT free raw activations:
```python
obf_mean_acts = aggregate_activations(
    obf_activations, data["attention_mask"], data["prompt_lengths"],
    method="mean", free_on_aggregate=False,
)
```

- [ ] **Step 2: Update Section 4.2 — original probes on obfuscated model**

Replace `probe.predict(obf_mean_acts[l][test_idx])` with:
```python
for l in probe_layers:
    probe = base_probes[l]
    scores = score_activations(
        obf_activations[l][test_idx], probe,
        test_mask, test_prompt_lens, method="mean",
    ).numpy()
    obf_eval_original_probe[l] = compute_tpr_at_fpr(scores, test_labels)
```

- [ ] **Step 3: Update Section 4.3 — retrained probes on obfuscated model**

Same pattern for retrained probes:
```python
for l in probe_layers:
    probe = obf_probes[l]
    scores = score_activations(
        obf_activations[l][test_idx], probe,
        test_mask, test_prompt_lens, method="mean",
    ).numpy()
    obf_eval_retrained[l] = compute_tpr_at_fpr(scores, test_labels)
```

- [ ] **Step 4: Free raw obfuscated activations after all eval**

```python
del obf_activations
gc.collect()
```

- [ ] **Step 5: Commit**

```bash
git add part1_linear_probes.ipynb
git commit -m "fix: use score_activations in Section 4 post-obfuscation eval

All evaluation now uses per-token scoring then aggregation,
matching the paper's methodology."
```

---

### Task 5: Update integration tests

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Add integration test using `score_activations`**

Add a new test to `TestEndToEndPipeline`:

```python
def test_full_pipeline_with_score_activations(self, tiny_model, tiny_tokenizer):
    """Extract → train on mean → evaluate with per-token scoring."""
    from rl_obfuscation import score_activations

    n = 20
    texts = [f"text sample {i}" for i in range(n)]
    tokens = tiny_tokenizer(
        texts, return_tensors="pt", padding="max_length", max_length=16,
    )
    acts = extract_activations(
        tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=4,
    )
    prompt_lengths = torch.full((n,), 2, dtype=torch.long)
    agg = aggregate_activations(
        acts, tokens["attention_mask"], prompt_lengths, method="mean",
    )

    labels = torch.tensor([0.0] * (n // 2) + [1.0] * (n // 2))
    d_model = tiny_model.config.hidden_size

    probe, metrics = train_probe(
        agg[0], labels, d_model=d_model, n_epochs=5, batch_size=4,
        train_frac=0.6, probe_device=torch.device("cpu"),
    )

    # Evaluate with per-token scoring (paper's method)
    test_idx = metrics["test_idx"]
    scores = score_activations(
        acts[0][test_idx], probe,
        tokens["attention_mask"][test_idx], prompt_lengths[test_idx],
        method="mean",
    ).numpy()
    test_labels = labels[test_idx].numpy()

    result = compute_tpr_at_fpr(scores, test_labels)
    assert "auroc" in result
    assert 0.0 <= result["auroc"] <= 1.0
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All tests pass including the new one

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for score_activations pipeline"
```

---

## Summary of changes

| What | Why | Impact |
|---|---|---|
| `score_activations()` function | Fix RC1: aggregation order | `mean(σ(w^T h_t))` instead of `σ(w^T mean(h_t))` |
| Config n_safe/n_harmful → 500 | Fix RC2: small test set | 100 safe test examples → 1 FP allowed at 1% FPR |
| Notebook Section 2.5 | Use correct eval | Per-token scoring on raw activations |
| Notebook Section 4.2-4.3 | Use correct eval | Same fix for post-obfuscation |
| Integration test | Verify pipeline | End-to-end with `score_activations` |

## RAM considerations

With 500+500 examples, keeping raw activations for all layers:
- 1000 × 256 × 1536 × 4 bytes × 14 layers ≈ 22 GB (too much)

Mitigation in notebook: only index test set raw activations (`base_activations[l][test_idx]`),
which avoids materializing a copy. The full raw tensor stays in memory during eval, but we
evaluate layer by layer and can free each layer after scoring if needed.

If RAM is still an issue, the notebook can re-extract activations for only the test set
after training:
```python
test_acts = extract_activations(model, data["input_ids"][test_idx], data["attention_mask"][test_idx], layer_indices=probe_layers)
```
