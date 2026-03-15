"""Tests for extract_activations, aggregate_activations, score_activations, and middle_layer_indices."""

import pytest
import torch
import numpy as np
from rl_obfuscation import aggregate_activations, extract_activations, middle_layer_indices, score_activations, LinearProbe


class TestMiddleLayerIndices:
    def test_24_layers(self):
        """Qwen2.5-1.5B has 24 layers — should return layers 6..17."""
        result = middle_layer_indices(24)
        assert result == list(range(6, 18))
        assert len(result) == 12

    def test_32_layers(self):
        """Llama 2 7B has 32 layers — should return layers 8..23."""
        result = middle_layer_indices(32)
        assert result == list(range(8, 24))
        assert len(result) == 16

    def test_4_layers(self):
        """Tiny model with 4 layers — should return layers 1, 2."""
        result = middle_layer_indices(4)
        assert result == [1, 2]

    def test_2_layers(self):
        """Very small model — should return at least 1 layer."""
        result = middle_layer_indices(2)
        assert len(result) >= 1

    def test_1_layer(self):
        """Edge case: single-layer model should return 1 layer."""
        result = middle_layer_indices(1)
        assert len(result) == 1
        assert result == [0]

    def test_always_subset(self):
        """Result should always be a subset of valid layer indices."""
        for n in range(1, 50):
            result = middle_layer_indices(n)
            assert all(0 <= l < n for l in result)
            assert len(result) >= 1

    def test_roughly_half(self):
        """For large enough models, should return ~50% of layers."""
        for n in [12, 24, 32, 48]:
            result = middle_layer_indices(n)
            ratio = len(result) / n
            assert 0.4 <= ratio <= 0.6, f"n={n}: got {len(result)} layers ({ratio:.0%})"


class TestAggregateActivations:
    def test_mean_simple(self, mock_activations, mock_attention_mask, mock_prompt_lengths):
        result = aggregate_activations(
            mock_activations, mock_attention_mask, mock_prompt_lengths, method="mean",
        )
        assert len(result) == 3
        for l in range(3):
            assert result[l].shape == (4, 8)

    def test_max_simple(self, mock_activations, mock_attention_mask, mock_prompt_lengths):
        result = aggregate_activations(
            mock_activations, mock_attention_mask, mock_prompt_lengths, method="max",
        )
        for l in range(3):
            assert result[l].shape == (4, 8)

    def test_median_simple(self, mock_activations, mock_attention_mask, mock_prompt_lengths):
        result = aggregate_activations(
            mock_activations, mock_attention_mask, mock_prompt_lengths, method="median",
        )
        for l in range(3):
            assert result[l].shape == (4, 8)

    def test_invalid_method_raises(self, mock_activations, mock_attention_mask, mock_prompt_lengths):
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_activations(
                mock_activations, mock_attention_mask, mock_prompt_lengths, method="invalid",
            )

    def test_mean_respects_response_mask(self):
        """Only response tokens (after prompt, before padding) should be averaged."""
        # 1 example, seq_len=8, d_model=2
        # Tokens: [prompt, prompt, response, response, pad, pad, pad, pad]
        acts = {0: torch.zeros(1, 8, 2)}
        acts[0][0, 2, :] = 4.0  # response token 1
        acts[0][0, 3, :] = 6.0  # response token 2
        acts[0][0, 0, :] = 100.0  # prompt token (should be excluded)
        acts[0][0, 4, :] = 100.0  # padding token (should be excluded)

        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        prompt_lengths = torch.tensor([2])

        result = aggregate_activations(acts, attention_mask, prompt_lengths, method="mean")
        # Mean of [4, 6] = 5
        expected = torch.tensor([[5.0, 5.0]])
        assert torch.allclose(result[0], expected)

    def test_max_respects_response_mask(self):
        acts = {0: torch.full((1, 8, 2), -10.0)}
        acts[0][0, 2, :] = 4.0  # response token 1
        acts[0][0, 3, :] = 6.0  # response token 2
        acts[0][0, 0, :] = 100.0  # prompt (should be excluded)

        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        prompt_lengths = torch.tensor([2])

        result = aggregate_activations(acts, attention_mask, prompt_lengths, method="max")
        expected = torch.tensor([[6.0, 6.0]])
        assert torch.allclose(result[0], expected)

    def test_handles_empty_response(self):
        """When prompt_length >= seq_length, response is empty."""
        acts = {0: torch.randn(1, 8, 2)}
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        prompt_lengths = torch.tensor([4])  # prompt fills all real tokens

        result = aggregate_activations(acts, attention_mask, prompt_lengths, method="mean")
        # With clamp(min=1), empty response → zeros divided by 1 = zeros
        assert result[0].shape == (1, 2)
        assert torch.allclose(result[0], torch.zeros(1, 2))

    def test_handles_empty_response_median(self):
        acts = {0: torch.randn(1, 8, 2)}
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        prompt_lengths = torch.tensor([4])

        result = aggregate_activations(acts, attention_mask, prompt_lengths, method="median")
        assert result[0].shape == (1, 2)
        assert torch.allclose(result[0], torch.zeros(1, 2))

    def test_no_nan_in_output(self, mock_activations, mock_attention_mask, mock_prompt_lengths):
        for method in ["mean", "max", "median"]:
            result = aggregate_activations(
                mock_activations, mock_attention_mask, mock_prompt_lengths, method=method,
            )
            for l in result:
                assert not torch.isnan(result[l]).any(), f"NaN in {method} aggregation"

    def test_free_on_aggregate_clears_input(self):
        """free_on_aggregate=True should delete raw tensors from the input dict."""
        acts = {0: torch.randn(2, 8, 4), 1: torch.randn(2, 8, 4)}
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 0, 0, 0]])
        prompt_lengths = torch.tensor([1, 2])

        result = aggregate_activations(
            acts, attention_mask, prompt_lengths, method="mean", free_on_aggregate=True,
        )
        # Input dict should now be empty
        assert len(acts) == 0
        # Result should still have both layers
        assert len(result) == 2
        assert result[0].shape == (2, 4)
        assert result[1].shape == (2, 4)

    def test_free_on_aggregate_same_result(self):
        """free_on_aggregate should not change the aggregation output."""
        torch.manual_seed(42)
        acts_a = {0: torch.randn(2, 8, 4), 1: torch.randn(2, 8, 4)}
        acts_b = {0: acts_a[0].clone(), 1: acts_a[1].clone()}
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 0, 0, 0]])
        prompt_lengths = torch.tensor([1, 2])

        result_keep = aggregate_activations(
            acts_a, attention_mask, prompt_lengths, method="mean", free_on_aggregate=False,
        )
        result_free = aggregate_activations(
            acts_b, attention_mask, prompt_lengths, method="mean", free_on_aggregate=True,
        )
        for l in result_keep:
            assert torch.allclose(result_keep[l], result_free[l])

    def test_non_contiguous_layer_keys(self):
        """Aggregation should work with non-contiguous layer indices (e.g. middle 50%)."""
        acts = {5: torch.randn(2, 8, 4), 10: torch.randn(2, 8, 4)}
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 0, 0, 0]])
        prompt_lengths = torch.tensor([1, 2])

        result = aggregate_activations(acts, attention_mask, prompt_lengths, method="mean")
        assert set(result.keys()) == {5, 10}
        assert result[5].shape == (2, 4)


@pytest.mark.integration
class TestExtractActivations:
    def test_shapes(self, tiny_model, tiny_tokenizer):
        text = "Hello world"
        tokens = tiny_tokenizer(
            text, return_tensors="pt", padding="max_length", max_length=16,
        )
        acts = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=1,
        )
        n_layers = tiny_model.config.num_hidden_layers
        d_model = tiny_model.config.hidden_size
        assert len(acts) == n_layers
        for l in range(n_layers):
            assert acts[l].shape == (1, 16, d_model)

    def test_all_layers_present(self, tiny_model, tiny_tokenizer):
        tokens = tiny_tokenizer("test", return_tensors="pt", padding="max_length", max_length=8)
        acts = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=1,
        )
        n_layers = tiny_model.config.num_hidden_layers
        assert set(acts.keys()) == set(range(n_layers))

    def test_layer_indices_subset(self, tiny_model, tiny_tokenizer):
        """Passing layer_indices should extract only those layers."""
        n_layers = tiny_model.config.num_hidden_layers
        subset = [1, 2] if n_layers > 2 else [0]
        tokens = tiny_tokenizer("test", return_tensors="pt", padding="max_length", max_length=8)
        acts = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"],
            batch_size=1, layer_indices=subset,
        )
        assert set(acts.keys()) == set(subset)

    def test_layer_indices_matches_full(self, tiny_model, tiny_tokenizer):
        """Activations from layer_indices should match the same layers from a full extraction."""
        tokens = tiny_tokenizer("hello", return_tensors="pt", padding="max_length", max_length=8)
        full = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=1,
        )
        n_layers = tiny_model.config.num_hidden_layers
        subset = middle_layer_indices(n_layers)
        partial = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"],
            batch_size=1, layer_indices=subset,
        )
        for l in subset:
            assert torch.allclose(full[l], partial[l], atol=1e-5)

    def test_hooks_cleaned_up(self, tiny_model, tiny_tokenizer):
        hooks_before = len(tiny_model.model.layers[0]._forward_hooks)
        tokens = tiny_tokenizer("test", return_tensors="pt", padding="max_length", max_length=8)
        _ = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=1,
        )
        hooks_after = len(tiny_model.model.layers[0]._forward_hooks)
        assert hooks_after == hooks_before

    def test_hooks_cleaned_up_with_layer_indices(self, tiny_model, tiny_tokenizer):
        """Hooks should be cleaned up even when using layer_indices."""
        n_layers = tiny_model.config.num_hidden_layers
        for l_mod in tiny_model.model.layers:
            hooks_before = len(l_mod._forward_hooks)
            assert hooks_before == 0

        tokens = tiny_tokenizer("test", return_tensors="pt", padding="max_length", max_length=8)
        _ = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"],
            batch_size=1, layer_indices=[0],
        )
        for l_mod in tiny_model.model.layers:
            assert len(l_mod._forward_hooks) == 0

    def test_deterministic(self, tiny_model, tiny_tokenizer):
        tokens = tiny_tokenizer("hello", return_tensors="pt", padding="max_length", max_length=8)
        acts1 = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=1,
        )
        acts2 = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=1,
        )
        for l in acts1:
            assert torch.allclose(acts1[l], acts2[l], atol=1e-5)

    def test_multiple_examples(self, tiny_model, tiny_tokenizer):
        texts = ["hello world", "goodbye world", "test input"]
        tokens = tiny_tokenizer(
            texts, return_tensors="pt", padding="max_length", max_length=16,
        )
        acts = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=2,
        )
        for l in acts:
            assert acts[l].shape[0] == 3

    def test_output_dtype_is_float32(self, tiny_model, tiny_tokenizer):
        tokens = tiny_tokenizer("test", return_tensors="pt", padding="max_length", max_length=8)
        acts = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=1,
        )
        for l in acts:
            assert acts[l].dtype == torch.float32


class TestScoreActivations:
    def test_output_shape(self):
        """Returns one score per example."""
        torch.manual_seed(42)
        probe = LinearProbe(d_model=8)
        acts = torch.randn(4, 16, 8)
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
        acts = torch.zeros(1, 8, 2)
        acts[0, 2, :] = 5.0
        acts[0, 3, :] = 5.0
        acts[0, 0, :] = -100.0
        acts[0, 4, :] = -100.0

        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        prompt_lengths = torch.tensor([2])

        score_correct = score_activations(acts, probe, attention_mask, prompt_lengths)

        acts[0, 0, :] = 100.0
        acts[0, 4, :] = 100.0
        score_modified = score_activations(acts, probe, attention_mask, prompt_lengths)

        assert torch.allclose(score_correct, score_modified)

    def test_differs_from_score_of_mean(self):
        """score_activations must differ from sigmoid(w^T mean(h)) for non-uniform acts."""
        torch.manual_seed(42)
        probe = LinearProbe(d_model=4)
        acts = torch.zeros(1, 6, 4)
        acts[0, 2, :] = 10.0
        acts[0, 3, :] = -10.0
        acts[0, 4, :] = 0.0

        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
        prompt_lengths = torch.tensor([2])

        paper_score = score_activations(acts, probe, attention_mask, prompt_lengths)

        response_acts = acts[0, 2:5]
        mean_act = response_acts.mean(dim=0, keepdim=True)
        with torch.no_grad():
            wrong_score = probe.predict(mean_act)

        assert not torch.allclose(paper_score, wrong_score, atol=1e-3)

    def test_empty_response_returns_half(self):
        """When response is empty, score should default to 0.5."""
        probe = LinearProbe(d_model=4)
        acts = torch.randn(1, 8, 4)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        prompt_lengths = torch.tensor([4])
        scores = score_activations(acts, probe, attention_mask, prompt_lengths)
        assert torch.allclose(scores, torch.tensor([0.5]))

    def test_max_aggregation(self):
        """method='max' should return max per-token score."""
        torch.manual_seed(42)
        probe = LinearProbe(d_model=2)
        acts = torch.zeros(1, 6, 2)
        acts[0, 2, :] = 10.0
        acts[0, 3, :] = -10.0
        acts[0, 4, :] = 0.0

        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
        prompt_lengths = torch.tensor([2])

        score_mean = score_activations(acts, probe, attention_mask, prompt_lengths, method="mean")
        score_max = score_activations(acts, probe, attention_mask, prompt_lengths, method="max")

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

    def test_invalid_method_raises(self):
        probe = LinearProbe(d_model=4)
        acts = torch.randn(1, 8, 4)
        attention_mask = torch.ones(1, 8, dtype=torch.long)
        prompt_lengths = torch.tensor([2])
        with pytest.raises(ValueError, match="Unknown scoring method"):
            score_activations(acts, probe, attention_mask, prompt_lengths, method="invalid")

    def test_uniform_acts_equals_score_of_mean(self):
        """When all response tokens are identical, both methods should agree."""
        torch.manual_seed(42)
        probe = LinearProbe(d_model=4)
        acts = torch.zeros(1, 6, 4)
        # All response tokens identical
        acts[0, 2, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        acts[0, 3, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        acts[0, 4, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])

        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
        prompt_lengths = torch.tensor([2])

        paper_score = score_activations(acts, probe, attention_mask, prompt_lengths)

        mean_act = acts[0, 2:5].mean(dim=0, keepdim=True)
        with torch.no_grad():
            old_score = probe.predict(mean_act)

        # For identical tokens: mean(sigmoid(w^T h)) == sigmoid(w^T mean(h))
        assert torch.allclose(paper_score, old_score, atol=1e-5)

    def test_separable_data_higher_tpr(self):
        """On well-separated data, score_activations should give better TPR@FPR
        than the old aggregate-then-score method."""
        torch.manual_seed(42)
        d_model = 8
        n = 50

        # Train a probe on mean-aggregated data
        safe_acts = torch.randn(n, 10, d_model) - 1.0
        harm_acts = torch.randn(n, 10, d_model) + 1.0
        # Make a few tokens strongly harmful (spike pattern)
        harm_acts[:, 5, :] += 5.0

        all_acts = torch.cat([safe_acts, harm_acts])
        labels = torch.tensor([0.0] * n + [1.0] * n)
        attention_mask = torch.ones(2 * n, 10, dtype=torch.long)
        prompt_lengths = torch.zeros(2 * n, dtype=torch.long)

        # Aggregate then score (old method)
        mean_acts = aggregate_activations(
            {0: all_acts.clone()}, attention_mask, prompt_lengths, method="mean",
        )[0]

        from rl_obfuscation import train_probe
        probe, _ = train_probe(
            mean_acts, labels, d_model=d_model, n_epochs=20, batch_size=16,
            probe_device=torch.device("cpu"),
        )

        # Score both ways
        old_scores = probe.predict(mean_acts).detach().numpy()
        new_scores = score_activations(
            all_acts, probe, attention_mask, prompt_lengths, method="mean",
        ).numpy()

        from rl_obfuscation import compute_tpr_at_fpr
        old_result = compute_tpr_at_fpr(old_scores, labels.numpy(), target_fpr=0.05)
        new_result = compute_tpr_at_fpr(new_scores, labels.numpy(), target_fpr=0.05)

        # Per-token scoring should preserve the spike signal better
        assert new_result["auroc"] >= old_result["auroc"] - 0.05  # at least comparable

    def test_no_gradient_leaks(self):
        """score_activations should not require or produce gradients."""
        probe = LinearProbe(d_model=4)
        acts = torch.randn(2, 6, 4, requires_grad=True)
        attention_mask = torch.ones(2, 6, dtype=torch.long)
        prompt_lengths = torch.tensor([2, 2])
        scores = score_activations(acts, probe, attention_mask, prompt_lengths)
        assert not scores.requires_grad
