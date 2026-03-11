"""Tests for extract_activations, aggregate_activations, and middle_layer_indices."""

import pytest
import torch
import numpy as np
from rl_obfuscation import aggregate_activations, extract_activations, middle_layer_indices


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
