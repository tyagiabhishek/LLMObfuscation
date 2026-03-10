"""Tests for extract_activations and aggregate_activations."""

import pytest
import torch
import numpy as np
from rl_obfuscation import aggregate_activations, extract_activations


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

    def test_hooks_cleaned_up(self, tiny_model, tiny_tokenizer):
        hooks_before = len(tiny_model.model.layers[0]._forward_hooks)
        tokens = tiny_tokenizer("test", return_tensors="pt", padding="max_length", max_length=8)
        _ = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=1,
        )
        hooks_after = len(tiny_model.model.layers[0]._forward_hooks)
        assert hooks_after == hooks_before

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
