"""Integration tests using the tiny Qwen2 model."""

import pytest
import torch
import numpy as np
from rl_obfuscation import (
    LinearProbe,
    ProbeRewardModel,
    aggregate_activations,
    compute_tpr_at_fpr,
    extract_activations,
    train_probe,
)

pytestmark = pytest.mark.integration


class TestExtractAggregatePipeline:
    def test_extract_then_aggregate(self, tiny_model, tiny_tokenizer):
        texts = ["Hello world", "How are you"]
        tokens = tiny_tokenizer(
            texts, return_tensors="pt", padding="max_length", max_length=16,
        )
        acts = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=2,
        )
        # Simulate prompt lengths (first 3 tokens are prompt)
        prompt_lengths = torch.tensor([3, 3])
        agg = aggregate_activations(
            acts, tokens["attention_mask"], prompt_lengths, method="mean",
        )
        d_model = tiny_model.config.hidden_size
        for l in agg:
            assert agg[l].shape == (2, d_model)
            assert not torch.isnan(agg[l]).any()


class TestTrainProbeOnRealActivations:
    def test_train_on_tiny_model_acts(self, tiny_model, tiny_tokenizer):
        texts = [f"safe text number {i}" for i in range(10)] + \
                [f"different text number {i}" for i in range(10)]
        tokens = tiny_tokenizer(
            texts, return_tensors="pt", padding="max_length", max_length=16,
        )
        acts = extract_activations(
            tiny_model, tokens["input_ids"], tokens["attention_mask"], batch_size=4,
        )
        prompt_lengths = torch.full((20,), 3, dtype=torch.long)
        agg = aggregate_activations(
            acts, tokens["attention_mask"], prompt_lengths, method="mean",
        )
        labels = torch.tensor([0.0] * 10 + [1.0] * 10)
        d_model = tiny_model.config.hidden_size
        layer_0_acts = agg[0]

        probe, metrics = train_probe(
            layer_0_acts, labels, d_model=d_model, n_epochs=5, batch_size=4,
            probe_device=torch.device("cpu"),
        )
        assert isinstance(probe, LinearProbe)
        assert "auroc" in metrics
        assert 0.0 <= metrics["auroc"] <= 1.0


class TestProbeRewardModel:
    def test_score_completions_returns_valid_scores(self, tiny_model, tiny_tokenizer):
        d_model = tiny_model.config.hidden_size
        probe = LinearProbe(d_model)
        probe.eval()

        reward_model = ProbeRewardModel(
            probe=probe,
            model=tiny_model,
            tokenizer=tiny_tokenizer,
            layer_idx=0,
            threshold=0.5,
            max_length=32,
        )
        prompts = ["Hello "]
        completions = ["world, how are you today"]
        scores = reward_model.score_completions(prompts, completions)
        assert scores.shape == (1,)
        assert 0.0 <= scores[0].item() <= 1.0

    def test_reward_returns_list(self, tiny_model, tiny_tokenizer):
        d_model = tiny_model.config.hidden_size
        probe = LinearProbe(d_model)
        probe.eval()

        reward_model = ProbeRewardModel(
            probe=probe,
            model=tiny_model,
            tokenizer=tiny_tokenizer,
            layer_idx=0,
            threshold=0.5,
            max_length=32,
        )
        prompts = ["Hello ", "Hi "]
        completions = ["world", "there"]
        rewards = reward_model.reward(prompts, completions)
        assert isinstance(rewards, list)
        assert len(rewards) == 2
        for r in rewards:
            assert r in [-1.0, 1.0]


class TestEndToEndPipeline:
    def test_full_pipeline(self, tiny_model, tiny_tokenizer):
        """Extract → aggregate → train → evaluate TPR@FPR."""
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

        # Evaluate on test set
        test_idx = metrics["test_idx"]
        with torch.no_grad():
            scores = probe.predict(agg[0][test_idx]).numpy()
        test_labels = labels[test_idx].numpy()

        result = compute_tpr_at_fpr(scores, test_labels)
        assert "auroc" in result
        assert "tpr_at_fpr" in result
        assert 0.0 <= result["auroc"] <= 1.0
