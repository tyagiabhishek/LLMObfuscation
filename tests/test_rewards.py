"""Tests for reward functions."""

import pytest
import numpy as np
from rl_obfuscation import keyword_quality_reward, length_reward


class TestKeywordQualityReward:
    def test_detects_refusal(self):
        texts = [
            "I cannot help you with that request.",
            "I'm sorry, but I can't assist with that.",
            "I apologize, but that's not appropriate.",
        ]
        rewards = keyword_quality_reward(texts)
        for r in rewards:
            assert r == -0.5

    def test_rewards_substantive_text(self):
        text = " ".join(["word"] * 100)  # 100 words, no refusal keywords
        rewards = keyword_quality_reward([text])
        assert rewards[0] > 0

    def test_short_text_penalized(self):
        text = "Very short response."  # < 20 words
        rewards = keyword_quality_reward([text])
        assert rewards[0] == -0.5

    def test_reward_range(self):
        texts = [
            "I cannot do that.",
            "short",
            " ".join(["word"] * 50),
            " ".join(["word"] * 300),
        ]
        rewards = keyword_quality_reward(texts)
        for r in rewards:
            assert -0.5 <= r <= 0.5

    def test_exactly_20_words_not_penalized(self):
        text = " ".join(["word"] * 20)
        rewards = keyword_quality_reward([text])
        assert rewards[0] >= 0  # Should not be penalized

    def test_refusal_keyword_case_insensitive(self):
        texts = [
            "i cannot do that",
            "I CANNOT do that",
            "i Cannot do that",
        ]
        rewards = keyword_quality_reward(texts)
        for r in rewards:
            assert r == -0.5

    def test_empty_completion(self):
        rewards = keyword_quality_reward([""])
        assert rewards[0] == -0.5  # empty = too short

    def test_max_reward_capped(self):
        text = " ".join(["word"] * 1000)
        rewards = keyword_quality_reward([text])
        assert rewards[0] == 0.5  # capped

    def test_batch_processing(self):
        texts = ["I cannot help", " ".join(["word"] * 50)]
        rewards = keyword_quality_reward(texts)
        assert len(rewards) == 2
        assert rewards[0] == -0.5
        assert rewards[1] > 0


class TestLengthReward:
    def test_at_target_length(self, tiny_tokenizer):
        """Token count exactly at target should give reward close to 1.0."""
        # Create text whose token count is close to target
        target = 50
        text = "word " * target
        rewards = length_reward([text], tiny_tokenizer, target=target, scale=0.1)
        # It won't be exactly 1.0 since tokenization isn't exactly 1:1,
        # but should be high
        assert rewards[0] > 0.5

    def test_far_from_target_low_reward(self, tiny_tokenizer):
        target = 200
        text = "hi"  # Very few tokens
        rewards = length_reward([text], tiny_tokenizer, target=target, scale=0.1)
        assert rewards[0] < 0.01

    def test_reward_range(self, tiny_tokenizer):
        texts = ["hi", "word " * 50, "word " * 200, "word " * 500]
        rewards = length_reward(texts, tiny_tokenizer, target=200, scale=0.1)
        for r in rewards:
            assert 0.0 <= r <= 1.0

    def test_symmetric_around_target(self, tiny_tokenizer):
        """Reward should be similar for equidistant points above/below target."""
        target = 100
        # Create texts of roughly target-30 and target+30 tokens
        short_text = "word " * 70
        long_text = "word " * 130
        rewards = length_reward(
            [short_text, long_text], tiny_tokenizer, target=target, scale=0.1,
        )
        # Won't be exactly equal due to tokenization, but should be comparable
        assert abs(rewards[0] - rewards[1]) < 0.3

    def test_batch_processing(self, tiny_tokenizer):
        texts = ["hi", "hello world", "word " * 100]
        rewards = length_reward(texts, tiny_tokenizer, target=100)
        assert len(rewards) == 3

    def test_returns_python_floats(self, tiny_tokenizer):
        rewards = length_reward(["hello"], tiny_tokenizer, target=10)
        assert isinstance(rewards[0], float)

    @pytest.fixture
    def tiny_tokenizer(self):
        """Local tokenizer fixture for non-integration tests."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
