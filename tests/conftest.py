import pytest
import torch
import numpy as np

TINY_MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="session")
def tiny_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL_ID)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


@pytest.fixture
def mock_activations():
    """Dict of random tensors mimicking extract_activations output.
    3 layers, 4 examples, seq_len=16, d_model=8."""
    torch.manual_seed(42)
    n_examples, seq_len, d_model = 4, 16, 8
    return {l: torch.randn(n_examples, seq_len, d_model) for l in range(3)}


@pytest.fixture
def mock_labels():
    """Balanced 0/1 labels for 4 examples."""
    return torch.tensor([0.0, 0.0, 1.0, 1.0])


@pytest.fixture
def mock_attention_mask():
    """Attention mask for 4 examples, seq_len=16. Varying real lengths."""
    mask = torch.zeros(4, 16, dtype=torch.long)
    mask[0, :12] = 1  # 12 real tokens
    mask[1, :10] = 1  # 10 real tokens
    mask[2, :14] = 1  # 14 real tokens
    mask[3, :8] = 1   # 8 real tokens
    return mask


@pytest.fixture
def mock_prompt_lengths():
    """Prompt lengths for 4 examples (response starts after these)."""
    return torch.tensor([4, 3, 5, 3], dtype=torch.long)
