"""Tests for Config dataclass."""

from rl_obfuscation import Config


def test_config_defaults():
    c = Config()
    assert c.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert c.n_safe == 200
    assert c.n_harmful == 200
    assert c.max_length == 256
    assert c.seed == 42


def test_config_field_types():
    c = Config()
    assert isinstance(c.model_name, str)
    assert isinstance(c.n_safe, int)
    assert isinstance(c.probe_lr, float)
    assert isinstance(c.seed, int)
    assert isinstance(c.lora_r, int)
    assert isinstance(c.rl_beta, float)


def test_config_override():
    c = Config(n_safe=50, n_harmful=50, seed=123)
    assert c.n_safe == 50
    assert c.n_harmful == 50
    assert c.seed == 123
    # Defaults unchanged
    assert c.probe_epochs == 40


def test_config_rl_params():
    c = Config()
    assert c.rl_wb_weight == 0.8
    assert c.rl_bb_weight == 1.0
    assert c.rl_length_weight == 2.0
    assert c.rl_beta == 0.05
    assert c.rl_learning_rate == 1e-5
    assert c.rl_num_generations == 8


def test_config_lora_params():
    c = Config()
    assert c.lora_r == 16
    assert c.lora_alpha == 32
    assert c.lora_dropout == 0.05
