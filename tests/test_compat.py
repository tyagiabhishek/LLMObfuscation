"""Compatibility tests: verify package APIs and versions work correctly."""

import importlib
from unittest import mock

import numpy as np
import pytest
import torch
from rl_obfuscation import (
    check_environment, _parse_version, REQUIRED_PACKAGES, CROSS_PACKAGE_CONSTRAINTS,
    BNB_PEFT_CONSTRAINT, DEEPSPEED_TORCH_CONSTRAINT,
)


# --- check_environment tests ---

def test_check_environment_returns_no_issues():
    """Current environment should pass validation."""
    issues = check_environment(verbose=False)
    assert issues == [], f"Environment issues: {issues}"


def test_check_environment_verbose_does_not_crash():
    """Verbose mode should print without errors."""
    check_environment(verbose=True)


def test_parse_version_simple():
    assert _parse_version("1.2.3") == (1, 2, 3)


def test_parse_version_dev():
    assert _parse_version("4.50.0.dev0") == (4, 50, 0)


def test_parse_version_rc():
    assert _parse_version("2.1.0rc1") == (2, 1, 0)


def test_parse_version_local_cuda():
    """PEP 440 local version: 2.1.1+cu121 should parse as (2, 1, 1)."""
    assert _parse_version("2.1.1+cu121") == (2, 1, 1)


def test_parse_version_local_rocm():
    assert _parse_version("2.4.0+rocm5.6") == (2, 4, 0)


def test_parse_version_local_cpu():
    assert _parse_version("2.1.0+cpu") == (2, 1, 0)


def test_parse_version_comparison():
    assert _parse_version("2.1.0") >= _parse_version("2.0.0")
    assert _parse_version("1.26.4") >= _parse_version("1.24.0")
    assert _parse_version("1.26.4") < _parse_version("2.0.0")


def test_parse_version_cuda_not_flagged_as_old():
    """2.1.1+cu121 should NOT be < 2.1.0 — this was the original bug."""
    assert _parse_version("2.1.1+cu121") >= _parse_version("2.1.0")


def test_torch_transformers_interop():
    """transformers must be able to detect torch for return_tensors='pt'."""
    from transformers.utils import is_torch_available
    assert is_torch_available(), (
        "transformers.utils.is_torch_available() is False — "
        "tokenizer(return_tensors='pt') will fail. "
        "transformers>=5.0 requires torch>=2.4.0."
    )


def test_cross_package_constraints_defined():
    """Cross-package constraints list should exist and have entries."""
    assert len(CROSS_PACKAGE_CONSTRAINTS) >= 1


def test_current_torch_meets_transformers_constraint():
    """Current torch version must satisfy transformers' internal requirement."""
    import transformers
    tf_ver = _parse_version(transformers.__version__)
    torch_ver = _parse_version(torch.__version__)
    for tf_min, torch_min, desc in CROSS_PACKAGE_CONSTRAINTS:
        if tf_ver >= _parse_version(tf_min):
            assert torch_ver >= _parse_version(torch_min), (
                f"torch {torch.__version__} is too old for transformers "
                f"{transformers.__version__}: {desc}"
            )


def test_required_packages_list_not_empty():
    assert len(REQUIRED_PACKAGES) >= 10


def test_trl_min_version_reflects_api():
    """trl minimum version should be >=0.15.0 (max_prompt_length removed)."""
    trl_entry = [r for r in REQUIRED_PACKAGES if r[0] == "trl"]
    assert len(trl_entry) == 1
    assert _parse_version(trl_entry[0][2]) >= _parse_version("0.15.0")


def test_peft_min_version_reflects_bnb_compat():
    """peft minimum version should be >=0.14.0 (sync_gpu fix)."""
    peft_entry = [r for r in REQUIRED_PACKAGES if r[0] == "peft"]
    assert len(peft_entry) == 1
    assert _parse_version(peft_entry[0][2]) >= _parse_version("0.14.0")


def test_grpo_config_no_max_prompt_length():
    """GRPOConfig in trl>=0.15.0 should not accept max_prompt_length."""
    from trl import GRPOConfig
    with pytest.raises(TypeError):
        GRPOConfig(output_dir="/tmp/test", max_prompt_length=256)


def test_grpo_config_accepts_max_completion_length():
    """GRPOConfig should accept max_completion_length (the replacement)."""
    from trl import GRPOConfig
    config = GRPOConfig(output_dir="/tmp/test", max_completion_length=200)
    assert config.max_completion_length == 200


# --- Optional package constraint detection tests ---

def test_bnb_peft_constraint_constants():
    """Constraint thresholds should be defined."""
    assert "bnb_breaks_at" in BNB_PEFT_CONSTRAINT
    assert "peft_fixed_at" in BNB_PEFT_CONSTRAINT
    assert _parse_version(BNB_PEFT_CONSTRAINT["bnb_breaks_at"]) == (0, 44, 0)
    assert _parse_version(BNB_PEFT_CONSTRAINT["peft_fixed_at"]) == (0, 14, 0)


def test_deepspeed_torch_constraint_constants():
    """Constraint thresholds should be defined."""
    assert "torch_breaks_at" in DEEPSPEED_TORCH_CONSTRAINT
    assert "deepspeed_fixed_at" in DEEPSPEED_TORCH_CONSTRAINT
    assert _parse_version(DEEPSPEED_TORCH_CONSTRAINT["torch_breaks_at"]) == (2, 4, 0)
    assert _parse_version(DEEPSPEED_TORCH_CONSTRAINT["deepspeed_fixed_at"]) == (0, 15, 0)


def _make_fake_module(version: str):
    """Create a mock module with a __version__ attribute."""
    mod = mock.MagicMock()
    mod.__version__ = version
    return mod


def test_check_environment_detects_bnb_peft_incompatibility():
    """check_environment should flag bitsandbytes>=0.44 + peft<0.14."""
    fake_bnb = _make_fake_module("0.44.1")
    fake_peft = _make_fake_module("0.12.0")
    original_import = importlib.import_module

    def patched_import(name, *args, **kwargs):
        if name == "bitsandbytes":
            return fake_bnb
        if name == "peft":
            return fake_peft
        return original_import(name, *args, **kwargs)

    with mock.patch("importlib.import_module", side_effect=patched_import):
        issues = check_environment(verbose=False)

    bnb_issues = [i for i in issues if "sync_gpu" in i]
    assert len(bnb_issues) == 1, f"Expected sync_gpu issue, got: {issues}"
    assert "peft>=0.14.0" in bnb_issues[0]


def test_check_environment_ok_with_compatible_bnb_peft():
    """No issue when peft is new enough for bitsandbytes."""
    fake_bnb = _make_fake_module("0.45.0")
    fake_peft = _make_fake_module("0.14.0")
    original_import = importlib.import_module

    def patched_import(name, *args, **kwargs):
        if name == "bitsandbytes":
            return fake_bnb
        if name == "peft":
            return fake_peft
        return original_import(name, *args, **kwargs)

    with mock.patch("importlib.import_module", side_effect=patched_import):
        issues = check_environment(verbose=False)

    bnb_issues = [i for i in issues if "sync_gpu" in i]
    assert len(bnb_issues) == 0


def test_check_environment_detects_deepspeed_torch_incompatibility():
    """check_environment should flag deepspeed<0.15 + torch>=2.4."""
    fake_ds = _make_fake_module("0.14.5")
    original_import = importlib.import_module

    def patched_import(name, *args, **kwargs):
        if name == "deepspeed":
            return fake_ds
        return original_import(name, *args, **kwargs)

    # Only triggers if torch >= 2.4.0 (current env has 2.10.0)
    torch_ver = _parse_version(torch.__version__)
    if torch_ver < _parse_version("2.4.0"):
        pytest.skip("torch < 2.4.0, deepspeed constraint not applicable")

    with mock.patch("importlib.import_module", side_effect=patched_import):
        issues = check_environment(verbose=False)

    ds_issues = [i for i in issues if "deepspeed" in i and "elastic" in i]
    assert len(ds_issues) == 1, f"Expected deepspeed/elastic issue, got: {issues}"
    assert "deepspeed>=0.15.0" in ds_issues[0]


def test_check_environment_ok_with_compatible_deepspeed():
    """No issue when deepspeed is new enough for torch."""
    fake_ds = _make_fake_module("0.16.0")
    original_import = importlib.import_module

    def patched_import(name, *args, **kwargs):
        if name == "deepspeed":
            return fake_ds
        return original_import(name, *args, **kwargs)

    with mock.patch("importlib.import_module", side_effect=patched_import):
        issues = check_environment(verbose=False)

    ds_issues = [i for i in issues if "deepspeed" in i and "elastic" in i]
    assert len(ds_issues) == 0


def test_check_environment_skips_missing_optional_packages():
    """No issue if bitsandbytes/deepspeed are not installed."""
    # In our env they're not installed, so check_environment should not flag them
    issues = check_environment(verbose=False)
    optional_issues = [i for i in issues if "sync_gpu" in i or "elastic" in i]
    assert len(optional_issues) == 0, f"False positives for uninstalled packages: {optional_issues}"


# --- numpy/torch/sklearn API tests ---


def test_numpy_default_rng_exists():
    rng = np.random.default_rng(42)
    assert hasattr(rng, "shuffle")


def test_numpy_shuffle_on_python_list():
    rng = np.random.default_rng(42)
    x = list(range(20))
    original = x.copy()
    rng.shuffle(x)
    assert x != original  # overwhelmingly likely with 20 elements
    assert sorted(x) == sorted(original)


def test_numpy_unique_on_float_array():
    arr = np.array([0.0, 0.0, 1.0, 1.0])
    unique = np.unique(arr)
    assert len(unique) == 2


def test_numpy_asarray_dtype_cast():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = np.asarray(arr, dtype=np.float64)
    assert result.dtype == np.float64


def test_torch_device_detection():
    d = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    assert isinstance(d, torch.device)


def test_torch_no_grad_context():
    x = torch.tensor([1.0], requires_grad=True)
    with torch.no_grad():
        y = x * 2
    assert not y.requires_grad


def test_sklearn_roc_curve_api():
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    assert len(fpr) > 0
    assert len(tpr) > 0
    auc = roc_auc_score(labels, scores)
    assert 0.0 <= auc <= 1.0


def test_sklearn_roc_curve_with_float64():
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = np.array([0, 0, 1, 1], dtype=np.float64)
    scores = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    assert 0.0 <= auc <= 1.0


def test_transformers_auto_classes_importable():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert callable(AutoModelForCausalLM.from_pretrained)
    assert callable(AutoTokenizer.from_pretrained)


def test_torch_register_forward_hook():
    """Verify forward hook API works as expected."""
    layer = torch.nn.Linear(4, 4)
    hook_data = {}

    def hook_fn(module, input, output):
        hook_data["output"] = output.detach()

    handle = layer.register_forward_hook(hook_fn)
    x = torch.randn(2, 4)
    _ = layer(x)
    handle.remove()

    assert "output" in hook_data
    assert hook_data["output"].shape == (2, 4)


def test_torch_binary_cross_entropy_with_logits():
    logits = torch.tensor([2.0, -1.0, 0.5])
    labels = torch.tensor([1.0, 0.0, 1.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    assert loss.item() > 0
    assert not torch.isnan(loss)
