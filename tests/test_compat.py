"""Compatibility tests: verify package APIs and versions work correctly."""

import numpy as np
import torch
from rl_obfuscation import check_environment, _parse_version, REQUIRED_PACKAGES


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


def test_parse_version_comparison():
    assert _parse_version("2.1.0") >= _parse_version("2.0.0")
    assert _parse_version("1.26.4") >= _parse_version("1.24.0")
    assert _parse_version("1.26.4") < _parse_version("2.0.0")


def test_required_packages_list_not_empty():
    assert len(REQUIRED_PACKAGES) >= 10


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
