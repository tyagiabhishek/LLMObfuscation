"""Tests for checkpoint save/load functions."""

import pytest
import torch
from pathlib import Path
from rl_obfuscation import (
    LinearProbe,
    save_probes,
    load_probes,
    save_eval_results,
    load_eval_results,
    DEFAULT_CHECKPOINT_DIR,
)


@pytest.fixture
def sample_probes():
    """Create a dict of probes mimicking train_probe output."""
    torch.manual_seed(42)
    d_model = 16
    probes = {}
    metrics = {}
    for l in [6, 7, 8]:
        probe = LinearProbe(d_model)
        probes[l] = probe
        metrics[l] = {"auroc": 0.9 + l * 0.01, "accuracy": 0.85 + l * 0.01}
    return probes, metrics, d_model


class TestSaveLoadProbes:
    def test_roundtrip(self, tmp_path, sample_probes):
        """Saved probes should load back with identical weights and metrics."""
        probes, metrics, d_model = sample_probes
        path = tmp_path / "probes.pt"

        save_probes(probes, metrics, path)
        loaded_probes, loaded_metrics = load_probes(path, probe_device=torch.device("cpu"))

        assert set(loaded_probes.keys()) == set(probes.keys())
        assert loaded_metrics == metrics
        for l in probes:
            orig_w = probes[l].linear.weight.data
            load_w = loaded_probes[l].linear.weight.data
            assert torch.allclose(orig_w, load_w)

    def test_loaded_probes_in_eval_mode(self, tmp_path, sample_probes):
        probes, metrics, _ = sample_probes
        path = tmp_path / "probes.pt"
        save_probes(probes, metrics, path)
        loaded, _ = load_probes(path, probe_device=torch.device("cpu"))
        for probe in loaded.values():
            assert not probe.training

    def test_loaded_probes_predict(self, tmp_path, sample_probes):
        """Loaded probes should produce valid predictions."""
        probes, metrics, d_model = sample_probes
        path = tmp_path / "probes.pt"
        save_probes(probes, metrics, path)
        loaded, _ = load_probes(path, probe_device=torch.device("cpu"))

        x = torch.randn(4, d_model)
        for l in loaded:
            preds = loaded[l].predict(x)
            assert preds.shape == (4,)
            assert (preds >= 0).all() and (preds <= 1).all()

    def test_creates_parent_dirs(self, tmp_path, sample_probes):
        probes, metrics, _ = sample_probes
        path = tmp_path / "nested" / "dir" / "probes.pt"
        save_probes(probes, metrics, path)
        assert path.exists()

    def test_non_contiguous_layer_indices(self, tmp_path):
        """Should handle non-contiguous layer indices (middle 50% case)."""
        torch.manual_seed(42)
        probes = {10: LinearProbe(8), 15: LinearProbe(8), 20: LinearProbe(8)}
        metrics = {10: {"auroc": 0.9}, 15: {"auroc": 0.95}, 20: {"auroc": 0.88}}
        path = tmp_path / "probes.pt"

        save_probes(probes, metrics, path)
        loaded, loaded_m = load_probes(path, probe_device=torch.device("cpu"))

        assert set(loaded.keys()) == {10, 15, 20}
        assert loaded_m[15]["auroc"] == 0.95

    def test_preserves_d_model(self, tmp_path):
        """d_model should be inferred correctly from loaded probes."""
        for d in [64, 128, 1536]:
            probes = {0: LinearProbe(d)}
            metrics = {0: {"auroc": 0.9}}
            path = tmp_path / f"probes_{d}.pt"
            save_probes(probes, metrics, path)
            loaded, _ = load_probes(path, probe_device=torch.device("cpu"))
            assert loaded[0].linear.in_features == d


class TestSaveLoadEvalResults:
    def test_roundtrip(self, tmp_path):
        results = {
            "base_eval": {6: {"auroc": 0.95, "tpr_at_fpr": 0.9}},
            "best_layer": 6,
        }
        path = tmp_path / "eval.pt"
        save_eval_results(results, path)
        loaded = load_eval_results(path)
        assert loaded["best_layer"] == 6
        assert loaded["base_eval"][6]["auroc"] == 0.95

    def test_handles_numpy_arrays(self, tmp_path):
        """Eval results often contain numpy arrays (fpr/tpr curves)."""
        import numpy as np
        results = {
            "curves": {0: {"fpr": np.array([0.0, 0.5, 1.0]), "tpr": np.array([0.0, 0.8, 1.0])}},
        }
        path = tmp_path / "eval.pt"
        save_eval_results(results, path)
        loaded = load_eval_results(path)
        assert (loaded["curves"][0]["fpr"] == results["curves"][0]["fpr"]).all()

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "eval.pt"
        save_eval_results({"key": "value"}, path)
        assert path.exists()


class TestDefaultCheckpointDir:
    def test_is_pathlib_path(self):
        assert isinstance(DEFAULT_CHECKPOINT_DIR, Path)

    def test_default_name(self):
        assert DEFAULT_CHECKPOINT_DIR.name == "checkpoints"
