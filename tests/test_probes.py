"""Tests for LinearProbe and train_probe."""

import numpy as np
import torch
from rl_obfuscation import LinearProbe, train_probe


class TestLinearProbe:
    def test_output_shape_single(self):
        probe = LinearProbe(d_model=8)
        x = torch.randn(8)
        out = probe(x)
        assert out.shape == ()

    def test_output_shape_batch(self):
        probe = LinearProbe(d_model=8)
        x = torch.randn(4, 8)
        out = probe(x)
        assert out.shape == (4,)

    def test_predict_range(self):
        probe = LinearProbe(d_model=8)
        x = torch.randn(10, 8)
        probs = probe.predict(x)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_shape(self):
        probe = LinearProbe(d_model=16)
        x = torch.randn(5, 16)
        probs = probe.predict(x)
        assert probs.shape == (5,)

    def test_gradient_flows(self):
        probe = LinearProbe(d_model=8)
        x = torch.randn(4, 8)
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        logits = probe(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        assert probe.linear.weight.grad is not None
        assert probe.linear.bias.grad is not None

    def test_different_d_model(self):
        for d in [4, 32, 128]:
            probe = LinearProbe(d_model=d)
            x = torch.randn(3, d)
            out = probe(x)
            assert out.shape == (3,)

    def test_parameter_count(self):
        d = 16
        probe = LinearProbe(d_model=d)
        n_params = sum(p.numel() for p in probe.parameters())
        # nn.Linear(d, 1) has d weights + 1 bias = d + 1
        assert n_params == d + 1


class TestTrainProbe:
    def test_returns_probe_and_metrics(self):
        torch.manual_seed(42)
        acts = torch.randn(40, 8)
        labels = torch.tensor([0.0] * 20 + [1.0] * 20)
        probe, metrics = train_probe(
            acts, labels, d_model=8, n_epochs=5, batch_size=8,
            probe_device=torch.device("cpu"),
        )
        assert isinstance(probe, LinearProbe)
        assert isinstance(metrics, dict)

    def test_metrics_keys(self):
        acts = torch.randn(40, 8)
        labels = torch.tensor([0.0] * 20 + [1.0] * 20)
        _, metrics = train_probe(
            acts, labels, d_model=8, n_epochs=5,
            probe_device=torch.device("cpu"),
        )
        assert "auroc" in metrics
        assert "accuracy" in metrics
        assert "test_idx" in metrics

    def test_separable_data_high_auroc(self):
        """Linearly separable data should give high AUROC."""
        torch.manual_seed(42)
        n = 100
        d = 8
        # Class 0: centered at -2, class 1: centered at +2
        acts_0 = torch.randn(n, d) - 2
        acts_1 = torch.randn(n, d) + 2
        acts = torch.cat([acts_0, acts_1])
        labels = torch.tensor([0.0] * n + [1.0] * n)
        probe, metrics = train_probe(
            acts, labels, d_model=d, n_epochs=20, batch_size=16,
            probe_device=torch.device("cpu"),
        )
        assert metrics["auroc"] > 0.9

    def test_probe_in_eval_mode_after_training(self):
        acts = torch.randn(40, 8)
        labels = torch.tensor([0.0] * 20 + [1.0] * 20)
        probe, _ = train_probe(
            acts, labels, d_model=8, n_epochs=2,
            probe_device=torch.device("cpu"),
        )
        assert not probe.training

    def test_train_frac_splits_data(self):
        n = 50
        acts = torch.randn(n, 8)
        labels = torch.tensor([0.0] * 25 + [1.0] * 25)
        _, metrics = train_probe(
            acts, labels, d_model=8, n_epochs=2, train_frac=0.6,
            probe_device=torch.device("cpu"),
        )
        # test_idx should have 40% of data = 20
        assert len(metrics["test_idx"]) == 20

    def test_seed_reproducibility(self):
        acts = torch.randn(40, 8)
        labels = torch.tensor([0.0] * 20 + [1.0] * 20)
        _, m1 = train_probe(
            acts, labels, d_model=8, n_epochs=5, seed=99,
            probe_device=torch.device("cpu"),
        )
        _, m2 = train_probe(
            acts, labels, d_model=8, n_epochs=5, seed=99,
            probe_device=torch.device("cpu"),
        )
        assert m1["auroc"] == m2["auroc"]
        assert m1["accuracy"] == m2["accuracy"]
