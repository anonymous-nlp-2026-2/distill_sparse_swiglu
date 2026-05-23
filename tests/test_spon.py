"""Unit tests for SPON — all CPU, no real model needed."""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spon import SPONBiasVectors, SPONPatcher
from predictor import SparsityPredictor


NUM_LAYERS = 2
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 128
BATCH = 2
SEQ = 4


# ---- fake model mimicking Llama MLP structure ----

class FakeLlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeLlamaLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.mlp = FakeLlamaMLP(hidden_size, intermediate_size)


class FakeLlamaModel(nn.Module):
    def __init__(self, num_layers, hidden_size, intermediate_size):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            FakeLlamaLayer(hidden_size, intermediate_size)
            for _ in range(num_layers)
        ])


def _make_model():
    torch.manual_seed(0)
    return FakeLlamaModel(NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)


def _make_predictors():
    torch.manual_seed(1)
    preds = nn.ModuleList([
        SparsityPredictor(HIDDEN_SIZE, INTERMEDIATE_SIZE, bottleneck_size=32)
        for _ in range(NUM_LAYERS)
    ])
    preds.eval()
    for p in preds.parameters():
        p.requires_grad = False
    return preds


# ---- SPONBiasVectors tests ----

class TestSPONBiasVectors:
    def test_zero_init_is_identity(self):
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        x = torch.randn(BATCH, SEQ, INTERMEDIATE_SIZE)
        assert torch.allclose(spon(0, x), x)

    def test_nonzero_bias_adds(self):
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        spon.biases[0].data.fill_(1.0)
        x = torch.zeros(BATCH, SEQ, INTERMEDIATE_SIZE)
        assert torch.allclose(spon(0, x), torch.ones(BATCH, SEQ, INTERMEDIATE_SIZE))

    def test_per_layer_independent(self):
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        spon.biases[0].data.fill_(1.0)
        spon.biases[1].data.fill_(2.0)
        x = torch.zeros(1, 1, INTERMEDIATE_SIZE)
        assert spon(0, x).mean().item() == pytest.approx(1.0)
        assert spon(1, x).mean().item() == pytest.approx(2.0)

    def test_param_count(self):
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        assert spon.param_count() == NUM_LAYERS * INTERMEDIATE_SIZE

    def test_bfloat16(self):
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE).bfloat16()
        x = torch.randn(BATCH, SEQ, INTERMEDIATE_SIZE, dtype=torch.bfloat16)
        assert spon(0, x).dtype == torch.bfloat16


# ---- fold_into_model tests ----

class TestFoldIntoModel:
    def test_creates_bias(self):
        model = _make_model()
        assert model.model.layers[0].mlp.down_proj.bias is None
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        spon.biases[0].data.fill_(0.1)
        spon.fold_into_model(model)
        assert model.model.layers[0].mlp.down_proj.bias is not None

    def test_correct_value(self):
        model = _make_model()
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        torch.manual_seed(42)
        spon.biases[0].data.normal_()
        expected = model.model.layers[0].mlp.down_proj.weight @ spon.biases[0].data
        spon.fold_into_model(model)
        assert torch.allclose(model.model.layers[0].mlp.down_proj.bias.data, expected, atol=1e-5)

    def test_output_equivalence(self):
        """down_proj(intermediate + alpha) == down_proj_folded(intermediate)"""
        torch.manual_seed(42)
        model = _make_model()
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        spon.biases[0].data.normal_(std=0.01)

        mlp = model.model.layers[0].mlp
        x = torch.randn(BATCH, SEQ, HIDDEN_SIZE)
        intermediate = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)

        out_with_spon = mlp.down_proj(intermediate + spon.biases[0])

        spon.fold_into_model(model)
        out_folded = mlp.down_proj(intermediate)

        assert torch.allclose(out_with_spon, out_folded, atol=1e-5)

    def test_accumulates_on_existing_bias(self):
        model = _make_model()
        mlp = model.model.layers[0].mlp
        mlp.down_proj.bias = nn.Parameter(torch.ones(HIDDEN_SIZE))

        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        spon.biases[0].data.fill_(0.1)
        expected_add = mlp.down_proj.weight @ spon.biases[0].data

        old_bias = mlp.down_proj.bias.data.clone()
        spon.fold_into_model(model)
        assert torch.allclose(mlp.down_proj.bias.data, old_bias + expected_add, atol=1e-5)


# ---- SPONPatcher tests ----

class TestSPONPatcher:
    def test_dense_mode_unchanged(self):
        model = _make_model()
        x = torch.randn(BATCH, SEQ, HIDDEN_SIZE)
        expected = model.model.layers[0].mlp(x).detach()

        patcher = SPONPatcher(model, _make_predictors(), SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE))
        patcher.patch()
        patcher.mode = "dense"
        actual = model.model.layers[0].mlp(x)
        patcher.unpatch()

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_sparse_spon_differs_from_sparse(self):
        model = _make_model()
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        torch.manual_seed(99)
        spon.biases[0].data.normal_(std=0.1)

        patcher = SPONPatcher(model, _make_predictors(), spon)
        patcher.patch()
        x = torch.randn(BATCH, SEQ, HIDDEN_SIZE)

        patcher.mode = "sparse"
        sparse_out = model.model.layers[0].mlp(x).clone()

        patcher.mode = "sparse_spon"
        spon_out = model.model.layers[0].mlp(x).clone()
        patcher.unpatch()

        assert not torch.allclose(sparse_out, spon_out)

    def test_zero_bias_sparse_spon_equals_sparse(self):
        model = _make_model()
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)

        patcher = SPONPatcher(model, _make_predictors(), spon)
        patcher.patch()
        x = torch.randn(BATCH, SEQ, HIDDEN_SIZE)

        patcher.mode = "sparse"
        sparse_out = model.model.layers[0].mlp(x).clone()

        patcher.mode = "sparse_spon"
        spon_out = model.model.layers[0].mlp(x).clone()
        patcher.unpatch()

        assert torch.allclose(sparse_out, spon_out)

    def test_context_manager_restores(self):
        model = _make_model()
        x = torch.randn(BATCH, SEQ, HIDDEN_SIZE)
        original_out = model.model.layers[0].mlp(x).detach().clone()

        with SPONPatcher(model, _make_predictors(), SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)) as p:
            p.mode = "sparse"
            patched_out = model.model.layers[0].mlp(x).detach().clone()

        restored_out = model.model.layers[0].mlp(x).detach()
        assert torch.allclose(restored_out, original_out, atol=1e-6)


# ---- gradient flow tests ----

class TestGradient:
    def test_gradient_flows_to_bias(self):
        model = _make_model()
        for p in model.parameters():
            p.requires_grad = False

        predictors = _make_predictors()
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)

        patcher = SPONPatcher(model, predictors, spon)
        patcher.patch()

        x = torch.randn(BATCH, SEQ, HIDDEN_SIZE)

        patcher.mode = "dense"
        with torch.no_grad():
            target = model.model.layers[0].mlp(x).detach()

        patcher.mode = "sparse_spon"
        output = model.model.layers[0].mlp(x)
        loss = (target - output).pow(2).mean()
        loss.backward()
        patcher.unpatch()

        assert spon.biases[0].grad is not None
        assert spon.biases[0].grad.abs().sum() > 0

    def test_model_params_stay_frozen(self):
        model = _make_model()
        for p in model.parameters():
            p.requires_grad = False

        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        patcher = SPONPatcher(model, _make_predictors(), spon)
        patcher.patch()

        x = torch.randn(BATCH, SEQ, HIDDEN_SIZE)
        patcher.mode = "sparse_spon"
        out = model.model.layers[0].mlp(x)
        out.sum().backward()
        patcher.unpatch()

        for p in model.parameters():
            assert p.grad is None

    def test_one_step_makes_bias_nonzero(self):
        model = _make_model()
        for p in model.parameters():
            p.requires_grad = False

        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        assert spon.biases[0].data.abs().sum() == 0

        patcher = SPONPatcher(model, _make_predictors(), spon)
        patcher.patch()

        optimizer = torch.optim.AdamW(spon.parameters(), lr=1e-3)
        x = torch.randn(BATCH, SEQ, HIDDEN_SIZE)

        patcher.mode = "dense"
        with torch.no_grad():
            target = model.model.layers[0].mlp(x).detach()

        patcher.mode = "sparse_spon"
        output = model.model.layers[0].mlp(x)
        loss = (target - output).pow(2).mean()
        loss.backward()
        optimizer.step()
        patcher.unpatch()

        assert spon.biases[0].data.abs().sum() > 0


# ---- save / load roundtrip ----

class TestSaveLoad:
    def test_roundtrip(self):
        spon = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
        torch.manual_seed(7)
        for b in spon.biases:
            b.data.normal_()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save({"spon_biases": spon.state_dict()}, path)
            spon2 = SPONBiasVectors(NUM_LAYERS, INTERMEDIATE_SIZE)
            spon2.load_state_dict(torch.load(path, weights_only=True)["spon_biases"])
            for i in range(NUM_LAYERS):
                assert torch.allclose(spon.biases[i], spon2.biases[i])
        finally:
            os.unlink(path)
