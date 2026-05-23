"""Tests for teal_activation_magnitude_masks -- CPU only, uses mini_llama fixture."""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluate import teal_activation_magnitude_masks

MINI_LLAMA_DIR = os.path.join(os.path.dirname(__file__), "mini_llama")


@pytest.fixture(scope="module")
def model():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MINI_LLAMA_DIR, torch_dtype=torch.float32)
    m.eval()
    return m


@pytest.fixture
def calib_ids():
    torch.manual_seed(42)
    return torch.randint(0, 128256, (4, 32))


class TestActivationMagnitudeMasks:
    def test_returns_all_layers(self, model, calib_ids):
        masks = teal_activation_magnitude_masks(model, calib_ids, sparsity_target=0.5)
        assert set(masks.keys()) == {0, 1}

    def test_mask_shape(self, model, calib_ids):
        masks = teal_activation_magnitude_masks(model, calib_ids, sparsity_target=0.5)
        for li, m in masks.items():
            assert m.shape == (512,), f"Layer {li}: expected (512,), got {m.shape}"

    def test_mask_binary(self, model, calib_ids):
        masks = teal_activation_magnitude_masks(model, calib_ids, sparsity_target=0.5)
        for li, m in masks.items():
            unique = m.unique()
            assert all(v in (0.0, 1.0) for v in unique.tolist()), \
                f"Layer {li}: non-binary values {unique}"

    def test_mask_dtype(self, model, calib_ids):
        masks = teal_activation_magnitude_masks(model, calib_ids, sparsity_target=0.5)
        for li, m in masks.items():
            assert m.dtype == torch.bfloat16

    def test_overall_sparsity_close_to_target(self, model, calib_ids):
        target = 0.5
        masks = teal_activation_magnitude_masks(model, calib_ids, sparsity_target=target)
        total_n = sum(m.numel() for m in masks.values())
        total_z = sum((m == 0).sum().item() for m in masks.values())
        actual = total_z / total_n
        assert abs(actual - target) < 0.15, \
            f"Sparsity {actual:.3f} too far from target {target}"

    def test_higher_target_fewer_neurons(self, model, calib_ids):
        masks_30 = teal_activation_magnitude_masks(model, calib_ids, sparsity_target=0.3)
        masks_70 = teal_activation_magnitude_masks(model, calib_ids, sparsity_target=0.7)
        kept_30 = sum(m.sum().item() for m in masks_30.values())
        kept_70 = sum(m.sum().item() for m in masks_70.values())
        assert kept_30 > kept_70, "Higher sparsity should keep fewer neurons"

    def test_zero_sparsity_keeps_all(self, model, calib_ids):
        masks = teal_activation_magnitude_masks(model, calib_ids, sparsity_target=0.0)
        for li, m in masks.items():
            assert m.sum().item() == m.numel(), \
                f"Layer {li}: zero sparsity should keep all neurons"

    def test_step_size_param(self, model, calib_ids):
        masks = teal_activation_magnitude_masks(
            model, calib_ids, sparsity_target=0.5, step_size=0.1)
        total_n = sum(m.numel() for m in masks.values())
        total_z = sum((m == 0).sum().item() for m in masks.values())
        actual = total_z / total_n
        assert actual > 0.0, "Should produce some sparsity with step_size=0.1"
