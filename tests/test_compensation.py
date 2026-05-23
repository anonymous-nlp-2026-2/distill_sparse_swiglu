"""Unit tests for ErrorCompensation — all CPU, no real model needed."""

import os
import sys
import tempfile

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from compensation import ErrorCompensation


BATCH, SEQ, DIM = 2, 4, 16
NUM_LAYERS = 3


def _make_compensation(num_layers=NUM_LAYERS, dim=DIM):
    """Build an ErrorCompensation with known mean activations."""
    means = {}
    torch.manual_seed(0)
    for i in range(num_layers):
        means[i] = torch.randn(dim)
    return ErrorCompensation(mean_activations=means)


def _rand_intermediate():
    return torch.randn(BATCH, SEQ, DIM)


# ---------- core behavior ----------

class TestCompensateBasic:
    def test_mask_all_ones_gives_original(self):
        """Mask = all 1 (keep everything) → output equals original intermediate."""
        comp = _make_compensation()
        x = _rand_intermediate()
        mask = torch.ones(BATCH, SEQ, DIM)
        out = comp.compensate(x, mask, layer_idx=0)
        assert torch.allclose(out, x), "All-ones mask should return original"

    def test_mask_all_zeros_gives_mean(self):
        """Mask = all 0 (prune everything) → output equals mean activation broadcast."""
        comp = _make_compensation()
        x = _rand_intermediate()
        mask = torch.zeros(BATCH, SEQ, DIM)
        out = comp.compensate(x, mask, layer_idx=0)
        expected = comp.mean_activations[0].expand_as(x)
        assert torch.allclose(out, expected), "All-zeros mask should return mean activation"

    def test_partial_mask_combines_correctly(self):
        """Partial mask: kept channels from x, pruned channels from mean."""
        comp = _make_compensation()
        x = _rand_intermediate()
        mask = torch.ones(BATCH, SEQ, DIM)
        mask[:, :, : DIM // 2] = 0

        out = comp.compensate(x, mask, layer_idx=0)
        mean = comp.mean_activations[0]
        expected = x * mask + mean * (1.0 - mask)
        assert torch.allclose(out, expected)

    def test_partial_mask_kept_channels_unchanged(self):
        """Kept channels should be identical to original values."""
        comp = _make_compensation()
        x = _rand_intermediate()
        mask = torch.ones(BATCH, SEQ, DIM)
        mask[:, :, : DIM // 2] = 0

        out = comp.compensate(x, mask, layer_idx=0)
        # Channels where mask=1 should match original x
        assert torch.allclose(out[:, :, DIM // 2 :], x[:, :, DIM // 2 :])

    def test_partial_mask_pruned_channels_are_mean(self):
        """Pruned channels should equal mean activation."""
        comp = _make_compensation()
        x = _rand_intermediate()
        mask = torch.ones(BATCH, SEQ, DIM)
        mask[:, :, : DIM // 2] = 0

        out = comp.compensate(x, mask, layer_idx=0)
        expected_pruned = comp.mean_activations[0][: DIM // 2].expand(BATCH, SEQ, -1)
        assert torch.allclose(out[:, :, : DIM // 2], expected_pruned)


# ---------- per-layer isolation ----------

class TestPerLayer:
    def test_different_layers_different_compensation(self):
        """Each layer should use its own mean activation vector."""
        comp = _make_compensation()
        x = _rand_intermediate()
        mask = torch.zeros(BATCH, SEQ, DIM)
        out0 = comp.compensate(x, mask, layer_idx=0)
        out1 = comp.compensate(x, mask, layer_idx=1)
        assert not torch.allclose(out0, out1), "Different layers should give different compensations"

    def test_missing_layer_falls_back_to_masked(self):
        """If layer_idx not in mean_activations, fall back to x * mask (no compensation)."""
        comp = _make_compensation(num_layers=1)  # only layer 0
        x = _rand_intermediate()
        mask = torch.zeros(BATCH, SEQ, DIM)
        out = comp.compensate(x, mask, layer_idx=99)
        assert torch.allclose(out, torch.zeros_like(x)), "Missing layer should fall back to x*mask"


# ---------- save / load ----------

class TestSaveLoad:
    def test_roundtrip(self):
        """save → load → compensate should give identical results."""
        comp = _make_compensation()
        x = _rand_intermediate()
        mask = torch.ones(BATCH, SEQ, DIM)
        mask[:, :, : DIM // 2] = 0
        expected = comp.compensate(x, mask, layer_idx=0)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            comp.save(path)
            comp2 = ErrorCompensation.load(path)
            result = comp2.compensate(x, mask, layer_idx=0)
            assert torch.allclose(expected, result), "Loaded compensation should match original"
        finally:
            os.unlink(path)

    def test_load_preserves_all_layers(self):
        comp = _make_compensation()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            comp.save(path)
            comp2 = ErrorCompensation.load(path)
            assert set(comp2.mean_activations.keys()) == set(comp.mean_activations.keys())
            for k in comp.mean_activations:
                assert torch.allclose(comp.mean_activations[k], comp2.mean_activations[k])
        finally:
            os.unlink(path)


# ---------- dtype / broadcasting ----------

class TestEdgeCases:
    def test_bfloat16(self):
        """Compensation should work with bfloat16 tensors."""
        comp = _make_compensation()
        x = _rand_intermediate().bfloat16()
        mask = torch.ones(BATCH, SEQ, DIM, dtype=torch.bfloat16)
        mask[:, :, : DIM // 2] = 0
        out = comp.compensate(x, mask, layer_idx=0)
        assert out.dtype == torch.bfloat16

    def test_broadcast_1d_mask(self):
        """A 1D mask [intermediate_size] should broadcast correctly."""
        comp = _make_compensation()
        x = _rand_intermediate()
        mask = torch.ones(DIM)
        mask[: DIM // 2] = 0
        out = comp.compensate(x, mask, layer_idx=0)
        assert out.shape == x.shape

    def test_zero_mean_activation(self):
        """If mean activation is zero, compensation on pruned channels is zero."""
        comp = ErrorCompensation({0: torch.zeros(DIM)})
        x = _rand_intermediate()
        mask = torch.zeros(BATCH, SEQ, DIM)
        out = comp.compensate(x, mask, layer_idx=0)
        assert torch.allclose(out, torch.zeros_like(x))
