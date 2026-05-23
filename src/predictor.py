# Sparsity predictor architecture: bottleneck MLP + Gumbel-sigmoid/STE mask + model wrapper.
# Compensation network: per-layer MLP correcting sparse MLP output errors (FastForward-style).
# Input: hidden_states (batch, seq, hidden_size) -> Output: mask logits (batch, seq, intermediate_size)

import torch
import torch.nn as nn


class SparsityPredictor(nn.Module):
    """Rank-bottleneck MLP predicting SwiGLU sparsity masks."""

    def __init__(self, hidden_size: int, intermediate_size: int, bottleneck_size: int = 128):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(hidden_states)))


class CompensationHead(nn.Module):
    """Per-layer MLP predicting the output error introduced by sparsification.

    Takes MLP input hidden state, outputs correction added to sparse MLP output.
    """

    def __init__(self, hidden_size: int, bottleneck_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size),
            nn.GELU(),
            nn.Linear(bottleneck_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class CompensationNetwork(nn.Module):
    """Per-layer compensation heads correcting sparse MLP output errors."""

    def __init__(self, num_layers: int, hidden_size: int, bottleneck_size: int = 256):
        super().__init__()
        self.heads = nn.ModuleList([
            CompensationHead(hidden_size, bottleneck_size)
            for _ in range(num_layers)
        ])

    def forward_layer(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.heads[layer_idx](hidden_states)


class GumbelSigmoidMask(nn.Module):
    """Differentiable binary mask via Gumbel-sigmoid relaxation.

    Training: soft mask with Gumbel noise and temperature annealing.
    Inference: hard threshold (sigmoid(logits) > 0.5 <=> logits > 0).
    """

    def __init__(self, tau: float = 1.0, hard: bool = False):
        super().__init__()
        self.tau = tau
        self.hard = hard

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training and not self.hard:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(u))
            y_soft = torch.sigmoid((logits + gumbel_noise) / self.tau)
            return y_soft
        else:
            return (logits > 0).to(logits.dtype)



class STEMask(nn.Module):
    """Straight-Through Estimator binary mask.

    Forward: hard threshold (sigmoid > 0.5).
    Backward: gradient flows through sigmoid (vanilla STE).
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            sigmoid_out = torch.sigmoid(logits)
            hard = (sigmoid_out > 0.5).to(logits.dtype)
            return hard - sigmoid_out.detach() + sigmoid_out
        else:
            return (logits > 0).to(logits.dtype)



class PredictorWrapper(nn.Module):
    """Wraps a frozen LlamaForCausalLM with per-layer sparsity predictors.

    Monkey-patches each MLP to optionally apply a predicted mask on the
    SwiGLU intermediate activation (act(gate_proj(x)) * up_proj(x)) before down_proj.

    Modes:
        sparse_mode=True:  apply Gumbel-sigmoid mask (training/eval)
        sparse_mode=False: passthrough; optionally capture intermediates for BCE
    """

    def __init__(self, model, bottleneck_size: int = 128):
        super().__init__()
        self.model = model
        for p in model.parameters():
            p.requires_grad = False

        config = model.config
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.predictors = nn.ModuleList([
            SparsityPredictor(hidden_size, intermediate_size, bottleneck_size)
            for _ in range(num_layers)
        ])
        self.gumbel_mask = GumbelSigmoidMask()

        self.sparse_mode = True
        self.capture_intermediates = False
        self._layer_masks = {}
        self._layer_intermediates = {}
        self._layer_inputs = {}

        # Compensation network (set externally, None by default)
        self.comp_network = None
        self.use_compensation = False

        self._patch_mlps()

    def _patch_mlps(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            self._patch_single_mlp(layer_idx, layer.mlp)

    def _patch_single_mlp(self, layer_idx, mlp):
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj
        down_proj = mlp.down_proj
        act_fn = mlp.act_fn
        wrapper = self
        predictor = self.predictors[layer_idx]

        def patched_forward(x):
            gate = act_fn(gate_proj(x))
            up = up_proj(x)
            intermediate = gate * up

            if wrapper.sparse_mode:
                logits = predictor(x)
                mask = wrapper.gumbel_mask(logits)
                wrapper._layer_masks[layer_idx] = mask
                result = down_proj(intermediate * mask)
                if wrapper.use_compensation and wrapper.comp_network is not None:
                    result = result + wrapper.comp_network.forward_layer(layer_idx, x)
                return result
            elif wrapper.capture_intermediates:
                wrapper._layer_intermediates[layer_idx] = intermediate.detach()
                wrapper._layer_inputs[layer_idx] = x.detach()

            return down_proj(intermediate)

        mlp.forward = patched_forward

    def forward(self, input_ids, attention_mask=None, labels=None):
        self._layer_masks.clear()
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def forward_dense(self, input_ids, attention_mask=None, capture_intermediates=False):
        """Forward without masks. Optionally captures per-layer intermediates & inputs."""
        self.sparse_mode = False
        self.capture_intermediates = capture_intermediates
        self._layer_intermediates.clear()
        self._layer_inputs.clear()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self.sparse_mode = True
        self.capture_intermediates = False
        return outputs

    def forward_sparse(self, input_ids, attention_mask=None, labels=None):
        """Forward with predicted sparsity masks."""
        self.sparse_mode = True
        return self.forward(input_ids, attention_mask, labels)

    def get_masks(self):
        return dict(self._layer_masks)

    def get_intermediates(self):
        return dict(self._layer_intermediates)

    def get_layer_inputs(self):
        return dict(self._layer_inputs)
