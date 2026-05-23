"""SPON: Spontaneous Neurons for sparse activation error compensation.

Per-layer learnable bias vectors added to sparsified MLP intermediates.
After training, biases fold into down_proj weights for zero inference overhead.
Reference: arXiv 2512.12744 (ICML 2026)
"""

import torch
import torch.nn as nn


class SPONBiasVectors(nn.Module):
    """Per-layer learnable bias vectors of shape [intermediate_size].

    Added to sparsified SwiGLU intermediate activations before down_proj.
    Initialized to zero so initial behavior matches the uncompensated sparse model.
    """

    def __init__(self, num_layers: int, intermediate_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(intermediate_size))
            for _ in range(num_layers)
        ])

    def forward(self, layer_idx: int, intermediate: torch.Tensor) -> torch.Tensor:
        return intermediate + self.biases[layer_idx]

    def fold_into_model(self, model):
        """Absorb biases into down_proj layers for zero inference overhead.

        For layer l: down_proj.bias = down_proj.weight @ alpha_l
        If down_proj already has a bias, the folded term is added to it.
        """
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= self.num_layers:
                break
            alpha = self.biases[layer_idx].data.to(layer.mlp.down_proj.weight.dtype)
            folded = layer.mlp.down_proj.weight @ alpha
            if layer.mlp.down_proj.bias is not None:
                layer.mlp.down_proj.bias.data.add_(folded.to(layer.mlp.down_proj.bias.dtype))
            else:
                layer.mlp.down_proj.bias = nn.Parameter(folded, requires_grad=False)

    def param_count(self) -> int:
        return sum(b.numel() for b in self.biases)


class SPONPatcher:
    """Patches MLP forwards to inject predictor masks and SPON biases.

    Modes:
        "dense":       original MLP forward (no mask, no bias)
        "sparse":      predictor hard mask only
        "sparse_spon": predictor hard mask + SPON bias compensation
    """

    def __init__(self, model, predictors: nn.ModuleList, spon_biases: SPONBiasVectors):
        self.model = model
        self.predictors = predictors
        self.spon_biases = spon_biases
        self._saved_forwards = {}
        self.mode = "dense"

    def patch(self):
        for idx, layer in enumerate(self.model.model.layers):
            if idx >= len(self.predictors):
                break
            self._saved_forwards[idx] = layer.mlp.forward
            self._make_patched_fwd(idx, layer.mlp)

    def unpatch(self):
        for idx, fwd in self._saved_forwards.items():
            self.model.model.layers[idx].mlp.forward = fwd
        self._saved_forwards.clear()

    def _make_patched_fwd(self, layer_idx, mlp):
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj
        down_proj = mlp.down_proj
        act_fn = mlp.act_fn
        predictor = self.predictors[layer_idx]
        spon = self.spon_biases
        patcher = self

        def fwd(x):
            gate = act_fn(gate_proj(x))
            up = up_proj(x)
            intermediate = gate * up

            if patcher.mode != "dense":
                with torch.no_grad():
                    mask = (predictor(x) > 0).to(intermediate.dtype)
                intermediate = intermediate * mask
                if patcher.mode == "sparse_spon":
                    intermediate = spon(layer_idx, intermediate)

            return down_proj(intermediate)

        mlp.forward = fwd

    def __enter__(self):
        self.patch()
        return self

    def __exit__(self, *args):
        self.unpatch()
