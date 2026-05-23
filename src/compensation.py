"""Error compensation for sparse SwiGLU activations (TEAL-style).

When neurons are masked out during sparsification, their expected contribution
(mean activation from calibration data) is substituted in, reducing output error.

For SwiGLU intermediate `a = act(gate_proj(x)) * up_proj(x)`:
  - Without compensation: y = down_proj(a * mask)
  - With compensation:    y = down_proj(a * mask + mean_a * (1 - mask))
"""

import torch


class ErrorCompensation:
    """Pre-computed mean activation vectors for error compensation.

    Calibrates on dense forward passes, then compensates masked intermediates
    by replacing pruned neurons with their channel-wise mean activation.
    """

    def __init__(self, mean_activations: dict[int, torch.Tensor] | None = None):
        self.mean_activations = mean_activations or {}

    @staticmethod
    def calibrate(wrapper, data_loader, device, num_batches: int = 32) -> "ErrorCompensation":
        """Compute per-layer mean activations from calibration data.

        Args:
            wrapper: PredictorWrapper with forward_dense / capture_intermediates.
            data_loader: yields dicts with "input_ids" tensors.
            device: torch device for computation.
            num_batches: number of calibration batches to average over.
        """
        accum = {}
        counts = {}

        data_iter = iter(data_loader)
        for _ in range(num_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            input_ids = batch["input_ids"].to(device)
            wrapper.forward_dense(input_ids, capture_intermediates=True)
            intermediates = wrapper.get_intermediates()

            for layer_idx, act in intermediates.items():
                # act: [batch, seq, intermediate_size] -> mean over batch & seq
                channel_sum = act.float().sum(dim=(0, 1))
                n_tokens = act.shape[0] * act.shape[1]
                if layer_idx not in accum:
                    accum[layer_idx] = channel_sum
                    counts[layer_idx] = n_tokens
                else:
                    accum[layer_idx] = accum[layer_idx] + channel_sum
                    counts[layer_idx] += n_tokens

        mean_activations = {}
        for layer_idx in sorted(accum):
            mean_activations[layer_idx] = (accum[layer_idx] / counts[layer_idx]).to(device)

        return ErrorCompensation(mean_activations)

    def compensate(
        self,
        intermediate: torch.Tensor,
        mask: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Replace masked-out neurons with their mean activation.

        Args:
            intermediate: [batch, seq, intermediate_size] — raw SwiGLU activation.
            mask: same shape or broadcastable — 1 = keep, 0 = prune.
            layer_idx: which layer's mean to use.

        Returns:
            Compensated intermediate, same shape as input.
        """
        if layer_idx not in self.mean_activations:
            return intermediate * mask

        mean_act = self.mean_activations[layer_idx].to(
            dtype=intermediate.dtype, device=intermediate.device
        )
        return intermediate * mask + mean_act * (1.0 - mask)

    def save(self, path: str) -> None:
        torch.save(
            {idx: v.cpu() for idx, v in self.mean_activations.items()},
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ErrorCompensation":
        data = torch.load(path, map_location=device, weights_only=True)
        return cls(mean_activations=data)
