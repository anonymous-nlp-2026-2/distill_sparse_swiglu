# Loss functions: KL distillation, BCE activation prediction, compensation MSE, sparsity regularizer.

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDistillLoss(nn.Module):
    """KL(P_dense || P_sparse) on output logit distributions.

    P = softmax(logits / T). Scaled by T^2 so gradient magnitude is
    independent of temperature.
    Normalized to per-token-per-dim scale.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, dense_logits: torch.Tensor, sparse_logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(dense_logits / self.temperature, dim=-1)
        log_q = F.log_softmax(sparse_logits / self.temperature, dim=-1)
        kl = F.kl_div(log_q, p, reduction="batchmean") * (self.temperature ** 2)
        seq_len = dense_logits.shape[-2]
        return kl / seq_len


class BCESparsityLoss(nn.Module):
    """BCE between predictor logits and top-k binary activation targets.

    For each position, the top (1-sparsity) fraction of neurons by absolute
    activation value are labeled 1 (keep); the rest 0 (prune).
    """

    def __init__(self, sparsity_target: float = 0.5):
        super().__init__()
        self.sparsity_target = sparsity_target

    def forward(self, predictor_logits: torch.Tensor, dense_intermediate: torch.Tensor) -> torch.Tensor:
        k = int(dense_intermediate.shape[-1] * (1 - self.sparsity_target))
        abs_vals = dense_intermediate.abs()
        topk_vals, _ = abs_vals.topk(k, dim=-1)
        threshold = topk_vals[..., -1:]
        binary_target = (abs_vals >= threshold).float()
        return F.binary_cross_entropy_with_logits(predictor_logits, binary_target)


class CompensationLoss(nn.Module):
    """MSE between (sparse_mlp_output + compensation) and dense_mlp_output.

    Operates per-layer in hidden_size space (after down_proj).
    """

    def forward(self, compensation: torch.Tensor, sparse_output: torch.Tensor,
                dense_output: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(sparse_output + compensation, dense_output)


class SparsityRegularizer(nn.Module):
    """L1 penalty driving average mask density toward (1 - target_sparsity)."""

    def __init__(self, target_sparsity: float = 0.5, weight: float = 1.0):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.weight = weight

    def forward(self, masks: dict) -> torch.Tensor:
        if not masks:
            return torch.tensor(0.0)
        penalties = []
        for mask in masks.values():
            density = mask.mean()
            penalties.append(torch.abs(density - (1.0 - self.target_sparsity)))
        return self.weight * torch.stack(penalties).mean()


class ReverseKLDistillLoss(nn.Module):
    """KL(P_sparse || P_dense) — mode-seeking divergence.

    Normalized to per-token scale (batchmean / seq_len), T^2 scaled.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, dense_logits: torch.Tensor, sparse_logits: torch.Tensor) -> torch.Tensor:
        p_sparse = F.softmax(sparse_logits / self.temperature, dim=-1)
        log_p_dense = F.log_softmax(dense_logits / self.temperature, dim=-1)
        # F.kl_div(input, target) = sum(target * (log(target) - input))
        # F.kl_div(log_p_dense, p_sparse) = KL(p_sparse || p_dense)
        kl = F.kl_div(log_p_dense, p_sparse, reduction="batchmean") * (self.temperature ** 2)
        seq_len = dense_logits.shape[-2]
        return kl / seq_len


class JSDDistillLoss(nn.Module):
    """Jensen-Shannon Divergence: 0.5*KL(P_dense||M) + 0.5*KL(P_sparse||M).

    M = 0.5*(P_dense + P_sparse). Symmetric, bounded [0, ln2].
    Normalized to per-token scale (batchmean / seq_len), T^2 scaled.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, dense_logits: torch.Tensor, sparse_logits: torch.Tensor) -> torch.Tensor:
        p_dense = F.softmax(dense_logits / self.temperature, dim=-1)
        p_sparse = F.softmax(sparse_logits / self.temperature, dim=-1)
        m = 0.5 * (p_dense + p_sparse)
        log_m = m.log()
        jsd = 0.5 * F.kl_div(log_m, p_dense, reduction="batchmean") + \
              0.5 * F.kl_div(log_m, p_sparse, reduction="batchmean")
        jsd = jsd * (self.temperature ** 2)
        seq_len = dense_logits.shape[-2]
        return jsd / seq_len


class PerLayerMSELoss(nn.Module):
    """Per-layer MSE between sparse and dense down_proj outputs.

    Measures ||down_proj(act * mask) - down_proj(act)||^2, averaged over elements.
    Wrapper for F.mse_loss to keep loss interface consistent with other loss classes.
    """

    def forward(self, sparse_out: torch.Tensor, dense_out: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(sparse_out, dense_out)
