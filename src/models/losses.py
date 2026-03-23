# src/models/losses.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss.

    Paper correspondence:
        L_rec = BPR(user_emb, pos_item_emb, neg_item_emb)

    Args:
        pos_scores: (B,)
        neg_scores: (B,)
        reduction: "mean" or "sum"

    Returns:
        scalar loss if reduction != 'none'
    """
    loss = -F.logsigmoid(pos_scores - neg_scores)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


def l2_regularization_loss(*tensors: torch.Tensor) -> torch.Tensor:
    """
    Standard L2 regularization on a set of tensors.

    Args:
        *tensors: arbitrary tensors

    Returns:
        scalar tensor
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided.")

    reg_loss = torch.zeros(1, device=tensors[0].device, dtype=tensors[0].dtype)
    for tensor in tensors:
        reg_loss = reg_loss + torch.sum(tensor ** 2)
    return reg_loss.squeeze(0)


def alignment_loss(
    delta_u: torch.Tensor,
    c_u: torch.Tensor,
    loss_type: str = "cosine",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Structure-aware alignment loss between semantic control signal Delta_u
    and structural context c_u.

    Paper correspondence:
        L_align(Delta_u, c_u)

    Args:
        delta_u: semantic control signal, shape (B, D)
        c_u: structural context, shape (B, D)
        loss_type: "cosine" or "mse"
        reduction: "mean", "sum", or "none"

    Returns:
        scalar loss if reduction != 'none'
    """
    if delta_u.shape != c_u.shape:
        raise ValueError(
            f"delta_u and c_u must have the same shape, got {delta_u.shape} vs {c_u.shape}"
        )

    if loss_type == "cosine":
        delta_norm = F.normalize(delta_u, p=2, dim=-1)
        c_norm = F.normalize(c_u, p=2, dim=-1)
        loss = 1.0 - torch.sum(delta_norm * c_norm, dim=-1)
    elif loss_type == "mse":
        loss = torch.mean((delta_u - c_u) ** 2, dim=-1)
    else:
        raise ValueError("loss_type must be either 'cosine' or 'mse'")

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


class SCALoss(nn.Module):
    """
    Combined loss for SCA:
        L = L_bpr + lambda_align * L_align + lambda_reg * L2

    Args:
        lambda_align: weight of alignment loss
        lambda_reg: weight of L2 regularization
        align_type: "cosine" or "mse"
    """

    def __init__(
        self,
        lambda_align: float = 0.1,
        lambda_reg: float = 1e-4,
        align_type: str = "cosine",
    ) -> None:
        super().__init__()
        self.lambda_align = lambda_align
        self.lambda_reg = lambda_reg
        self.align_type = align_type

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        delta_u: torch.Tensor,
        c_u: torch.Tensor,
        *reg_tensors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pos_scores: (B,)
            neg_scores: (B,)
            delta_u: (B, D)
            c_u: (B, D)
            *reg_tensors: tensors to regularize

        Returns:
            dict with:
                loss
                bpr_loss
                align_loss
                reg_loss
        """
        rec = bpr_loss(pos_scores, neg_scores, reduction="mean")
        align = alignment_loss(
            delta_u=delta_u,
            c_u=c_u,
            loss_type=self.align_type,
            reduction="mean",
        )

        if len(reg_tensors) > 0:
            reg = l2_regularization_loss(*reg_tensors)
        else:
            reg = torch.zeros((), device=pos_scores.device, dtype=pos_scores.dtype)

        total = rec + self.lambda_align * align + self.lambda_reg * reg

        return {
            "loss": total,
            "bpr_loss": rec,
            "align_loss": align,
            "reg_loss": reg,
        }