# src/models/sca.py

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lightgcn import LightGCN
from src.models.semantic_encoder import SemanticEncoder
from src.models.gate import SemanticGate


class SCA(nn.Module):
    """
    Structure-aware Control Alignment model.

    Paper correspondence:
        1) z_u = LLM(x_u)
        2) Delta_u = W z_u
        3) c_u = Aggregate(N(u))
        4) g_u = sigmoid(W_g [e_u || c_u || Delta_u])
        5) e_tilde_u = e_u + g_u ⊙ Delta_u

    This model wraps:
        - LightGCN backbone for collaborative embeddings
        - SemanticEncoder for z_u
        - Linear projector for Delta_u
        - SemanticGate for g_u

    Args:
        backbone: LightGCN backbone
        semantic_encoder: semantic encoder module
        embedding_dim: collaborative embedding dimension D
        semantic_dim: semantic encoder output dim
        gate_hidden_dim: optional hidden dim for gate MLP
        gate_type: "vector" or "scalar"
        control_scale: optional global alpha for semantic control strength
        dropout: dropout used in projector
    """

    def __init__(
        self,
        backbone: LightGCN,
        semantic_encoder: SemanticEncoder,
        embedding_dim: int,
        semantic_dim: int,
        gate_hidden_dim: int | None = None,
        gate_type: str = "vector",
        control_scale: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.semantic_encoder = semantic_encoder
        self.embedding_dim = embedding_dim
        self.semantic_dim = semantic_dim
        self.control_scale = control_scale

        self.projector = nn.Linear(semantic_dim, embedding_dim)
        self.projector_dropout = nn.Dropout(dropout)

        self.gate = SemanticGate(
            embedding_dim=embedding_dim,
            gate_type=gate_type,
            hidden_dim=gate_hidden_dim,
            dropout=dropout,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.projector.weight)
        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)

    def get_collaborative_embeddings(
        self,
        norm_adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get propagated collaborative embeddings from backbone.

        Args:
            norm_adj: sparse normalized adjacency matrix
                      shape (num_users + num_items, num_users + num_items)

        Returns:
            user_all_embeddings: (num_users, D)
            item_all_embeddings: (num_items, D)
        """
        return self.backbone(norm_adj)

    def project_semantic_signal(self, user_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute z_u and Delta_u.

        Paper:
            z_u = LLM(x_u)
            Delta_u = W z_u

        Args:
            user_ids: (B,)

        Returns:
            z_u: (B, semantic_dim)
            delta_u: (B, D)
        """
        z_u = self.semantic_encoder(user_ids)
        delta_u = self.projector(self.projector_dropout(z_u))
        return z_u, delta_u

    def aggregate_structural_context(
        self,
        user_ids: torch.Tensor,
        item_all_embeddings: torch.Tensor,
        user_item_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate structural context c_u from user interaction neighborhood.

        Engineering assumption:
        - user_item_matrix is a sparse or dense matrix of shape (num_users, num_items)
        - c_u is computed as normalized weighted average of interacted item embeddings

        Args:
            user_ids: (B,)
            item_all_embeddings: (num_items, D)
            user_item_matrix: (num_users, num_items)

        Returns:
            c_u: (B, D)
        """
        user_item_dense = user_item_matrix.to_dense()
        user_item_sub = user_item_dense[user_ids]  # (B, num_items)

        c_u = torch.matmul(user_item_sub, item_all_embeddings)
        row_sum = user_item_sub.sum(dim=1, keepdim=True)

        row_sum = row_sum.clamp_min(1.0)
        c_u = c_u / row_sum
        return c_u

    def fuse_user_representation(
        self,
        e_u: torch.Tensor,
        c_u: torch.Tensor,
        delta_u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply semantic control injection.

        Paper:
            g_u = sigmoid(W_g [e_u || c_u || Delta_u])
            e_tilde_u = e_u + g_u ⊙ Delta_u

        Args:
            e_u: (B, D)
            c_u: (B, D)
            delta_u: (B, D)

        Returns:
            e_tilde_u: (B, D)
            g_u: (B, D) or (B, 1)
        """
        g_u = self.gate(e_u, c_u, delta_u)
        e_tilde_u = e_u + self.control_scale * (g_u * delta_u)
        return e_tilde_u, g_u

    def forward(
        self,
        norm_adj: torch.Tensor,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        user_item_matrix: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pairwise BPR training.

        Args:
            norm_adj:
                Sparse normalized bipartite adjacency matrix
                shape (num_users + num_items, num_users + num_items)

            user_ids:
                shape (B,)

            pos_item_ids:
                shape (B,)

            neg_item_ids:
                shape (B,)

            user_item_matrix:
                user-item interaction matrix
                shape (num_users, num_items)

        Returns:
            A dict containing:
                user_emb: controlled user embedding (B, D)
                pos_item_emb: positive item embedding (B, D)
                neg_item_emb: negative item embedding (B, D)
                e_u: raw collaborative user embedding (B, D)
                c_u: structural context (B, D)
                z_u: semantic representation (B, semantic_dim)
                delta_u: semantic control signal (B, D)
                g_u: gate values
                pos_scores: (B,)
                neg_scores: (B,)
                user_all_embeddings: (num_users, D)
                item_all_embeddings: (num_items, D)
        """
        user_all_embeddings, item_all_embeddings = self.get_collaborative_embeddings(norm_adj)

        e_u = user_all_embeddings[user_ids]
        pos_item_emb = item_all_embeddings[pos_item_ids]
        neg_item_emb = item_all_embeddings[neg_item_ids]

        z_u, delta_u = self.project_semantic_signal(user_ids)
        c_u = self.aggregate_structural_context(
            user_ids=user_ids,
            item_all_embeddings=item_all_embeddings,
            user_item_matrix=user_item_matrix,
        )
        user_emb, g_u = self.fuse_user_representation(e_u, c_u, delta_u)

        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=-1)

        return {
            "user_emb": user_emb,
            "pos_item_emb": pos_item_emb,
            "neg_item_emb": neg_item_emb,
            "e_u": e_u,
            "c_u": c_u,
            "z_u": z_u,
            "delta_u": delta_u,
            "g_u": g_u,
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "user_all_embeddings": user_all_embeddings,
            "item_all_embeddings": item_all_embeddings,
        }

    @torch.no_grad()
    def full_sort_predict(
        self,
        norm_adj: torch.Tensor,
        user_ids: torch.Tensor,
        user_item_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores between controlled user embeddings and all items.

        Used in evaluation.

        Args:
            norm_adj: sparse normalized adjacency matrix
            user_ids: (B,)
            user_item_matrix: (num_users, num_items)

        Returns:
            scores: (B, num_items)
        """
        user_all_embeddings, item_all_embeddings = self.get_collaborative_embeddings(norm_adj)

        e_u = user_all_embeddings[user_ids]
        _, delta_u = self.project_semantic_signal(user_ids)
        c_u = self.aggregate_structural_context(
            user_ids=user_ids,
            item_all_embeddings=item_all_embeddings,
            user_item_matrix=user_item_matrix,
        )
        user_emb, _ = self.fuse_user_representation(e_u, c_u, delta_u)

        scores = torch.matmul(user_emb, item_all_embeddings.t())
        return scores