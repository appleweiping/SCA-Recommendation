# src/models/lightgcn.py

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """
    LightGCN backbone for collaborative filtering.

    This module learns collaborative user/item embeddings from a bipartite
    interaction graph and returns final propagated representations.

    Expected usage:
        model = LightGCN(
            num_users=...,
            num_items=...,
            embedding_dim=64,
            num_layers=3,
        )
        user_all, item_all = model(norm_adj)

    Args:
        num_users: Number of users.
        num_items: Number of items.
        embedding_dim: Embedding dimension for users/items.
        num_layers: Number of graph propagation layers.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_ego_embeddings(self) -> torch.Tensor:
        """
        Returns concatenated raw embeddings [user_emb; item_emb].

        Output:
            Tensor of shape (num_users + num_items, embedding_dim)
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        return torch.cat([user_emb, item_emb], dim=0)

    def propagate(self, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        Perform LightGCN propagation and layer-wise averaging.

        Args:
            norm_adj:
                Sparse normalized adjacency matrix of shape
                (num_users + num_items, num_users + num_items).

        Returns:
            all_emb:
                Final propagated embeddings of shape
                (num_users + num_items, embedding_dim)
        """
        if not norm_adj.is_sparse:
            raise ValueError("norm_adj must be a sparse torch tensor.")

        emb_0 = self.get_ego_embeddings()
        embs = [emb_0]

        emb_k = emb_0
        for _ in range(self.num_layers):
            emb_k = torch.sparse.mm(norm_adj, emb_k)
            embs.append(emb_k)

        all_emb = torch.stack(embs, dim=0).mean(dim=0)
        return all_emb

    def forward(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            norm_adj:
                Sparse normalized adjacency matrix.

        Returns:
            user_all_embeddings:
                Shape (num_users, embedding_dim)
            item_all_embeddings:
                Shape (num_items, embedding_dim)
        """
        all_emb = self.propagate(norm_adj)
        user_all_embeddings, item_all_embeddings = torch.split(
            all_emb, [self.num_users, self.num_items], dim=0
        )
        return user_all_embeddings, item_all_embeddings

    def get_user_item_embeddings(
        self,
        norm_adj: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch propagated embeddings for a batch of users and items.

        Args:
            norm_adj: Sparse normalized adjacency matrix.
            user_ids: Tensor of shape (batch_size,)
            item_ids: Tensor of shape (batch_size,)

        Returns:
            user_emb: (batch_size, embedding_dim)
            item_emb: (batch_size, embedding_dim)
        """
        user_all_embeddings, item_all_embeddings = self.forward(norm_adj)
        user_emb = user_all_embeddings[user_ids]
        item_emb = item_all_embeddings[item_ids]
        return user_emb, item_emb

    def score(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dot-product scoring.

        Args:
            user_emb: (batch_size, embedding_dim)
            item_emb: (batch_size, embedding_dim)

        Returns:
            scores: (batch_size,)
        """
        return torch.sum(user_emb * item_emb, dim=-1)

    def full_sort_scores(
        self,
        norm_adj: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores between a batch of users and all items.

        Args:
            norm_adj: Sparse normalized adjacency matrix.
            user_ids: (batch_size,)

        Returns:
            scores: (batch_size, num_items)
        """
        user_all_embeddings, item_all_embeddings = self.forward(norm_adj)
        user_emb = user_all_embeddings[user_ids]  # (B, D)
        scores = torch.matmul(user_emb, item_all_embeddings.t())  # (B, I)
        return scores