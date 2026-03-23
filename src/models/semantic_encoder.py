# src/models/semantic_encoder.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticEncoder(nn.Module):
    """
    Semantic encoder for user text profiles.

    This module corresponds to the paper's:
        z_u = LLM(x_u)

    Engineering assumption:
    - We do NOT run a full LLM inside training.
    - Instead, each user already has an offline semantic feature vector
      extracted from text profile x_u.
    - This module transforms that raw feature into train-time semantic
      representation z_u.

    Two modes are supported:
    1) Feature mode:
       user_semantic_features is provided externally via set_user_features(...)
    2) Fallback embedding mode:
       if no external features are provided, learn a user semantic embedding
       directly, only for debugging / minimal runnable pipeline.

    Args:
        num_users: Number of users.
        input_dim: Dimension of raw offline semantic features.
        output_dim: Dimension of output semantic representation z_u.
        dropout: Dropout ratio.
        use_mlp: Whether to use 2-layer MLP instead of single linear projection.
        normalize: Whether to L2-normalize z_u.
    """

    def __init__(
        self,
        num_users: int,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        use_mlp: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize

        self.dropout = nn.Dropout(dropout)

        if use_mlp:
            hidden_dim = max(output_dim, input_dim)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.encoder = nn.Linear(input_dim, output_dim)

        # Fallback trainable semantic embedding for minimal runnable setup.
        self.fallback_user_embedding = nn.Embedding(num_users, input_dim)
        nn.init.xavier_uniform_(self.fallback_user_embedding.weight)

        # External offline semantic features buffer.
        # Shape should be: (num_users, input_dim)
        self.register_buffer("user_semantic_features", None, persistent=False)

    def set_user_features(self, features: torch.Tensor) -> None:
        """
        Register offline user semantic features.

        Args:
            features: Tensor of shape (num_users, input_dim)

        Raises:
            ValueError: If shape mismatches.
        """
        if features.dim() != 2:
            raise ValueError(
                f"features must be 2D, got shape {tuple(features.shape)}"
            )
        if features.size(0) != self.num_users:
            raise ValueError(
                f"features.size(0) must equal num_users={self.num_users}, "
                f"got {features.size(0)}"
            )
        if features.size(1) != self.input_dim:
            raise ValueError(
                f"features.size(1) must equal input_dim={self.input_dim}, "
                f"got {features.size(1)}"
            )

        self.user_semantic_features = features

    def get_raw_features(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Get raw semantic features for selected users.

        Priority:
        - external offline features if available
        - otherwise fallback trainable user semantic embedding

        Args:
            user_ids: Tensor of shape (batch_size,)

        Returns:
            raw_features: Tensor of shape (batch_size, input_dim)
        """
        if self.user_semantic_features is not None:
            return self.user_semantic_features[user_ids]
        return self.fallback_user_embedding(user_ids)

    def encode(self, raw_features: torch.Tensor) -> torch.Tensor:
        """
        Transform raw semantic features into z_u.

        Args:
            raw_features: (batch_size, input_dim)

        Returns:
            z_u: (batch_size, output_dim)
        """
        z_u = self.encoder(self.dropout(raw_features))
        if self.normalize:
            z_u = F.normalize(z_u, p=2, dim=-1)
        return z_u

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: Tensor of shape (batch_size,)

        Returns:
            z_u: Semantic representation of shape (batch_size, output_dim)
        """
        raw_features = self.get_raw_features(user_ids)
        z_u = self.encode(raw_features)
        return z_u

    def get_all_user_semantics(self) -> torch.Tensor:
        """
        Get z_u for all users.

        Returns:
            all_z: Tensor of shape (num_users, output_dim)
        """
        device = next(self.parameters()).device
        user_ids = torch.arange(self.num_users, device=device)
        return self.forward(user_ids)