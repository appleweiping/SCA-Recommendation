# src/models/gate.py

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticGate(nn.Module):
    """
    Semantic gate for structure-aware control injection.

    Paper correspondence:
        g_u = sigmoid(W_g [e_u || c_u || Delta_u])

    Inputs:
        e_u:     collaborative user embedding, shape (B, D)
        c_u:     structural context embedding, shape (B, D)
        delta_u: semantic control signal, shape (B, D)

    Output:
        g_u: gate values, shape (B, D) if gate_type='vector'
             or shape (B, 1) if gate_type='scalar'
    """

    def __init__(
        self,
        embedding_dim: int,
        gate_type: str = "vector",
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if gate_type not in {"vector", "scalar"}:
            raise ValueError("gate_type must be either 'vector' or 'scalar'")

        self.embedding_dim = embedding_dim
        self.gate_type = gate_type
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        input_dim = embedding_dim * 3
        output_dim = embedding_dim if gate_type == "vector" else 1

        if hidden_dim is None:
            self.gate_layer = nn.Linear(input_dim, output_dim)
        else:
            self.gate_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        self.activation = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        e_u: torch.Tensor,
        c_u: torch.Tensor,
        delta_u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gate value.

        Args:
            e_u: Collaborative user embedding, shape (B, D)
            c_u: Structural context embedding, shape (B, D)
            delta_u: Semantic control signal, shape (B, D)

        Returns:
            gate: shape (B, D) for vector gate, or (B, 1) for scalar gate
        """
        if e_u.shape != c_u.shape or e_u.shape != delta_u.shape:
            raise ValueError(
                f"Shape mismatch: e_u={e_u.shape}, c_u={c_u.shape}, delta_u={delta_u.shape}"
            )

        gate_input = torch.cat([e_u, c_u, delta_u], dim=-1)
        gate_input = self.dropout(gate_input)
        gate = self.activation(self.gate_layer(gate_input))
        return gate