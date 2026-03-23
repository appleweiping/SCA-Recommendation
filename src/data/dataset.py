# src/data/dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class InteractionDataBundle:
    """
    Container for all interaction-related training data.

    Attributes:
        num_users:
            Total number of users.
        num_items:
            Total number of items.

        train_pairs:
            List of (user, item) positive interactions for training.
        valid_pairs:
            List of (user, item) positive interactions for validation.
        test_pairs:
            List of (user, item) positive interactions for testing.

        train_user_pos_dict:
            Dict[user_id] -> set of positive item_ids in train set.
        valid_user_pos_dict:
            Dict[user_id] -> set of positive item_ids in valid set.
        test_user_pos_dict:
            Dict[user_id] -> set of positive item_ids in test set.

        user_item_matrix:
            Sparse user-item interaction matrix of shape (num_users, num_items).
            Used by SCA to aggregate structural context c_u.

        norm_adj:
            Sparse normalized adjacency matrix of shape
            (num_users + num_items, num_users + num_items).
            Used by LightGCN backbone.

    Shape notes:
        user_item_matrix: (U, I)
        norm_adj:         (U + I, U + I)
    """

    num_users: int
    num_items: int

    train_pairs: List[Tuple[int, int]]
    valid_pairs: List[Tuple[int, int]]
    test_pairs: List[Tuple[int, int]]

    train_user_pos_dict: Dict[int, Set[int]]
    valid_user_pos_dict: Dict[int, Set[int]]
    test_user_pos_dict: Dict[int, Set[int]]

    user_item_matrix: torch.Tensor
    norm_adj: torch.Tensor


class InteractionDataset(Dataset):
    """
    Dataset for positive user-item training pairs.

    Each sample returns one positive interaction:
        {
            "user_id": LongTensor scalar,
            "pos_item_id": LongTensor scalar
        }

    Negative sampling is NOT done here.
    It is delegated to the sampler / collate function.

    Args:
        pairs:
            List of (user_id, item_id) training positive pairs.
    """

    def __init__(self, pairs: List[Tuple[int, int]]) -> None:
        super().__init__()
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        user_id, pos_item_id = self.pairs[index]
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "pos_item_id": torch.tensor(pos_item_id, dtype=torch.long),
        }


def _read_interaction_csv(file_path: str | Path) -> List[Tuple[int, int]]:
    """
    Read interaction csv file and return list of (user, item) pairs.

    Supported columns:
        - user, item
        - user, item, rating
        - user, item, timestamp
        - user_id, item_id
        - userId, itemId

    Only user/item ids are used.

    Args:
        file_path:
            Path to csv file.

    Returns:
        pairs: List[(user_id, item_id)]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Interaction file not found: {file_path}")

    df = pd.read_csv(file_path)

    possible_user_cols = ["user", "user_id", "userId"]
    possible_item_cols = ["item", "item_id", "itemId"]

    user_col = None
    item_col = None

    for col in possible_user_cols:
        if col in df.columns:
            user_col = col
            break

    for col in possible_item_cols:
        if col in df.columns:
            item_col = col
            break

    if user_col is None or item_col is None:
        raise ValueError(
            f"Could not find user/item columns in {file_path}. "
            f"Found columns: {list(df.columns)}"
        )

    pairs = list(zip(df[user_col].astype(int).tolist(), df[item_col].astype(int).tolist()))
    return pairs


def _build_user_pos_dict(
    pairs: List[Tuple[int, int]],
    num_users: int,
) -> Dict[int, Set[int]]:
    """
    Build user -> positive item set dictionary.

    Args:
        pairs:
            List of (user_id, item_id)
        num_users:
            Total number of users

    Returns:
        dict[user_id] -> set(item_ids)
    """
    user_pos_dict: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
    for user_id, item_id in pairs:
        user_pos_dict[user_id].add(item_id)
    return user_pos_dict


def _infer_num_users_items(
    *pair_lists: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Infer number of users/items from all provided interaction pairs.

    Assumption:
        user/item ids are zero-based contiguous or at least max-id-addressable.

    Returns:
        num_users, num_items
    """
    max_user = -1
    max_item = -1
    for pairs in pair_lists:
        for u, i in pairs:
            if u > max_user:
                max_user = u
            if i > max_item:
                max_item = i

    if max_user < 0 or max_item < 0:
        raise ValueError("Empty interaction pairs; cannot infer num_users/num_items.")

    return max_user + 1, max_item + 1


def _build_sparse_user_item_matrix(
    train_pairs: List[Tuple[int, int]],
    num_users: int,
    num_items: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build sparse user-item interaction matrix R of shape (U, I).

    Values are binary 1.0 for observed interactions.

    Args:
        train_pairs:
            Training positive pairs
        num_users:
            Number of users
        num_items:
            Number of items
        device:
            Optional target device

    Returns:
        Sparse COO tensor of shape (num_users, num_items)
    """
    if len(train_pairs) == 0:
        raise ValueError("train_pairs is empty.")

    rows = torch.tensor([u for u, _ in train_pairs], dtype=torch.long)
    cols = torch.tensor([i for _, i in train_pairs], dtype=torch.long)
    vals = torch.ones(len(train_pairs), dtype=torch.float32)

    indices = torch.stack([rows, cols], dim=0)
    matrix = torch.sparse_coo_tensor(
        indices,
        vals,
        size=(num_users, num_items),
        dtype=torch.float32,
        device=device,
    )
    return matrix.coalesce()


def _build_sparse_norm_adj(
    train_pairs: List[Tuple[int, int]],
    num_users: int,
    num_items: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build symmetric normalized adjacency matrix for LightGCN.

    Graph:
        A = [[0, R],
             [R^T, 0]]

    Normalization:
        A_hat = D^{-1/2} A D^{-1/2}

    Args:
        train_pairs:
            Training positive pairs
        num_users:
            Number of users
        num_items:
            Number of items
        device:
            Optional target device

    Returns:
        Sparse normalized adjacency matrix of shape (U+I, U+I)
    """
    if len(train_pairs) == 0:
        raise ValueError("train_pairs is empty.")

    user_indices = torch.tensor([u for u, _ in train_pairs], dtype=torch.long)
    item_indices = torch.tensor([i for _, i in train_pairs], dtype=torch.long) + num_users

    # Build symmetric edges:
    # user -> item and item -> user
    row = torch.cat([user_indices, item_indices], dim=0)
    col = torch.cat([item_indices, user_indices], dim=0)
    val = torch.ones(row.size(0), dtype=torch.float32)

    num_nodes = num_users + num_items

    adj = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=val,
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
        device=device,
    ).coalesce()

    # Degree
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # (U+I,)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    indices = adj.indices()
    values = adj.values()

    row_idx = indices[0]
    col_idx = indices[1]
    norm_values = deg_inv_sqrt[row_idx] * values * deg_inv_sqrt[col_idx]

    norm_adj = torch.sparse_coo_tensor(
        indices=indices,
        values=norm_values,
        size=adj.size(),
        dtype=torch.float32,
        device=device,
    ).coalesce()

    return norm_adj


def build_interaction_data_bundle(
    train_path: str | Path,
    valid_path: Optional[str | Path] = None,
    test_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> InteractionDataBundle:
    """
    Build the full interaction data bundle from processed csv files.

    Minimal runnable convention:
        train.csv / valid.csv / test.csv
    each contain at least:
        user,item

    Args:
        train_path:
            Path to training csv
        valid_path:
            Optional path to validation csv
        test_path:
            Optional path to test csv
        device:
            Device for sparse matrices

    Returns:
        InteractionDataBundle
    """
    train_pairs = _read_interaction_csv(train_path)
    valid_pairs = _read_interaction_csv(valid_path) if valid_path is not None else []
    test_pairs = _read_interaction_csv(test_path) if test_path is not None else []

    num_users, num_items = _infer_num_users_items(train_pairs, valid_pairs, test_pairs)

    train_user_pos_dict = _build_user_pos_dict(train_pairs, num_users)
    valid_user_pos_dict = _build_user_pos_dict(valid_pairs, num_users)
    test_user_pos_dict = _build_user_pos_dict(test_pairs, num_users)

    user_item_matrix = _build_sparse_user_item_matrix(
        train_pairs=train_pairs,
        num_users=num_users,
        num_items=num_items,
        device=device,
    )

    norm_adj = _build_sparse_norm_adj(
        train_pairs=train_pairs,
        num_users=num_users,
        num_items=num_items,
        device=device,
    )

    return InteractionDataBundle(
        num_users=num_users,
        num_items=num_items,
        train_pairs=train_pairs,
        valid_pairs=valid_pairs,
        test_pairs=test_pairs,
        train_user_pos_dict=train_user_pos_dict,
        valid_user_pos_dict=valid_user_pos_dict,
        test_user_pos_dict=test_user_pos_dict,
        user_item_matrix=user_item_matrix,
        norm_adj=norm_adj,
    )