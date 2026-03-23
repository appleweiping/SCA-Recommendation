# src/data/sampler.py

from __future__ import annotations

import random
from typing import Dict, List, Set

import torch


class BPRTrainCollator:
    """
    Collate function for BPR training.

    Input:
        A batch from InteractionDataset, where each element is:
            {
                "user_id": LongTensor scalar,
                "pos_item_id": LongTensor scalar
            }

    Output:
        {
            "user_ids":     LongTensor of shape (B,)
            "pos_item_ids": LongTensor of shape (B,)
            "neg_item_ids": LongTensor of shape (B,)
        }

    Negative sampling rule:
        For each (u, i_pos), randomly sample one i_neg from [0, num_items),
        such that i_neg not in user's positive interaction set.

    Engineering default:
        - Uniform negative sampling
        - 1 negative per positive
        - Uses train_user_pos_dict only

    Args:
        num_items:
            Total number of items
        user_pos_dict:
            Dict[user_id] -> set(item_ids) from training interactions
        num_negatives:
            Number of negatives per positive. For minimal runnable BPR,
            this should stay 1.
    """

    def __init__(
        self,
        num_items: int,
        user_pos_dict: Dict[int, Set[int]],
        num_negatives: int = 1,
    ) -> None:
        if num_negatives != 1:
            raise ValueError(
                "This minimal BPR collator currently supports only num_negatives=1."
            )

        self.num_items = num_items
        self.user_pos_dict = user_pos_dict
        self.num_negatives = num_negatives

    def _sample_one_negative(self, user_id: int) -> int:
        """
        Sample one negative item for the given user.

        Args:
            user_id: int

        Returns:
            neg_item_id: int
        """
        pos_items = self.user_pos_dict[user_id]

        if len(pos_items) >= self.num_items:
            raise ValueError(
                f"User {user_id} has interacted with all items; cannot sample negative."
            )

        while True:
            neg_item_id = random.randint(0, self.num_items - 1)
            if neg_item_id not in pos_items:
                return neg_item_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Convert a list of positive samples into a BPR training batch.

        Args:
            batch:
                List of dicts from InteractionDataset.__getitem__()

        Returns:
            batch_dict with:
                user_ids:     (B,)
                pos_item_ids: (B,)
                neg_item_ids: (B,)
        """
        user_ids: List[int] = []
        pos_item_ids: List[int] = []
        neg_item_ids: List[int] = []

        for sample in batch:
            user_id = int(sample["user_id"].item())
            pos_item_id = int(sample["pos_item_id"].item())
            neg_item_id = self._sample_one_negative(user_id)

            user_ids.append(user_id)
            pos_item_ids.append(pos_item_id)
            neg_item_ids.append(neg_item_id)

        batch_dict = {
            "user_ids": torch.tensor(user_ids, dtype=torch.long),
            "pos_item_ids": torch.tensor(pos_item_ids, dtype=torch.long),
            "neg_item_ids": torch.tensor(neg_item_ids, dtype=torch.long),
        }
        return batch_dict