from __future__ import annotations

import time
from typing import Any, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.trainers.trainer_base import BaseTrainer
from src.data.dataset import InteractionDataset
from src.data.sampler import BPRTrainCollator


class LightGCNTrainer(BaseTrainer):
    """
    Pure LightGCN trainer for baseline experiments.

    This trainer is aligned with the current project structure:
    - BaseTrainer handles device / checkpoint / gradient clipping utilities
    - InteractionDataset(train_pairs) provides (user_id, pos_item_id)
    - BPRTrainCollator samples one negative item per positive interaction
    - LightGCN.forward(norm_adj) returns (user_all_embeddings, item_all_embeddings)
    """

    def __init__(
        self,
        model: nn.Module,
        train_pairs: List[tuple[int, int]],
        user_pos_dict: Dict[int, set[int]],
        norm_adj: torch.Tensor,
        num_users: int,
        num_items: int,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device = "cpu",
        num_workers: int = 0,
        shuffle: bool = True,
        scheduler: Any | None = None,
        grad_clip_norm: float | None = None,
        save_dir: str | None = None,
        weight_decay: float = 1e-4,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            grad_clip_norm=grad_clip_norm,
            save_dir=save_dir,
        )

        self.train_pairs = train_pairs
        self.user_pos_dict = user_pos_dict
        self.norm_adj = norm_adj.coalesce().to(self.device)
        self.num_users = num_users
        self.num_items = num_items

        self.num_workers = num_workers
        self.shuffle = shuffle
        self.weight_decay = weight_decay
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.max_batches_per_epoch = None

        self.train_loader = self._build_train_loader()

    def _build_train_loader(self) -> DataLoader:
        dataset = InteractionDataset(self.train_pairs)
        collator = BPRTrainCollator(
            num_items=self.num_items,
            user_pos_dict=self.user_pos_dict,
            num_negatives=1,
        )

        return DataLoader(
            dataset,
            batch_size=1024 if len(dataset) > 1024 else max(len(dataset), 1),
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def set_batch_size(self, batch_size: int) -> None:
        """
        Rebuild train loader with a user-specified batch size.
        Useful because current run.py passes batch_size externally.
        """
        dataset = InteractionDataset(self.train_pairs)
        collator = BPRTrainCollator(
            num_items=self.num_items,
            user_pos_dict=self.user_pos_dict,
            num_negatives=1,
        )

        self.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
    
    def set_max_batches_per_epoch(self, max_batches: int | None) -> None:
        self.max_batches_per_epoch = max_batches

    def _compute_all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_all_embeddings, item_all_embeddings = self.model(self.norm_adj)
        return user_all_embeddings, item_all_embeddings

    @staticmethod
    def _bpr_loss(
        user_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_scores = torch.sum(user_emb * pos_emb, dim=-1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=-1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()
        return loss, pos_scores, neg_scores

    def _l2_reg_loss(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_emb_ego = self.model.user_embedding(user_ids)
        pos_emb_ego = self.model.item_embedding(pos_item_ids)
        neg_emb_ego = self.model.item_embedding(neg_item_ids)

        reg = (
            user_emb_ego.pow(2).sum(dim=1)
            + pos_emb_ego.pow(2).sum(dim=1)
            + neg_emb_ego.pow(2).sum(dim=1)
        ).mean()

        return 0.5 * reg

    def inspect_one_batch(self) -> Dict[str, Any]:
        self.model.eval()

        batch = next(iter(self.train_loader))
        batch = self.move_batch_to_device(batch)

        user_ids = batch["user_ids"]
        pos_item_ids = batch["pos_item_ids"]
        neg_item_ids = batch["neg_item_ids"]

        with torch.no_grad():
            user_all_embeddings, item_all_embeddings = self._compute_all_embeddings()

            user_emb = user_all_embeddings[user_ids]
            pos_emb = item_all_embeddings[pos_item_ids]
            neg_emb = item_all_embeddings[neg_item_ids]

            bpr_loss, pos_scores, neg_scores = self._bpr_loss(user_emb, pos_emb, neg_emb)

        return {
            "batch_size": int(user_ids.shape[0]),
            "user_ids_shape": tuple(user_ids.shape),
            "pos_item_ids_shape": tuple(pos_item_ids.shape),
            "neg_item_ids_shape": tuple(neg_item_ids.shape),
            "user_all_embeddings_shape": tuple(user_all_embeddings.shape),
            "item_all_embeddings_shape": tuple(item_all_embeddings.shape),
            "user_emb_shape": tuple(user_emb.shape),
            "pos_emb_shape": tuple(pos_emb.shape),
            "neg_emb_shape": tuple(neg_emb.shape),
            "bpr_loss": float(bpr_loss.item()),
            "pos_mean": float(pos_scores.mean().item()),
            "neg_mean": float(neg_scores.mean().item()),
            "pos_gt_neg": float((pos_scores > neg_scores).float().mean().item()),
        }

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        start_time = time.time()

        total_loss = 0.0
        total_bpr_loss = 0.0
        total_reg_loss = 0.0
        total_pos_mean = 0.0
        total_neg_mean = 0.0
        total_pos_gt_neg = 0.0
        num_batches = 0



        for batch_idx, batch in enumerate(self.train_loader):
            if self.max_batches_per_epoch is not None and batch_idx >= self.max_batches_per_epoch:
                break
            batch = self.move_batch_to_device(batch)

            user_ids = batch["user_ids"]
            pos_item_ids = batch["pos_item_ids"]
            neg_item_ids = batch["neg_item_ids"]

            user_all_embeddings, item_all_embeddings = self._compute_all_embeddings()

            user_emb = user_all_embeddings[user_ids]
            pos_emb = item_all_embeddings[pos_item_ids]
            neg_emb = item_all_embeddings[neg_item_ids]

            bpr_loss, pos_scores, neg_scores = self._bpr_loss(user_emb, pos_emb, neg_emb)
            reg_loss = self._l2_reg_loss(user_ids, pos_item_ids, neg_item_ids)

            loss = bpr_loss + self.weight_decay * reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.clip_gradients()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_bpr_loss += float(bpr_loss.item())
            total_reg_loss += float(reg_loss.item())
            total_pos_mean += float(pos_scores.mean().item())
            total_neg_mean += float(neg_scores.mean().item())
            total_pos_gt_neg += float((pos_scores > neg_scores).float().mean().item())
            num_batches += 1

        self.step_scheduler()

        epoch_time = time.time() - start_time
        denom = max(num_batches, 1)

        return {
            "epoch": float(epoch),
            "loss": total_loss / denom,
            "bpr_loss": total_bpr_loss / denom,
            "reg_loss": total_reg_loss / denom,
            "pos_scores_mean": total_pos_mean / denom,
            "neg_scores_mean": total_neg_mean / denom,
            "pos_gt_neg_ratio": total_pos_gt_neg / denom,
            "epoch_time_sec": epoch_time,
        }

    def fit(self, num_epochs: int) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for epoch in range(1, num_epochs + 1):
            stats = self.train_one_epoch(epoch)
            history.append(stats)
        return history