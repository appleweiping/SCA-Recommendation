# src/trainers/trainer_lightgcn.py
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
    Pure LightGCN trainer for baseline experiments on implicit-feedback recommendation.

    This trainer is designed to fit the current SCA-Recommendation project structure:
    - training data comes from InteractionDataset(train_pairs)
    - negative sampling is handled by BPRTrainCollator
    - model is src.models.lightgcn.LightGCN
    - graph input is the sparse normalized adjacency matrix norm_adj

    Training objective:
        BPR loss + L2 regularization on ego embeddings

    Returned logs are intentionally simple so they can be printed directly
    or saved later into log.txt / metrics.json by run.py.
    """

    def __init__(
        self,
        model: nn.Module,
        train_pairs: List[tuple[int, int]],
        user_pos_dict: Dict[int, set[int]],
        norm_adj: torch.Tensor,
        num_users: int,
        num_items: int,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        super().__init__()

        self.model = model.to(device)
        self.train_pairs = train_pairs
        self.user_pos_dict = user_pos_dict
        self.norm_adj = norm_adj.coalesce().to(device)
        self.num_users = num_users
        self.num_items = num_items
        self.config = config
        self.device = device

        train_cfg = config.get("train", {})
        model_cfg = config.get("model", {})

        self.batch_size = int(train_cfg.get("batch_size", 1024))
        self.lr = float(train_cfg.get("lr", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-6))
        self.num_workers = int(train_cfg.get("num_workers", 0))
        self.embedding_dim = int(model_cfg.get("embedding_dim", 64))

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.0,  # keep explicit reg term for classic LightGCN-style BPR
        )

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
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collator,
            drop_last=False,
        )

    def _compute_all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN.forward(norm_adj) returns:
            user_all_embeddings: (num_users, embedding_dim)
            item_all_embeddings: (num_items, embedding_dim)
        """
        user_all_embeddings, item_all_embeddings = self.model(self.norm_adj)
        return user_all_embeddings, item_all_embeddings

    @staticmethod
    def _bpr_loss(
        user_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard BPR loss.
        """
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
        """
        L2 regularization on ego embeddings (raw trainable embeddings),
        which is the usual choice in LightGCN-style training.
        """
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
        """
        Minimal sanity check before formal training.

        Useful for verifying:
        - batch keys
        - embedding shapes
        - score ranges
        - whether pos_score > neg_score is at least numerically valid
        """
        self.model.eval()

        batch = next(iter(self.train_loader))
        user_ids = batch["user_ids"].to(self.device)
        pos_item_ids = batch["pos_item_ids"].to(self.device)
        neg_item_ids = batch["neg_item_ids"].to(self.device)

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
        """
        Train LightGCN for one epoch.

        Returns averaged epoch-level statistics.
        """
        self.model.train()
        start_time = time.time()

        total_loss = 0.0
        total_bpr_loss = 0.0
        total_reg_loss = 0.0
        total_pos_mean = 0.0
        total_neg_mean = 0.0
        total_pos_gt_neg = 0.0
        num_batches = 0

        for batch in self.train_loader:
            user_ids = batch["user_ids"].to(self.device)
            pos_item_ids = batch["pos_item_ids"].to(self.device)
            neg_item_ids = batch["neg_item_ids"].to(self.device)

            user_all_embeddings, item_all_embeddings = self._compute_all_embeddings()

            user_emb = user_all_embeddings[user_ids]
            pos_emb = item_all_embeddings[pos_item_ids]
            neg_emb = item_all_embeddings[neg_item_ids]

            bpr_loss, pos_scores, neg_scores = self._bpr_loss(user_emb, pos_emb, neg_emb)
            reg_loss = self._l2_reg_loss(user_ids, pos_item_ids, neg_item_ids)

            loss = bpr_loss + self.weight_decay * reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_bpr_loss += float(bpr_loss.item())
            total_reg_loss += float(reg_loss.item())
            total_pos_mean += float(pos_scores.mean().item())
            total_neg_mean += float(neg_scores.mean().item())
            total_pos_gt_neg += float((pos_scores > neg_scores).float().mean().item())
            num_batches += 1

        epoch_time = time.time() - start_time
        denom = max(num_batches, 1)

        return {
            "epoch": float(epoch),
            "loss": total_loss / denom,
            "bpr_loss": total_bpr_loss / denom,
            "reg_loss": total_reg_loss / denom,
            "pos_mean": total_pos_mean / denom,
            "neg_mean": total_neg_mean / denom,
            "pos_gt_neg": total_pos_gt_neg / denom,
            "epoch_time_sec": epoch_time,
        }

    def fit(self, num_epochs: int) -> List[Dict[str, float]]:
        """
        Optional convenience wrapper.
        If run.py already loops externally by epoch, this method can be unused.
        """
        history: List[Dict[str, float]] = []
        for epoch in range(1, num_epochs + 1):
            stats = self.train_one_epoch(epoch)
            history.append(stats)
        return history