# src/trainers/trainer_sca.py

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.data.dataset import InteractionDataBundle, InteractionDataset
from src.data.sampler import BPRTrainCollator
from src.models.losses import SCALoss
from src.trainers.trainer_base import BaseTrainer


class SCATrainer(BaseTrainer):
    """
    Minimal runnable trainer for SCA.

    Current version:
    - standard per-batch recompute for collaborative embeddings
    - no retain_graph
    - adds debug statistics into epoch metrics
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data_bundle: InteractionDataBundle,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device = "cpu",
        num_workers: int = 0,
        shuffle: bool = True,
        scheduler: Optional[object] = None,
        grad_clip_norm: Optional[float] = None,
        save_dir: Optional[str] = None,
        lambda_align: float = 0.1,
        lambda_reg: float = 1e-4,
        align_type: str = "cosine",
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

        self.data_bundle = data_bundle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.criterion = SCALoss(
            lambda_align=lambda_align,
            lambda_reg=lambda_reg,
            align_type=align_type,
        ).to(self.device)

        self.norm_adj = data_bundle.norm_adj.to(self.device)
        self.user_item_matrix = data_bundle.user_item_matrix.to(self.device)

        self.train_dataset = InteractionDataset(data_bundle.train_pairs)
        self.train_collator = BPRTrainCollator(
            num_items=data_bundle.num_items,
            user_pos_dict=data_bundle.train_user_pos_dict,
            num_negatives=1,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.train_collator,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def _forward_one_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward one batch using per-batch recompute.

        Input batch dict:
            user_ids:     (B,)
            pos_item_ids: (B,)
            neg_item_ids: (B,)

        Returns dict:
            user_emb:     (B, D)
            pos_item_emb: (B, D)
            neg_item_emb: (B, D)
            e_u:          (B, D)
            c_u:          (B, D)
            z_u:          (B, semantic_dim)
            delta_u:      (B, D)
            g_u:          (B, D) or (B, 1)
            pos_scores:   (B,)
            neg_scores:   (B,)
        """
        user_ids = batch["user_ids"]          # (B,)
        pos_item_ids = batch["pos_item_ids"]  # (B,)
        neg_item_ids = batch["neg_item_ids"]  # (B,)

        # 1) recompute collaborative embeddings every batch
        user_all_embeddings, item_all_embeddings = self.model.get_collaborative_embeddings(
            self.norm_adj
        )

        # 2) index collaborative embeddings
        e_u = user_all_embeddings[user_ids]               # (B, D)
        pos_item_emb = item_all_embeddings[pos_item_ids]  # (B, D)
        neg_item_emb = item_all_embeddings[neg_item_ids]  # (B, D)

        # 3) semantic branch
        z_u, delta_u = self.model.project_semantic_signal(user_ids)

        # 4) structural context
        c_u = self.model.aggregate_structural_context(
            user_ids=user_ids,
            item_all_embeddings=item_all_embeddings,
            user_item_matrix=self.user_item_matrix,
        )

        # 5) control injection
        user_emb, g_u = self.model.fuse_user_representation(
            e_u=e_u,
            c_u=c_u,
            delta_u=delta_u,
        )

        # 6) pairwise scores
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)  # (B,)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=-1)  # (B,)

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
        }

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch and return scalar metrics.

        Added debug stats:
            - pos_scores_mean
            - neg_scores_mean
            - pos_gt_neg_ratio
            - delta_abs_mean
            - gate_mean
            - gate_std
            - control_shift_mean = (user_emb - e_u).abs().mean()
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_bpr = 0.0
        total_align = 0.0
        total_reg = 0.0
        total_examples = 0

        # debug stats
        total_pos_scores_mean = 0.0
        total_neg_scores_mean = 0.0
        total_pos_gt_neg_ratio = 0.0
        total_delta_abs_mean = 0.0
        total_gate_mean = 0.0
        total_gate_std = 0.0
        total_control_shift_mean = 0.0

        num_batches = len(self.train_loader)
        if num_batches == 0:
            raise ValueError("train_loader is empty; cannot train.")

        for batch in self.train_loader:
            batch = self.move_batch_to_device(batch)
            batch_size = batch["user_ids"].size(0)
            total_examples += batch_size

            outputs = self._forward_one_batch(batch=batch)

            loss_dict = self.criterion(
                outputs["pos_scores"],
                outputs["neg_scores"],
                outputs["delta_u"],
                outputs["c_u"],
                outputs["user_emb"],
                outputs["pos_item_emb"],
                outputs["neg_item_emb"],
            )

            loss = loss_dict["loss"]
            loss.backward()

            self.clip_gradients()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # base losses
            total_loss += float(loss.detach().item()) * batch_size
            total_bpr += float(loss_dict["bpr_loss"].detach().item()) * batch_size
            total_align += float(loss_dict["align_loss"].detach().item()) * batch_size
            total_reg += float(loss_dict["reg_loss"].detach().item()) * batch_size

            # debug stats
            pos_scores_mean = outputs["pos_scores"].detach().mean().item()
            neg_scores_mean = outputs["neg_scores"].detach().mean().item()
            pos_gt_neg_ratio = (
                (outputs["pos_scores"].detach() > outputs["neg_scores"].detach())
                .float()
                .mean()
                .item()
            )
            delta_abs_mean = outputs["delta_u"].detach().abs().mean().item()
            gate_mean = outputs["g_u"].detach().mean().item()
            gate_std = outputs["g_u"].detach().std().item()
            control_shift_mean = (
                (outputs["user_emb"].detach() - outputs["e_u"].detach())
                .abs()
                .mean()
                .item()
            )

            total_pos_scores_mean += pos_scores_mean * batch_size
            total_neg_scores_mean += neg_scores_mean * batch_size
            total_pos_gt_neg_ratio += pos_gt_neg_ratio * batch_size
            total_delta_abs_mean += delta_abs_mean * batch_size
            total_gate_mean += gate_mean * batch_size
            total_gate_std += gate_std * batch_size
            total_control_shift_mean += control_shift_mean * batch_size

        self.step_scheduler()

        metrics = {
            "epoch": float(epoch),
            "loss": total_loss / max(total_examples, 1),
            "bpr_loss": total_bpr / max(total_examples, 1),
            "align_loss": total_align / max(total_examples, 1),
            "reg_loss": total_reg / max(total_examples, 1),
            "num_examples": float(total_examples),
            "num_batches": float(num_batches),
            # debug metrics
            "pos_scores_mean": total_pos_scores_mean / max(total_examples, 1),
            "neg_scores_mean": total_neg_scores_mean / max(total_examples, 1),
            "pos_gt_neg_ratio": total_pos_gt_neg_ratio / max(total_examples, 1),
            "delta_abs_mean": total_delta_abs_mean / max(total_examples, 1),
            "gate_mean": total_gate_mean / max(total_examples, 1),
            "gate_std": total_gate_std / max(total_examples, 1),
            "control_shift_mean": total_control_shift_mean / max(total_examples, 1),
        }
        return metrics

    @torch.no_grad()
    def inspect_one_batch(self) -> Dict[str, tuple]:
        """
        Optional debugging helper:
        Returns tensor shapes from one batch forward pass.
        """
        self.model.eval()

        batch = next(iter(self.train_loader))
        batch = self.move_batch_to_device(batch)

        outputs = self._forward_one_batch(batch=batch)

        shape_dict = {
            "user_ids": tuple(batch["user_ids"].shape),
            "pos_item_ids": tuple(batch["pos_item_ids"].shape),
            "neg_item_ids": tuple(batch["neg_item_ids"].shape),
            "user_emb": tuple(outputs["user_emb"].shape),
            "pos_item_emb": tuple(outputs["pos_item_emb"].shape),
            "neg_item_emb": tuple(outputs["neg_item_emb"].shape),
            "e_u": tuple(outputs["e_u"].shape),
            "c_u": tuple(outputs["c_u"].shape),
            "z_u": tuple(outputs["z_u"].shape),
            "delta_u": tuple(outputs["delta_u"].shape),
            "g_u": tuple(outputs["g_u"].shape),
            "pos_scores": tuple(outputs["pos_scores"].shape),
            "neg_scores": tuple(outputs["neg_scores"].shape),
        }
        return shape_dict