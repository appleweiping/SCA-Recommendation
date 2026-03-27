from __future__ import annotations

import random
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.data.dataset import InteractionDataBundle, InteractionDataset
from src.data.sampler import BPRTrainCollator
from src.models.losses import SCALoss
from src.trainers.trainer_base import BaseTrainer


class SCATrainer(BaseTrainer):
    """
    Accelerated SCA trainer.

    Key change:
    - compute full-graph collaborative / semantic / control representations
      ONCE per epoch
    - then score train pairs in chunks
    - accumulate one epoch loss
    - backward ONLY once per epoch

    This is much faster than recomputing the whole graph every batch.
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

        # precompute user degree once
        self.user_degree = torch.sparse.sum(self.user_item_matrix, dim=1).to_dense().unsqueeze(1)
        self.user_degree = self.user_degree.clamp_min(1.0).to(self.device)

        self.num_users = data_bundle.num_users
        self.num_items = data_bundle.num_items

        self.train_pairs = data_bundle.train_pairs
        self.train_user_pos_dict = data_bundle.train_user_pos_dict

        # for optional quick smoke test
        self.max_batches_per_epoch: Optional[int] = None

        # keep dataloader only for inspect_one_batch
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

    def set_max_batches_per_epoch(self, max_batches: Optional[int]) -> None:
        self.max_batches_per_epoch = max_batches

    def _sample_negative_items(
        self,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample one negative item for each user in the given user_ids.

        This runs once per epoch, not once per mini-batch.
        """
        neg_items = []
        for u in user_ids.tolist():
            pos_set = self.train_user_pos_dict[u]
            neg = random.randrange(self.num_items)
            while neg in pos_set:
                neg = random.randrange(self.num_items)
            neg_items.append(neg)

        return torch.tensor(neg_items, dtype=torch.long, device=self.device)

    def _prepare_epoch_pairs(self) -> Dict[str, torch.Tensor]:
        """
        Build training triples for the whole epoch once.
        Optionally truncate for smoke test using max_batches_per_epoch.
        """
        if self.max_batches_per_epoch is None:
            pairs = self.train_pairs
        else:
            max_pairs = min(len(self.train_pairs), self.max_batches_per_epoch * self.batch_size)
            pairs = self.train_pairs[:max_pairs]

        user_ids = torch.tensor([u for u, _ in pairs], dtype=torch.long, device=self.device)
        pos_item_ids = torch.tensor([i for _, i in pairs], dtype=torch.long, device=self.device)
        neg_item_ids = self._sample_negative_items(user_ids)

        if self.shuffle and user_ids.numel() > 1:
            perm = torch.randperm(user_ids.size(0), device=self.device)
            user_ids = user_ids[perm]
            pos_item_ids = pos_item_ids[perm]
            neg_item_ids = neg_item_ids[perm]

        return {
            "user_ids": user_ids,
            "pos_item_ids": pos_item_ids,
            "neg_item_ids": neg_item_ids,
        }

    def _forward_all_users(self) -> Dict[str, torch.Tensor]:
        """
        Full-graph forward ONCE per epoch.
        """
        all_user_ids = torch.arange(self.num_users, device=self.device)

        # collaborative embeddings
        user_all_embeddings, item_all_embeddings = self.model.get_collaborative_embeddings(
            self.norm_adj
        )

        # semantic branch for all users
        z_all, delta_all = self.model.project_semantic_signal(all_user_ids)

        # structural context for all users
        c_all = self.model.aggregate_structural_context(
            user_ids=all_user_ids,
            item_all_embeddings=item_all_embeddings,
            user_item_matrix=self.user_item_matrix,
            user_degree=self.user_degree,
        )

        # control injection for all users
        user_emb_all, g_all = self.model.fuse_user_representation(
            e_u=user_all_embeddings,
            c_u=c_all,
            delta_u=delta_all,
        )

        return {
            "user_emb_all": user_emb_all,
            "item_all_embeddings": item_all_embeddings,
            "e_all": user_all_embeddings,
            "c_all": c_all,
            "z_all": z_all,
            "delta_all": delta_all,
            "g_all": g_all,
        }

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One epoch = one full-graph forward + one optimizer step.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch_pairs = self._prepare_epoch_pairs()
        user_ids_all = epoch_pairs["user_ids"]
        pos_item_ids_all = epoch_pairs["pos_item_ids"]
        neg_item_ids_all = epoch_pairs["neg_item_ids"]

        total_examples = int(user_ids_all.size(0))
        if total_examples == 0:
            raise ValueError("No training pairs available for this epoch.")

        # full forward once
        full_outputs = self._forward_all_users()

        user_emb_all = full_outputs["user_emb_all"]
        item_all_embeddings = full_outputs["item_all_embeddings"]
        e_all = full_outputs["e_all"]
        c_all = full_outputs["c_all"]
        delta_all = full_outputs["delta_all"]
        g_all = full_outputs["g_all"]

        # accumulate tensor loss for one backward
        total_loss_tensor = torch.zeros([], device=self.device)

        # scalar stats
        total_loss = 0.0
        total_bpr = 0.0
        total_align = 0.0
        total_reg = 0.0

        total_pos_scores_mean = 0.0
        total_neg_scores_mean = 0.0
        total_pos_gt_neg_ratio = 0.0
        total_delta_abs_mean = 0.0
        total_gate_mean = 0.0
        total_gate_std = 0.0
        total_control_shift_mean = 0.0

        num_chunks = 0

        for start in range(0, total_examples, self.batch_size):
            end = min(start + self.batch_size, total_examples)

            user_ids = user_ids_all[start:end]
            pos_item_ids = pos_item_ids_all[start:end]
            neg_item_ids = neg_item_ids_all[start:end]

            user_emb = user_emb_all[user_ids]
            pos_item_emb = item_all_embeddings[pos_item_ids]
            neg_item_emb = item_all_embeddings[neg_item_ids]

            e_u = e_all[user_ids]
            c_u = c_all[user_ids]
            delta_u = delta_all[user_ids]
            g_u = g_all[user_ids]

            pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
            neg_scores = torch.sum(user_emb * neg_item_emb, dim=-1)

            loss_dict = self.criterion(
                pos_scores,
                neg_scores,
                delta_u,
                c_u,
                user_emb,
                pos_item_emb,
                neg_item_emb,
            )

            chunk_size = end - start
            weight = chunk_size / total_examples
            total_loss_tensor = total_loss_tensor + loss_dict["loss"] * weight

            # detached scalar stats
            total_loss += float(loss_dict["loss"].detach().item()) * chunk_size
            total_bpr += float(loss_dict["bpr_loss"].detach().item()) * chunk_size
            total_align += float(loss_dict["align_loss"].detach().item()) * chunk_size
            total_reg += float(loss_dict["reg_loss"].detach().item()) * chunk_size

            pos_scores_mean = pos_scores.detach().mean().item()
            neg_scores_mean = neg_scores.detach().mean().item()
            pos_gt_neg_ratio = (pos_scores.detach() > neg_scores.detach()).float().mean().item()
            delta_abs_mean = delta_u.detach().abs().mean().item()
            gate_mean = g_u.detach().mean().item()
            gate_std = g_u.detach().std().item()
            control_shift_mean = (user_emb.detach() - e_u.detach()).abs().mean().item()

            total_pos_scores_mean += pos_scores_mean * chunk_size
            total_neg_scores_mean += neg_scores_mean * chunk_size
            total_pos_gt_neg_ratio += pos_gt_neg_ratio * chunk_size
            total_delta_abs_mean += delta_abs_mean * chunk_size
            total_gate_mean += gate_mean * chunk_size
            total_gate_std += gate_std * chunk_size
            total_control_shift_mean += control_shift_mean * chunk_size

            num_chunks += 1

        # one backward only
        total_loss_tensor.backward()
        self.clip_gradients()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step_scheduler()

        denom = max(total_examples, 1)

        metrics = {
            "epoch": float(epoch),
            "loss": total_loss / denom,
            "bpr_loss": total_bpr / denom,
            "align_loss": total_align / denom,
            "reg_loss": total_reg / denom,
            "num_examples": float(total_examples),
            "num_batches": float(num_chunks),
            "pos_scores_mean": total_pos_scores_mean / denom,
            "neg_scores_mean": total_neg_scores_mean / denom,
            "pos_gt_neg_ratio": total_pos_gt_neg_ratio / denom,
            "delta_abs_mean": total_delta_abs_mean / denom,
            "gate_mean": total_gate_mean / denom,
            "gate_std": total_gate_std / denom,
            "control_shift_mean": total_control_shift_mean / denom,
        }
        return metrics

    @torch.no_grad()
    def inspect_one_batch(self) -> Dict[str, tuple]:
        """
        Debug helper using one sampled batch.
        """
        self.model.eval()

        batch = next(iter(self.train_loader))
        batch = self.move_batch_to_device(batch)

        full_outputs = self._forward_all_users()

        user_ids = batch["user_ids"]
        pos_item_ids = batch["pos_item_ids"]
        neg_item_ids = batch["neg_item_ids"]

        user_emb = full_outputs["user_emb_all"][user_ids]
        pos_item_emb = full_outputs["item_all_embeddings"][pos_item_ids]
        neg_item_emb = full_outputs["item_all_embeddings"][neg_item_ids]
        e_u = full_outputs["e_all"][user_ids]
        c_u = full_outputs["c_all"][user_ids]
        z_u = full_outputs["z_all"][user_ids]
        delta_u = full_outputs["delta_all"][user_ids]
        g_u = full_outputs["g_all"][user_ids]
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=-1)

        shape_dict = {
            "user_ids": tuple(batch["user_ids"].shape),
            "pos_item_ids": tuple(batch["pos_item_ids"].shape),
            "neg_item_ids": tuple(batch["neg_item_ids"].shape),
            "user_emb": tuple(user_emb.shape),
            "pos_item_emb": tuple(pos_item_emb.shape),
            "neg_item_emb": tuple(neg_item_emb.shape),
            "e_u": tuple(e_u.shape),
            "c_u": tuple(c_u.shape),
            "z_u": tuple(z_u.shape),
            "delta_u": tuple(delta_u.shape),
            "g_u": tuple(g_u.shape),
            "pos_scores": tuple(pos_scores.shape),
            "neg_scores": tuple(neg_scores.shape),
        }
        return shape_dict