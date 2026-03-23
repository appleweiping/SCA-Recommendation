# src/trainers/trainer_base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class BaseTrainer(ABC):
    """
    Base trainer with common utilities.

    Minimal responsibilities:
    - keep model / optimizer / device
    - move batch to device
    - save / load checkpoint
    - define abstract train_one_epoch()

    Subclasses are expected to implement task-specific training logic.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device = "cpu",
        scheduler: Optional[Any] = None,
        grad_clip_norm: Optional[float] = None,
        save_dir: Optional[str | Path] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.device = torch.device(device)

        self.model.to(self.device)

        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move all tensor values in a batch dict to target device.

        Args:
            batch: dict of arbitrary values

        Returns:
            new batch dict with tensors moved to self.device
        """
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def clip_gradients(self) -> None:
        """
        Clip gradients if grad_clip_norm is set.
        """
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

    def step_scheduler(self) -> None:
        """
        Step scheduler if provided.
        """
        if self.scheduler is not None:
            self.scheduler.step()

    def save_checkpoint(
        self,
        file_name: str,
        epoch: int,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save model / optimizer / scheduler states.

        Args:
            file_name: checkpoint file name
            epoch: current epoch
            extra_state: any extra metadata

        Returns:
            saved checkpoint path
        """
        if self.save_dir is None:
            raise ValueError("save_dir is not set; cannot save checkpoint.")

        ckpt_path = self.save_dir / file_name
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        if extra_state is not None:
            state["extra_state"] = extra_state

        torch.save(state, ckpt_path)
        return ckpt_path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        map_location: Optional[str | torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: path to checkpoint file
            load_optimizer: whether to restore optimizer
            load_scheduler: whether to restore scheduler
            map_location: torch.load map_location

        Returns:
            loaded checkpoint dict
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location if map_location is not None else self.device,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            load_scheduler
            and self.scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    @abstractmethod
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one training epoch and return scalar metrics.
        """
        raise NotImplementedError