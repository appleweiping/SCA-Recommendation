# run.py

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from src.data.dataset import build_interaction_data_bundle
from src.models.lightgcn import LightGCN
from src.models.semantic_encoder import SemanticEncoder
from src.models.sca import SCA
from src.trainers.trainer_sca import SCATrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal runnable pipeline for SCA.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sca_default.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level YAML mapping.")
    return config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Minimal runnable setting; deterministic flags are omitted for simplicity.


def ensure_required_keys(config: Dict[str, Any]) -> None:
    required_keys = [
        "data",
        "model",
        "train",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config section: '{key}'")

    data_keys = ["train_path", "valid_path", "test_path"]
    for key in data_keys:
        if key not in config["data"]:
            raise KeyError(f"Missing required config key: data.{key}")

    model_keys = ["embedding_dim", "num_layers", "semantic_input_dim", "semantic_dim"]
    for key in model_keys:
        if key not in config["model"]:
            raise KeyError(f"Missing required config key: model.{key}")

    train_keys = ["lr", "batch_size", "epochs"]
    for key in train_keys:
        if key not in config["train"]:
            raise KeyError(f"Missing required config key: train.{key}")


def maybe_make_save_dir(config: Dict[str, Any]) -> str | None:
    save_dir = config.get("output", {}).get("save_dir", None)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir


def build_sca_model(
    config: Dict[str, Any],
    num_users: int,
    num_items: int,
) -> torch.nn.Module:
    """
    Build SCA model from config.

    Reasonable default:
    - SemanticEncoder runs in fallback/trainable-user-semantic mode unless you
      later inject offline semantic features.
    """
    model_cfg = config["model"]

    backbone = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=model_cfg["embedding_dim"],
        num_layers=model_cfg["num_layers"],
    )

    semantic_encoder = SemanticEncoder(
        num_users=num_users,
        input_dim=model_cfg["semantic_input_dim"],
        output_dim=model_cfg["semantic_dim"],
        dropout=model_cfg.get("semantic_dropout", 0.0),
        use_mlp=model_cfg.get("semantic_use_mlp", False),
        normalize=model_cfg.get("semantic_normalize", True),
    )

    # Try to be robust to small constructor changes in SCA.
    sca_kwargs = {
        "backbone": backbone,
        "semantic_encoder": semantic_encoder,
        "embedding_dim": model_cfg["embedding_dim"],
        "semantic_dim": model_cfg["semantic_dim"],
        "gate_hidden_dim": model_cfg.get("gate_hidden_dim", None),
        "gate_type": model_cfg.get("gate_type", "vector"),
        "control_scale": model_cfg.get("control_scale", 1.0),
        "dropout": model_cfg.get("dropout", 0.0),
    }

    sca_signature = inspect.signature(SCA.__init__)
    filtered_kwargs = {
        k: v for k, v in sca_kwargs.items()
        if k in sca_signature.parameters
    }

    model = SCA(**filtered_kwargs)
    return model


def build_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    return torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_required_keys(config)

    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    save_dir = maybe_make_save_dir(config)

    # ------------------------------------------------------------------
    # 1) Build interaction data bundle
    # ------------------------------------------------------------------
    data_cfg = config["data"]
    data_bundle = build_interaction_data_bundle(
        train_path=data_cfg["train_path"],
        valid_path=data_cfg["valid_path"],
        test_path=data_cfg["test_path"],
        device=device,
    )

    print(
        "[INFO] Data loaded | "
        f"num_users={data_bundle.num_users}, "
        f"num_items={data_bundle.num_items}, "
        f"train_pairs={len(data_bundle.train_pairs)}"
    )

    # ------------------------------------------------------------------
    # 2) Build model
    # ------------------------------------------------------------------
    model = build_sca_model(
        config=config,
        num_users=data_bundle.num_users,
        num_items=data_bundle.num_items,
    )

    # ------------------------------------------------------------------
    # 3) Build optimizer
    # ------------------------------------------------------------------
    optimizer = build_optimizer(model, config)

    # ------------------------------------------------------------------
    # 4) Build trainer
    # ------------------------------------------------------------------
    train_cfg = config["train"]
    trainer = SCATrainer(
        model=model,
        data_bundle=data_bundle,
        batch_size=train_cfg["batch_size"],
        optimizer=optimizer,
        device=device,
        num_workers=train_cfg.get("num_workers", 0),
        shuffle=train_cfg.get("shuffle", True),
        scheduler=None,
        grad_clip_norm=train_cfg.get("grad_clip_norm", None),
        save_dir=save_dir,
        lambda_align=train_cfg.get("lambda_align", 0.1),
        lambda_reg=train_cfg.get("lambda_reg", 1e-4),
        align_type=train_cfg.get("align_type", "cosine"),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=train_cfg.get("drop_last", False),
    )

    # Optional sanity check
    try:
        shape_info = trainer.inspect_one_batch()
        print(f"[INFO] Batch inspection: {shape_info}")
    except Exception as e:
        print(f"[WARN] inspect_one_batch failed: {e}")

    # ------------------------------------------------------------------
    # 5) Train epochs
    # ------------------------------------------------------------------
    epochs = train_cfg["epochs"]
    for epoch in range(1, epochs + 1):
        metrics = trainer.train_one_epoch(epoch)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={metrics['loss']:.6f} | "
            f"bpr={metrics['bpr_loss']:.6f} | "
            f"align={metrics['align_loss']:.6f} | "
            f"reg={metrics['reg_loss']:.6f}"
        )

        if save_dir is not None and train_cfg.get("save_every_epoch", False):
            trainer.save_checkpoint(
                file_name=f"sca_epoch_{epoch}.pt",
                epoch=epoch,
                extra_state={"metrics": metrics, "config": config},
            )

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()