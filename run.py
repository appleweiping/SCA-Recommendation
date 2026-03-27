from __future__ import annotations

from src.evaluation.evaluator import RankingEvaluator
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
from src.trainers.trainer_lightgcn import LightGCNTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal runnable pipeline for SCA / LightGCN.")
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


def ensure_required_keys(config: Dict[str, Any]) -> None:
    required_keys = ["data", "model", "train"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config section: '{key}'")

    data_keys = ["train_path", "valid_path", "test_path"]
    for key in data_keys:
        if key not in config["data"]:
            raise KeyError(f"Missing required config key: data.{key}")

    model_type = config["model"].get("name", "sca").lower()

    # shared model keys
    shared_model_keys = ["embedding_dim", "num_layers"]
    for key in shared_model_keys:
        if key not in config["model"]:
            raise KeyError(f"Missing required config key: model.{key}")

    # sca-only keys
    if model_type == "sca":
        sca_model_keys = ["semantic_input_dim", "semantic_dim"]
        for key in sca_model_keys:
            if key not in config["model"]:
                raise KeyError(f"Missing required config key for SCA: model.{key}")

    train_keys = ["lr", "batch_size", "epochs"]
    for key in train_keys:
        if key not in config["train"]:
            raise KeyError(f"Missing required config key: train.{key}")


def maybe_make_save_dir(config: Dict[str, Any]) -> str | None:
    save_dir = config.get("output", {}).get("save_dir", None)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir


def build_lightgcn_model(
    config: Dict[str, Any],
    num_users: int,
    num_items: int,
) -> torch.nn.Module:
    model_cfg = config["model"]

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=model_cfg["embedding_dim"],
        num_layers=model_cfg["num_layers"],
    )
    return model


def build_sca_model(
    config: Dict[str, Any],
    num_users: int,
    num_items: int,
) -> torch.nn.Module:
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


def inspect_data_bundle(data_bundle) -> None:
    """
    Diagnose whether the processed split is valid for recommendation evaluation.
    """
    train_user_pos = data_bundle.train_user_pos_dict
    valid_user_pos = data_bundle.valid_user_pos_dict
    test_user_pos = data_bundle.test_user_pos_dict

    train_users = [u for u, items in train_user_pos.items() if len(items) > 0]
    valid_users = [u for u, items in valid_user_pos.items() if len(items) > 0]
    test_users = [u for u, items in test_user_pos.items() if len(items) > 0]

    print("[DIAG] ===== Data Split Inspection =====")
    print(f"[DIAG] num_users={data_bundle.num_users}, num_items={data_bundle.num_items}")
    print(f"[DIAG] train_pairs={len(data_bundle.train_pairs)}")
    print(f"[DIAG] valid_pairs={len(data_bundle.valid_pairs)}")
    print(f"[DIAG] test_pairs={len(data_bundle.test_pairs)}")
    print(f"[DIAG] train_users={len(train_users)}")
    print(f"[DIAG] valid_users={len(valid_users)}")
    print(f"[DIAG] test_users={len(test_users)}")

    train_valid_overlap_users = 0
    train_test_overlap_users = 0
    valid_test_overlap_users = 0

    train_valid_overlap_items = 0
    train_test_overlap_items = 0
    valid_test_overlap_items = 0

    for u in range(data_bundle.num_users):
        train_items = train_user_pos.get(u, set())
        valid_items = valid_user_pos.get(u, set())
        test_items = test_user_pos.get(u, set())

        tv = train_items & valid_items
        tt = train_items & test_items
        vt = valid_items & test_items

        if len(tv) > 0:
            train_valid_overlap_users += 1
            train_valid_overlap_items += len(tv)
        if len(tt) > 0:
            train_test_overlap_users += 1
            train_test_overlap_items += len(tt)
        if len(vt) > 0:
            valid_test_overlap_users += 1
            valid_test_overlap_items += len(vt)

    print(f"[DIAG] train-valid overlap users={train_valid_overlap_users}, items={train_valid_overlap_items}")
    print(f"[DIAG] train-test overlap users={train_test_overlap_users}, items={train_test_overlap_items}")
    print(f"[DIAG] valid-test overlap users={valid_test_overlap_users}, items={valid_test_overlap_items}")

    valid_gt_counts = [len(valid_user_pos[u]) for u in valid_users]
    test_gt_counts = [len(test_user_pos[u]) for u in test_users]

    if len(valid_gt_counts) > 0:
        print(
            "[DIAG] valid gt per user | "
            f"min={min(valid_gt_counts)}, max={max(valid_gt_counts)}, "
            f"avg={sum(valid_gt_counts)/len(valid_gt_counts):.4f}"
        )

    if len(test_gt_counts) > 0:
        print(
            "[DIAG] test gt per user | "
            f"min={min(test_gt_counts)}, max={max(test_gt_counts)}, "
            f"avg={sum(test_gt_counts)/len(test_gt_counts):.4f}"
        )

    candidate_sizes = []
    for u in test_users[:20]:
        seen_items = set()
        seen_items |= train_user_pos.get(u, set())
        seen_items |= valid_user_pos.get(u, set())
        candidate_size = data_bundle.num_items - len(seen_items)
        candidate_sizes.append(candidate_size)
        print(
            f"[DIAG] user={u} | train={len(train_user_pos.get(u, set()))} | "
            f"valid={len(valid_user_pos.get(u, set()))} | "
            f"test={len(test_user_pos.get(u, set()))} | "
            f"candidate={candidate_size}"
        )

    if len(candidate_sizes) > 0:
        print(
            "[DIAG] sampled candidate size | "
            f"min={min(candidate_sizes)}, max={max(candidate_sizes)}, "
            f"avg={sum(candidate_sizes)/len(candidate_sizes):.2f}"
        )

    print("[DIAG] ===== End Inspection =====")


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


def build_trainer(
    model: torch.nn.Module,
    config: Dict[str, Any],
    data_bundle,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    save_dir: str | None,
):
    model_type = config["model"].get("name", "sca").lower()
    train_cfg = config["train"]

    if model_type == "lightgcn":
        trainer = LightGCNTrainer(
            model=model,
            train_pairs=data_bundle.train_pairs,
            user_pos_dict=data_bundle.train_user_pos_dict,
            norm_adj=data_bundle.norm_adj,
            num_users=data_bundle.num_users,
            num_items=data_bundle.num_items,
            optimizer=optimizer,
            device=device,
            num_workers=train_cfg.get("num_workers", 0),
            shuffle=train_cfg.get("shuffle", True),
            scheduler=None,
            grad_clip_norm=train_cfg.get("grad_clip_norm", None),
            save_dir=save_dir,
            weight_decay=train_cfg.get("lambda_reg", 1e-4),
            pin_memory=train_cfg.get("pin_memory", True),
            drop_last=train_cfg.get("drop_last", False),
        )
        trainer.set_batch_size(train_cfg["batch_size"])
        return trainer

    if model_type == "sca":
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
        return trainer

    raise ValueError(f"Unsupported model.name: {model_type}")


def print_epoch_metrics(model_type: str, epoch: int, metrics: Dict[str, float]) -> None:
    if model_type == "lightgcn":
        print(
            f"[Epoch {epoch:03d}] "
            f"loss={metrics['loss']:.6f} | "
            f"bpr={metrics['bpr_loss']:.6f} | "
            f"reg={metrics['reg_loss']:.6f} | "
            f"pos_mean={metrics['pos_scores_mean']:.6f} | "
            f"neg_mean={metrics['neg_scores_mean']:.6f} | "
            f"pos>neg={metrics['pos_gt_neg_ratio']:.6f}"
        )
    else:
        print(
            f"[Epoch {epoch:03d}] "
            f"loss={metrics['loss']:.6f} | "
            f"bpr={metrics['bpr_loss']:.6f} | "
            f"align={metrics['align_loss']:.6f} | "
            f"reg={metrics['reg_loss']:.6f} | "
            f"pos_mean={metrics['pos_scores_mean']:.6f} | "
            f"neg_mean={metrics['neg_scores_mean']:.6f} | "
            f"pos>neg={metrics['pos_gt_neg_ratio']:.6f} | "
            f"delta_abs={metrics['delta_abs_mean']:.6f} | "
            f"gate_mean={metrics['gate_mean']:.6f} | "
            f"gate_std={metrics['gate_std']:.6f} | "
            f"ctrl_shift={metrics['control_shift_mean']:.6f}"
        )


def patch_lightgcn_for_ranking_eval_if_needed(model: torch.nn.Module) -> torch.nn.Module:
    """
    RankingEvaluator expects:
        model.full_sort_predict(norm_adj=..., user_ids=..., user_item_matrix=...)
    But pure LightGCN currently exposes:
        model.full_sort_scores(norm_adj, user_ids)

    This patch adds a compatible method at runtime without changing model design.
    """
    if hasattr(model, "full_sort_predict"):
        return model

    if hasattr(model, "full_sort_scores"):
        def _full_sort_predict(norm_adj, user_ids, user_item_matrix=None):
            return model.full_sort_scores(norm_adj=norm_adj, user_ids=user_ids)

        setattr(model, "full_sort_predict", _full_sort_predict)
        return model

    raise AttributeError(
        "Model has neither `full_sort_predict` nor `full_sort_scores`, "
        "cannot run ranking evaluation."
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_required_keys(config)

    seed = config.get("seed", 42)
    set_seed(seed)

    model_type = config["model"].get("name", "sca").lower()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Model type: {model_type}")

    save_dir = maybe_make_save_dir(config)

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
    inspect_data_bundle(data_bundle)

    if model_type == "lightgcn":
        model = build_lightgcn_model(
            config=config,
            num_users=data_bundle.num_users,
            num_items=data_bundle.num_items,
        )
    elif model_type == "sca":
        model = build_sca_model(
            config=config,
            num_users=data_bundle.num_users,
            num_items=data_bundle.num_items,
        )
    else:
        raise ValueError(f"Unsupported model.name: {model_type}")

    optimizer = build_optimizer(model, config)

    trainer = build_trainer(
        model=model,
        config=config,
        data_bundle=data_bundle,
        optimizer=optimizer,
        device=device,
        save_dir=save_dir,
    )

    try:
        shape_info = trainer.inspect_one_batch()
        print(f"[INFO] Batch inspection: {shape_info}")
    except Exception as e:
        print(f"[WARN] inspect_one_batch failed: {e}")

    epochs = config["train"]["epochs"]
    for epoch in range(1, epochs + 1):
        metrics = trainer.train_one_epoch(epoch)
        print_epoch_metrics(model_type=model_type, epoch=epoch, metrics=metrics)

        if save_dir is not None and config["train"].get("save_every_epoch", False):
            file_prefix = "lightgcn" if model_type == "lightgcn" else "sca"
            trainer.save_checkpoint(
                file_name=f"{file_prefix}_epoch_{epoch}.pt",
                epoch=epoch,
                extra_state={"metrics": metrics, "config": config},
            )

    print("[INFO] Training finished.")
    print("[INFO] Start evaluation...")

    eval_model = model
    if model_type == "lightgcn":
        eval_model = patch_lightgcn_for_ranking_eval_if_needed(model)

    evaluator = RankingEvaluator(k_list=[10], device=device)

    results = evaluator.evaluate(
        model=eval_model,
        data_bundle=data_bundle,
        split="test"
    )

    print("[RESULT] Test Metrics:")
    print(results)


if __name__ == "__main__":
    main()