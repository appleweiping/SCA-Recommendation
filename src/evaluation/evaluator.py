from typing import Dict, List

import numpy as np
import torch

from src.evaluation.metrics import recall_at_k, ndcg_at_k, hit_rate_at_k
from src.data.dataset import InteractionDataBundle


class RankingEvaluator:
    def __init__(self, k_list=[10], device="cpu"):
        self.k_list = k_list
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model,
        data_bundle: InteractionDataBundle,
        split: str = "test",
    ) -> Dict[str, float]:

        model.eval()

        norm_adj = data_bundle.norm_adj.to(self.device)
        user_item_matrix = data_bundle.user_item_matrix.to(self.device)

        if split == "test":
            user_pos_dict = data_bundle.test_user_pos_dict
        elif split == "valid":
            user_pos_dict = data_bundle.valid_user_pos_dict
        else:
            raise ValueError(f"Unknown split: {split}")

        train_user_pos_dict = data_bundle.train_user_pos_dict

        # ✅ 修复1：过滤没有 ground-truth 的用户
        user_ids = [u for u, items in user_pos_dict.items() if len(items) > 0]

        results = {}
        for k in self.k_list:
            results[f"Recall@{k}"] = []
            results[f"NDCG@{k}"] = []
            results[f"HR@{k}"] = []

        batch_size = 256

        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i + batch_size]
            batch_users_tensor = torch.tensor(batch_users, dtype=torch.long).to(self.device)

            # ⭐ 核心：调用 SCA full sort
            scores = model.full_sort_predict(
                norm_adj=norm_adj,
                user_ids=batch_users_tensor,
                user_item_matrix=user_item_matrix,
            )  # (B, num_items)

            scores = scores.cpu()

            for idx, user_id in enumerate(batch_users):
                # ✅ 修复2：clone，避免污染 batch tensor
                user_scores = scores[idx].clone()

                # ❗mask training interactions
                train_items = train_user_pos_dict.get(user_id, set())
                if len(train_items) > 0:
                    user_scores[list(train_items)] = -1e9

                ranked_items = torch.argsort(user_scores, descending=True).tolist()

                gt_items = user_pos_dict[user_id]

                # ✅ 修复3：直接支持 multi-ground-truth（更规范）
                for k in self.k_list:
                    results[f"Recall@{k}"].append(
                        recall_at_k(ranked_items, gt_items, k)
                    )
                    results[f"NDCG@{k}"].append(
                        ndcg_at_k(ranked_items, gt_items, k)
                    )
                    results[f"HR@{k}"].append(
                        hit_rate_at_k(ranked_items, gt_items, k)
                    )

        final_results = {
            metric: float(np.mean(values)) for metric, values in results.items()
        }

        return final_results