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
        valid_user_pos_dict = data_bundle.valid_user_pos_dict

        # 只保留当前 split 中有 ground-truth 的用户
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

            scores = model.full_sort_predict(
                norm_adj=norm_adj,
                user_ids=batch_users_tensor,
                user_item_matrix=user_item_matrix,
            )  # (B, num_items)

            scores = scores.cpu()

            for idx, user_id in enumerate(batch_users):
                user_scores = scores[idx].clone()

                # valid 阶段：mask train
                # test 阶段：mask train + valid
                if split == "test":
                    seen_items = set()
                    seen_items |= train_user_pos_dict.get(user_id, set())
                    seen_items |= valid_user_pos_dict.get(user_id, set())
                else:
                    seen_items = train_user_pos_dict.get(user_id, set())

                if len(seen_items) > 0:
                    user_scores[list(seen_items)] = -1e9

                ranked_items = torch.argsort(user_scores, descending=True).tolist()

                gt_items = user_pos_dict[user_id]
                if len(gt_items) == 0:
                    continue

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

        final_results = {}
        for metric, values in results.items():
            final_results[metric] = float(np.mean(values)) if len(values) > 0 else 0.0

        return final_results