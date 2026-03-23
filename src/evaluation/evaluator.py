from typing import Dict, List, Optional

import numpy as np
import torch

from src.evaluation.metrics import recall_at_k, hit_rate_at_k, ndcg_at_k


class RankingEvaluator:
    """
    Full-ranking evaluator for top-K recommendation.

    Assumptions:
    1. model can provide user/item embeddings
    2. ground_truth maps user_id -> one test item
    3. train_user_items maps user_id -> list of interacted items in training
    """

    def __init__(
        self,
        k_list: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        self.k_list = k_list if k_list is not None else [10]
        self.device = device

    def evaluate(
        self,
        model,
        user_ids: List[int],
        train_user_items: Dict[int, List[int]],
        ground_truth: Dict[int, int],
    ) -> Dict[str, float]:
        """
        Evaluate model with full ranking.

        Args:
            model: recommendation model
            user_ids: users to evaluate
            train_user_items: dict[user_id] = list of training items
            ground_truth: dict[user_id] = test item

        Returns:
            dict of metrics, e.g. Recall@10, NDCG@10, HR@10
        """
        model.eval()

        with torch.no_grad():
            user_emb, item_emb = self._get_user_item_embeddings(model)

            results = {}
            for k in self.k_list:
                results[f"Recall@{k}"] = []
                results[f"NDCG@{k}"] = []
                results[f"HR@{k}"] = []

            for user_id in user_ids:
                if user_id not in ground_truth:
                    continue

                scores = self._compute_scores(user_emb, item_emb, user_id)

                # mask training interactions
                interacted_items = train_user_items.get(user_id, [])
                if len(interacted_items) > 0:
                    scores[interacted_items] = -1e9

                ranked_items = torch.argsort(scores, descending=True).cpu().tolist()
                gt_item = ground_truth[user_id]

                for k in self.k_list:
                    results[f"Recall@{k}"].append(recall_at_k(ranked_items, gt_item, k))
                    results[f"NDCG@{k}"].append(ndcg_at_k(ranked_items, gt_item, k))
                    results[f"HR@{k}"].append(hit_rate_at_k(ranked_items, gt_item, k))

        final_results = {}
        for metric_name, values in results.items():
            final_results[metric_name] = float(np.mean(values)) if len(values) > 0 else 0.0

        return final_results

    def _get_user_item_embeddings(self, model):
        """
        Try to get final user/item embeddings from model.
        You may adjust this depending on your project implementation.
        """
        if hasattr(model, "get_user_item_embeddings"):
            user_emb, item_emb = model.get_user_item_embeddings()
        elif hasattr(model, "forward_eval"):
            user_emb, item_emb = model.forward_eval()
        else:
            # fallback: assume model() returns embeddings
            user_emb, item_emb = model()

        return user_emb.to(self.device), item_emb.to(self.device)

    def _compute_scores(self, user_emb, item_emb, user_id: int) -> torch.Tensor:
        """
        Compute scores of one user against all items.
        """
        user_vector = user_emb[user_id]               # (d,)
        scores = torch.matmul(item_emb, user_vector)  # (num_items,)
        return scores