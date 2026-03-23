import math
from typing import List


def recall_at_k(ranked_items: List[int], ground_truth_item: int, k: int) -> float:
    """
    Recall@K for single ground-truth item.
    Since each user has exactly one test item, Recall@K is either 0 or 1.
    """
    topk_items = ranked_items[:k]
    return 1.0 if ground_truth_item in topk_items else 0.0


def hit_rate_at_k(ranked_items: List[int], ground_truth_item: int, k: int) -> float:
    """
    HR@K is equivalent to Recall@K when there is one ground-truth item.
    """
    topk_items = ranked_items[:k]
    return 1.0 if ground_truth_item in topk_items else 0.0


def ndcg_at_k(ranked_items: List[int], ground_truth_item: int, k: int) -> float:
    """
    NDCG@K for single ground-truth item.
    If the ground-truth item appears at rank r (0-based) in top-K,
    NDCG = 1 / log2(r + 2). Otherwise 0.
    """
    topk_items = ranked_items[:k]
    if ground_truth_item in topk_items:
        rank = topk_items.index(ground_truth_item)
        return 1.0 / math.log2(rank + 2.0)
    return 0.0