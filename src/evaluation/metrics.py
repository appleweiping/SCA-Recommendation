from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Set


def _to_item_set(ground_truth_items: int | Sequence[int] | Set[int]) -> Set[int]:
    """
    Normalize ground-truth items into a set.

    Supports:
        - single int
        - list / tuple
        - set

    Returns:
        set of item ids
    """
    if isinstance(ground_truth_items, int):
        return {ground_truth_items}

    if isinstance(ground_truth_items, set):
        return ground_truth_items

    return set(ground_truth_items)


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, but got: {k}")


def recall_at_k(
    ranked_items: Sequence[int],
    ground_truth_items: int | Sequence[int] | Set[int],
    k: int,
) -> float:
    """
    Compute Recall@K.

    Definition:
        Recall@K = (# relevant items in top-K) / (# relevant items)

    Notes:
        - If there is exactly one ground-truth item, Recall@K is 0 or 1.
        - If multiple ground-truth items are provided, this computes the standard recall.

    Args:
        ranked_items:
            Ranked recommendation list in descending order of scores.
        ground_truth_items:
            One relevant item id or a collection of relevant item ids.
        k:
            Top-K cutoff.

    Returns:
        Recall@K in [0, 1].
    """
    _validate_k(k)

    gt_set = _to_item_set(ground_truth_items)
    if len(gt_set) == 0:
        return 0.0

    topk_items = ranked_items[:k]
    hit_count = sum(1 for item in topk_items if item in gt_set)

    return float(hit_count) / float(len(gt_set))


def hit_rate_at_k(
    ranked_items: Sequence[int],
    ground_truth_items: int | Sequence[int] | Set[int],
    k: int,
) -> float:
    """
    Compute Hit Rate@K.

    Definition:
        HR@K = 1 if any relevant item appears in top-K, else 0.

    Notes:
        - For the common leave-one-out setting with one test item per user,
          HR@K is numerically identical to Recall@K.

    Args:
        ranked_items:
            Ranked recommendation list.
        ground_truth_items:
            One relevant item id or a collection of relevant item ids.
        k:
            Top-K cutoff.

    Returns:
        1.0 if there is at least one hit in top-K, else 0.0.
    """
    _validate_k(k)

    gt_set = _to_item_set(ground_truth_items)
    if len(gt_set) == 0:
        return 0.0

    topk_items = ranked_items[:k]
    hit = any(item in gt_set for item in topk_items)

    return 1.0 if hit else 0.0


def dcg_at_k(
    ranked_items: Sequence[int],
    ground_truth_items: int | Sequence[int] | Set[int],
    k: int,
) -> float:
    """
    Compute DCG@K.

    Definition:
        DCG@K = sum_{i=1..K} rel_i / log2(i+1)

    Here:
        rel_i = 1 if the item at rank i is relevant, else 0.

    Args:
        ranked_items:
            Ranked recommendation list.
        ground_truth_items:
            One relevant item id or a collection of relevant item ids.
        k:
            Top-K cutoff.

    Returns:
        DCG@K.
    """
    _validate_k(k)

    gt_set = _to_item_set(ground_truth_items)
    if len(gt_set) == 0:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k]):
        if item in gt_set:
            dcg += 1.0 / math.log2(rank + 2.0)
    return dcg


def idcg_at_k(
    ground_truth_items: int | Sequence[int] | Set[int],
    k: int,
) -> float:
    """
    Compute ideal DCG@K.

    This is the maximum possible DCG for the given number of relevant items.

    Args:
        ground_truth_items:
            One relevant item id or a collection of relevant item ids.
        k:
            Top-K cutoff.

    Returns:
        IDCG@K.
    """
    _validate_k(k)

    gt_set = _to_item_set(ground_truth_items)
    num_relevant = min(len(gt_set), k)

    if num_relevant == 0:
        return 0.0

    idcg = 0.0
    for rank in range(num_relevant):
        idcg += 1.0 / math.log2(rank + 2.0)
    return idcg


def ndcg_at_k(
    ranked_items: Sequence[int],
    ground_truth_items: int | Sequence[int] | Set[int],
    k: int,
) -> float:
    """
    Compute NDCG@K.

    Definition:
        NDCG@K = DCG@K / IDCG@K

    Notes:
        - For one ground-truth item, this reduces to:
              1 / log2(rank + 2)   if hit in top-K
              0                    otherwise
        - For multiple ground-truth items, this is the standard normalized DCG.

    Args:
        ranked_items:
            Ranked recommendation list.
        ground_truth_items:
            One relevant item id or a collection of relevant item ids.
        k:
            Top-K cutoff.

    Returns:
        NDCG@K in [0, 1].
    """
    _validate_k(k)

    dcg = dcg_at_k(ranked_items, ground_truth_items, k)
    ideal_dcg = idcg_at_k(ground_truth_items, k)

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg