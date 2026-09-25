#!/usr/bin/env python3
"""
Macro-averaged F_0.5 Metric Evaluator for Business Entity Resolution.
Strictly implements the official challenge evaluation metric:
  F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Singletons (entities with 0 ground truth matches):
  - Correctly predicting empty set ([]) -> 1.0
  - Predicting ANY false match           -> 0.0
"""

from typing import Dict, Iterable, List, Optional, Set, Union
import numpy as np


def compute_entity_f05(
    true_matches: Set[str],
    pred_matches: Set[str]
) -> float:
    """
    Compute F_0.5 for a single Source 1 entity.
    """
    n_true = len(true_matches)
    n_pred = len(pred_matches)

    # Singleton case (0 true matches)
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0

    # Non-singleton but predicted empty
    if n_pred == 0:
        return 0.0

    # True positives
    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0

    precision = tp / n_pred
    recall = tp / n_true

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    countries: Optional[Dict[str, str]] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluate Macro F_0.5 across all Source 1 entities in ground truth.

    Args:
        ground_truth: mapping of {s1_id: set_of_true_matched_ids}
        predictions:  mapping of {s1_id: set_of_predicted_matched_ids}
        countries:    optional mapping of {s1_id: country_name} for per-country reporting

    Returns:
        dict with:
          - 'macro_f05': overall score
          - 'singleton_score': score strictly on singletons
          - 'non_singleton_f05': score strictly on non-singletons
          - 'macro_precision': precision on non-singletons
          - 'macro_recall': recall on non-singletons
          - 'by_country': dict of per-country macro F_0.5 (if countries provided)
    """
    entity_scores = []
    singleton_scores = []
    non_singleton_scores = []
    precisions = []
    recalls = []

    country_scores = {}

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        entity_scores.append(score)

        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)
            if len(pred_set) > 0:
                tp = len(true_set & pred_set)
                precisions.append(tp / len(pred_set))
                recalls.append(tp / len(true_set))
            else:
                precisions.append(0.0)
                recalls.append(0.0)

        if countries and s1_id in countries:
            c = countries[s1_id]
            if c not in country_scores:
                country_scores[c] = []
            country_scores[c].append(score)

    results = {
        "macro_f05": float(np.mean(entity_scores)) if entity_scores else 0.0,
        "n_entities": len(entity_scores),
        "singleton_score": float(np.mean(singleton_scores)) if singleton_scores else 1.0,
        "n_singletons": len(singleton_scores),
        "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        "n_non_singletons": len(non_singleton_scores),
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
    }

    if country_scores:
        results["by_country"] = {c: float(np.mean(s)) for c, s in country_scores.items()}

    return results


if __name__ == "__main__":
    # Test case from official challenge description
    # Predicted: [S2-00047, S2-00193, S3-00812]
    # Ground truth: [S2-00047, S3-00812]
    # Precision = 2/3, Recall = 1.0 -> F_0.5 = 0.714
    score = compute_entity_f05(
        true_matches={"S2-00047", "S3-00812"},
        pred_matches={"S2-00047", "S2-00193", "S3-00812"}
    )
    print(f"Official Example Test: score = {score:.4f} (Expected: ~0.7143)")
    assert abs(score - 0.7142857) < 1e-5, "Official test failed!"

    # Singleton test cases
    assert compute_entity_f05(set(), set()) == 1.0, "Singleton empty match should be 1.0"
    assert compute_entity_f05(set(), {"S2-00001"}) == 0.0, "Singleton with false match should be 0.0"

    print("All metric tests passed successfully!")
