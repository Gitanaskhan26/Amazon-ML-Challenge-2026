#!/usr/bin/env python3
"""
LightGBM Training, Calibration, and Macro-F0.5 Optimizer.

Pipeline:
  1. Loads candidate pairs (data/val/candidate_pairs.tsv) and validation source tables.
  2. Extracts 22+ tabular similarity features for each candidate pair.
  3. Trains LightGBM Binary Classifier (utilizing multi-threading across 16 cores).
  4. Applies Greedy 1-to-at-most-1 Conflict Resolution.
  5. Calibrates per-entity expected-F0.5 prefix selection.
  6. Evaluates and reports Macro-F0.5 (Overall, US, India, and Singletons).
  7. Saves trained model to data/lgb_model.txt.

Compatible with Windows and Linux/macOS.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger, write_submission_tsv
from src.metrics import evaluate_macro_f05, compute_entity_f05
from src.preprocessing import clean_name_multiview, clean_address_multiview
from src.features import extract_pair_features
from src.train import solve_greedy_bipartite_assignment, select_optimal_prefix_per_entity

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM Entity Resolution Model.")
    parser.add_argument("--val-dir", type=str, default="./data/val", help="Path to validation directory")
    parser.add_argument("--candidate-file", type=str, default="./data/val/candidate_pairs.tsv", help="Path to candidate pairs file")
    parser.add_argument("--model-out", type=str, default="./data/lgb_model.txt", help="Path to save trained LightGBM model")
    parser.add_argument("--n-jobs", type=int, default=16, help="Number of CPU threads to use (default: 16)")
    return parser.parse_args()


def load_val_records(val_dir: Path):
    """Load and preprocess S1, S2, S3, and Ground Truth."""
    logger.info("Loading validation source files...")
    s1_path = val_dir / "val_source1.tsv"
    s2_path = val_dir / "val_source2.tsv"
    s3_path = val_dir / "val_source3.tsv"
    gt_path = val_dir / "val_ground_truth.tsv"

    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str)

    gt_map = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            s1_id = p[0].strip()
            if len(p) > 1 and p[1].strip():
                gt_map[s1_id] = {m.strip() for m in p[1].split(",") if m.strip()}
            else:
                gt_map[s1_id] = set()

    # Preprocess S1
    s1_records = {}
    countries = {}
    for _, r in tqdm(df_s1.iterrows(), total=len(df_s1), desc="Preprocessing S1"):
        c = r.get("country", "")
        e_id = r["entity_id"]
        nv = clean_name_multiview(r.get("business_name", ""))
        av = clean_address_multiview(r.get("business_address", ""), country=c)
        s1_records[e_id] = {"entity_id": e_id, "country": c, **nv, **av}
        countries[e_id] = c

    # Preprocess pool (S2 + S3)
    pool_records = {}
    for df_pool, name in [(df_s2, "S2"), (df_s3, "S3")]:
        for _, r in tqdm(df_pool.iterrows(), total=len(df_pool), desc=f"Preprocessing {name}"):
            c = r.get("country", "")
            e_id = r["entity_id"]
            nv = clean_name_multiview(r.get("business_name", ""))
            av = clean_address_multiview(r.get("business_address", ""), country=c)
            pool_records[e_id] = {"entity_id": e_id, "country": c, **nv, **av}

    return s1_records, pool_records, gt_map, countries


def train_pipeline():
    args = parse_args()
    if not HAS_LIGHTGBM:
        logger.error("LightGBM is not installed! Please run: pip install lightgbm")
        return

    val_dir = Path(args.val_dir).resolve()
    cand_file = Path(args.candidate_file).resolve()
    model_out = Path(args.model_out).resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)

    with Timer("Loading validation records"):
        s1_records, pool_records, gt_map, countries = load_val_records(val_dir)

    # Read candidate pairs
    logger.info(f"Reading candidate pairs from: {cand_file}")
    candidate_pairs = []
    with open(cand_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            s1_id = p[0].strip()
            if len(p) > 1 and p[1].strip():
                cands = [c.strip() for c in p[1].split(",") if c.strip()]
                for c in cands:
                    candidate_pairs.append((s1_id, c))

    logger.info(f"Total candidate pairs loaded: {len(candidate_pairs):,}")

    # Entity-level train/validation split (80% train, 20% holdout validation)
    all_s1_ids = list(s1_records.keys())
    np.random.seed(42)
    np.random.shuffle(all_s1_ids)

    n_train = int(len(all_s1_ids) * 0.80)
    train_s1 = set(all_s1_ids[:n_train])
    val_s1 = set(all_s1_ids[n_train:])

    logger.info(f"Training S1 entities: {len(train_s1):,}, Holdout Evaluation S1 entities: {len(val_s1):,}")

    # Extract features
    X_train, y_train = [], []
    eval_pairs = []
    X_val, y_val = [], []

    feature_names = None

    with Timer("Extracting pairwise tabular features"):
        for s1_id, cand_id in tqdm(candidate_pairs, desc="Extracting Features"):
            if s1_id not in s1_records or cand_id not in pool_records:
                continue

            feats = extract_pair_features(s1_records[s1_id], pool_records[cand_id])
            if feature_names is None:
                feature_names = sorted(feats.keys())

            feat_vec = [feats[k] for k in feature_names]
            label = 1.0 if cand_id in gt_map.get(s1_id, set()) else 0.0

            if s1_id in train_s1:
                X_train.append(feat_vec)
                y_train.append(label)
            else:
                X_val.append(feat_vec)
                y_val.append(label)
                eval_pairs.append((s1_id, cand_id))

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)

    logger.info(f"Training set: {X_train.shape[0]:,} pairs ({int(y_train.sum()):,} positives, {int(len(y_train) - y_train.sum()):,} negatives)")
    logger.info(f"Holdout set:  {X_val.shape[0]:,} pairs ({int(y_val.sum()):,} positives, {int(len(y_val) - y_val.sum()):,} negatives)")

    # Train LightGBM
    with Timer("Training LightGBM Classifier"):
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "learning_rate": 0.08,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "num_threads": args.n_jobs,
            "verbose": -1
        }

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=300,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False), lgb.log_evaluation(period=50)]
        )

        booster.save_model(str(model_out))
        logger.info(f"Trained LightGBM model saved to: {model_out}")

    # Feature Importance
    importance = booster.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    logger.info("--- Top 10 Most Important Features (by Gain) ---")
    for fname, imp in feat_imp[:10]:
        logger.info(f"  {fname:<25}: {imp:.2f}")

    # Inference & Metric Calibration on Holdout Evaluation Set
    with Timer("Holdout Inference & Macro-F0.5 Calibration"):
        val_preds = booster.predict(X_val)

        scored_eval_pairs = [
            (eval_pairs[i][0], eval_pairs[i][1], float(val_preds[i]))
            for i in range(len(eval_pairs))
        ]

        # 1. Greedy 1-to-at-most-1 Bipartite Conflict Resolution
        assigned_mapping = solve_greedy_bipartite_assignment(scored_eval_pairs)

        # 2. Threshold Calibration Grid Search
        best_tau = 0.50
        best_score = 0.0
        best_results = None

        cand_scores_by_s1 = defaultdict(list)
        for s1_id, cand_id, sc in scored_eval_pairs:
            cand_scores_by_s1[s1_id].append((cand_id, sc))

        holdout_gt = {s1_id: gt_map[s1_id] for s1_id in val_s1}
        holdout_countries = {s1_id: countries[s1_id] for s1_id in val_s1}

        # Test threshold range
        for tau in np.arange(0.35, 0.85, 0.05):
            preds_tau = {}
            for s1_id in val_s1:
                survived = assigned_mapping.get(s1_id, set())
                cand_list = [(c, sc) for c, sc in cand_scores_by_s1[s1_id] if c in survived and sc >= tau]
                preds_tau[s1_id] = {c for c, _ in cand_list}

            res = evaluate_macro_f05(holdout_gt, preds_tau, countries=holdout_countries)
            score = res["macro_f05"]
            if score > best_score:
                best_score = score
                best_tau = tau
                best_results = res

        logger.info("=" * 60)
        logger.info(f"OPTIMAL DECISION THRESHOLD (tau*): {best_tau:.2f}")
        logger.info(f">>> BEST HOLDOUT MACRO F_0.5 SCORE: {best_results['macro_f05']:.4f} <<<")
        logger.info(f"  Singletons Score:    {best_results['singleton_score']:.4f} ({best_results['n_singletons']:,} entities)")
        logger.info(f"  Non-Singleton F0.5:  {best_results['non_singleton_f05']:.4f}")
        logger.info(f"  Macro Precision:     {best_results['macro_precision']:.4f}")
        logger.info(f"  Macro Recall:        {best_results['macro_recall']:.4f}")
        if "by_country" in best_results:
            logger.info("--- Country Breakdown ---")
            for c, sc in best_results["by_country"].items():
                logger.info(f"  {c:<10}: {sc:.4f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    train_pipeline()
