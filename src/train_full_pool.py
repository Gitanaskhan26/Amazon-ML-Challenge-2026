#!/usr/bin/env python3
"""
Full-Pool Hard Negative LightGBM Training & Calibration Pipeline.

Solves the Generalization Gap:
  Instead of training on a small distractor pool (where 80% of candidates are true matches),
  this pipeline blocks S1 entities against the FULL 10.3M training pool (train_source2 + train_source3).
  This exposes LightGBM to millions of authentic hard negative distractors:
    - Same business names in different cities/states/PIN codes (chain stores, franchisees)
    - Same street names with different business names
    - Shared frequent keywords
    - Transliterated or OCR-distorted negative pairs

Outputs:
  - data/lgb_model.txt (Trained 33-feature LightGBM model)
  - data/calibration_results.json (Optimal threshold and validation metrics)

Compatible with Windows (Intel i9 + 128 GB RAM) and Linux/macOS.
"""

import os
import sys
import gc
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger, write_submission_tsv, find_file
from src.metrics import evaluate_macro_f05
from src.preprocessing import clean_name_multiview, clean_address_multiview
from src.features import extract_pair_features
from src.blocking import BlockingEngine

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM on Full Pool with Authentic Hard Negatives.")
    parser.add_argument(
        "--train-dir",
        type=str,
        default="/Users/anaskhan/Downloads/student_resource/dataset/train",
        help="Path to training dataset directory containing source1, source2, source3, and ground_truth"
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="./data/lgb_model.txt",
        help="Path to save trained LightGBM model"
    )
    parser.add_argument(
        "--train-sample-s1",
        type=int,
        default=80000,
        help="Number of Source 1 entities for training (default: 80,000)"
    )
    parser.add_argument(
        "--val-sample-s1",
        type=int,
        default=20000,
        help="Number of Source 1 entities for holdout validation (default: 20,000)"
    )
    parser.add_argument(
        "--adaptive-cap",
        type=int,
        default=35,
        help="Maximum candidates per Source 1 entity in blocking stage (default: 35)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16,
        help="Number of CPU threads to use (default: 16)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_s1_and_gt(train_dir: Path):
    """Load S1 entity metadata and ground truth mapping."""
    logger.info("Locating train source1 and ground_truth files...")
    s1_path = find_file(train_dir, ["train_source1.tsv", "source1.tsv", "train_source_1.tsv", "source_1.tsv"])
    gt_path = find_file(train_dir, ["train_ground_truth.tsv", "ground_truth.tsv", "groundtruth.tsv", "train_groundtruth.tsv"])
    logger.info(f"Using S1: {s1_path.name}")
    logger.info(f"Using GT: {gt_path.name}")

    with Timer("Loading Source 1 metadata"):
        df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)

    with Timer("Loading Ground Truth pairs"):
        gt_map = {}
        with open(gt_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                s1_id = parts[0].strip()
                if len(parts) > 1 and parts[1].strip():
                    gt_map[s1_id] = {m.strip() for m in parts[1].split(",") if m.strip()}
                else:
                    gt_map[s1_id] = set()

    return df_s1, gt_map


def sample_train_val_s1(
    df_s1: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    n_train: int,
    n_val: int,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified sampling of S1 entities matching the real distribution:
      - 60% US, 40% India
      - ~5.58% singletons
    """
    np.random.seed(seed)
    total_needed = n_train + n_val

    df_s1 = df_s1.copy()
    df_s1["num_matches"] = df_s1["entity_id"].map(lambda x: len(gt_map.get(x, set())))
    df_s1["is_singleton"] = df_s1["num_matches"] == 0

    n_us = int(total_needed * 0.60)
    n_in = total_needed - n_us

    us_single = df_s1[(df_s1["country"] == "US") & (df_s1["is_singleton"])]
    us_matched = df_s1[(df_s1["country"] == "US") & (~df_s1["is_singleton"])]
    in_single = df_s1[(df_s1["country"] == "India") & (df_s1["is_singleton"])]
    in_matched = df_s1[(df_s1["country"] == "India") & (~df_s1["is_singleton"])]

    n_us_single = int(n_us * 0.0558)
    n_us_matched = n_us - n_us_single
    n_in_single = int(n_in * 0.0558)
    n_in_matched = n_in - n_in_single

    sample_us_s = us_single.sample(n=min(n_us_single, len(us_single)), random_state=seed)
    sample_us_m = us_matched.sample(n=min(n_us_matched, len(us_matched)), random_state=seed)
    sample_in_s = in_single.sample(n=min(n_in_single, len(in_single)), random_state=seed)
    sample_in_m = in_matched.sample(n=min(n_in_matched, len(in_matched)), random_state=seed)

    sampled = pd.concat([sample_us_s, sample_us_m, sample_in_s, sample_in_m]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df_train = sampled.iloc[:n_train].reset_index(drop=True)
    df_val = sampled.iloc[n_train:total_needed].reset_index(drop=True)

    logger.info(f"Sampled {len(df_train):,} Training S1 entities:")
    logger.info(f"  US: {(df_train['country'] == 'US').sum():,}, India: {(df_train['country'] == 'India').sum():,}, Singletons: {df_train['is_singleton'].sum():,}")
    logger.info(f"Sampled {len(df_val):,} Validation S1 entities:")
    logger.info(f"  US: {(df_val['country'] == 'US').sum():,}, India: {(df_val['country'] == 'India').sum():,}, Singletons: {df_val['is_singleton'].sum():,}")

    return df_train, df_val


def process_country_partition(
    country: str,
    train_dir: Path,
    df_s1_train: pd.DataFrame,
    df_s1_val: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    adaptive_cap: int = 35
) -> Tuple[List[List[float]], List[float], List[List[float]], List[float], List[Tuple[str, str]], List[str]]:
    """
    Process one country partition against the full pool:
      1. Ingest full S2/S3 for this country (~4M-5M records)
      2. Build 5-pass Inverted Index
      3. Block S1 train and S1 val entities
      4. Label candidates against Ground Truth (1 = True Match, 0 = Authentic Hard Negative)
      5. Extract 33 pairwise tabular features
    """
    logger.info("=" * 60)
    logger.info(f"Processing Full Pool Partition: {country}")

    s2_path = find_file(train_dir, ["train_source2.tsv", "source2.tsv", "train_source_2.tsv", "source_2.tsv"])
    s3_path = find_file(train_dir, ["train_source3.tsv", "source3.tsv", "train_source_3.tsv", "source_3.tsv"])

    # Preprocess S1 records
    s1_preprocessed = {}
    s1_is_train = {}

    for is_tr, sub_df in [(True, df_s1_train[df_s1_train["country"] == country]), (False, df_s1_val[df_s1_val["country"] == country])]:
        sub_ids = sub_df["entity_id"].tolist()
        sub_names = sub_df["business_name"].fillna("").tolist()
        sub_addrs = sub_df["business_address"].fillna("").tolist()
        for e_id, b_name, b_addr in zip(sub_ids, sub_names, sub_addrs):
            nv = clean_name_multiview(b_name)
            av = clean_address_multiview(b_addr, country=country)
            s1_preprocessed[e_id] = {
                "entity_id": e_id,
                "country": country,
                "core_words": set(w for w in nv["core_name"].split() if len(w) >= 2),
                **nv,
                **av
            }
            s1_is_train[e_id] = is_tr

    logger.info(f"  {country} S1 count: {len(s1_preprocessed):,} ({sum(s1_is_train.values()):,} train, {len(s1_preprocessed) - sum(s1_is_train.values()):,} val)")

    # Ingest S2 and S3 for this country from full pool
    pool_records = []
    with Timer(f"Ingesting full pool S2/S3 for {country}"):
        for s_name, s_path in [("Source2", s2_path), ("Source3", s3_path)]:
            chunksize = 500000
            for chunk in pd.read_csv(s_path, sep="\t", dtype=str, chunksize=chunksize):
                sub = chunk[chunk["country"] == country]
                if len(sub) == 0:
                    continue
                sub_ids = sub["entity_id"].tolist()
                sub_names = sub["business_name"].fillna("").tolist()
                sub_addrs = sub["business_address"].fillna("").tolist()
                for e_id, b_name, b_addr in zip(sub_ids, sub_names, sub_addrs):
                    nv = clean_name_multiview(b_name)
                    av = clean_address_multiview(b_addr, country=country)
                    pool_records.append({
                        "entity_id": e_id,
                        "country": country,
                        "core_words": set(w for w in nv["core_name"].split() if len(w) >= 2),
                        **nv,
                        **av
                    })

    logger.info(f"  Total candidate pool (S2+S3) for {country}: {len(pool_records):,}")
    pool_lookup = {r["entity_id"]: r for r in pool_records}

    # Build Inverted Indexes
    with Timer(f"Building 5-Pass Inverted Indexes for {country}"):
        engine = BlockingEngine(adaptive_cap=adaptive_cap)
        engine.build_indexes(pool_records)

    # Candidate Retrieval & Hard Negative Mining
    pairs_train = []
    pairs_val = []
    true_pairs_retrieved = 0
    total_true_pairs_in_gt = 0

    with Timer(f"Blocking {len(s1_preprocessed):,} S1 entities against full pool"):
        for s1_id, s1_rec in tqdm(s1_preprocessed.items(), desc=f"Blocking ({country})"):
            cands, _ = engine.retrieve_candidates_for_s1(s1_rec)
            gt_targets = gt_map.get(s1_id, set())
            total_true_pairs_in_gt += len(gt_targets)

            is_tr = s1_is_train[s1_id]
            for cand_id in cands:
                if cand_id in pool_lookup:
                    is_true = cand_id in gt_targets
                    if is_true:
                        true_pairs_retrieved += 1
                    if is_tr:
                        pairs_train.append((s1_id, cand_id, 1.0 if is_true else 0.0))
                    else:
                        pairs_val.append((s1_id, cand_id, 1.0 if is_true else 0.0))

    blocking_recall = true_pairs_retrieved / total_true_pairs_in_gt if total_true_pairs_in_gt > 0 else 1.0
    logger.info(f"  Blocking Recall on full pool ({country}): {blocking_recall*100:.2f}% ({true_pairs_retrieved:,} / {total_true_pairs_in_gt:,})")
    logger.info(f"  Candidate pairs: {len(pairs_train):,} train, {len(pairs_val):,} val")

    # Extract Tabular Features
    X_train, y_train = [], []
    X_val, y_val = [], []
    eval_pairs = []
    feature_names = None

    with Timer(f"Feature Extraction for {len(pairs_train) + len(pairs_val):,} pairs in {country}"):
        for s1_id, cand_id, label in tqdm(pairs_train, desc=f"Feats Train ({country})"):
            feats = extract_pair_features(s1_preprocessed[s1_id], pool_lookup[cand_id])
            if feature_names is None:
                feature_names = sorted(feats.keys())
            X_train.append([feats[k] for k in feature_names])
            y_train.append(label)

        for s1_id, cand_id, label in tqdm(pairs_val, desc=f"Feats Val ({country})"):
            feats = extract_pair_features(s1_preprocessed[s1_id], pool_lookup[cand_id])
            X_val.append([feats[k] for k in feature_names])
            y_val.append(label)
            eval_pairs.append((s1_id, cand_id))

    # Free memory
    del pool_records
    del pool_lookup
    del s1_preprocessed
    del engine
    del pairs_train
    del pairs_val
    gc.collect()

    return X_train, y_train, X_val, y_val, eval_pairs, feature_names


def train_full_pool():
    args = parse_args()
    if not HAS_LIGHTGBM:
        logger.error("LightGBM is not installed! Please run: pip install lightgbm")
        return

    train_dir = Path(args.train_dir).resolve()
    model_out = Path(args.model_out).resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Full-Pool Hard Negative LightGBM Training ===")
    logger.info(f"Train Directory: {train_dir}")
    logger.info(f"Model Output:    {model_out}")

    # 1. Load S1 and GT
    df_s1, gt_map = load_s1_and_gt(train_dir)
    logger.info(f"Total S1 entities: {len(df_s1):,}, Total GT entries: {len(gt_map):,}")

    # 2. Sample Train and Validation S1
    df_s1_train, df_s1_val = sample_train_val_s1(
        df_s1, gt_map,
        n_train=args.train_sample_s1,
        n_val=args.val_sample_s1,
        seed=args.seed
    )

    # 3. Process US and India partitions
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    all_eval_pairs = []
    feature_names = None

    for country in ["US", "India"]:
        X_tr, y_tr, X_v, y_v, eval_p, fnames = process_country_partition(
            country=country,
            train_dir=train_dir,
            df_s1_train=df_s1_train,
            df_s1_val=df_s1_val,
            gt_map=gt_map,
            adaptive_cap=args.adaptive_cap
        )
        if feature_names is None:
            feature_names = fnames
        all_X_train.extend(X_tr)
        all_y_train.extend(y_tr)
        all_X_val.extend(X_v)
        all_y_val.extend(y_v)
        all_eval_pairs.extend(eval_p)

    X_train = np.array(all_X_train, dtype=np.float32)
    y_train = np.array(all_y_train, dtype=np.float32)
    X_val = np.array(all_X_val, dtype=np.float32)
    y_val = np.array(all_y_val, dtype=np.float32)

    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    logger.info("=" * 60)
    logger.info(f"COMBINED TRAINING SET: {len(y_train):,} pairs")
    logger.info(f"  True Positives: {n_pos_train:,} ({n_pos_train/len(y_train)*100:.2f}%)")
    logger.info(f"  Hard Negatives: {n_neg_train:,} ({n_neg_train/len(y_train)*100:.2f}%)")
    logger.info(f"COMBINED VALIDATION SET: {len(y_val):,} pairs ({int(y_val.sum()):,} positives, {len(y_val) - int(y_val.sum()):,} negatives)")

    # 4. Train LightGBM Binary Classifier
    with Timer("Training LightGBM Classifier with Hard Negatives"):
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "learning_rate": 0.08,
            "num_leaves": 63,
            "max_depth": 8,
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
            num_boost_round=400,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True), lgb.log_evaluation(period=50)]
        )

        booster.save_model(str(model_out))
        logger.info(f"Trained LightGBM model successfully saved to: {model_out}")

    # Feature Importance
    importance = booster.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    logger.info("--- Top 15 Most Important Features (by Gain) ---")
    for fname, imp in feat_imp[:15]:
        logger.info(f"  {fname:<30}: {imp:.2f}")

    # 5. Holdout Evaluation & Threshold Calibration
    with Timer(f"Holdout Evaluation & Threshold Grid Search on {len(df_s1_val):,} entities"):
        val_preds = booster.predict(X_val)

        val_pairs_scored = [
            (all_eval_pairs[i][0], all_eval_pairs[i][1], float(val_preds[i]))
            for i in range(len(all_eval_pairs))
        ]

        val_s1_ids = set(df_s1_val["entity_id"])
        val_gt = {s1_id: gt_map[s1_id] for s1_id in val_s1_ids}
        val_countries = dict(zip(df_s1_val["entity_id"], df_s1_val["country"]))

        # Sort candidate scores descending for 1-to-at-most-1 bipartite resolution
        val_pairs_scored.sort(key=lambda x: x[2], reverse=True)

        best_tau = 0.55
        best_score = 0.0
        best_metrics = {}

        logger.info("--- Threshold Calibration Results (Macro-F0.5) ---")
        for tau in np.arange(0.40, 0.80, 0.03):
            tau = round(float(tau), 2)
            assigned_s23 = set()
            s1_s2_count = defaultdict(int)
            s1_s3_count = defaultdict(int)
            preds_tau = defaultdict(set)
            for s1_id, cand_id, score in val_pairs_scored:
                if score < tau:
                    break
                if cand_id not in assigned_s23:
                    is_s2 = cand_id.startswith("S2")
                    if is_s2 and s1_s2_count[s1_id] >= 5:
                        continue
                    if not is_s2 and s1_s3_count[s1_id] >= 5:
                        continue
                    if (s1_s2_count[s1_id] + s1_s3_count[s1_id]) >= 8:
                        continue

                    assigned_s23.add(cand_id)
                    if is_s2:
                        s1_s2_count[s1_id] += 1
                    else:
                        s1_s3_count[s1_id] += 1
                    preds_tau[s1_id].add(cand_id)

            for s1_id in val_s1_ids:
                if s1_id not in preds_tau:
                    preds_tau[s1_id] = set()

            res = evaluate_macro_f05(val_gt, preds_tau, countries=val_countries)
            f05 = res["macro_f05"]
            prec = res["macro_precision"]
            rec = res["macro_recall"]
            logger.info(f"  tau = {tau:.2f} -> Macro-F0.5: {f05:.4f} (Precision: {prec:.4f}, Recall: {rec:.4f})")

            if f05 > best_score:
                best_score = f05
                best_tau = tau
                best_metrics = res

        logger.info("=" * 60)
        logger.info(f">>> OPTIMAL DECISION THRESHOLD: tau* = {best_tau:.2f} (Macro-F0.5: {best_score:.4f}) <<<")
        logger.info(f"  Holdout Macro Precision: {best_metrics.get('macro_precision', 0):.4f}")
        logger.info(f"  Holdout Macro Recall:    {best_metrics.get('macro_recall', 0):.4f}")
        logger.info(f"  Singleton F0.5 Score:    {best_metrics.get('singleton_f05', 0):.4f}")
        logger.info(f"  Country Breakdown:")
        for c, sc in best_metrics.get("by_country", {}).items():
            logger.info(f"    {c}: {sc:.4f}")

        # Save calibration metadata
        cal_path = Path("./data/calibration_results.json")
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump({
                "optimal_threshold": best_tau,
                "macro_f05": best_score,
                "metrics": best_metrics,
                "num_features": len(feature_names),
                "features": feature_names
            }, f, indent=2)
        logger.info(f"Saved calibration metadata to: {cal_path}")


if __name__ == "__main__":
    train_full_pool()
