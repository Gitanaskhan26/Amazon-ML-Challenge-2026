#!/usr/bin/env python3
"""
Full Local Validation Benchmark.

Runs the exact end-to-end inference pipeline on data/val,
and evaluates against val_ground_truth.tsv using the official Macro F_0.5 metric.
Reports exact Macro F_0.5, Macro Precision, Macro Recall, and per-country metrics.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
import gc

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger
from src.preprocessing import clean_name_multiview, clean_address_multiview
from src.features import extract_pair_features
from src.blocking import BlockingEngine
from src.metrics import evaluate_macro_f05


def run_local_evaluation(val_dir: Path, model_path: Path, threshold: float = 0.48, adaptive_cap: int = 35):
    logger.info("=== Starting Full Local Validation Benchmark ===")
    logger.info(f"Validation Dir: {val_dir}")
    logger.info(f"Model Path:     {model_path}")
    logger.info(f"Threshold:      {threshold}")

    s1_path = val_dir / "val_source1.tsv"
    s2_path = val_dir / "val_source2.tsv"
    s3_path = val_dir / "val_source3.tsv"
    gt_path = val_dir / "val_ground_truth.tsv"

    # 1. Load Ground Truth
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

    total_gt_pairs = sum(len(v) for v in gt_map.values())
    logger.info(f"Total S1 entities in Ground Truth: {len(gt_map):,}")
    logger.info(f"Total True GT Pairs:               {total_gt_pairs:,}")

    # 2. Load S1
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    countries = dict(zip(df_s1["entity_id"], df_s1["country"]))

    # 3. Load Model
    booster = lgb.Booster(model_file=str(model_path))
    feature_names = booster.feature_name()
    logger.info(f"Loaded LightGBM model with {len(feature_names)} features.")

    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str)

    final_candidates = {}
    final_matches = {}
    blocked_gt_pairs = set()

    for country in ["US", "India"]:
        logger.info("=" * 60)
        logger.info(f"Evaluating Partition: {country}")
        s1_country = df_s1[df_s1["country"] == country]
        s2_country = df_s2[df_s2["country"] == country]
        s3_country = df_s3[df_s3["country"] == country]

        # Ingest pool
        pool_records = []
        for df_pool in [s2_country, s3_country]:
            ids = df_pool["entity_id"].tolist()
            names = df_pool["business_name"].fillna("").tolist()
            addrs = df_pool["business_address"].fillna("").tolist()
            for eid, name, addr in zip(ids, names, addrs):
                nv = clean_name_multiview(name)
                av = clean_address_multiview(addr, country=country)
                pool_records.append({
                    "entity_id": eid,
                    "country": country,
                    "core_words": set(w for w in nv["core_name"].split() if len(w) >= 2),
                    **nv,
                    **av
                })

        pool_lookup = {r["entity_id"]: r for r in pool_records}

        # Build Inverted Indexes
        engine = BlockingEngine(adaptive_cap=adaptive_cap)
        engine.build_indexes(pool_records)

        # Preprocess S1
        s1_preprocessed = []
        s1_ids = s1_country["entity_id"].tolist()
        s1_names = s1_country["business_name"].fillna("").tolist()
        s1_addrs = s1_country["business_address"].fillna("").tolist()
        for eid, name, addr in zip(s1_ids, s1_names, s1_addrs):
            nv = clean_name_multiview(name)
            av = clean_address_multiview(addr, country=country)
            s1_preprocessed.append({
                "entity_id": eid,
                "country": country,
                "core_words": set(w for w in nv["core_name"].split() if len(w) >= 2),
                **nv,
                **av
            })

        s1_lookup = {r["entity_id"]: r for r in s1_preprocessed}
        pairs_to_score = []

        # Candidate Retrieval
        with Timer(f"Blocking {len(s1_preprocessed):,} entities in {country}"):
            for s1_rec in tqdm(s1_preprocessed, desc=f"Blocking ({country})"):
                s1_id = s1_rec["entity_id"]
                cands, _ = engine.retrieve_candidates_for_s1(s1_rec)
                final_candidates[s1_id] = cands
                for cid in cands:
                    if cid in pool_lookup:
                        pairs_to_score.append((s1_id, cid))
                        if cid in gt_map.get(s1_id, set()):
                            blocked_gt_pairs.add((s1_id, cid))

        logger.info(f"Candidate pairs to score for {country}: {len(pairs_to_score):,}")

        # Batched Scoring
        candidate_scores = []
        batch_size = 100000
        with Timer(f"Scoring batches for {country}"):
            for start_idx in range(0, len(pairs_to_score), batch_size):
                batch_pairs = pairs_to_score[start_idx : start_idx + batch_size]
                feat_matrix = []
                for s1_id, cid in batch_pairs:
                    feats = extract_pair_features(s1_lookup[s1_id], pool_lookup[cid])
                    feat_matrix.append([feats.get(k, 0.0) for k in feature_names])
                probs = booster.predict(np.array(feat_matrix, dtype=np.float32))
                for (s1_id, cid), prob in zip(batch_pairs, probs):
                    candidate_scores.append((s1_id, cid, float(prob)))

        # 1-to-at-most-1 Bipartite Resolution & Thresholding
        candidate_scores.sort(key=lambda x: x[2], reverse=True)
        assigned_s23 = set()
        for s1_id, cid, score in candidate_scores:
            if score < threshold:
                break
            if cid not in assigned_s23:
                assigned_s23.add(cid)
                if s1_id not in final_matches:
                    final_matches[s1_id] = set()
                final_matches[s1_id].add(cid)

        for s1_rec in s1_preprocessed:
            s1_id = s1_rec["entity_id"]
            if s1_id not in final_matches:
                final_matches[s1_id] = set()

        del candidate_scores
        del pool_records
        del pool_lookup
        del pairs_to_score
        del s1_lookup
        del s1_preprocessed
        gc.collect()

    # Blocking Recall
    blocking_recall = len(blocked_gt_pairs) / total_gt_pairs
    logger.info("=" * 60)
    logger.info(f">>> BLOCKING PAIR-LEVEL RECALL: {blocking_recall:.2%} ({len(blocked_gt_pairs):,} / {total_gt_pairs:,}) <<<")

    # Evaluate Macro F_0.5
    logger.info("Calculating official Macro F_0.5 metrics...")
    res = evaluate_macro_f05(gt_map, final_matches, countries=countries)

    logger.info("=" * 60)
    logger.info(f">>> OVERALL MACRO F_0.5 SCORE: {res['macro_f05']:.4f} <<<")
    logger.info(f"  Macro Precision:    {res['macro_precision']:.4f}")
    logger.info(f"  Macro Recall:       {res['macro_recall']:.4f}")
    logger.info(f"  Singleton Score:    {res['singleton_score']:.4f} ({res['n_singletons']:,} entities)")
    logger.info(f"  Non-Singleton F0.5: {res['non_singleton_f05']:.4f}")

    if "by_country" in res:
        logger.info("--- Metrics By Country ---")
        for c, sc in res["by_country"].items():
            logger.info(f"  {c:<10}: {sc:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", type=str, default="./data/val")
    parser.add_argument("--model-path", type=str, default="./data/lgb_model.txt")
    parser.add_argument("--threshold", type=float, default=0.48)
    parser.add_argument("--adaptive-cap", type=int, default=35)
    args = parser.parse_args()
    run_local_evaluation(Path(args.val_dir), Path(args.model_path), args.threshold, args.adaptive_cap)
