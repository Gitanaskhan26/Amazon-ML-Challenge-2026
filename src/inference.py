#!/usr/bin/env python3
"""
Full Test Inference Pipeline.

Generates the official submission deliverables:
  1. output/candidate_pairs.tsv (blocking candidates fed to model)
  2. output/matching_results.tsv (final scored entity matches)
Runs official validator (utils/validate_submission.py) automatically.

Designed to scale seamlessly to the 11.7M test set on Windows (Intel i9 + 128 GB RAM)
and Linux/macOS.
"""

import os
import sys
import gc
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger, write_submission_tsv, find_file
from src.preprocessing import clean_name_multiview, clean_address_multiview
from src.features import extract_pair_features
from src.blocking import BlockingEngine
from src.train import (
    heuristic_score_pair,
    solve_greedy_bipartite_assignment,
    select_optimal_prefix_per_entity
)



def parse_args():
    parser = argparse.ArgumentParser(description="Run Full Test Inference Pipeline.")
    parser.add_argument(
        "--test-dir",
        type=str,
        default="/Users/anaskhan/Downloads/student_resource/dataset/test",
        help="Path to test dataset directory (containing test_source1.tsv, test_source2.tsv, test_source3.tsv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save matching_results.tsv and candidate_pairs.tsv"
    )
    parser.add_argument(
        "--adaptive-cap",
        type=int,
        default=35,
        help="Maximum candidates per Source 1 entity in blocking stage (default: 35)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./data/lgb_model.txt",
        help="Path to trained LightGBM model (if exists, uses model; otherwise uses heuristic)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for matching (default: calibrated optimal threshold or 0.58)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run utils/validate_submission.py on generated outputs"
    )
    return parser.parse_args()



def run_pipeline():
    args = parse_args()
    test_dir = Path(args.test_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_path = find_file(test_dir, ["test_source1.tsv", "source1.tsv", "test_source_1.tsv", "source_1.tsv"])
    s2_path = find_file(test_dir, ["test_source2.tsv", "source2.tsv", "test_source_2.tsv", "source_2.tsv"])
    s3_path = find_file(test_dir, ["test_source3.tsv", "source3.tsv", "test_source_3.tsv", "source_3.tsv"])


    logger.info("=== Starting Business Entity Resolution Pipeline ===")
    logger.info(f"Test Directory: {test_dir}")
    logger.info(f"Output Directory: {out_dir}")

    # Load S1
    with Timer("Loading test_source1.tsv"):
        df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    logger.info(f"Total Test Source 1 Entities: {len(df_s1):,}")

    # Detect countries (US, India, France)
    countries = list(df_s1["country"].unique())
    logger.info(f"Countries present in test set: {countries}")

    # Check for trained LightGBM model
    model_path = Path(args.model_path).resolve()
    booster = None
    feature_names = None
    if model_path.exists():
        try:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(model_path))
            feature_names = booster.feature_name()
            logger.info(f"Loaded trained LightGBM model from: {model_path} ({len(feature_names)} features)")
        except Exception as e:
            logger.warning(f"Could not load LightGBM model: {e}. Falling back to heuristic scorer.")
    else:
        logger.info("No trained LightGBM model found. Using heuristic scoring engine.")

    # Resolve Decision Thresholds
    cal_path = Path(__file__).resolve().parent.parent / "data" / "calibration_results.json"
    cal_by_country = {}
    default_threshold = args.threshold
    if cal_path.exists():
        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                cal_data = json.load(f)
                cal_by_country = cal_data.get("optimal_thresholds_by_country", {})
                if default_threshold is None:
                    cal_tau = cal_data.get("optimal_threshold")
                    if cal_tau is not None:
                        default_threshold = float(cal_tau)
        except Exception:
            pass

    if default_threshold is None:
        default_threshold = 0.58
    logger.info(f"Default Matching Threshold: {default_threshold:.2f}")
    if cal_by_country:
        logger.info(f"Country-Specific Calibrated Thresholds: {cal_by_country}")

    final_candidates = {}
    final_matches = {}

    for country in countries:
        logger.info("=" * 60)
        logger.info(f"Processing Country Partition: {country}")
        if args.threshold is not None:
            country_tau = args.threshold
        elif country in cal_by_country:
            country_tau = float(cal_by_country[country])
            logger.info(f"  Using Country-Calibrated Threshold for {country}: tau = {country_tau:.2f}")
        else:
            country_tau = default_threshold
            logger.info(f"  Using Default Threshold for {country}: tau = {country_tau:.2f}")

        s1_country = df_s1[df_s1["country"] == country]
        logger.info(f"  {country} S1 count: {len(s1_country):,}")

        # Stream S2 and S3 for this country
        pool_records = []
        with Timer(f"Ingesting & Preprocessing S2/S3 for {country}"):
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

        # Build Inverted Indexes for this country
        with Timer(f"Building 5-Pass Inverted Indexes for {country}"):
            engine = BlockingEngine(adaptive_cap=args.adaptive_cap)
            engine.build_indexes(pool_records)

        # Preprocess S1 records
        s1_preprocessed = []
        s1_ids = s1_country["entity_id"].tolist()
        s1_names = s1_country["business_name"].fillna("").tolist()
        s1_addrs = s1_country["business_address"].fillna("").tolist()
        for e_id, b_name, b_addr in zip(s1_ids, s1_names, s1_addrs):
            nv = clean_name_multiview(b_name)
            av = clean_address_multiview(b_addr, country=country)
            s1_preprocessed.append({
                "entity_id": e_id,
                "country": country,
                "core_words": set(w for w in nv["core_name"].split() if len(w) >= 2),
                **nv,
                **av
            })


        # 1. Fast Candidate Retrieval (Pure Inverted Index, ~2,000+ it/s)
        s1_lookup = {r["entity_id"]: r for r in s1_preprocessed}
        pairs_to_score = []

        with Timer(f"Candidate Retrieval for {len(s1_preprocessed):,} S1 entities in {country}"):
            for s1_rec in tqdm(s1_preprocessed, desc=f"Blocking ({country})"):
                s1_id = s1_rec["entity_id"]
                cands, _ = engine.retrieve_candidates_for_s1(s1_rec)
                final_candidates[s1_id] = cands
                for cand_id in cands:
                    if cand_id in pool_lookup:
                        pairs_to_score.append((s1_id, cand_id))

        logger.info(f"Total candidate pairs to score for {country}: {len(pairs_to_score):,}")

        # 2. Batched Scoring with LightGBM (chunks of 100,000 pairs!)
        candidate_scores = []
        with Timer(f"Batched Scoring of {len(pairs_to_score):,} pairs in {country}"):
            if booster is not None:
                batch_size = 100000
                for start_idx in tqdm(range(0, len(pairs_to_score), batch_size), desc=f"Scoring Batches ({country})"):
                    batch_pairs = pairs_to_score[start_idx : start_idx + batch_size]
                    feat_matrix = []
                    for s1_id, cand_id in batch_pairs:
                        feats = extract_pair_features(s1_lookup[s1_id], pool_lookup[cand_id])
                        feat_matrix.append([feats.get(k, 0.0) for k in feature_names])

                    probs = booster.predict(np.array(feat_matrix, dtype=np.float32))
                    for (s1_id, cand_id), prob in zip(batch_pairs, probs):
                        candidate_scores.append((s1_id, cand_id, float(prob)))
            else:
                for s1_id, cand_id in pairs_to_score:
                    score = heuristic_score_pair(s1_lookup[s1_id], pool_lookup[cand_id])
                    candidate_scores.append((s1_id, cand_id, score))


        # Enforce 1-to-at-most-1 Bipartite Invariant & Calibrated Thresholding
        with Timer(f"Enforcing 1-to-at-most-1 Assignment (tau={country_tau:.2f}) for {country}"):
            candidate_scores.sort(key=lambda x: x[2], reverse=True)
            assigned_s23 = set()
            s1_s2_count = defaultdict(int)
            s1_s3_count = defaultdict(int)
            for s1_id, cand_id, score in candidate_scores:
                if score < country_tau:
                    break  # since candidate_scores is sorted descending
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

                    if s1_id not in final_matches:
                        final_matches[s1_id] = set()
                    final_matches[s1_id].add(cand_id)

            for s1_rec in s1_preprocessed:
                s1_id = s1_rec["entity_id"]
                if s1_id not in final_matches:
                    final_matches[s1_id] = set()

        # Free memory before next country partition
        del candidate_scores
        del pool_records
        del pool_lookup
        del pairs_to_score
        del s1_lookup
        del s1_preprocessed
        gc.collect()


    # Write Deliverables
    cand_path = out_dir / "candidate_pairs.tsv"
    match_path = out_dir / "matching_results.tsv"

    logger.info("=" * 60)
    logger.info("Writing Official Deliverables...")

    with Timer("Saving candidate_pairs.tsv"):
        write_submission_tsv(
            cand_path,
            final_candidates,
            col_s1="source1_entity_id",
            col_target="candidate_entity_ids"
        )
    logger.info(f"Saved: {cand_path}")

    with Timer("Saving matching_results.tsv"):
        write_submission_tsv(
            match_path,
            final_matches,
            col_s1="source1_entity_id",
            col_target="matched_entity_ids"
        )
    logger.info(f"Saved: {match_path}")

    # Validate deliverables
    if args.validate:
        logger.info("=" * 60)
        logger.info("Running Official Validation Script (utils/validate_submission.py)...")
        val_script = Path(__file__).resolve().parent.parent / "utils" / "validate_submission.py"
        cmd = [
            sys.executable,
            str(val_script),
            "--matching", str(match_path),
            "--candidate", str(cand_path),
            "--test-dir", str(test_dir)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            logger.info(">>> VALIDATION STATUS: PASS (Exit code 0). SUBMISSION IS 100% VALID! <<<")
        else:
            logger.error(">>> VALIDATION FAILED! Check output above. <<<")


if __name__ == "__main__":
    run_pipeline()
