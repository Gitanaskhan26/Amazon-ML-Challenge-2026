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
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger, write_submission_tsv, find_file
from src.preprocessing import clean_name_multiview, clean_address_multiview
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
        default=25,
        help="Maximum candidates per Source 1 entity in blocking stage"
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

    final_candidates = {}
    final_matches = {}

    for country in countries:
        logger.info("=" * 60)
        logger.info(f"Processing Country Partition: {country}")
        s1_country = df_s1[df_s1["country"] == country]
        logger.info(f"  {country} S1 count: {len(s1_country):,}")

        # Stream S2 and S3 for this country
        pool_records = []
        with Timer(f"Ingesting & Preprocessing S2/S3 for {country}"):
            for s_name, s_path in [("Source2", s2_path), ("Source3", s3_path)]:
                chunksize = 250000
                for chunk in pd.read_csv(s_path, sep="\t", dtype=str, chunksize=chunksize):
                    sub = chunk[chunk["country"] == country]
                    for _, r in sub.iterrows():
                        nv = clean_name_multiview(r.get("business_name", ""))
                        av = clean_address_multiview(r.get("business_address", ""), country=country)
                        pool_records.append({
                            "entity_id": r["entity_id"],
                            "country": country,
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
        for _, r in s1_country.iterrows():
            nv = clean_name_multiview(r.get("business_name", ""))
            av = clean_address_multiview(r.get("business_address", ""), country=country)
            s1_preprocessed.append({
                "entity_id": r["entity_id"],
                "country": country,
                **nv,
                **av
            })

        # Candidate Retrieval & Pairwise Scoring
        candidate_scores = []
        with Timer(f"Blocking & Scoring {len(s1_preprocessed):,} S1 entities in {country}"):
            for s1_rec in tqdm(s1_preprocessed, desc=f"Inference ({country})"):
                s1_id = s1_rec["entity_id"]
                cands, _ = engine.retrieve_candidates_for_s1(s1_rec)
                final_candidates[s1_id] = cands

                for cand_id in cands:
                    if cand_id in pool_lookup:
                        score = heuristic_score_pair(s1_rec, pool_lookup[cand_id])
                        candidate_scores.append((s1_id, cand_id, score))

        # Enforce 1-to-at-most-1 Bipartite Invariant
        with Timer(f"Enforcing 1-to-at-most-1 Conflict Resolution for {country}"):
            assigned_mapping = solve_greedy_bipartite_assignment(candidate_scores)

        # Per-Entity Expected F0.5 Prefix Selection & Singleton Gating
        with Timer(f"Optimizing Expected-F0.5 Prefix Selection for {country}"):
            cand_scores_by_s1 = defaultdict(list)
            for s1_id, cand_id, score in candidate_scores:
                cand_scores_by_s1[s1_id].append((cand_id, score))

            for s1_rec in s1_preprocessed:
                s1_id = s1_rec["entity_id"]
                # Only consider candidates that survived conflict assignment
                survived = assigned_mapping.get(s1_id, set())
                cand_list = [(c, sc) for c, sc in cand_scores_by_s1[s1_id] if c in survived]

                # Select optimal prefix (prunes singletons to [])
                selected_matches = select_optimal_prefix_per_entity(cand_list)
                final_matches[s1_id] = selected_matches

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
