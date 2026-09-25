#!/usr/bin/env python3
"""
Validation Split & Leave-One-Country-Out (LOCO) Probe Generator.

Generates a representative 100,000-entity local validation split:
  - Stratified by country (60% US, 40% India)
  - Stratified by singleton status (~5.58% singletons with 0 matches)
  - Extracts matching S2/S3 records + 25% unlinked distractor pool
  - Generates Leave-One-Country-Out (LOCO) probes to benchmark France zero-shot transfer penalty

Compatible with both Windows and Linux/macOS.
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger, write_submission_tsv, find_file


def parse_args():
    parser = argparse.ArgumentParser(description="Create stratified validation split and LOCO probes.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/anaskhan/Downloads/student_resource/dataset/train",
        help="Path to training dataset directory containing source1, source2, source3, and ground_truth files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/val",
        help="Directory to save the validation split and LOCO probes"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=100000,
        help="Number of Source 1 entities in validation split (default: 100,000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_s1_and_gt(data_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """Load S1 entity metadata and ground truth matches, auto-detecting file names."""
    logger.info("Locating source1 and ground_truth files...")
    s1_path = find_file(data_dir, ["train_source1.tsv", "source1.tsv", "train_source_1.tsv", "source_1.tsv"])
    gt_path = find_file(data_dir, ["train_ground_truth.tsv", "ground_truth.tsv", "groundtruth.tsv", "train_groundtruth.tsv"])
    logger.info(f"Using S1 file: {s1_path.name}")
    logger.info(f"Using GT file: {gt_path.name}")


    # Read S1
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)

    # Read Ground Truth
    gt_map = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                gt_map[s1_id] = {m.strip() for m in parts[1].split(",") if m.strip()}
            else:
                gt_map[s1_id] = set()

    return df_s1, gt_map


def create_split():
    args = parse_args()
    random.seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with Timer("Loading S1 and Ground Truth"):
        df_s1, gt_map = load_s1_and_gt(data_dir)

    logger.info(f"Total S1 entities loaded: {len(df_s1):,}")

    # Annotate singleton status
    df_s1["num_matches"] = df_s1["entity_id"].map(lambda x: len(gt_map.get(x, set())))
    df_s1["is_singleton"] = df_s1["num_matches"] == 0

    # Stratified sampling
    # Target proportions:
    # US: 60%, India: 40%
    # Singletons: 5.58%
    n_us = int(args.val_size * 0.60)
    n_in = args.val_size - n_us

    us_single = df_s1[(df_s1["country"] == "US") & (df_s1["is_singleton"])]
    us_matched = df_s1[(df_s1["country"] == "US") & (~df_s1["is_singleton"])]

    in_single = df_s1[(df_s1["country"] == "India") & (df_s1["is_singleton"])]
    in_matched = df_s1[(df_s1["country"] == "India") & (~df_s1["is_singleton"])]

    n_us_single = int(n_us * 0.0558)
    n_us_matched = n_us - n_us_single

    n_in_single = int(n_in * 0.0558)
    n_in_matched = n_in - n_in_single

    val_us_single = us_single.sample(n=n_us_single, random_state=args.seed)
    val_us_matched = us_matched.sample(n=n_us_matched, random_state=args.seed)
    val_in_single = in_single.sample(n=n_in_single, random_state=args.seed)
    val_in_matched = in_matched.sample(n=n_in_matched, random_state=args.seed)

    val_s1 = pd.concat([val_us_single, val_us_matched, val_in_single, val_in_matched]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_s1_ids = set(val_s1["entity_id"])

    logger.info(f"Validation S1 created: {len(val_s1):,} records")
    logger.info(f"  US: {(val_s1['country'] == 'US').sum():,} ({(val_s1['country'] == 'US').mean()*100:.1f}%)")
    logger.info(f"  India: {(val_s1['country'] == 'India').sum():,} ({(val_s1['country'] == 'India').mean()*100:.1f}%)")
    logger.info(f"  Singletons: {val_s1['is_singleton'].sum():,} ({val_s1['is_singleton'].mean()*100:.2f}%)")

    # Save validation S1
    val_s1_path = out_dir / "val_source1.tsv"
    val_s1[["entity_id", "business_name", "business_address", "country"]].to_csv(val_s1_path, sep="\t", index=False)
    logger.info(f"Saved: {val_s1_path}")

    # Save validation Ground Truth
    val_gt = {s1_id: gt_map[s1_id] for s1_id in val_s1_ids}
    val_gt_path = out_dir / "val_ground_truth.tsv"
    write_submission_tsv(val_gt_path, val_gt, col_s1="source1_entity_id", col_target="matched_entity_ids")
    logger.info(f"Saved: {val_gt_path}")

    # Collect needed S2 and S3 IDs
    needed_s2 = set()
    needed_s3 = set()
    for s1_id in val_s1_ids:
        for mid in gt_map.get(s1_id, set()):
            if mid.startswith("S2-"):
                needed_s2.add(mid)
            elif mid.startswith("S3-"):
                needed_s3.add(mid)

    logger.info(f"Needed true match targets: {len(needed_s2):,} S2, {len(needed_s3):,} S3")

    # Add 25% unlinked distractor ratio
    n_distractors_s2 = int(len(needed_s2) * 0.25)
    n_distractors_s3 = int(len(needed_s3) * 0.25)

    # Stream S2 and S3 to extract targets + distractors
    for src_name, needed_set, n_dist in [("source2", needed_s2, n_distractors_s2), ("source3", needed_s3, n_distractors_s3)]:
        src_path = find_file(data_dir, [f"train_{src_name}.tsv", f"{src_name}.tsv", f"train_{src_name[:6]}_{src_name[6:]}.tsv", f"{src_name[:6]}_{src_name[6:]}.tsv"])
        out_src_path = out_dir / f"val_{src_name}.tsv"
        logger.info(f"Filtering {src_name} ({src_path.name}) for validation pool...")


        matched_rows = []
        candidate_distractors = []

        chunksize = 250000
        for chunk in tqdm(pd.read_csv(src_path, sep="\t", dtype=str, chunksize=chunksize), desc=f"Scanning {src_name}"):
            # Matches
            is_match = chunk["entity_id"].isin(needed_set)
            if is_match.any():
                matched_rows.append(chunk[is_match])

            # Sample potential distractors (rows not in GT needed_set)
            non_match = chunk[~is_match]
            if len(candidate_distractors) < n_dist * 2:
                sample_k = min(len(non_match), 10000)
                candidate_distractors.append(non_match.sample(n=sample_k, random_state=args.seed))

        df_matched = pd.concat(matched_rows)
        df_dist = pd.concat(candidate_distractors).sample(n=min(n_dist, sum(len(x) for x in candidate_distractors)), random_state=args.seed)

        df_val_src = pd.concat([df_matched, df_dist]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        df_val_src.to_csv(out_src_path, sep="\t", index=False)
        logger.info(f"Saved: {out_src_path} ({len(df_val_src):,} records, {len(df_matched):,} true matches + {len(df_dist):,} distractors)")

    # Save LOCO probe splits
    # US probe: 15,000 US entities
    # India probe: 15,000 India entities
    loco_dir = out_dir / "loco_probes"
    loco_dir.mkdir(parents=True, exist_ok=True)

    us_val = val_s1[val_s1["country"] == "US"].head(15000)
    in_val = val_s1[val_s1["country"] == "India"].head(15000)

    us_val[["entity_id", "business_name", "business_address", "country"]].to_csv(loco_dir / "probe_us_s1.tsv", sep="\t", index=False)
    in_val[["entity_id", "business_name", "business_address", "country"]].to_csv(loco_dir / "probe_in_s1.tsv", sep="\t", index=False)

    write_submission_tsv(loco_dir / "probe_us_gt.tsv", {x: gt_map[x] for x in us_val["entity_id"]})
    write_submission_tsv(loco_dir / "probe_in_gt.tsv", {x: gt_map[x] for x in in_val["entity_id"]})

    logger.info(f"LOCO probe splits successfully saved to: {loco_dir}")
    logger.info("Validation split generation complete!")


if __name__ == "__main__":
    create_split()
