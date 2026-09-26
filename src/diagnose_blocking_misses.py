#!/usr/bin/env python3
"""
Diagnostic Script: Analyze Exact False Negative Matches Missed by Blocking.

Samples Indian entities from train, runs blocking, isolates the true ground truth pairs
that were missed by all blocking passes, and prints detailed forensic analysis of why
they were missed (shared tokens, missing postal codes, phonetic variants, etc.).
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import clean_name_multiview, clean_address_multiview
from src.blocking import BlockingEngine
from src.utils import Timer, logger, find_file


def diagnose(train_dir_path: str, sample_s1: int = 2000, adaptive_cap: int = 80):
    train_dir = Path(train_dir_path).resolve()
    logger.info(f"=== Diagnosing Blocking False Negatives on {train_dir} (Adaptive Cap = {adaptive_cap}) ===")

    # 1. Load S1
    s1_file = find_file(train_dir, ["train_source1.tsv", "source1.tsv"])
    gt_file = find_file(train_dir, ["train_ground_truth.tsv", "ground_truth.tsv"])
    s2_file = find_file(train_dir, ["train_source2.tsv", "source2.tsv"])
    s3_file = find_file(train_dir, ["train_source3.tsv", "source3.tsv"])

    logger.info("Loading S1 and GT...")
    df_s1 = pd.read_csv(s1_file, sep="\t", dtype=str)
    s1_india = df_s1[df_s1["country"] == "India"].sample(n=min(sample_s1, len(df_s1)), random_state=42)
    s1_ids = set(s1_india["entity_id"])

    # Load GT for these S1
    gt_map = defaultdict(set)
    target_cand_ids = set()
    with open(gt_file, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in s1_ids and len(parts) > 1 and parts[1].strip():
                targets = {x.strip() for x in parts[1].split(",") if x.strip()}
                gt_map[parts[0]] = targets
                target_cand_ids.update(targets)

    logger.info(f"Sampled {len(s1_india):,} India S1 entities with {sum(len(v) for v in gt_map.values()):,} true GT links.")

    # 2. Ingest S2/S3 records that match target_cand_ids + a background pool
    logger.info("Ingesting candidate records from S2 and S3...")
    pool_records = []
    found_targets = set()
    for s_name, s_path in [("Source2", s2_file), ("Source3", s3_file)]:
        chunksize = 200000
        for chunk in pd.read_csv(s_path, sep="\t", dtype=str, chunksize=chunksize):
            sub = chunk[chunk["country"] == "India"]
            if len(sub) == 0:
                continue
            # Keep records that are in target_cand_ids OR sample a few to simulate index density
            mask = sub["entity_id"].isin(target_cand_ids)
            relevant = sub[mask]
            for _, row in relevant.iterrows():
                e_id = row["entity_id"]
                found_targets.add(e_id)
                nv = clean_name_multiview(str(row.get("business_name", "")))
                av = clean_address_multiview(str(row.get("business_address", "")), country="India")
                pool_records.append({
                    "entity_id": e_id,
                    "country": "India",
                    "core_words": set(w for w in nv["core_name"].split() if len(w) >= 2),
                    **nv,
                    **av
                })
            # Also keep 5,000 distractors to simulate index
            distractors = sub[~mask].head(5000)
            for _, row in distractors.iterrows():
                e_id = row["entity_id"]
                nv = clean_name_multiview(str(row.get("business_name", "")))
                av = clean_address_multiview(str(row.get("business_address", "")), country="India")
                pool_records.append({
                    "entity_id": e_id,
                    "country": "India",
                    "core_words": set(w for w in nv["core_name"].split() if len(w) >= 2),
                    **nv,
                    **av
                })

    logger.info(f"Built pool of {len(pool_records):,} records containing {len(found_targets):,} / {len(target_cand_ids):,} target GT entities.")
    pool_lookup = {r["entity_id"]: r for r in pool_records}

    # 3. Build Blocking Engine
    engine = BlockingEngine(adaptive_cap=adaptive_cap)
    engine.build_indexes(pool_records)

    # 4. Preprocess S1
    s1_preprocessed = []
    for _, row in s1_india.iterrows():
        e_id = row["entity_id"]
        nv = clean_name_multiview(str(row.get("business_name", "")))
        av = clean_address_multiview(str(row.get("business_address", "")), country="India")
        s1_preprocessed.append({
            "entity_id": e_id,
            "raw_name": str(row.get("business_name", "")),
            "raw_addr": str(row.get("business_address", "")),
            "country": "India",
            "core_words": set(w for w in nv["core_name"].split() if len(w) >= 2),
            **nv,
            **av
        })

    # 5. Check misses
    missed_pairs = []
    total_true_checked = 0
    raw_retrieved_count = 0
    capped_retrieved_count = 0
    missed_due_to_capping = 0
    missed_due_to_index = 0

    for s1_rec in s1_preprocessed:
        s1_id = s1_rec["entity_id"]
        cands, per_pass = engine.retrieve_candidates_for_s1(s1_rec)
        raw_cands = set().union(*per_pass.values())
        gt_targets = gt_map.get(s1_id, set()) & found_targets
        for tgt in gt_targets:
            total_true_checked += 1
            in_raw = tgt in raw_cands
            in_capped = tgt in cands
            if in_raw:
                raw_retrieved_count += 1
            if in_capped:
                capped_retrieved_count += 1
            else:
                missed_pairs.append((s1_rec, pool_lookup[tgt], in_raw))
                if in_raw:
                    missed_due_to_capping += 1
                else:
                    missed_due_to_index += 1

    raw_recall = raw_retrieved_count / total_true_checked if total_true_checked > 0 else 0
    capped_recall = capped_retrieved_count / total_true_checked if total_true_checked > 0 else 0

    logger.info(f"\n============================================================")
    logger.info(f"DIAGNOSTIC BLOCKING RESULTS ON SAMPLE (Cap = {adaptive_cap}):")
    logger.info(f"Total True Pairs Evaluated:                 {total_true_checked}")
    logger.info(f"Raw Recall (Before Capping - All Passes):   {raw_retrieved_count} ({raw_recall*100:.2f}%)")
    logger.info(f"Capped Recall (After Cap {adaptive_cap}):             {capped_retrieved_count} ({capped_recall*100:.2f}%)")
    logger.info(f"Missed PURELY DUE TO CAPPING:                 {missed_due_to_capping} ({(missed_due_to_capping/total_true_checked)*100:.2f}%)")
    logger.info(f"Missed by Inverted Index (True Gap):        {missed_due_to_index} ({(missed_due_to_index/total_true_checked)*100:.2f}%)")
    logger.info(f"============================================================\n")

    # Forensic Analysis of first 15 misses
    print("--- DETAILED FORENSIC AUDIT OF MISSED TRUE MATCHES ---")
    for i, (s1, tgt, in_raw) in enumerate(missed_pairs[:15], 1):
        status = "[DROPPED BY CAPPING]" if in_raw else "[MISSED BY ALL PASSES]"
        print(f"\n[MISSED PAIR #{i}]  {status}")
        print(f"  S1 ID:          {s1['entity_id']}")
        print(f"  S1 Name (raw):  {s1.get('raw_name', '')}")
        print(f"  S1 Core Name:   {s1['core_name']}")
        print(f"  S1 First Token: {s1['first_token']}")
        print(f"  S1 Street Nums: {s1.get('street_numbers', set())}")
        print(f"  S1 Postal Code: {s1.get('postal_code', '')}")
        print(f"  S1 Street Tok:  {s1.get('first_street_token', '')}")
        print(f"  S1 Addr (raw):  {s1.get('raw_addr', '')}")
        print(f"  --------------------------------------------------")
        print(f"  Tgt ID:         {tgt['entity_id']}")
        print(f"  Tgt Core Name:  {tgt['core_name']}")
        print(f"  Tgt First Token:{tgt['first_token']}")
        print(f"  Tgt Street Nums:{tgt.get('street_numbers', set())}")
        print(f"  Tgt Postal Code:{tgt.get('postal_code', '')}")
        print(f"  Tgt Street Tok: {tgt.get('first_street_token', '')}")
        print(f"  Tgt Clean Addr: {tgt.get('clean_addr', '')}")
        print(f"  --------------------------------------------------")
        shared_name = s1["core_words"] & tgt["core_words"]
        shared_addr = s1["addr_tokens"] & tgt["addr_tokens"]
        print(f"  Shared Name Words: {shared_name}")
        print(f"  Shared Addr Words: {shared_addr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="./dataset/train")
    parser.add_argument("--sample-s1", type=int, default=2000)
    parser.add_argument("--adaptive-cap", type=int, default=80)
    args = parser.parse_args()
    diagnose(args.train_dir, args.sample_s1, args.adaptive_cap)
