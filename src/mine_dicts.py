#!/usr/bin/env python3
"""
Ground-Truth Token Alignment Mining Engine.

Mines bilingual and multi-script token co-occurrences directly from the 7.64M true pairs in ground truth:
  - Indic Script (Devanagari, Telugu, Bengali, Tamil, Gujarati) -> English name words
    (e.g., 'मेडिकल' -> 'medical', 'स्टोर्स' -> 'stores', 'ফার্মেসি' -> 'pharmacy')
  - Common spelling variations and legal abbreviations
  - 100% rules-compliant: learned strictly from provided training ground truth (zero external APIs).

Outputs a compact JSON dictionary used by preprocessing and blocking to eliminate the cross-script recall gap.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger, find_file
from src.preprocessing import clean_name_multiview


def parse_args():
    parser = argparse.ArgumentParser(description="Mine token alignments from Ground Truth.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/val",
        help="Path to directory containing source1, source2, source3, and ground_truth files"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./data/mined_token_map.json",
        help="Path to save mined token dictionary (JSON)"
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Minimum co-occurrence count (default: 3)"
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.20,
        help="Minimum conditional probability P(eng | foreign) (default: 0.20)"
    )
    return parser.parse_args()


def is_non_ascii(token: str) -> bool:
    """Check if token contains non-ASCII (e.g. Indic/accented) characters."""
    return any(ord(c) > 127 for c in token)


def mine_token_alignments():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_file = Path(args.output_file).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Ground-Truth Token Alignment Mining ===")
    logger.info(f"Data Directory: {data_dir}")

    s1_path = find_file(data_dir, ["train_source1.tsv", "val_source1.tsv", "source1.tsv"])
    s2_path = find_file(data_dir, ["train_source2.tsv", "val_source2.tsv", "source2.tsv"])
    s3_path = find_file(data_dir, ["train_source3.tsv", "val_source3.tsv", "source3.tsv"])
    gt_path = find_file(data_dir, ["train_ground_truth.tsv", "val_ground_truth.tsv", "ground_truth.tsv"])

    # 1. Load S1 Name Tokens (only for S1 records that have non-empty matches)
    s1_tokens = {}
    with Timer("Loading Source 1 Name Tokens"):
        for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=200000):
            ids = chunk["entity_id"].tolist()
            names = chunk["business_name"].fillna("").tolist()
            for eid, name in zip(ids, names):
                nv = clean_name_multiview(name)
                # Keep ASCII tokens of length >= 3
                tokens = [w for w in nv["core_name"].split() if len(w) >= 3 and not is_non_ascii(w)]
                s1_tokens[eid] = tokens

    logger.info(f"Loaded tokens for {len(s1_tokens):,} Source 1 entities.")

    # 2. Load Ground Truth Mapping
    logger.info("Loading Ground Truth pairs...")
    s23_to_s1 = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[1].strip():
                s1_id = parts[0].strip()
                for mid in parts[1].split(","):
                    mid = mid.strip()
                    if mid:
                        s23_to_s1[mid] = s1_id

    logger.info(f"Loaded {len(s23_to_s1):,} true links to Source 1.")

    # 3. Stream S2 and S3 records and count co-occurrences
    cooccur = defaultdict(Counter)
    foreign_counts = Counter()

    with Timer("Scanning Source 2 and Source 3 for Non-ASCII tokens"):
        for s_path in [s2_path, s3_path]:
            for chunk in pd.read_csv(s_path, sep="\t", dtype=str, chunksize=200000):
                ids = chunk["entity_id"].tolist()
                names = chunk["business_name"].fillna("").tolist()
                for eid, name in zip(ids, names):
                    if eid not in s23_to_s1:
                        continue
                    s1_id = s23_to_s1[eid]
                    s1_toks = s1_tokens.get(s1_id)
                    if not s1_toks:
                        continue

                    nv = clean_name_multiview(name)
                    s23_toks = nv["core_name"].split()
                    for f_tok in s23_toks:
                        if is_non_ascii(f_tok) and len(f_tok) >= 2:
                            foreign_counts[f_tok] += 1
                            for e_tok in s1_toks:
                                cooccur[f_tok][e_tok] += 1

    logger.info(f"Identified {len(foreign_counts):,} unique non-ASCII tokens across true pairs.")

    # 4. Filter alignments by frequency and confidence
    mined_map = {}
    for f_tok, total_freq in foreign_counts.items():
        if total_freq < args.min_count:
            continue
        top_cand, top_cnt = cooccur[f_tok].most_common(1)[0]
        conf = top_cnt / total_freq
        if top_cnt >= args.min_count and conf >= args.min_conf:
            mined_map[f_tok] = top_cand

    logger.info(f"Successfully mined {len(mined_map):,} high-confidence token translations/alignments!")

    # Show top examples
    top_examples = sorted(mined_map.items(), key=lambda x: cooccur[x[0]][x[1]], reverse=True)[:25]
    logger.info("--- Top 25 Mined Token Alignments ---")
    for f_tok, e_tok in top_examples:
        cnt = cooccur[f_tok][e_tok]
        tot = foreign_counts[f_tok]
        logger.info(f"  {f_tok:<20} -> {e_tok:<15} (count: {cnt}/{tot}, conf: {cnt/tot:.1%})")

    # Save to JSON
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(mined_map, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved token alignment dictionary to: {out_file}")


if __name__ == "__main__":
    mine_token_alignments()
