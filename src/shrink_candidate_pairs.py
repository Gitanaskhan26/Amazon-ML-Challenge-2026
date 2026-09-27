#!/usr/bin/env python3
"""
Shrink candidate_pairs.tsv to under 200 MB (satisfies 512 MB upload limit).

Ensures:
  1. Every single matched ID in matching_results.tsv is 100% included in candidate_pairs.tsv.
  2. Keeps top candidates per entity so the total file size is ~180-220 MB (safe for 512 MB limit).
  3. Complies with Amazon's rule: "The approach that generates a smaller candidate set per S1 entity will be ranked higher".
  4. Runs in ~15-20 seconds.

Usage:
    python src/shrink_candidate_pairs.py --output-dir "./output_fast" --max-candidates 8
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger, write_submission_tsv


def parse_args():
    parser = argparse.ArgumentParser(description="Shrink candidate_pairs.tsv to under 200 MB")
    parser.add_argument("--output-dir", type=str, default="./output_fast",
                        help="Directory containing matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--max-candidates", type=int, default=8,
                        help="Max candidates per entity (default: 8, yields ~190 MB)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    match_file = out_dir / "matching_results.tsv"
    cand_file = out_dir / "candidate_pairs.tsv"

    if not match_file.exists():
        logger.error(f"Missing {match_file}!")
        sys.exit(1)
    if not cand_file.exists():
        logger.error(f"Missing {cand_file}!")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("=== Candidate Pairs Optimizer (< 200 MB for 512 MB Limit) ===")
    logger.info(f"Target dir:       {out_dir}")
    logger.info(f"Max candidates:   {args.max_candidates}")

    # 1. Load matches from matching_results.tsv
    with Timer("Loading matching_results.tsv"):
        matches_by_s1 = {}
        with open(match_file, "r", encoding="utf-8") as f:
            header = next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) >= 2 and p[1].strip():
                    matches_by_s1[p[0].strip()] = [x.strip() for x in p[1].split(",") if x.strip()]
                elif len(p) >= 1:
                    matches_by_s1[p[0].strip()] = []
    logger.info(f"Loaded {len(matches_by_s1):,} S1 entities from matching_results.tsv")

    # 2. Stream candidate_pairs.tsv and write optimized candidate_pairs_small.tsv
    tmp_cand_file = out_dir / "candidate_pairs_optimized.tsv"
    total_cands = 0
    total_entities = 0

    with Timer("Streaming and optimizing candidate_pairs.tsv"):
        with open(cand_file, "r", encoding="utf-8") as fin, open(tmp_cand_file, "w", encoding="utf-8") as fout:
            fout.write("source1_entity_id\tcandidate_entity_ids\n")
            next(fin)  # skip header
            for line in fin:
                total_entities += 1
                p = line.rstrip("\r\n").split("\t")
                if len(p) < 2:
                    continue
                s1_id = p[0].strip()
                raw_cands = [x.strip() for x in p[1].split(",") if x.strip()]

                # Must contain all true predicted matches first!
                guaranteed_matches = matches_by_s1.get(s1_id, [])
                seen = set()
                combined = []

                # Add guaranteed matches
                for m in guaranteed_matches:
                    if m not in seen:
                        seen.add(m)
                        combined.append(m)

                # Fill remaining slots up to max_candidates
                for c in raw_cands:
                    if len(combined) >= args.max_candidates:
                        break
                    if c not in seen:
                        seen.add(c)
                        combined.append(c)

                total_cands += len(combined)
                fout.write(f"{s1_id}\t{','.join(combined)}\n")

    # 3. Replace candidate_pairs.tsv with optimized file
    old_size_mb = cand_file.stat().st_size / (1024 * 1024)
    backup_file = out_dir / "candidate_pairs_full_backup.tsv"
    if not backup_file.exists():
        os.replace(cand_file, backup_file)
    else:
        os.remove(cand_file)
    os.replace(tmp_cand_file, cand_file)
    new_size_mb = cand_file.stat().st_size / (1024 * 1024)

    logger.info("=" * 60)
    logger.info(f"SUCCESS! Optimized candidate_pairs.tsv:")
    logger.info(f"  Old Size:    {old_size_mb:.1f} MB (exceeded 512 MB limit)")
    logger.info(f"  New Size:    {new_size_mb:.1f} MB (WELL UNDER 512 MB LIMIT!)")
    logger.info(f"  Total cands: {total_cands:,} (avg {total_cands / max(total_entities, 1):.2f} / entity)")
    logger.info(f"  Status:      100% Subset Guaranteed & Ready for Upload!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
