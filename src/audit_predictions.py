#!/usr/bin/env python3
"""
Diagnostic Script: Audit Output Predictions vs Test S1 Distribution.

Breaks down matching_results.tsv by country:
  - Prediction counts: singletons (0 matches), 1 match, 2 matches, 3+ matches
  - Match count per S1 entity
  - S2 vs S3 match distribution
"""

import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import find_file


def audit_results(test_dir: Path, output_dir: Path):
    s1_path = find_file(test_dir, ["test_source1.tsv", "source1.tsv"])
    match_path = output_dir / "matching_results.tsv"
    cand_path = output_dir / "candidate_pairs.tsv"

    print("Loading test_source1.tsv...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    countries = dict(zip(df_s1["entity_id"], df_s1["country"]))
    total_s1 = len(df_s1)

    print(f"Total S1 entities: {total_s1:,}")
    country_counts = Counter(df_s1["country"])
    for c, cnt in country_counts.items():
        print(f"  {c}: {cnt:,} ({cnt/total_s1:.1%})")

    print("\nAuditing matching_results.tsv...")
    matches_by_country = defaultdict(list)
    s2_by_country = defaultdict(int)
    s3_by_country = defaultdict(int)
    empty_by_country = defaultdict(int)

    with open(match_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0].strip()
            c = countries.get(s1_id, "Unknown")
            if len(parts) > 1 and parts[1].strip():
                m_ids = [m.strip() for m in parts[1].split(",") if m.strip()]
                matches_by_country[c].append(len(m_ids))
                for m in m_ids:
                    if m.startswith("S2-"):
                        s2_by_country[c] += 1
                    elif m.startswith("S3-"):
                        s3_by_country[c] += 1
            else:
                matches_by_country[c].append(0)
                empty_by_country[c] += 1

    print("\n" + "=" * 60)
    print("MATCHING RESULTS BREAKDOWN BY COUNTRY")
    print("=" * 60)
    for c in sorted(country_counts.keys()):
        counts = matches_by_country[c]
        n_c = country_counts[c]
        n_empty = empty_by_country[c]
        c_dist = Counter(counts)
        avg_matches = sum(counts) / n_c if n_c > 0 else 0.0

        print(f"Country: {c} (Total: {n_c:,})")
        print(f"  Predicted Singletons (0 matches): {n_empty:,} ({n_empty/n_c:.1%})")
        print(f"  1 match:    {c_dist[1]:,} ({c_dist[1]/n_c:.1%})")
        print(f"  2 matches:  {c_dist[2]:,} ({c_dist[2]/n_c:.1%})")
        print(f"  3+ matches: {sum(cnt for k, cnt in c_dist.items() if k >= 3):,} ({sum(cnt for k, cnt in c_dist.items() if k >= 3)/n_c:.1%})")
        print(f"  Average matches per S1: {avg_matches:.2f}")
        print(f"  Total S2 links: {s2_by_country[c]:,}")
        print(f"  Total S3 links: {s3_by_country[c]:,}")
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default="./dataset/test")
    parser.add_argument("--output-dir", type=str, default="./output")
    args = parser.parse_args()
    audit_results(Path(args.test_dir), Path(args.output_dir))
