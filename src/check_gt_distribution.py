import sys
from pathlib import Path
from collections import Counter

gt_path = Path("data/val/val_ground_truth.tsv")
if not gt_path.exists():
    gt_path = Path("C:/Users/NCC-HPC8/Amazon-ML-Challenge-2026/data/val/val_ground_truth.tsv")

print("Reading ground truth from:", gt_path)
match_counts = Counter()
s2_counts = Counter()
s3_counts = Counter()

with open(gt_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 1 and parts[1].strip():
            m_ids = [m.strip() for m in parts[1].split(",") if m.strip()]
            s2_cnt = sum(1 for m in m_ids if m.startswith("S2-"))
            s3_cnt = sum(1 for m in m_ids if m.startswith("S3-"))
            match_counts[len(m_ids)] += 1
            s2_counts[s2_cnt] += 1
            s3_counts[s3_cnt] += 1
        else:
            match_counts[0] += 1
            s2_counts[0] += 1
            s3_counts[0] += 1

total = sum(match_counts.values())
print(f"Total S1 entities in Ground Truth: {total:,}")
print("\n--- Match Count per S1 Entity in Ground Truth ---")
for k in sorted(match_counts.keys()):
    print(f"  {k} matches: {match_counts[k]:,} ({match_counts[k]/total:.2%})")

print("\n--- S2 Matches per S1 Entity ---")
for k in sorted(s2_counts.keys()):
    print(f"  {k} S2 matches: {s2_counts[k]:,} ({s2_counts[k]/total:.2%})")

print("\n--- S3 Matches per S1 Entity ---")
for k in sorted(s3_counts.keys()):
    print(f"  {k} S3 matches: {s3_counts[k]:,} ({s3_counts[k]/total:.2%})")
