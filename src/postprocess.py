#!/usr/bin/env python3
"""
Post-Processing Threshold Sweep for Maximum Macro-F0.5.

Re-sweeps decision thresholds and singleton veto thresholds on the scored pairs
exported by inference.py, WITHOUT re-running blocking or model scoring.

Usage:
    python src/postprocess.py --output-dir ./output --test-dir ./dataset/test

    # Try different thresholds:
    python src/postprocess.py --output-dir ./output --test-dir ./dataset/test \
        --us-tau 0.67 --india-tau 0.58 --france-tau 0.65 --veto-margin 0.12
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger, write_submission_tsv


def parse_args():
    parser = argparse.ArgumentParser(description="Post-Processing Threshold Sweep.")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory containing scored_pairs_*.tsv from inference")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Path to test dataset directory (for validation)")
    parser.add_argument("--us-tau", type=float, default=None,
                        help="Threshold for US (overrides calibration)")
    parser.add_argument("--india-tau", type=float, default=None,
                        help="Threshold for India (overrides calibration)")
    parser.add_argument("--france-tau", type=float, default=None,
                        help="Threshold for France (overrides calibration)")
    parser.add_argument("--veto-margin", type=float, default=0.12,
                        help="Margin above threshold for singleton veto (default: 0.12)")
    parser.add_argument("--no-veto", action="store_true",
                        help="Disable singleton veto")
    parser.add_argument("--no-graph", action="store_true",
                        help="Disable graph consistency checks")
    return parser.parse_args()


def load_scored_pairs(output_dir: Path):
    """Load all scored_pairs_*.tsv from the output directory."""
    all_pairs = {}  # country -> list of (s1_id, cand_id, score)
    for f in sorted(output_dir.glob("scored_pairs_*.tsv")):
        country = f.stem.replace("scored_pairs_", "")
        logger.info(f"Loading scored pairs for {country} from {f}")
        pairs = []
        with open(f, "r", encoding="utf-8") as fin:
            next(fin)  # skip header
            for line in fin:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    pairs.append((parts[0], parts[1], float(parts[2])))
        all_pairs[country] = pairs
        logger.info(f"  {country}: {len(pairs):,} scored pairs")
    return all_pairs


def load_pool_postals(test_dir: Path):
    """Load postal codes from S2/S3 for graph consistency checks."""
    pool_postals = {}
    for src in ["test_source2.tsv", "test_source3.tsv"]:
        src_path = test_dir / src
        if not src_path.exists():
            continue
        for chunk in pd.read_csv(src_path, sep="\t", dtype=str, chunksize=500000):
            for _, row in chunk.iterrows():
                eid = str(row.get("entity_id", ""))
                addr = str(row.get("business_address", ""))
                # Extract postal code (last word that looks like a number)
                postal = ""
                for word in reversed(addr.split()):
                    cleaned = word.strip().replace("-", "")
                    if cleaned.isdigit() and len(cleaned) >= 4:
                        postal = cleaned
                        break
                pool_postals[eid] = postal
    return pool_postals


def assign_matches(pairs, tau, veto_margin=0.12, enable_veto=True,
                   enable_graph=True, pool_postals=None):
    """Run bipartite assignment with singleton veto and graph consistency."""
    pairs.sort(key=lambda x: x[2], reverse=True)
    assigned_s23 = set()
    s1_s2_count = defaultdict(int)
    s1_s3_count = defaultdict(int)
    final_matches = defaultdict(set)
    s1_scored = defaultdict(list)

    for s1_id, cand_id, score in pairs:
        if score < tau:
            break
        if cand_id not in assigned_s23:
            is_s2 = cand_id.startswith("S2")
            if is_s2 and s1_s2_count[s1_id] >= 5:
                continue
            if not is_s2 and s1_s3_count[s1_id] >= 6:
                continue
            if (s1_s2_count[s1_id] + s1_s3_count[s1_id]) >= 8:
                continue

            assigned_s23.add(cand_id)
            if is_s2:
                s1_s2_count[s1_id] += 1
            else:
                s1_s3_count[s1_id] += 1
            final_matches[s1_id].add(cand_id)
            s1_scored[s1_id].append((cand_id, score))

    # Singleton Veto
    n_vetoed = 0
    if enable_veto:
        singleton_veto_tau = tau + veto_margin
        for s1_id in list(final_matches.keys()):
            matches = final_matches[s1_id]
            if len(matches) == 0:
                continue
            scored = s1_scored.get(s1_id, [])
            if not scored:
                continue
            top_score = scored[0][1]
            if len(matches) == 1 and top_score < singleton_veto_tau:
                final_matches[s1_id] = set()
                n_vetoed += 1

    # Graph Consistency
    n_graph_drops = 0
    if enable_graph and pool_postals:
        for s1_id, matches in final_matches.items():
            if len(matches) < 2:
                continue
            scored = s1_scored.get(s1_id, [])
            s2_matches = [(c, s) for c, s in scored if c.startswith("S2") and c in matches]
            s3_matches = [(c, s) for c, s in scored if c.startswith("S3") and c in matches]
            if not s2_matches or not s3_matches:
                continue
            s2_postals = {pool_postals.get(c, "") for c, _ in s2_matches} - {""}
            s3_postals = {pool_postals.get(c, "") for c, _ in s3_matches} - {""}
            if s2_postals and s3_postals and not (s2_postals & s3_postals):
                s2_min = min(s2_matches, key=lambda x: x[1])
                s3_min = min(s3_matches, key=lambda x: x[1])
                if s2_min[1] < s3_min[1]:
                    matches.discard(s2_min[0])
                else:
                    matches.discard(s3_min[0])
                n_graph_drops += 1

    return final_matches, n_vetoed, n_graph_drops


def count_distribution(matches):
    """Count match distribution."""
    from collections import Counter
    counts = Counter()
    for s1_id, m in matches.items():
        counts[len(m)] += 1
    return counts


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    logger.info("=" * 60)
    logger.info("=== Post-Processing Threshold Sweep ===")

    # Load scored pairs
    all_pairs = load_scored_pairs(output_dir)
    if not all_pairs:
        logger.error("No scored_pairs_*.tsv files found! Run inference.py first.")
        return

    # Load pool postals for graph consistency
    pool_postals = {}
    if args.test_dir and not args.no_graph:
        test_dir = Path(args.test_dir).resolve()
        with Timer("Loading pool postal codes for graph consistency"):
            pool_postals = load_pool_postals(test_dir)
        logger.info(f"  Loaded {len(pool_postals):,} postal codes")

    # Load S1 IDs for all countries
    all_s1_ids = set()
    if args.test_dir:
        test_dir = Path(args.test_dir).resolve()
        for fname in ["test_source1.tsv", "source1.tsv", "test_source_1.tsv", "source_1.tsv"]:
            candidate_path = test_dir / fname
            if candidate_path.exists():
                df_s1 = pd.read_csv(candidate_path, sep="\t", dtype=str, usecols=["entity_id", "country"])
                all_s1_ids = set(df_s1["entity_id"])
                logger.info(f"  Total S1 entities loaded from {fname}: {len(all_s1_ids):,}")
                break

    # Fallback to existing candidate_pairs.tsv or matching_results.tsv
    if not all_s1_ids:
        cand_file = output_dir / "candidate_pairs.tsv"
        if not cand_file.exists():
            cand_file = output_dir / "matching_results.tsv"
        if cand_file.exists():
            logger.info(f"Loading all S1 entities from {cand_file.name} fallback...")
            with open(cand_file, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.split("\t")
                    if parts and parts[0].strip():
                        all_s1_ids.add(parts[0].strip())
            logger.info(f"  Total S1 entities from {cand_file.name}: {len(all_s1_ids):,}")

    # Resolve thresholds
    import json
    cal_path = output_dir.parent / "data" / "calibration_results.json"
    default_thresholds = {"US": 0.64, "India": 0.55, "France": 0.64}
    if cal_path.exists():
        with open(cal_path, "r") as f:
            cal = json.load(f)
            cal_thresholds = cal.get("optimal_thresholds_by_country", {})
            for k, v in cal_thresholds.items():
                default_thresholds[k] = float(v)

    # Override with CLI args
    if args.us_tau is not None:
        default_thresholds["US"] = args.us_tau
    if args.india_tau is not None:
        default_thresholds["India"] = args.india_tau
    if args.france_tau is not None:
        default_thresholds["France"] = args.france_tau

    logger.info(f"  Thresholds: {default_thresholds}")
    logger.info(f"  Veto margin: {args.veto_margin}")
    logger.info(f"  Singleton veto: {'disabled' if args.no_veto else 'enabled'}")
    logger.info(f"  Graph consistency: {'disabled' if args.no_graph else 'enabled'}")

    # Run assignment per country
    all_final_matches = {}
    for country, pairs in all_pairs.items():
        tau = default_thresholds.get(country, 0.60)
        logger.info(f"\nProcessing {country} ({len(pairs):,} pairs, tau={tau:.2f})...")
        matches, n_veto, n_graph = assign_matches(
            pairs, tau,
            veto_margin=args.veto_margin,
            enable_veto=not args.no_veto,
            enable_graph=not args.no_graph,
            pool_postals=pool_postals
        )
        all_final_matches.update(matches)
        logger.info(f"  {country}: {len(matches):,} entities, {n_veto:,} vetoed, {n_graph:,} graph drops")

        dist = count_distribution(matches)
        for k in sorted(dist.keys()):
            logger.info(f"    {k} matches: {dist[k]:,}")

    # Add singletons for any S1 entities not in matches
    for s1_id in all_s1_ids:
        if s1_id not in all_final_matches:
            all_final_matches[s1_id] = set()

    # Write output
    match_path = output_dir / "matching_results.tsv"
    with Timer("Saving matching_results.tsv"):
        write_submission_tsv(
            match_path,
            all_final_matches,
            col_s1="source1_entity_id",
            col_target="matched_entity_ids"
        )

    total_entities = len(all_final_matches)
    total_matches = sum(len(m) for m in all_final_matches.values())
    total_singletons = sum(1 for m in all_final_matches.values() if len(m) == 0)
    logger.info(f"\n=== RESULTS ===")
    logger.info(f"  Total entities: {total_entities:,}")
    logger.info(f"  Total matches: {total_matches:,}")
    logger.info(f"  Avg matches/entity: {total_matches/max(total_entities,1):.2f}")
    logger.info(f"  Singletons: {total_singletons:,} ({100*total_singletons/max(total_entities,1):.1f}%)")
    logger.info(f"  Output: {match_path}")

    # Overall distribution
    dist = count_distribution(all_final_matches)
    logger.info(f"\n=== Match Distribution ===")
    for k in sorted(dist.keys()):
        logger.info(f"  {k} matches: {dist[k]:>10,} ({100*dist[k]/total_entities:.1f}%)")

    # Run official validator on postprocessed output
    val_script = Path(__file__).resolve().parent.parent / "utils" / "validate_submission.py"
    cand_path = output_dir / "candidate_pairs.tsv"
    if val_script.exists() and args.test_dir:
        import subprocess
        logger.info("=" * 60)
        logger.info("Running Official Validation Script (utils/validate_submission.py)...")
        cmd = [
            sys.executable,
            str(val_script),
            "--matching", str(match_path),
            "--test-dir", str(args.test_dir)
        ]
        if cand_path.exists():
            cmd.extend(["--candidate", str(cand_path)])
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            logger.info(">>> VALIDATION STATUS: PASS (Exit code 0). POSTPROCESSED SUBMISSION IS 100% VALID! <<<")
        else:
            logger.warning(">>> VALIDATION DIAGNOSTIC: Review validator output above. <<<")


if __name__ == "__main__":
    main()
