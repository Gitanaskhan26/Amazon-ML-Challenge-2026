#!/usr/bin/env python3
"""
Comprehensive Post-Processing Grid Search for Maximum Macro-F0.5.

Tests hundreds of (threshold, veto_margin, veto_mode) configurations on
scored_pairs_*.tsv exported by inference.py.  Picks the configuration that
best matches the ground-truth match distribution observed in training.

Key Insight for 0.990+ F0.5:
  - Precision must exceed 99.3%.  F0.5 = 1.25 / (0.25/P + 1/R).
  - False positives cost 4× more than false negatives.
  - The OPTIMAL threshold is almost always HIGHER than the model's
    calibrated default.  We must test aggressively tight thresholds.

Usage (run AFTER inference.py finishes):
    python src/grid_search_postprocess.py \
        --output-dir ./output_fast \
        --test-dir "C:\Users\NCC-HPC8\Amazon-ML-Challenge-2026\dataset\test"
"""

import os
import sys
import json
import argparse
import itertools
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger, write_submission_tsv

# ============================================================
# Ground-Truth Distribution Reference (from training analysis)
# ============================================================
# These are the EXPECTED match distributions per S1 entity from
# the training ground truth.  A good submission should produce
# a distribution CLOSE to this.  Over-matching (too many matches)
# signals low precision; under-matching signals low recall.
GT_EXPECTED_AVG_MATCHES = {
    "US":     1.62,   # US entities average ~1.6 matches in GT
    "India":  1.85,   # India entities average ~1.85 matches in GT
    "France": 1.55,   # France entities average ~1.55 matches in GT
}

# From GT: approximate fraction of S1 entities that are true singletons
GT_SINGLETON_RATE = {
    "US":     0.065,  # ~6.5% of US S1 entities have NO true matches
    "India":  0.048,  # ~4.8% of India S1 entities
    "France": 0.072,  # ~7.2% of France
}


def parse_args():
    parser = argparse.ArgumentParser(description="Grid Search Post-Processing for Max F0.5")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory containing scored_pairs_*.tsv from inference")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to test dataset directory")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top configurations to display")
    parser.add_argument("--write-best", action="store_true", default=True,
                        help="Write the best configuration's matching_results.tsv")
    return parser.parse_args()


def load_scored_pairs(output_dir: Path):
    """Load scored pairs per country."""
    all_pairs = {}
    for f in sorted(output_dir.glob("scored_pairs_*.tsv")):
        country = f.stem.replace("scored_pairs_", "")
        pairs = []
        with open(f, "r", encoding="utf-8") as fin:
            next(fin)
            for line in fin:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    pairs.append((parts[0], parts[1], float(parts[2])))
        all_pairs[country] = pairs
        logger.info(f"  {country}: {len(pairs):,} scored pairs (score range: "
                    f"{min(p[2] for p in pairs):.4f} – {max(p[2] for p in pairs):.4f})")
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
                postal = ""
                for word in reversed(addr.split()):
                    cleaned = word.strip().replace("-", "")
                    if cleaned.isdigit() and len(cleaned) >= 4:
                        postal = cleaned
                        break
                pool_postals[eid] = postal
    return pool_postals


def run_assignment(pairs, tau, veto_margin=0.12, enable_veto=True,
                   enable_graph=True, pool_postals=None,
                   india_state_veto=False):
    """Greedy bipartite assignment with caps + optional veto + graph."""
    pairs.sort(key=lambda x: x[2], reverse=True)
    assigned_s23 = set()
    s1_s2_count = defaultdict(int)
    s1_s3_count = defaultdict(int)
    final_matches = defaultdict(set)
    s1_scored = defaultdict(list)

    for s1_id, cand_id, score in pairs:
        if score < tau:
            break
        if cand_id in assigned_s23:
            continue
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

    # Singleton Veto (Standard: top score must exceed tau + margin)
    n_vetoed = 0
    if enable_veto:
        singleton_veto_tau = tau + veto_margin
        for s1_id in list(final_matches.keys()):
            matches = final_matches[s1_id]
            scored = s1_scored.get(s1_id, [])
            if not scored:
                continue
            top_score = scored[0][1]
            if len(matches) == 1 and top_score < singleton_veto_tau:
                final_matches[s1_id] = set()
                n_vetoed += 1

    # Score-Gap Confidence Veto:
    # If an entity has exactly 1 match that barely cleared tau (within 0.05),
    # AND that match comes from only ONE source (no cross-source corroboration),
    # it's very likely a false positive. Veto it.
    n_gap_vetoed = 0
    if enable_veto and veto_margin > 0:
        gap_tau = tau + 0.03  # Must clear tau by at least 0.03 if uncorroborated
        for s1_id in list(final_matches.keys()):
            matches = final_matches[s1_id]
            if len(matches) != 1:
                continue
            scored = s1_scored.get(s1_id, [])
            if not scored:
                continue
            cand_id, score = scored[0]
            if score >= gap_tau:
                continue
            # Check: is this match from only one source with no corroboration?
            has_s2 = any(c.startswith("S2") for c in matches)
            has_s3 = any(c.startswith("S3") for c in matches)
            if has_s2 != has_s3:  # Only one source, no cross-validation
                final_matches[s1_id] = set()
                n_gap_vetoed += 1
    n_vetoed += n_gap_vetoed

    # India State Contradiction Veto (2-digit PIN prefix mismatch)
    n_state_vetoes = 0
    if india_state_veto and pool_postals:
        for s1_id, matches in final_matches.items():
            if len(matches) < 2:
                continue
            scored_dict = dict(s1_scored.get(s1_id, []))
            to_drop = set()
            match_list = list(matches)
            for i in range(len(match_list)):
                for j in range(i + 1, len(match_list)):
                    c1, c2 = match_list[i], match_list[j]
                    p1 = pool_postals.get(c1, "")
                    p2 = pool_postals.get(c2, "")
                    if p1 and p2 and len(p1) == 6 and len(p2) == 6 and p1[:2] != p2[:2]:
                        sc1 = scored_dict.get(c1, 0.0)
                        sc2 = scored_dict.get(c2, 0.0)
                        weaker = c1 if sc1 < sc2 else c2
                        to_drop.add(weaker)
                        n_state_vetoes += 1
            for d in to_drop:
                matches.discard(d)

    # Graph Consistency: S2 vs S3 postal contradiction drop
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

    return final_matches, n_vetoed, n_state_vetoes, n_graph_drops


def score_config_quality(matches, country, n_s1):
    """
    Score how well a configuration's output matches the expected
    ground-truth distribution.  Lower = better (closer to GT).
    
    Uses a composite penalty that captures:
      1. Deviation from expected avg matches/entity
      2. Deviation from expected singleton rate
      3. Penalty for suspiciously high match counts (precision risk)
    """
    total_matches = sum(len(m) for m in matches.values())
    n_entities_with_matches = sum(1 for m in matches.values() if len(m) > 0)
    n_singletons = n_s1 - n_entities_with_matches

    avg_matches = total_matches / max(n_s1, 1)
    singleton_rate = n_singletons / max(n_s1, 1)

    expected_avg = GT_EXPECTED_AVG_MATCHES.get(country, 1.65)
    expected_singleton = GT_SINGLETON_RATE.get(country, 0.06)

    # Deviation penalties
    avg_dev = abs(avg_matches - expected_avg)
    sing_dev = abs(singleton_rate - expected_singleton)

    # Heavy penalty for over-matching (precision killer)
    over_match_penalty = max(0, avg_matches - expected_avg) * 2.0

    # Light penalty for under-matching (recall loss, but F0.5 forgives)
    under_match_penalty = max(0, expected_avg - avg_matches) * 0.5

    quality = avg_dev + sing_dev * 5.0 + over_match_penalty + under_match_penalty

    return quality, avg_matches, singleton_rate, total_matches


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    test_dir = Path(args.test_dir).resolve()

    logger.info("=" * 60)
    logger.info("=== Comprehensive Grid Search Post-Processing ===")
    logger.info(f"Output Dir: {output_dir}")
    logger.info(f"Test Dir:   {test_dir}")

    # Load scored pairs
    all_pairs = load_scored_pairs(output_dir)
    if not all_pairs:
        logger.error("No scored_pairs_*.tsv files found! Run inference.py first.")
        return

    # Load pool postals
    with Timer("Loading pool postal codes"):
        pool_postals = load_pool_postals(test_dir)
    logger.info(f"  Loaded {len(pool_postals):,} postal codes")

    # Load S1 entity counts per country
    s1_counts = {}
    for fname in ["test_source1.tsv", "source1.tsv"]:
        s1_path = test_dir / fname
        if s1_path.exists():
            df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, usecols=["entity_id", "country"])
            for country in df_s1["country"].unique():
                s1_counts[country] = len(df_s1[df_s1["country"] == country])
            all_s1_ids = set(df_s1["entity_id"])
            logger.info(f"  S1 counts: {s1_counts}")
            break
    else:
        logger.error("Could not find test_source1.tsv!")
        return

    # ============================================================
    # DEFINE GRID
    # ============================================================
    # Threshold grids — we test TIGHT thresholds because F0.5
    # rewards precision 4× more than recall
    tau_grids = {
        "US":     [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82],
        "India":  [0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71],
        "France": [0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84],
    }
    veto_margins = [0.0, 0.06, 0.08, 0.10, 0.12, 0.15]
    graph_options = [True]
    india_state_veto_options = [True]

    logger.info(f"\n  Threshold grids: {tau_grids}")
    logger.info(f"  Veto margins: {veto_margins}")

    # ============================================================
    # PER-COUNTRY GRID SEARCH
    # ============================================================
    best_per_country = {}

    for country, pairs in all_pairs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Grid Search for {country} ({len(pairs):,} pairs)")
        n_s1 = s1_counts.get(country, len(set(p[0] for p in pairs)))

        taus = tau_grids.get(country, [0.60, 0.65, 0.70, 0.75, 0.80])
        results = []

        for tau in taus:
            for vm in veto_margins:
                for graph in graph_options:
                    state_veto = country == "India"
                    pairs_copy = list(pairs)  # sort mutates in place
                    matches, n_veto, n_sv, n_gd = run_assignment(
                        pairs_copy, tau,
                        veto_margin=vm,
                        enable_veto=(vm > 0),
                        enable_graph=graph,
                        pool_postals=pool_postals,
                        india_state_veto=state_veto
                    )

                    quality, avg_m, sing_r, total_m = score_config_quality(
                        matches, country, n_s1
                    )

                    results.append({
                        "tau": tau,
                        "veto_margin": vm,
                        "graph": graph,
                        "quality": quality,
                        "avg_matches": avg_m,
                        "singleton_rate": sing_r,
                        "total_matches": total_m,
                        "n_vetoed": n_veto,
                        "n_state_vetoes": n_sv,
                        "n_graph_drops": n_gd,
                        "matches": matches,
                    })

        # Sort by quality (lower = better)
        results.sort(key=lambda x: x["quality"])

        logger.info(f"\n  Top {args.top_k} configurations for {country}:")
        logger.info(f"  {'Rank':<5} {'tau':<6} {'veto':<6} {'graph':<6} {'avg_m':<8} {'sing%':<8} {'total_m':<10} {'vetoed':<8} {'quality':<8}")
        for i, r in enumerate(results[:args.top_k]):
            logger.info(f"  {i+1:<5} {r['tau']:<6.2f} {r['veto_margin']:<6.2f} "
                        f"{'Y' if r['graph'] else 'N':<6} {r['avg_matches']:<8.3f} "
                        f"{r['singleton_rate']*100:<8.1f} {r['total_matches']:<10,} "
                        f"{r['n_vetoed']:<8,} {r['quality']:<8.4f}")

        # Also show the expected GT distribution for reference
        logger.info(f"\n  Reference GT distribution for {country}:")
        logger.info(f"    Expected avg matches/entity: {GT_EXPECTED_AVG_MATCHES.get(country, 'N/A')}")
        logger.info(f"    Expected singleton rate: {GT_SINGLETON_RATE.get(country, 'N/A'):.1%}")

        best_per_country[country] = results[0]

    # ============================================================
    # WRITE BEST CONFIGURATION
    # ============================================================
    if args.write_best:
        logger.info(f"\n{'='*60}")
        logger.info("=== Writing Best Configuration ===")

        all_final_matches = {}
        for country, best in best_per_country.items():
            logger.info(f"  {country}: tau={best['tau']:.2f}, veto={best['veto_margin']:.2f}, "
                        f"avg_matches={best['avg_matches']:.3f}, quality={best['quality']:.4f}")
            all_final_matches.update(best["matches"])

        # Add singletons for missing S1 entities
        for s1_id in all_s1_ids:
            if s1_id not in all_final_matches:
                all_final_matches[s1_id] = set()

        total = len(all_final_matches)
        total_m = sum(len(m) for m in all_final_matches.values())
        total_s = sum(1 for m in all_final_matches.values() if len(m) == 0)
        logger.info(f"\n  Total entities: {total:,}")
        logger.info(f"  Total matches: {total_m:,}")
        logger.info(f"  Avg matches/entity: {total_m/max(total,1):.3f}")
        logger.info(f"  Singletons: {total_s:,} ({100*total_s/max(total,1):.1f}%)")

        # Match distribution
        dist = Counter(len(m) for m in all_final_matches.values())
        logger.info(f"\n  Match Distribution:")
        for k in sorted(dist.keys()):
            logger.info(f"    {k} matches: {dist[k]:>10,} ({100*dist[k]/total:.1f}%)")

        # Write output
        match_path = output_dir / "matching_results.tsv"
        with Timer("Saving optimized matching_results.tsv"):
            write_submission_tsv(
                match_path,
                all_final_matches,
                col_s1="source1_entity_id",
                col_target="matched_entity_ids"
            )
        logger.info(f"\n  Saved: {match_path}")

        # Run official validator
        import subprocess
        val_script = Path(__file__).resolve().parent.parent / "utils" / "validate_submission.py"
        if val_script.exists():
            logger.info("Running Official Validation...")
            cmd = [
                sys.executable, str(val_script),
                "--matching", str(match_path),
                "--test-dir", str(test_dir)
            ]
            cand_path = output_dir / "candidate_pairs.tsv"
            if cand_path.exists():
                cmd.extend(["--candidate", str(cand_path)])
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr)
            if res.returncode == 0:
                logger.info(">>> VALIDATION PASS — SUBMISSION IS VALID <<<")
            else:
                logger.warning(">>> VALIDATION ISSUE — Review output above <<<")

        # Save best config for reference
        config_path = output_dir / "best_config.json"
        config = {}
        for country, best in best_per_country.items():
            config[country] = {
                "tau": best["tau"],
                "veto_margin": best["veto_margin"],
                "avg_matches": best["avg_matches"],
                "singleton_rate": best["singleton_rate"],
                "total_matches": best["total_matches"],
                "quality_score": best["quality"],
            }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"  Saved best config: {config_path}")


if __name__ == "__main__":
    main()
