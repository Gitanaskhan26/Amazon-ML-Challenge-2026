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

try:
    from src.transformer_matcher import DenseSemanticMatcher, HAS_SENTENCE_TRANSFORMERS
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    DenseSemanticMatcher = None



def are_cross_source_twins(r2: dict, r3: dict, country: str = "") -> bool:
    """
    Determines if an S2 entity and an S3 entity represent the exact same physical business.
    Used for transitive recovery of borderline matches with high precision.
    """
    p2 = r2.get("postal_code", "")
    p3 = r3.get("postal_code", "")
    # Hard contradiction for India: different 2-digit PIN circle (different states)
    if country == "India" and p2 and p3 and len(p2) == 6 and len(p3) == 6 and p2[:2] != p3[:2]:
        return False

    core2 = r2.get("core_name", "")
    core3 = r3.get("core_name", "")
    compact2 = core2.replace(" ", "")
    compact3 = core3.replace(" ", "")
    exact_name = bool(compact2 and compact2 == compact3)

    skel2 = r2.get("consonant_skel", "")
    skel3 = r3.get("consonant_skel", "")
    exact_skel = bool(skel2 and len(skel2) >= 3 and skel2 == skel3)

    words2 = r2.get("core_words", set())
    words3 = r3.get("core_words", set())
    inter = len(words2 & words3)
    min_words = min(len(words2), len(words3))
    containment = inter / min_words if min_words > 0 else 0.0

    nums2 = r2.get("street_numbers", set())
    nums3 = r3.get("street_numbers", set())
    has_num_match = bool(nums2 and nums3 and (nums2 & nums3))

    toks2 = r2.get("addr_tokens", set())
    toks3 = r3.get("addr_tokens", set())
    addr_inter = len(toks2 & toks3)

    postal_match = bool(p2 and p3 and p2 == p3)

    # Condition 1: Exact Name or Skeleton match + physical verification
    if exact_name or exact_skel:
        if postal_match or has_num_match or addr_inter >= 1:
            return True

    # Condition 2: High token containment (brand within longer legal name) + physical verification
    if containment >= 0.80 and inter >= 1:
        if postal_match or has_num_match or addr_inter >= 2:
            return True

    # Condition 3: Same building (exact number + exact postal) + at least 1 common name token
    if postal_match and has_num_match and inter >= 1:
        return True

    return False



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
        default=160,
        help="Maximum candidates per Source 1 entity in blocking stage (default: 160)"
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
    parser.add_argument(
        "--enable-veto",
        action="store_true",
        default=False,
        help="Enable heuristic singleton veto (default: False, preserves calibrated matches)"
    )
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        default=False,
        help="Enable S2/S3 postal contradiction graph filtering (default: False)"
    )
    parser.add_argument(
        "--enable-transitive",
        dest="enable_transitive",
        action="store_true",
        default=True,
        help="Enable Multi-Source S2<->S3 Transitive Closure & Intra-Clique Consistency (default: True)"
    )
    parser.add_argument(
        "--no-transitive",
        dest="enable_transitive",
        action="store_false",
        help="Disable Multi-Source S2<->S3 Transitive Closure"
    )
    parser.add_argument(
        "--enable-transformer",
        action="store_true",
        default=False,
        help="Enable Dense Multilingual Transformer Semantic Similarity Rescoring (requires sentence-transformers)"
    )
    parser.add_argument(
        "--transformer-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="HuggingFace model for Dense Semantic Matcher"
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

    # Check for trained CatBoost model (for Ensemble)
    cb_path = model_path.parent / "cb_model.cbm"
    cb_booster = None
    if cb_path.exists():
        try:
            from catboost import CatBoostClassifier
            cb_booster = CatBoostClassifier()
            cb_booster.load_model(str(cb_path))
            # Safety: ensure CatBoost feature count matches LightGBM
            if feature_names and cb_booster.feature_count_ != len(feature_names):
                logger.warning(f"CatBoost model has {cb_booster.feature_count_} features but LightGBM has {len(feature_names)}. Disabling CatBoost ensemble.")
                cb_booster = None
            else:
                logger.info(f"Loaded trained CatBoost model from: {cb_path} (50/50 Ensemble Enabled!)")
        except Exception as e:
            logger.warning(f"Could not load CatBoost model: {e}")

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

        # Optional Dense Transformer Semantic Encoding
        s1_embs = None
        pool_embs = None
        if getattr(args, "enable_transformer", False) and HAS_SENTENCE_TRANSFORMERS and DenseSemanticMatcher is not None:
            with Timer(f"Dense Transformer Semantic Encoding for {country}"):
                try:
                    t_matcher = DenseSemanticMatcher(model_name=args.transformer_model)
                    s1_embs = t_matcher.encode_entities(s1_lookup)
                    active_pool_ids = set(cand_id for _, cand_id in pairs_to_score)
                    active_pool_dict = {cid: pool_lookup[cid] for cid in active_pool_ids if cid in pool_lookup}
                    pool_embs = t_matcher.encode_entities(active_pool_dict)
                    logger.info(f"  Successfully encoded {len(s1_embs):,} S1 and {len(pool_embs):,} pool entities with {args.transformer_model}")
                except Exception as e:
                    logger.warning(f"Could not run transformer encoder: {e}. Falling back to pure GBDT ensemble.")
                    s1_embs = None
                    pool_embs = None

        # 2. Batched Scoring with LightGBM + CatBoost (chunks of 100,000 pairs!)
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

                    X_np = np.array(feat_matrix, dtype=np.float32)
                    probs = booster.predict(X_np)
                    if cb_booster is not None:
                        cb_probs = cb_booster.predict_proba(X_np)[:, 1]
                        probs = 0.5 * probs + 0.5 * cb_probs

                    # Blend Transformer Dense Semantic Similarity if enabled
                    if s1_embs is not None and pool_embs is not None:
                        emb_dim = 384
                        vecs1 = np.array([s1_embs.get(s1_id, np.zeros(emb_dim, dtype=np.float32)) for s1_id, _ in batch_pairs], dtype=np.float32)
                        vecs2 = np.array([pool_embs.get(cand_id, np.zeros(emb_dim, dtype=np.float32)) for _, cand_id in batch_pairs], dtype=np.float32)
                        sem_sims = np.clip(np.sum(vecs1 * vecs2, axis=1), 0.0, 1.0)
                        # 80% GBDT Calibrated Score + 20% Transformer Semantic Similarity
                        probs = 0.80 * probs + 0.20 * sem_sims

                    for (s1_id, cand_id), prob in zip(batch_pairs, probs):
                        candidate_scores.append((s1_id, cand_id, float(prob)))
            else:
                for s1_id, cand_id in pairs_to_score:
                    score = heuristic_score_pair(s1_lookup[s1_id], pool_lookup[cand_id])
                    candidate_scores.append((s1_id, cand_id, score))

        if s1_embs is not None:
            del s1_embs
            del pool_embs
            gc.collect()


        # ================================================================
        # PHASE 1: Global Greedy Bipartite Assignment with GT-Based Caps
        # ================================================================
        # Sort ALL pairs descending by score. Walk through and assign each
        # S2/S3 candidate to its best S1 match (1-to-at-most-1 for candidates).
        # Per-S1 limits based on ground truth distribution:
        #   S2: up to 5, S3: up to 6, Total: up to 8
        with Timer(f"Phase 1: Greedy Bipartite Assignment (tau={country_tau:.2f}) for {country}"):
            candidate_scores.sort(key=lambda x: x[2], reverse=True)
            assigned_s23 = set()
            s1_s2_count = defaultdict(int)
            s1_s3_count = defaultdict(int)
            s1_scored = defaultdict(list)  # s1_id -> [(cand_id, score), ...] for post-processing

            for s1_id, cand_id, score in candidate_scores:
                if score < country_tau:
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

                    if s1_id not in final_matches:
                        final_matches[s1_id] = set()
                    final_matches[s1_id].add(cand_id)
                    s1_scored[s1_id].append((cand_id, score))

        # ================================================================
        # PHASE 1.5: Multi-Source Transitive Closure & Intra-Clique Consistency
        # ================================================================
        if getattr(args, "enable_transitive", True):
            with Timer(f"Phase 1.5: S2<->S3 Transitive Closure & Consistency for {country}"):
                # Group borderline candidate scores (0.35 <= score < country_tau) by s1_id
                s1_borderline = defaultdict(list)
                for s1_id, cand_id, score in candidate_scores:
                    if 0.35 <= score < country_tau:
                        s1_borderline[s1_id].append((cand_id, score))

                n_transitive_rescues = 0
                n_state_vetoes = 0

                # 1. Transitive Candidate Recovery (S2 <-> S3 Cross-Validation)
                for s1_id, matches in final_matches.items():
                    if not matches:
                        continue
                    borderline = s1_borderline.get(s1_id, [])
                    if not borderline:
                        continue

                    accepted_s2 = [c for c in matches if c.startswith("S2")]
                    accepted_s3 = [c for c in matches if c.startswith("S3")]

                    # Case A: Rescue S3 matches if high-confidence S2 matches exist
                    if accepted_s2 and len(accepted_s3) < 6:
                        for s3_id, sc in borderline:
                            if not s3_id.startswith("S3") or s3_id in assigned_s23:
                                continue
                            if (s1_s2_count[s1_id] + s1_s3_count[s1_id]) >= 8 or s1_s3_count[s1_id] >= 6:
                                break
                            r3 = pool_lookup.get(s3_id, {})
                            is_twin = False
                            for s2_id in accepted_s2:
                                r2 = pool_lookup.get(s2_id, {})
                                if are_cross_source_twins(r2, r3, country):
                                    is_twin = True
                                    break
                            if is_twin:
                                assigned_s23.add(s3_id)
                                s1_s3_count[s1_id] += 1
                                matches.add(s3_id)
                                s1_scored[s1_id].append((s3_id, sc))
                                n_transitive_rescues += 1

                    # Case B: Rescue S2 matches if high-confidence S3 matches exist
                    if accepted_s3 and len(accepted_s2) < 5:
                        for s2_id, sc in borderline:
                            if not s2_id.startswith("S2") or s2_id in assigned_s23:
                                continue
                            if (s1_s2_count[s1_id] + s1_s3_count[s1_id]) >= 8 or s1_s2_count[s1_id] >= 5:
                                break
                            r2 = pool_lookup.get(s2_id, {})
                            is_twin = False
                            for s3_id in accepted_s3:
                                r3 = pool_lookup.get(s3_id, {})
                                if are_cross_source_twins(r2, r3, country):
                                    is_twin = True
                                    break
                            if is_twin:
                                assigned_s23.add(s2_id)
                                s1_s2_count[s1_id] += 1
                                matches.add(s2_id)
                                s1_scored[s1_id].append((s2_id, sc))
                                n_transitive_rescues += 1

                # 2. Intra-Clique State Contradiction Veto (India):
                # Businesses in different 2-digit PIN circles are in different states.
                if country == "India":
                    for s1_id, matches in final_matches.items():
                        if len(matches) < 2:
                            continue
                        scored_dict = dict(s1_scored.get(s1_id, []))
                        to_drop = set()
                        match_list = list(matches)
                        for i in range(len(match_list)):
                            for j in range(i + 1, len(match_list)):
                                c1, c2 = match_list[i], match_list[j]
                                p1 = pool_lookup.get(c1, {}).get("postal_code", "")
                                p2 = pool_lookup.get(c2, {}).get("postal_code", "")
                                if p1 and p2 and len(p1) == 6 and len(p2) == 6 and p1[:2] != p2[:2]:
                                    sc1 = scored_dict.get(c1, 0.0)
                                    sc2 = scored_dict.get(c2, 0.0)
                                    weaker = c1 if sc1 < sc2 else c2
                                    to_drop.add(weaker)
                                    n_state_vetoes += 1
                        for d in to_drop:
                            matches.discard(d)

                logger.info(f"  Transitive rescues: {n_transitive_rescues:,}, Intra-clique state vetoes: {n_state_vetoes:,}")

        # ================================================================
        # PHASE 2: Singleton Veto & Graph Consistency Post-Processing (Optional)
        # ================================================================
        if args.enable_veto or args.enable_graph:
            with Timer(f"Phase 2: Singleton Veto & Consistency for {country}"):
                # 2a. Singleton Veto: if top score is weak, predict empty
                n_vetoed = 0
                if args.enable_veto:
                    singleton_veto_tau = country_tau + 0.12
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

                # 2b. Graph Consistency: if S2 and S3 matches have contradicting postal codes, drop weaker
                n_graph_drops = 0
                if args.enable_graph:
                    for s1_id, matches in final_matches.items():
                        if len(matches) < 2:
                            continue
                        scored = s1_scored.get(s1_id, [])
                        s2_matches = [(c, s) for c, s in scored if c.startswith("S2") and c in matches]
                        s3_matches = [(c, s) for c, s in scored if c.startswith("S3") and c in matches]
                        if not s2_matches or not s3_matches:
                            continue
                        s2_postals = set()
                        s3_postals = set()
                        for cid, _ in s2_matches:
                            r = pool_lookup.get(cid, {})
                            p = r.get("postal_code", "")
                            if p:
                                s2_postals.add(p)
                        for cid, _ in s3_matches:
                            r = pool_lookup.get(cid, {})
                            p = r.get("postal_code", "")
                            if p:
                                s3_postals.add(p)
                        if s2_postals and s3_postals and not (s2_postals & s3_postals):
                            s2_min = min(s2_matches, key=lambda x: x[1])
                            s3_min = min(s3_matches, key=lambda x: x[1])
                            if s2_min[1] < s3_min[1]:
                                matches.discard(s2_min[0])
                            else:
                                matches.discard(s3_min[0])
                            n_graph_drops += 1

                logger.info(f"  Singleton vetoes: {n_vetoed:,}, Graph consistency drops: {n_graph_drops:,}")

        # Fill in entities with no candidates as singletons
        for s1_rec in s1_preprocessed:
            s1_id = s1_rec["entity_id"]
            if s1_id not in final_matches:
                final_matches[s1_id] = set()

        # ================================================================
        # SAVE scored pairs for optional fast threshold sweeps later
        # ================================================================
        scored_path = out_dir / f"scored_pairs_{country}.tsv"
        with Timer(f"Saving scored pairs for {country}"):
            with open(scored_path, "w", encoding="utf-8") as fout:
                fout.write("source1_entity_id\tcandidate_entity_id\tscore\n")
                for s1_id, cid, sc in candidate_scores:
                    if sc < 0.25:
                        break
                    fout.write(f"{s1_id}\t{cid}\t{sc:.6f}\n")
        logger.info(f"  Saved scored pairs: {scored_path}")

        # Free memory before next country partition
        del candidate_scores
        del pool_records
        del pool_lookup
        del pairs_to_score
        del s1_lookup
        del s1_preprocessed
        del s1_scored
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
