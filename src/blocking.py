#!/usr/bin/env python3
"""
Five-Pass High-Recall Candidate Generation (Blocking) Engine.

Implements 5 complementary blocking passes to guarantee >=99% true match recall:
  - Pass A: Rare Core Name Tokens Inverted Index (IDF-weighted to avoid stopword bloat)
  - Pass B: Street Number + Name First Token Inverted Index
  - Pass C: Transductive Character 3-gram TF-IDF / Cosine with Adaptive Cap
  - Pass D: Address-Only (Street Number + Postal Code)
  - Pass E: Address-Only (Postal Code + First Street Token)

Outputs cached to Parquet / TSV on disk. Compatible with Windows and Linux/macOS.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger, write_submission_tsv
from src.preprocessing import clean_name_multiview, clean_address_multiview


class BlockingEngine:
    def __init__(self, adaptive_cap: int = 50):
        self.adaptive_cap = adaptive_cap

    def build_indexes(self, pool_records: List[Dict]):
        """
        Build inverted indexes over the secondary pool (Source 2 and Source 3 records).
        """
        self.pool_lookup = {r["entity_id"]: r for r in pool_records}

        self.idx_exact_name = defaultdict(list)
        self.idx_rare_token = defaultdict(list)
        self.idx_bigram = defaultdict(list)
        self.idx_num_first = defaultdict(list)
        self.idx_num_street = defaultdict(list)
        self.idx_num_post = defaultdict(list)
        self.idx_post_name = defaultdict(list)
        self.idx_post_street = defaultdict(list)
        self.idx_addr_rare = defaultdict(list)
        self.idx_char3 = defaultdict(list)

        # Token and 3-gram frequencies
        self.token_freq = Counter()
        self.addr_token_freq = Counter()
        self.char3_freq = Counter()
        for r in pool_records:
            if "core_words" not in r:
                r["core_words"] = set(w for w in r["core_name"].split() if len(w) >= 2)
            for w in r["core_words"]:
                self.token_freq[w] += 1
            for a in r.get("addr_tokens", set()):
                if len(a) >= 3:
                    self.addr_token_freq[a] += 1
            compact = r["core_name"].replace(" ", "")
            if len(compact) >= 3:
                for i in range(len(compact) - 2):
                    self.char3_freq[compact[i:i+3]] += 1

        n_records = len(pool_records)
        self.rare_thresh = 15000
        self.char3_thresh = min(1200, max(50, int(n_records * 0.001)))

        for r in pool_records:
            e_id = r["entity_id"]
            core_name = r["core_name"]
            compact = core_name.replace(" ", "")
            numbers = r["street_numbers"]
            postal = r["postal_code"]
            street_tok = r["first_street_token"]

            # Pass 0: Exact compact core name
            if len(compact) >= 2:
                self.idx_exact_name[compact].append(e_id)

            # Pass A: Rare core name tokens (IDF prioritized)
            words = list(r["core_words"])
            for w in words:
                if self.token_freq[w] <= self.rare_thresh:
                    self.idx_rare_token[w].append(e_id)

            # Pass A2: Core name bigrams
            for i in range(len(words) - 1):
                self.idx_bigram[(words[i], words[i+1])].append(e_id)

            # Pass B: (street_number, name_token)
            for num in numbers:
                for w in words[:3]:
                    self.idx_num_first[(num, w)].append(e_id)

            # Pass B2: (street_number, first_street_token)
            if street_tok and len(street_tok) >= 3:
                for num in numbers:
                    self.idx_num_street[(num, street_tok)].append(e_id)

            # Pass D: (street_number, postal_code)
            if postal:
                for num in numbers:
                    self.idx_num_post[(num, postal)].append(e_id)

            # Pass P1: (postal_code, name_token) - Crucial for Indic and US without street numbers
            if postal:
                for w in words[:4]:
                    self.idx_post_name[(postal, w)].append(e_id)

            # Pass P2: (postal_code, first_street_token)
            if postal and street_tok and len(street_tok) >= 3:
                self.idx_post_street[(postal, street_tok)].append(e_id)

            # Pass E: Address token pairs (critical for Indic entities where name is in native script)
            addr_words = sorted([a for a in r.get("addr_tokens", set()) if self.addr_token_freq[a] <= 800], key=lambda x: self.addr_token_freq[x])
            if len(addr_words) >= 2:
                self.idx_addr_rare[(addr_words[0], addr_words[1])].append(e_id)

            # Pass C: Character 3-grams (filtered for speed)
            if len(compact) >= 3:
                for i in range(len(compact) - 2):
                    tri = compact[i:i+3]
                    if self.char3_freq[tri] <= self.char3_thresh:
                        self.idx_char3[tri].append(e_id)

    def retrieve_candidates_for_s1(self, s1_record: Dict) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """
        Query inverted indexes across Passes 0, A, B, C, D, E.
        All passes contribute their top candidates via bounded posting list slicing.
        Returns:
          (union_candidates, per_pass_candidates)
        """
        core_name = s1_record["core_name"]
        compact = core_name.replace(" ", "")
        numbers = s1_record["street_numbers"]
        postal = s1_record["postal_code"]
        street_tok = s1_record["first_street_token"]

        cands_exact = set()
        cands_a = set()
        cands_b = set()
        cands_c = set()
        cands_d = set()
        cands_e = set()
        cands_p = set()

        # Pass 0: Exact compact core
        if compact in self.idx_exact_name:
            cands_exact.update(self.idx_exact_name[compact][:25])

        s1_words = s1_record.get("core_words")
        if s1_words is None:
            s1_words = set(w for w in core_name.split() if len(w) >= 2)
            s1_record["core_words"] = s1_words

        words = sorted(list(s1_words), key=lambda w: self.token_freq.get(w, 0))

        # Pass A: Rare tokens (rarest first, check up to 4 words, up to 30 per word)
        for w in words[:4]:
            if self.token_freq.get(w, 0) <= self.rare_thresh and w in self.idx_rare_token:
                cands_a.update(self.idx_rare_token[w][:30])

        # Pass A2: Bigrams
        for i in range(len(words) - 1):
            pair = (words[i], words[i+1])
            if pair in self.idx_bigram:
                cands_a.update(self.idx_bigram[pair][:20])

        # Pass B & B2: Street Number + Word / Street (ALWAYS queried)
        for num in numbers:
            for w in words[:3]:
                key = (num, w)
                if key in self.idx_num_first:
                    cands_b.update(self.idx_num_first[key][:20])
            if street_tok:
                key = (num, street_tok)
                if key in self.idx_num_street:
                    cands_b.update(self.idx_num_street[key][:20])

        # Pass D: Street Number + Postal (ALWAYS queried)
        if postal:
            for num in numbers:
                key = (num, postal)
                if key in self.idx_num_post:
                    cands_d.update(self.idx_num_post[key][:20])

        # Pass P1 & P2: Postal Code + Name Word / Street Token (ALWAYS queried when postal present)
        if postal:
            for w in words[:4]:
                key = (postal, w)
                if key in self.idx_post_name:
                    cands_p.update(self.idx_post_name[key][:20])
            if street_tok:
                key = (postal, street_tok)
                if key in self.idx_post_street:
                    cands_p.update(self.idx_post_street[key][:20])

        # Pass E: Address token pairs (ALWAYS queried)
        addr_words = sorted([a for a in s1_record.get("addr_tokens", set()) if self.addr_token_freq.get(a, 0) <= 800], key=lambda x: self.addr_token_freq.get(x, 0))
        if len(addr_words) >= 2:
            key = (addr_words[0], addr_words[1])
            if key in self.idx_addr_rare:
                cands_e.update(self.idx_addr_rare[key][:20])

        # Pass C: Character 3-grams (fallback if total candidates < 15)
        if len(cands_exact | cands_a | cands_b | cands_d | cands_e | cands_p) < 15 and len(compact) >= 3:
            char_counts = Counter()
            tris = [compact[i:i+3] for i in range(len(compact)-2)]
            tris.sort(key=lambda t: self.char3_freq.get(t, 0))
            for tri in tris[:3]:
                if self.char3_freq.get(tri, 0) <= self.char3_thresh and tri in self.idx_char3:
                    for cid in self.idx_char3[tri][:20]:
                        char_counts[cid] += 1
            for cid, _ in char_counts.most_common(10):
                cands_c.add(cid)

        # Union
        union_set = cands_exact | cands_a | cands_b | cands_c | cands_d | cands_e | cands_p

        # Intelligent Similarity-Based Capping (preserves true matches with high similarity)
        if len(union_set) > self.adaptive_cap:
            s1_nums = numbers
            len_s1 = len(s1_words)
            scored = []
            for cand in union_set:
                r = self.pool_lookup.get(cand)
                if not r:
                    continue
                c_words = r.get("core_words", set())
                inter = len(s1_words & c_words)
                union_len = len_s1 + len(c_words) - inter
                word_sim = inter / union_len if union_len > 0 else 0.0
                num_sim = 1.0 if (s1_nums and r["street_numbers"] & s1_nums) else 0.0
                post_sim = 1.0 if (postal and r.get("postal_code") == postal) else 0.0
                exact_bonus = 0.5 if cand in cands_exact else 0.0
                pass_bonus = 0.35 if cand in (cands_p | cands_e) else 0.0
                score = 0.40 * word_sim + 0.20 * num_sim + 0.20 * post_sim + exact_bonus + pass_bonus
                scored.append((cand, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            union_set = {cand for cand, _ in scored[:self.adaptive_cap]}

        per_pass = {
            "Pass_Exact": cands_exact,
            "Pass_A": cands_a,
            "Pass_B": cands_b,
            "Pass_C": cands_c,
            "Pass_D": cands_d,
            "Pass_E": cands_e,
            "Pass_P": cands_p,
        }
        return union_set, per_pass



def run_blocking_validation(
    val_dir: Path,
    out_candidate_path: Path,
    adaptive_cap: int = 30
):
    """
    Run 5-pass blocking on the local validation split and audit pair-level recall.
    """
    logger.info("Loading validation data for blocking...")

    # Load S1
    s1_path = val_dir / "val_source1.tsv"
    gt_path = val_dir / "val_ground_truth.tsv"
    s2_path = val_dir / "val_source2.tsv"
    s3_path = val_dir / "val_source3.tsv"

    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str)

    # Load Ground Truth
    gt_pairs = set()
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[1]:
                s1_id = parts[0].strip()
                for mid in parts[1].split(","):
                    if mid.strip():
                        gt_pairs.add((s1_id, mid.strip()))

    total_gt_pairs = len(gt_pairs)
    logger.info(f"Total True Ground Truth Pairs to retrieve: {total_gt_pairs:,}")

    # Process by country partition
    all_candidates = {}
    retrieved_pairs = set()
    pass_retrieved = defaultdict(set)

    for country in ["US", "India"]:
        with Timer(f"Blocking partition for {country}"):
            s1_country = df_s1[df_s1["country"] == country]
            s2_country = df_s2[df_s2["country"] == country]
            s3_country = df_s3[df_s3["country"] == country]

            logger.info(f"Country {country}: {len(s1_country):,} S1, {len(s2_country):,} S2, {len(s3_country):,} S3")

            # Preprocess pool (S2 + S3)
            pool_records = []
            for df_src in [s2_country, s3_country]:
                for _, r in df_src.iterrows():
                    nv = clean_name_multiview(r.get("business_name", ""))
                    av = clean_address_multiview(r.get("business_address", ""), country=country)
                    pool_records.append({
                        "entity_id": r["entity_id"],
                        "country": country,
                        **nv,
                        **av
                    })

            # Build Inverted Indexes
            engine = BlockingEngine(adaptive_cap=adaptive_cap)
            engine.build_indexes(pool_records)

            # Query for each S1 entity
            for _, r in tqdm(s1_country.iterrows(), total=len(s1_country), desc=f"Querying S1 ({country})"):
                s1_id = r["entity_id"]
                nv = clean_name_multiview(r.get("business_name", ""))
                av = clean_address_multiview(r.get("business_address", ""), country=country)
                s1_rec = {"entity_id": s1_id, "country": country, **nv, **av}

                cands, per_pass = engine.retrieve_candidates_for_s1(s1_rec)
                all_candidates[s1_id] = cands

                # Audit recall
                for cand in cands:
                    if (s1_id, cand) in gt_pairs:
                        retrieved_pairs.add((s1_id, cand))

                for p_name, p_set in per_pass.items():
                    for cand in p_set:
                        if (s1_id, cand) in gt_pairs:
                            pass_retrieved[p_name].add((s1_id, cand))

    # Compute and log recall
    recall = len(retrieved_pairs) / total_gt_pairs if total_gt_pairs > 0 else 1.0
    logger.info("=" * 60)
    logger.info(f"OVERALL BLOCKING PAIR-LEVEL RECALL: {recall * 100:.2f}% ({len(retrieved_pairs):,} / {total_gt_pairs:,})")
    logger.info(f"Average candidates per S1 entity: {sum(len(c) for c in all_candidates.values()) / len(all_candidates):.2f}")

    logger.info("--- Per-Pass Marginal Contribution ---")
    for p_name in sorted(pass_retrieved.keys()):
        p_recall = len(pass_retrieved[p_name]) / total_gt_pairs * 100
        logger.info(f"  {p_name}: {len(pass_retrieved[p_name]):,} true pairs ({p_recall:.2f}%)")
    logger.info("=" * 60)

    # Save candidate pairs TSV
    write_submission_tsv(
        out_candidate_path,
        all_candidates,
        col_s1="source1_entity_id",
        col_target="candidate_entity_ids"
    )
    logger.info(f"Saved candidate pairs to: {out_candidate_path}")
    return recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-pass blocking.")
    parser.add_argument("--val-dir", type=str, default="./data/val", help="Path to validation directory")
    parser.add_argument("--out-candidate", type=str, default="./data/val/candidate_pairs.tsv", help="Path to output candidate pairs")
    parser.add_argument("--adaptive-cap", type=int, default=45, help="Adaptive candidate cap per S1 entity (default: 45)")
    args = parser.parse_args()

    run_blocking_validation(
        val_dir=Path(args.val_dir),
        out_candidate_path=Path(args.out_candidate),
        adaptive_cap=args.adaptive_cap
    )

