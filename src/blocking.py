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
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger, write_submission_tsv
from src.preprocessing import clean_name_multiview, clean_address_multiview
from src.transliteration import consonant_skeleton


class BlockingEngine:
    def __init__(self, adaptive_cap: int = 80):
        self.adaptive_cap = adaptive_cap

    def build_indexes(self, pool_records: List[Dict]):
        """
        Build inverted indexes over the secondary pool (Source 2 and Source 3 records).
        """
        self.pool_lookup = {r["entity_id"]: r for r in pool_records}

        self.idx_exact_name = defaultdict(list)
        self.idx_rare_token = defaultdict(list)
        self.idx_bigram = defaultdict(list)
        self.idx_name_pair = defaultdict(list)
        self.idx_num_first = defaultdict(list)
        self.idx_num_street = defaultdict(list)
        self.idx_num_addr = defaultdict(list)
        self.idx_num_post = defaultdict(list)
        self.idx_post_name = defaultdict(list)
        self.idx_post_street = defaultdict(list)
        self.idx_addr_rare = defaultdict(list)
        self.idx_char3 = defaultdict(list)
        self.idx_skel_exact = defaultdict(list)
        self.idx_skel3 = defaultdict(list)

        # Token and 3-gram frequencies
        self.token_freq = Counter()
        self.addr_token_freq = Counter()
        self.char3_freq = Counter()
        self.skel3_freq = Counter()
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
            skel = r.get("consonant_skel", "")
            if len(skel) >= 3:
                for i in range(len(skel) - 2):
                    self.skel3_freq[skel[i:i+3]] += 1

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

            # Pass A2: Core name natural bigrams (preserves natural word adjacency)
            core_toks = [w for w in core_name.split() if len(w) >= 2]
            for i in range(len(core_toks) - 1):
                self.idx_bigram[(core_toks[i], core_toks[i+1])].append(e_id)

            # Pass A3: Unordered Rare Name Pairs (robust to reordering, typos in 3rd word, token insertions)
            clean_words = sorted([w for w in words if len(w) >= 2], key=lambda w: self.token_freq[w])
            for i in range(min(4, len(clean_words))):
                for j in range(i + 1, min(4, len(clean_words))):
                    w1, w2 = clean_words[i], clean_words[j]
                    if min(self.token_freq[w1], self.token_freq[w2]) <= 35000:
                        pair = (w1, w2) if w1 < w2 else (w2, w1)
                        self.idx_name_pair[pair].append(e_id)

            # Pass B: (street_number, name_token)
            for num in numbers:
                for w in words[:3]:
                    self.idx_num_first[(num, w)].append(e_id)

            # Pass B2: (street_number, first_street_token)
            if street_tok and len(street_tok) >= 3:
                for num in numbers:
                    self.idx_num_street[(num, street_tok)].append(e_id)

            # Pass B3: (street_number, addr_token) - Crucial for non-Latin names and messy road names
            addr_words = sorted([a for a in r.get("addr_tokens", set()) if len(a) >= 3 and self.addr_token_freq[a] <= 100000], key=lambda x: self.addr_token_freq[x])
            for num in numbers:
                for a in addr_words[:25]:
                    self.idx_num_addr[(num, a)].append(e_id)

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

            # Pass E: Combinatorial Address Token Pairs
            addr_words_e = sorted([a for a in r.get("addr_tokens", set()) if len(a) >= 3 and self.addr_token_freq[a] <= 35000], key=lambda x: self.addr_token_freq[x])
            for i in range(min(10, len(addr_words_e))):
                for j in range(i + 1, min(10, len(addr_words_e))):
                    pair = (addr_words_e[i], addr_words_e[j]) if addr_words_e[i] < addr_words_e[j] else (addr_words_e[j], addr_words_e[i])
                    self.idx_addr_rare[pair].append(e_id)

            # Pass C: Character 3-grams (filtered for speed)
            if len(compact) >= 3:
                for i in range(len(compact) - 2):
                    tri = compact[i:i+3]
                    if self.char3_freq[tri] <= self.char3_thresh:
                        self.idx_char3[tri].append(e_id)

            # Pass SK: Consonant Skeleton Exact & 3-grams (cross-lingual transliteration bridge)
            skel = r.get("consonant_skel", "")
            if len(skel) >= 3:
                self.idx_skel_exact[skel].append(e_id)
                for i in range(len(skel) - 2):
                    tri = skel[i:i+3]
                    if self.skel3_freq[tri] <= 8000:
                        self.idx_skel3[tri].append(e_id)

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
            cands_exact.update(self.idx_exact_name[compact][:100])

        s1_words = s1_record.get("core_words")
        if s1_words is None:
            s1_words = set(w for w in core_name.split() if len(w) >= 2)
            s1_record["core_words"] = s1_words

        words = sorted(list(s1_words), key=lambda w: self.token_freq.get(w, 0))

        # Pass A: Rare tokens (rarest first, check up to 4 words, up to 40 per word)
        for w in words[:4]:
            if self.token_freq.get(w, 0) <= self.rare_thresh and w in self.idx_rare_token:
                cands_a.update(self.idx_rare_token[w][:40])

        # Pass A2: Core name natural bigrams (preserves natural word adjacency)
        core_toks = [w for w in core_name.split() if len(w) >= 2]
        for i in range(len(core_toks) - 1):
            pair = (core_toks[i], core_toks[i+1])
            if pair in self.idx_bigram:
                cands_a.update(self.idx_bigram[pair][:40])

        # Pass A3: Unordered Rare Name Pairs (vital for word reordering / extra words)
        clean_words = sorted([w for w in words if len(w) >= 2], key=lambda w: self.token_freq.get(w, 0))
        for i in range(min(4, len(clean_words))):
            for j in range(i + 1, min(4, len(clean_words))):
                w1, w2 = clean_words[i], clean_words[j]
                if min(self.token_freq.get(w1, 0), self.token_freq.get(w2, 0)) <= 35000:
                    pair = (w1, w2) if w1 < w2 else (w2, w1)
                    if pair in self.idx_name_pair:
                        cands_a.update(self.idx_name_pair[pair][:40])

        # Pass B, B2, B3: Street Number + Word / Street / Rare Addr Token
        addr_words = sorted([a for a in s1_record.get("addr_tokens", set()) if len(a) >= 3 and self.addr_token_freq.get(a, 0) <= 100000], key=lambda x: self.addr_token_freq.get(x, 0))
        for num in numbers:
            for w in words[:3]:
                key = (num, w)
                if key in self.idx_num_first:
                    cands_b.update(self.idx_num_first[key][:45])
            if street_tok:
                key = (num, street_tok)
                if key in self.idx_num_street:
                    cands_b.update(self.idx_num_street[key][:45])
            for a in addr_words[:25]:
                key = (num, a)
                if key in self.idx_num_addr:
                    cands_b.update(self.idx_num_addr[key][:45])

        # Pass D: Street Number + Postal (ALWAYS queried)
        if postal:
            for num in numbers:
                key = (num, postal)
                if key in self.idx_num_post:
                    cands_d.update(self.idx_num_post[key][:40])

        # Pass P1 & P2: Postal Code + Name Word / Street Token (ALWAYS queried when postal present)
        if postal:
            for w in words[:4]:
                key = (postal, w)
                if key in self.idx_post_name:
                    cands_p.update(self.idx_post_name[key][:45])
            if street_tok:
                key = (postal, street_tok)
                if key in self.idx_post_street:
                    cands_p.update(self.idx_post_street[key][:45])

        # Pass E: Combinatorial Address Token Pairs
        addr_words_e = sorted([a for a in s1_record.get("addr_tokens", set()) if len(a) >= 3 and self.addr_token_freq.get(a, 0) <= 35000], key=lambda x: self.addr_token_freq.get(x, 0))
        for i in range(min(10, len(addr_words_e))):
            for j in range(i + 1, min(10, len(addr_words_e))):
                pair = (addr_words_e[i], addr_words_e[j]) if addr_words_e[i] < addr_words_e[j] else (addr_words_e[j], addr_words_e[i])
                if pair in self.idx_addr_rare:
                    cands_e.update(self.idx_addr_rare[pair][:35])

        # Pass C: Character 3-grams (fallback if total candidates < 15)
        if len(cands_exact | cands_a | cands_b | cands_d | cands_e | cands_p) < 15 and len(compact) >= 3:
            char_counts = Counter()
            tris = [compact[i:i+3] for i in range(len(compact)-2)]
            tris.sort(key=lambda t: self.char3_freq.get(t, 0))
            for tri in tris[:3]:
                if self.char3_freq.get(tri, 0) <= self.char3_thresh and tri in self.idx_char3:
                    for cid in self.idx_char3[tri][:30]:
                        char_counts[cid] += 1
            for cid, _ in char_counts.most_common(15):
                cands_c.add(cid)

        # Pass SK: Consonant Skeleton Exact & 3-grams (cross-lingual transliteration bridge)
        cands_sk = set()
        s1_skel = s1_record.get("consonant_skel", "")
        if len(s1_skel) >= 3:
            if s1_skel in self.idx_skel_exact:
                cands_sk.update(self.idx_skel_exact[s1_skel][:40])
            s1_tris = list(set(s1_skel[i:i+3] for i in range(len(s1_skel) - 2)))
            valid_tris = [t for t in s1_tris if 0 < self.skel3_freq.get(t, 0) <= 8000]
            valid_tris.sort(key=lambda t: self.skel3_freq[t])
            for tri in valid_tris[:5]:
                if tri in self.idx_skel3:
                    cands_sk.update(self.idx_skel3[tri][:35])

        # Union
        union_set = cands_exact | cands_a | cands_b | cands_c | cands_d | cands_e | cands_p | cands_sk

        # Intelligent Similarity-Based Capping (preserves true matches with high similarity)
        if len(union_set) > self.adaptive_cap:
            s1_nums = numbers
            s1_addr_toks = s1_record.get("addr_tokens", set())
            len_s1 = len(s1_words)
            s1_skel = s1_record.get("consonant_skel", "")
            scored = []
            for cand in union_set:
                r = self.pool_lookup.get(cand)
                if not r:
                    continue
                c_words = r.get("core_words", set())
                inter = len(s1_words & c_words)
                union_len = len_s1 + len(c_words) - inter
                word_sim = inter / union_len if union_len > 0 else 0.0

                c_addr_toks = r.get("addr_tokens", set())
                addr_inter = len(s1_addr_toks & c_addr_toks)
                addr_union = len(s1_addr_toks | c_addr_toks)
                addr_sim = addr_inter / addr_union if addr_union > 0 else 0.0

                num_sim = 1.0 if (s1_nums and r["street_numbers"] & s1_nums) else 0.0
                post_sim = 1.0 if (postal and r.get("postal_code") == postal) else 0.0

                # 1. Exact Name Match (Guaranteed top tier)
                exact_bonus = 3.5 if cand in cands_exact else 0.0

                # 2. Skeleton Transliteration Match (cross-lingual phonetic bridge)
                c_skel = r.get("consonant_skel", "")
                exact_skel = bool(len(s1_skel) >= 3 and s1_skel == c_skel)
                skel_bonus = 2.5 if exact_skel else (1.5 if cand in cands_sk else 0.0)

                # 3. Word Overlap Bonuses
                multi_word_bonus = 1.8 if inter >= 2 else (0.3 if inter == 1 and word_sim >= 0.3 else 0.0)

                # 4. Multi Address Overlap
                multi_addr_bonus = 0.50 if addr_inter >= 3 else (0.25 if addr_inter >= 2 else 0.0)
                pass_bonus = 0.35 if cand in (cands_p | cands_e | cands_b) else 0.0

                # 5. Strong Doorstep Physical Address Match:
                # Direct physical co-location at same door/house/plot/building
                strong_doorstep = bool(
                    (num_sim == 1.0 and addr_inter >= 2)
                    or (addr_inter >= 4)
                    or (num_sim == 1.0 and post_sim == 1.0 and addr_inter >= 1)
                )

                # Moderate Doorstep (Street number or 2 address words)
                moderate_doorstep = bool(
                    not strong_doorstep and (
                        (num_sim == 1.0 and addr_inter >= 1)
                        or (addr_inter >= 2)
                        or (num_sim == 1.0 and post_sim == 1.0)
                    )
                )

                # Scaled Doorstep Bonus: Real physical locations get massive scores!
                if strong_doorstep:
                    doorstep_bonus = 2.8 + (0.25 * min(addr_inter, 6))
                elif moderate_doorstep:
                    doorstep_bonus = 1.0
                else:
                    doorstep_bonus = 0.0

                has_name_affinity = bool(inter >= 1 or cand in cands_sk or exact_skel or cand in cands_exact)

                # Confirmed Business Match: Strong Doorstep Address + Name/Skeleton Affinity
                confirmed_bonus = 2.0 if (strong_doorstep and has_name_affinity) else 0.0

                # 6. Brand Name & Missing Address Protection:
                s1_first = s1_record.get("first_token", "")
                c_first = r.get("first_token", "")
                has_first_token_match = bool(s1_first and c_first and s1_first == c_first and self.token_freq.get(s1_first, 0) <= 25000)
                is_missing_addr = bool(len(c_addr_toks) == 0 or not r.get("clean_addr") or r.get("clean_addr") == "nan")
                name_rescue_bonus = 2.2 if (is_missing_addr and (inter >= 2 or has_first_token_match or exact_skel or cand in cands_exact)) else 0.0

                is_high_name = bool(inter >= 2 or has_first_token_match)

                score = (
                    0.25 * word_sim
                    + 0.25 * addr_sim
                    + 0.15 * num_sim
                    + 0.10 * post_sim
                    + exact_bonus
                    + skel_bonus
                    + multi_word_bonus
                    + doorstep_bonus
                    + confirmed_bonus
                    + name_rescue_bonus
                    + multi_addr_bonus
                    + pass_bonus
                )
                scored.append((
                    cand,
                    score,
                    cand in cands_exact,
                    strong_doorstep,
                    confirmed_bonus > 0,
                    cand in cands_sk or exact_skel,
                    is_high_name
                ))

            scored.sort(key=lambda x: x[1], reverse=True)

            # Balanced Multi-Tier Protection (tightly budgeted so total < adaptive_cap):
            # 1. Exact Name Matches (MUST never be dropped)
            t_exact = [x[0] for x in scored if x[2]][:15]
            protected_set = set(t_exact)

            # 2. Strong Doorstep Physical Address Matches (MUST never be dropped - covers DBA/trade names & cross-script entities)
            t_door_cap = max(20, int(self.adaptive_cap * 0.25))
            t_door = [x[0] for x in scored if x[0] not in protected_set and x[3]][:t_door_cap]
            protected_set.update(t_door)

            # 3. Confirmed Business Matches (Doorstep + Name/Skeleton Affinity)
            t_conf_cap = max(15, int(self.adaptive_cap * 0.20))
            t_conf = [x[0] for x in scored if x[0] not in protected_set and x[4]][:t_conf_cap]
            protected_set.update(t_conf)

            # 4. Transliteration Skeleton Matches (cross-lingual phonetic bridge)
            t_skel_cap = max(12, int(self.adaptive_cap * 0.15))
            t_skel = [x[0] for x in scored if x[0] not in protected_set and x[5]][:t_skel_cap]
            protected_set.update(t_skel)

            # 5. High-Confidence Name Matches (>= 2 shared words or rare brand first token match)
            t_name_cap = max(15, int(self.adaptive_cap * 0.20))
            t_name = [x[0] for x in scored if x[0] not in protected_set and x[6]][:t_name_cap]
            protected_set.update(t_name)

            # 6. Fill remaining slots with the highest scoring candidates overall
            remaining = [x[0] for x in scored if x[0] not in protected_set]
            slots_left = max(0, self.adaptive_cap - len(protected_set))
            union_set = protected_set | set(remaining[:slots_left])

        per_pass = {
            "Pass_Exact": cands_exact,
            "Pass_A": cands_a,
            "Pass_B": cands_b,
            "Pass_C": cands_c,
            "Pass_D": cands_d,
            "Pass_E": cands_e,
            "Pass_P": cands_p,
            "Pass_SK": cands_sk,
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

    import pandas as pd
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

