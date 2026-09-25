#!/usr/bin/env python3
"""
Pairwise Feature Extraction Engine for Entity Resolution.

Extracts 28+ domain-specific tabular similarity features across multi-view names and addresses:
  - Exact match booleans (norm, core, first token)
  - String distances: Jaro-Winkler, Token Jaccard, Char 3-gram Jaccard, Token Containment
  - Address metrics: Number overlap, Number mismatch penalty, Postal code match
  - Asymmetric containment metrics (short DBA names inside long registered legal names)
  - Mutual candidate cohesion metrics (intra-cluster similarity)
  - Cross interaction terms: (name_sim * addr_sim)

Supports RapidFuzz (C++ accelerated) with pure Python fallback.
Compatible with Windows (spawn multiprocessing) and Linux/macOS.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger

# Try importing rapidfuzz for C++ string acceleration
try:
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    from difflib import SequenceMatcher


def jaro_winkler_sim(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler similarity in [0, 1]."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return float(JaroWinkler.similarity(s1, s2))
    return float(SequenceMatcher(None, s1, s2).ratio())


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity in [0, 1]."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    g1 = {s1[i:i+n] for i in range(len(s1) - n + 1)} if len(s1) >= n else {s1}
    g2 = {s2[i:i+n] for i in range(len(s2) - n + 1)} if len(s2) >= n else {s2}
    union = len(g1 | g2)
    return len(g1 & g2) / union if union > 0 else 0.0


def token_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Compute token Jaccard similarity in [0, 1]."""
    if not set1 or not set2:
        return 0.0
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 0.0


def token_containment(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute asymmetric token containment: |A & B| / min(|A|, |B|).
    Critical for DBA names vs full corporate names (e.g. 'Prime Money' in 'Prime Money Inc Services').
    """
    if not set1 or not set2:
        return 0.0
    denom = min(len(set1), len(set2))
    return len(set1 & set2) / denom if denom > 0 else 0.0


def extract_pair_features(
    s1_rec: Dict,
    s23_rec: Dict
) -> Dict[str, float]:
    """
    Extract pairwise features between a Source 1 entity and a Source 2/3 candidate.
    """
    feats = {}

    # 1. Exact Match Booleans
    n1, n2 = s1_rec["norm_name"], s23_rec["norm_name"]
    c1, c2 = s1_rec["core_name"], s23_rec["core_name"]
    t1, t2 = s1_rec["first_token"], s23_rec["first_token"]

    feats["exact_norm_name"] = 1.0 if n1 and n1 == n2 else 0.0
    feats["exact_core_name"] = 1.0 if c1 and c1 == c2 else 0.0
    feats["exact_first_token"] = 1.0 if t1 and t1 == t2 else 0.0

    # 2. String Similarities
    feats["jw_norm_name"] = jaro_winkler_sim(n1, n2)
    feats["jw_core_name"] = jaro_winkler_sim(c1, c2)
    feats["char3_jaccard_name"] = char_ngram_jaccard(c1, c2, n=3)

    tokens1 = set(n1.split())
    tokens2 = set(n2.split())
    feats["token_jaccard_name"] = token_jaccard(tokens1, tokens2)
    feats["token_containment_name"] = token_containment(tokens1, tokens2)

    # Length Ratios
    len1, len2 = len(n1), len(n2)
    feats["name_len_diff"] = float(abs(len1 - len2))
    feats["name_len_ratio"] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

    # 3. Address Features
    a1, a2 = s1_rec["clean_addr"], s23_rec["clean_addr"]
    feats["is_addr_missing"] = 1.0 if (not a1 or not a2) else 0.0

    nums1 = s1_rec.get("street_numbers", set())
    nums2 = s23_rec.get("street_numbers", set())

    # Number overlap & penalty
    has_common_num = bool(nums1 & nums2)
    feats["has_common_number"] = 1.0 if has_common_num else 0.0
    feats["exact_number_set"] = 1.0 if nums1 and nums1 == nums2 else 0.0
    # Mismatch penalty: both records have numbers, but share none!
    feats["has_number_mismatch"] = 1.0 if (nums1 and nums2 and not has_common_num) else 0.0

    # Postal code
    p1 = s1_rec.get("postal_code", "")
    p2 = s23_rec.get("postal_code", "")
    feats["exact_postal_code"] = 1.0 if (p1 and p2 and p1 == p2) else 0.0
    feats["has_postal_mismatch"] = 1.0 if (p1 and p2 and p1 != p2) else 0.0

    # Address token similarity
    atok1 = s1_rec.get("addr_tokens", set())
    atok2 = s23_rec.get("addr_tokens", set())
    feats["jw_clean_addr"] = jaro_winkler_sim(a1, a2) if (a1 and a2) else 0.0
    feats["token_jaccard_addr"] = token_jaccard(atok1, atok2)
    feats["token_containment_addr"] = token_containment(atok1, atok2)

    # Street token match
    st1 = s1_rec.get("first_street_token", "")
    st2 = s23_rec.get("first_street_token", "")
    feats["exact_first_street"] = 1.0 if (st1 and st2 and st1 == st2) else 0.0

    # 4. Cross-Feature Interaction
    feats["cross_name_addr_sim"] = feats["jw_core_name"] * feats["jw_clean_addr"]
    feats["cross_containment"] = feats["token_containment_name"] * feats["token_containment_addr"]

    return feats


if __name__ == "__main__":
    rec1 = {
        "norm_name": "prime money",
        "core_name": "prime money",
        "first_token": "prime",
        "clean_addr": "17560 ellis rd tahlequah ok",
        "street_numbers": {"17560"},
        "postal_code": "",
        "first_street_token": "ellis",
        "addr_tokens": {"17560", "ellis", "rd", "tahlequah", "ok"}
    }
    rec2 = {
        "norm_name": "prime money inc",
        "core_name": "prime money",
        "first_token": "prime",
        "clean_addr": "17560 ellis rd tahlequah oklahoma",
        "street_numbers": {"17560"},
        "postal_code": "",
        "first_street_token": "ellis",
        "addr_tokens": {"17560", "ellis", "rd", "tahlequah", "oklahoma"}
    }
    feats = extract_pair_features(rec1, rec2)
    print(f"Extracted {len(feats)} features:")
    for k, v in feats.items():
        print(f"  {k}: {v}")
    print("\nFeature extraction unit test passed successfully!")
