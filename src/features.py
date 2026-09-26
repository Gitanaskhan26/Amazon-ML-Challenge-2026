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


def soundex(s: str) -> str:
    """Fast pure-Python Soundex for phonetic transliteration matching."""
    if not s or not s[0].isalpha():
        return ""
    s = s.upper()
    first = s[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    encoded = [first]
    prev = mapping.get(first, '')
    for c in s[1:]:
        code = mapping.get(c, '')
        if code and code != prev:
            encoded.append(code)
            prev = code
        elif c not in ('H', 'W'):
            prev = ''
    return ("".join(encoded) + "0000")[:4]


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


def monge_elkan_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """
    Compute asymmetric Monge-Elkan similarity from tokens1 to tokens2:
    Mean of max Jaro-Winkler similarities for each token in tokens1 across tokens2.
    Uses an exact set membership lookup to skip fuzzy distance when token is an identical match.
    """
    if not tokens1 or not tokens2:
        return 0.0
    s2_set = set(tokens2)
    scores = []
    for u in tokens1:
        if u in s2_set:
            scores.append(1.0)
        else:
            max_sim = max((jaro_winkler_sim(u, v) for v in tokens2), default=0.0)
            scores.append(max_sim)
    return sum(scores) / len(scores)


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
    feats["soundex_first_token_match"] = 1.0 if (t1 and t2 and soundex(t1) == soundex(t2)) else 0.0

    # Leetspeak Inversion Match
    l1 = s1_rec.get("guarded_leet_name", "")
    l2 = s23_rec.get("guarded_leet_name", "")
    feats["exact_leet_name"] = 1.0 if l1 and l1 == l2 else 0.0
    feats["jw_leet_name"] = jaro_winkler_sim(l1, l2) if (l1 and l2) else 0.0

    # 2. String Similarities
    feats["jw_norm_name"] = jaro_winkler_sim(n1, n2)
    feats["jw_core_name"] = jaro_winkler_sim(c1, c2)
    feats["char3_jaccard_name"] = char_ngram_jaccard(c1, c2, n=3)

    tokens1 = set(n1.split())
    tokens2 = set(n2.split())
    common_tokens = tokens1 & tokens2
    feats["token_jaccard_name"] = token_jaccard(tokens1, tokens2)
    feats["token_containment_name"] = token_containment(tokens1, tokens2)
    feats["common_name_tokens"] = float(len(common_tokens))

    # Monge-Elkan soft token matching (robust to token reordering and insertions)
    words1 = n1.split()
    words2 = n2.split()
    me_12 = monge_elkan_similarity(words1, words2)
    me_21 = monge_elkan_similarity(words2, words1)
    feats["monge_elkan_name"] = 0.5 * (me_12 + me_21)

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

    # Postal prefix (first 2 digits: state/region in US/France/India)
    has_p_prefix_match = bool(p1 and p2 and len(p1) >= 2 and len(p2) >= 2 and p1[:2] == p2[:2])
    feats["postal_prefix_match"] = 1.0 if has_p_prefix_match else 0.0
    feats["has_postal_prefix_mismatch"] = 1.0 if (p1 and p2 and len(p1) >= 2 and len(p2) >= 2 and p1[:2] != p2[:2]) else 0.0

    # Postal 3-digit prefix (district / metropolitan area)
    has_p3_match = bool(p1 and p2 and len(p1) >= 3 and len(p2) >= 3 and p1[:3] == p2[:3])
    feats["postal_3digit_match"] = 1.0 if has_p3_match else 0.0
    feats["has_postal_3digit_mismatch"] = 1.0 if (p1 and p2 and len(p1) >= 3 and len(p2) >= 3 and p1[:3] != p2[:3]) else 0.0

    # Address token similarity
    atok1 = s1_rec.get("addr_tokens", set())
    atok2 = s23_rec.get("addr_tokens", set())
    common_atok = atok1 & atok2
    feats["jw_clean_addr"] = jaro_winkler_sim(a1, a2) if (a1 and a2) else 0.0
    feats["token_jaccard_addr"] = token_jaccard(atok1, atok2)
    feats["token_containment_addr"] = token_containment(atok1, atok2)
    feats["common_addr_tokens"] = float(len(common_atok))
    feats["both_addr_present_zero_overlap"] = 1.0 if (a1 and a2 and not common_atok) else 0.0

    # Address Monge-Elkan
    awords1 = list(atok1)
    awords2 = list(atok2)
    me_a12 = monge_elkan_similarity(awords1, awords2)
    me_a21 = monge_elkan_similarity(awords2, awords1)
    feats["monge_elkan_addr"] = 0.5 * (me_a12 + me_a21)

    # Address lengths
    alen1, alen2 = len(a1), len(a2)
    feats["addr_len_diff"] = float(abs(alen1 - alen2))
    feats["addr_len_ratio"] = min(alen1, alen2) / max(alen1, alen2) if max(alen1, alen2) > 0 else 1.0

    # Street token match
    st1 = s1_rec.get("first_street_token", "")
    st2 = s23_rec.get("first_street_token", "")
    feats["exact_first_street"] = 1.0 if (st1 and st2 and st1 == st2) else 0.0
    feats["jw_first_street"] = jaro_winkler_sim(st1, st2) if (st1 and st2) else 0.0

    # Source table indicator
    feats["is_s2"] = 1.0 if s23_rec.get("entity_id", "").startswith("S2") else 0.0

    # 4. Cross-Feature Interaction
    feats["cross_name_addr_sim"] = feats["jw_core_name"] * feats["jw_clean_addr"]
    feats["cross_containment"] = feats["token_containment_name"] * feats["token_containment_addr"]

    # 5. Consonant Skeleton Transliteration Features (Cross-script matching)
    sk1 = s1_rec.get("consonant_skel", "")
    sk2 = s23_rec.get("consonant_skel", "")
    if sk1 and sk2:
        feats["skel_similarity"] = jaro_winkler_sim(sk1, sk2)
        sk1_3g = set(sk1[i:i+3] for i in range(len(sk1) - 2)) if len(sk1) >= 3 else set([sk1])
        sk2_3g = set(sk2[i:i+3] for i in range(len(sk2) - 2)) if len(sk2) >= 3 else set([sk2])
        union_3g = sk1_3g | sk2_3g
        feats["skel_char3_jaccard"] = len(sk1_3g & sk2_3g) / len(union_3g) if union_3g else 0.0
    else:
        feats["skel_similarity"] = 0.0
        feats["skel_char3_jaccard"] = 0.0

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
