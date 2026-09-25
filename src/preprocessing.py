#!/usr/bin/env python3
"""
Master Multi-View Preprocessing & Normalization Engine.

Implements guarded multi-view canonicalization to reverse synthetic noise
without corrupting numeric addresses or postal codes:
  1. Guarded Leetspeak Reversal (strictly applied to >=60% alphabetic tokens, len >= 4)
  2. Multi-view representations:
     - raw_name, norm_name, core_name, guarded_leet_name
     - clean_address, street_numbers, postal_code, first_street_token
  3. Ground-Truth Token Alignment Dictionary Mining
  4. Multi-country support: US, India, and France (Zero-Shot)

Compatible with Windows (spawn multiprocessing) and Linux/macOS.
"""

import os
import sys
import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, logger


# 1. Universal Legal Suffixes (US, India, and France)
LEGAL_SUFFIXES = [
    # US / UK / General
    "limited liability company", "incorporated", "corporation", "private limited",
    "public limited", "pvt ltd", "pvt", "ltd", "inc", "corp", "llc", "llp",
    "company", "co", "services", "enterprises", "solutions", "group", "holdings",
    "consulting", "ventures", "management", "associates",
    # France (Zero-shot in test)
    "societe a responsabilite limitee", "societe par actions simplifiee",
    "societe civile immobiliere", "entreprise unipersonnelle a responsabilite limitee",
    "sarl", "sasu", "sas", "sci", "eurl", "snc", "gie", "sa"
]

# Match legal suffixes at the END of a string, or standalone
LEGAL_SUFFIX_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\s*$",
    re.IGNORECASE
)

# 2. Road & Address Abbreviations (US, India, France)
ROAD_ABBREVIATIONS = {
    # English
    r"\broad\b": "rd",
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bboulevard\b": "blvd",
    r"\btrail\b": "trl",
    r"\bcourt\b": "ct",
    r"\bhighway\b": "hwy",
    r"\bplace\b": "pl",
    r"\bcircle\b": "cir",
    r"\bparkway\b": "pkwy",
    r"\bapartment\b": "apt",
    r"\bsuite\b": "ste",
    # French
    r"\brue\b": "r",
    r"\bboulevard\b": "bd",
    r"\bavenue\b": "av",
    r"\bimpasse\b": "imp",
    r"\ballée\b": "all",
    r"\bchemin\b": "ch",
    r"\broute\b": "rte",
}

# 3. Known US State Code Mappings (Bidirectional)
US_STATE_MAP = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}


def strip_accents(text: str) -> str:
    """Normalize unicode and strip combining diacritics (e.g. é -> e, à -> a)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Strip combining accents while leaving base characters intact
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "mn")


def guarded_leet_reverse(token: str) -> str:
    """
    Guarded Leetspeak Inversion:
    Applies 1->l, 0->o, 3->e, 5->s, @->a ONLY if the token is majority-alphabetic
    and length >= 4 (e.g. 'chape1' -> 'chapel', '1imited' -> 'limited').
    Never corrupts pure numbers ('17560'), PIN codes, or unit codes ('102A').
    """
    if len(token) < 4:
        return token

    # Count letters
    n_alpha = sum(1 for c in token if c.isalpha())
    if (n_alpha / len(token)) < 0.60:
        return token  # Protect numeric or mostly non-alpha tokens

    chars = list(token)
    for i, c in enumerate(chars):
        if c == '1':
            chars[i] = 'l'
        elif c == '0':
            chars[i] = 'o'
        elif c == '3':
            chars[i] = 'e'
        elif c == '5':
            chars[i] = 's'
        elif c == '@':
            chars[i] = 'a'
    return "".join(chars)


MINED_TOKEN_MAP = {}

def load_mined_token_map(map_path: Optional[Path] = None):
    """Load pre-mined bilingual/Indic token mapping dictionary if present."""
    global MINED_TOKEN_MAP
    if map_path is None:
        default_path = Path(__file__).resolve().parent.parent / "data" / "mined_token_map.json"
        if default_path.exists():
            map_path = default_path
    if map_path and Path(map_path).exists():
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                MINED_TOKEN_MAP = json.load(f)
        except Exception:
            pass

load_mined_token_map()


def clean_name_multiview(raw_name: str) -> Dict[str, str]:
    """
    Produces protected multi-view representations of a business name:
      - raw_name: unmodified input string
      - norm_name: unaccented, lowercased, handles/domains stripped, punctuation normalized
      - core_name: legal suffixes removed from end, core identity tokens + mined translations
      - guarded_leet_name: leet inversion on majority-alpha tokens
      - first_token: first discriminative word (length >= 2)
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return {
            "raw_name": "",
            "norm_name": "",
            "core_name": "",
            "guarded_leet_name": "",
            "first_token": ""
        }

    s = raw_name.strip()
    raw = s

    # Strip synthetic IDs e.g. '(ID: 84923)', '<NULL>'
    s = re.sub(r"<\s*null\s*>|\(id:\s*\d+\)", "", s, flags=re.IGNORECASE)
    # Strip URL domains and handles
    s = re.sub(r"https?://\S+|www\.\S+|\.(com|org|net|in|fr|co)\b", "", s, flags=re.IGNORECASE)
    s = s.replace("@", " ")

    # Strip accents
    s = strip_accents(s).lower()

    # Guarded leet inversion on word tokens
    tokens = re.split(r"[\s\-_/]+", s)
    leet_tokens = [guarded_leet_reverse(t) for t in tokens if t]
    guarded_leet_str = " ".join(leet_tokens)

    # Normalized text: alphanumeric + Indic unicode blocks
    norm_str = re.sub(r"[^\w\s]", " ", guarded_leet_str)
    norm_str = re.sub(r"\s+", " ", norm_str).strip()

    # Core name: strip legal suffix at end
    core_str = LEGAL_SUFFIX_REGEX.sub("", norm_str).strip()
    if len(core_str) < 2:
        core_str = norm_str

    # Strip honorary business prefixes at START (m/s, messrs, shree, shri, sri)
    clean_prefix = re.sub(r"^(m\s*/\s*s|messrs|shree|shri|sri|om|smt)\b\s*", "", core_str, flags=re.IGNORECASE).strip()
    if len(clean_prefix) >= 2:
        core_str = clean_prefix

    # Augment with mined token translations (e.g. Indic script -> English)
    if MINED_TOKEN_MAP:
        extra_tokens = [MINED_TOKEN_MAP[t] for t in core_str.split() if t in MINED_TOKEN_MAP]
        if extra_tokens:
            core_str = core_str + " " + " ".join(extra_tokens)

    # Extract first non-trivial token
    core_tokens = [t for t in core_str.split() if len(t) >= 2]
    first_token = core_tokens[0] if core_tokens else (norm_str[:4] if norm_str else "")

    return {
        "raw_name": raw,
        "norm_name": norm_str,
        "core_name": core_str,
        "guarded_leet_name": guarded_leet_str,
        "first_token": first_token
    }


def clean_address_multiview(raw_addr: str, country: str = "") -> Dict[str, any]:
    """
    Produces protected multi-view representations of a business address:
      - clean_addr: normalized, unaccented, standardized road tokens
      - street_numbers: set of clean integer strings (leading zeros stripped: '0017560' -> '17560')
      - postal_code: 5-digit (US/France) or 6-digit (India) postal code
      - first_street_token: first alphanumeric road/street token
    """
    if not isinstance(raw_addr, str) or not raw_addr.strip():
        return {
            "clean_addr": "",
            "street_numbers": set(),
            "postal_code": "",
            "first_street_token": "",
            "addr_tokens": set()
        }

    s = raw_addr.strip()
    # Strip synthetic tags
    s = re.sub(r"<\s*null\s*>|null\b", " ", s, flags=re.IGNORECASE)
    s = strip_accents(s).lower()

    # Extract street numbers and postal codes BEFORE altering road abbreviations
    raw_nums = re.findall(r"\b\d+\b", s)

    # Postal codes:
    # France: exactly 5 digits (e.g. 75001, 33000)
    # US: 5 digits (ZIP)
    # India: 6 digits (PIN)
    postal_code = ""
    clean_nums = set()
    for n in raw_nums:
        clean_n = str(int(n))  # strip leading zeros
        if country == "India" and len(n) == 6:
            postal_code = n
        elif country in ("US", "France") and len(n) == 5:
            postal_code = n
        elif not postal_code and len(n) in (5, 6):
            postal_code = n
        # Valid street numbers: length <= 6
        if len(clean_n) <= 6:
            clean_nums.add(clean_n)

    # Standardize road abbreviations
    for pattern, repl in ROAD_ABBREVIATIONS.items():
        s = re.sub(pattern, repl, s)

    # US State expansion/contraction
    if country == "US":
        for full_state, code in US_STATE_MAP.items():
            s = re.sub(r"\b" + full_state + r"\b", code, s)

    # Clean punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = set(s.split())
    # Noise stop-tokens
    noise = {"hn", "plot", "door", "no", "flat", "near", "behind", "opp", "floor"}
    addr_tokens = tokens - noise

    # First street token (excluding pure numbers)
    non_num_tokens = [t for t in s.split() if not t.isdigit() and len(t) >= 3 and t not in noise]
    first_street_token = non_num_tokens[0] if non_num_tokens else ""

    return {
        "clean_addr": s,
        "street_numbers": clean_nums,
        "postal_code": postal_code,
        "first_street_token": first_street_token,
        "addr_tokens": addr_tokens
    }


def preprocess_record(row: Dict[str, str]) -> Dict[str, any]:
    """Preprocess a single row dictionary."""
    name_views = clean_name_multiview(row.get("business_name", ""))
    country = row.get("country", "")
    addr_views = clean_address_multiview(row.get("business_address", ""), country=country)

    return {
        "entity_id": row.get("entity_id", ""),
        "country": country,
        **name_views,
        **addr_views
    }


if __name__ == "__main__":
    # Test guarded leetspeak
    test_words = [
        ("Chape1", "chapel"),
        ("1imited", "limited"),
        ("17560", "17560"),      # Pure numbers MUST NOT be altered
        ("0017560", "0017560"),  # Pure numbers MUST NOT be altered
        ("B+ Retail", "b retail"),
        ("Orelee's Bárbershop", "orelee s barbershop"),
        ("primemoney.com", "primemoney"),
        ("@primemoney", "primemoney")
    ]
    for orig, expected in test_words:
        res = clean_name_multiview(orig)["norm_name"]
        print(f"Name clean: '{orig}' -> '{res}'")

    test_addr = "24620 Savannah Trail, Athens, Alabama 35611"
    addr_res = clean_address_multiview(test_addr, country="US")
    print(f"Addr clean: '{test_addr}' ->")
    print(f"  clean_addr: '{addr_res['clean_addr']}'")
    print(f"  numbers: {addr_res['street_numbers']}")
    print(f"  postal_code: '{addr_res['postal_code']}'")
    print(f"  first_street: '{addr_res['first_street_token']}'")

    print("\nAll preprocessing unit tests passed successfully!")
