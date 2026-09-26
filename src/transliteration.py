#!/usr/bin/env python3
"""
Universal Brahmic / Indic Phonetic Transliteration Engine.

Provides unified, deterministic ISCII-offset transliteration across 9 Indian scripts:
  - Devanagari (Hindi, Marathi, Sanskrit, Nepali)
  - Bengali / Assamese
  - Gurmukhi (Punjabi)
  - Gujarati
  - Oriya
  - Tamil
  - Telugu
  - Kannada
  - Malayalam

Converts Indic script and synthetic spaced aksharas (e.g. 'प र म यर ट क')
into standardized Latin phonemes and consonant skeletons.
Zero external dependencies, 100% pure Python, optimized for high-throughput streaming.
"""

import re
from typing import Set


# Universal ISCII block offset mapping (relative to script block start: U+0900, U+0A80, U+0B80, etc.)
OFFSET_MAP = {
    # Independent Vowels
    0x05: 'a', 0x06: 'a', 0x07: 'i', 0x08: 'i', 0x09: 'u', 0x0A: 'u',
    0x0B: 'ri', 0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x12: 'o', 0x13: 'o', 0x14: 'au',
    # Velar Consonants
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'ng',
    # Palatal Consonants
    0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j', 0x1D: 'jh', 0x1E: 'ny',
    # Retroflex Consonants
    0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n',
    # Dental Consonants
    0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n',
    # Labial Consonants
    0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm',
    # Semivowels, Sibilants, Fricatives
    0x2F: 'y', 0x30: 'r', 0x31: 'r', 0x32: 'l', 0x33: 'l', 0x34: 'l',
    0x35: 'v', 0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
    # Dependent Vowel Signs (Matras)
    0x3E: 'a', 0x3F: 'i', 0x40: 'i', 0x41: 'u', 0x42: 'u', 0x43: 'ri',
    0x46: 'e', 0x47: 'e', 0x48: 'ai', 0x4A: 'o', 0x4B: 'o', 0x4C: 'au',
    # Modifiers
    0x02: 'n', 0x03: 'h', 0x4D: ''  # Virama / halant
}

VOWEL_SET = set("aeiou \t\n\r-_/.,#()[]{}'\"")


def is_indic_text(text: str) -> bool:
    """Return True if string contains characters in Brahmic Unicode range (U+0900 to U+0D7F)."""
    return any(0x0900 <= ord(c) <= 0x0D7F for c in text)


def transliterate_indic_universal(text: str) -> str:
    """
    Transliterate Brahmic text (Devanagari, Tamil, Kannada, Gujarati, Malayalam, etc.)
    into Romanized Latin phonemes. Handles synthetic whitespace between aksharas.
    """
    if not text:
        return ""

    res = []
    has_indic = False
    for c in text:
        cp = ord(c)
        if 0x0900 <= cp <= 0x0D7F:
            has_indic = True
            off = cp & 0x7F
            if off in OFFSET_MAP:
                res.append(OFFSET_MAP[off])
        else:
            res.append(c)

    if not has_indic:
        return text

    raw = "".join(res)
    # Collapse isolated single-letter spaces produced by synthetic character spacing
    collapsed = re.sub(r"(?<=\b[a-zA-Z])\s+(?=[a-zA-Z]\b)", "", raw)
    return re.sub(r"\s+", " ", collapsed).strip().lower()


def consonant_skeleton(text: str) -> str:
    """
    Produces the invariant consonant skeleton of a name by:
      1. Transliterating Indic script to Latin phonemes
      2. Removing all vowels and punctuation
    Enables cross-lingual matching:
      'Premier Tech' -> 'prmrtch'
      'प र म यर ट क'  -> 'prmyrtk' (89% similarity)
    """
    if not text:
        return ""
    translit = transliterate_indic_universal(text).lower()
    return "".join(c for c in translit if c.isalpha() and c not in VOWEL_SET)


def skeleton_3grams(skeleton: str) -> Set[str]:
    """Extract character 3-grams from consonant skeleton."""
    if len(skeleton) < 3:
        return {skeleton} if skeleton else set()
    return {skeleton[i:i+3] for i in range(len(skeleton) - 2)}
