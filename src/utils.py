#!/usr/bin/env python3
"""
Cross-platform utility functions for data loading, Parquet I/O, timing, and memory management.
Designed to run seamlessly on both Windows and POSIX (Linux/macOS).
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False
    pd = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ER")


def find_file(directory: Path, candidate_names: List[str]) -> Path:
    """Find the first matching file among candidate names in a directory (case-insensitive)."""
    for name in candidate_names:
        p = directory / name
        if p.exists():
            return p
    # Try case-insensitive search
    dir_files = {f.name.lower(): f for f in directory.iterdir() if f.is_file()} if directory.exists() else {}
    for name in candidate_names:
        if name.lower() in dir_files:
            return dir_files[name.lower()]
    raise FileNotFoundError(f"None of {candidate_names} found in {directory}")



class Timer:
    """Context manager for timing execution blocks."""
    def __init__(self, description: str):
        self.description = description
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting: {self.description}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(elapsed, 60)
        if mins > 0:
            logger.info(f"Finished: {self.description} in {int(mins)}m {secs:.2f}s")
        else:
            logger.info(f"Finished: {self.description} in {secs:.2f}s")


def load_tsv_mapping(tsv_path: Path, col_s1: str = "source1_entity_id", col_target: str = "matched_entity_ids") -> Dict[str, Set[str]]:
    """
    Fast reader for ground truth or results TSV into a {s1_id: set_of_target_ids} dictionary.
    Handles singletons (empty string / NaN) cleanly.
    """
    mapping = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            idx_s1 = header.index(col_s1)
            idx_target = header.index(col_target)
        except ValueError:
            raise ValueError(f"Expected columns {[col_s1, col_target]} in {tsv_path}, got {header}")

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[idx_s1].strip()
            if len(parts) > idx_target and parts[idx_target].strip():
                targets = {x.strip() for x in parts[idx_target].split(",") if x.strip()}
            else:
                targets = set()
            mapping[s1_id] = targets
    return mapping


def write_submission_tsv(
    output_path: Path,
    mapping: Dict[str, Set[str]],
    col_s1: str = "source1_entity_id",
    col_target: str = "matched_entity_ids"
):
    """
    Write a submission file in exact official TSV format.
    Ensures:
      - Separated by single TAB
      - No quotes
      - Comma-separated target IDs with NO duplicates
      - Empty string for singletons
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{col_s1}\t{col_target}\n")
        for s1_id in sorted(mapping.keys()):
            target_ids = sorted(mapping[s1_id])
            target_str = ",".join(target_ids) if target_ids else ""
            f.write(f"{s1_id}\t{target_str}\n")


def to_parquet_fast(df: "pd.DataFrame", path: Path):
    """Save dataframe to Parquet with snappy compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)


def read_parquet_fast(path: Path) -> "pd.DataFrame":
    """Read Parquet file using pyarrow."""
    return pd.read_parquet(path, engine="pyarrow")
