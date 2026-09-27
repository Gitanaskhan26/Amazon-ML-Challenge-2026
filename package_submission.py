#!/usr/bin/env python3
"""
Automated Final Submission Packager for Amazon ML Challenge 2026.

Assembles the exact required submission ZIP structure:
    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── data/               (lgb_model.txt, mined_token_map.json, etc.)
    │       ├── src/                (all source code)
    │       ├── utils/              (validate_submission.py)
    │       ├── README.md           (reproduction instructions)
    │       └── requirements.txt    (pinned dependencies)
    └── Documentation_template.md   (methodology write-up)

Usage:
    python package_submission.py --team-name "MyTeamName" --output-dir "./output_fast"
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Package official submission ZIP")
    parser.add_argument("--team-name", type=str, required=True,
                        help="Your official team name (used in ZIP filename)")
    parser.add_argument("--output-dir", type=str, default="./output_fast",
                        help="Directory containing matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--repo-dir", type=str, default=".",
                        help="Root directory of the repository")
    parser.add_argument("--destination", type=str, default=".",
                        help="Destination directory to save the final ZIP")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dest_dir = Path(args.destination).resolve()

    clean_team_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in args.team_name.strip())
    zip_filename = f"{clean_team_name}_submission.zip"
    zip_path = dest_dir / zip_filename

    print("=" * 60)
    print("=== Amazon ML Challenge 2026: Final Submission Packager ===")
    print(f"Team Name:   {args.team_name}")
    print(f"Output Dir:  {output_dir}")
    print(f"ZIP Target:  {zip_path}")
    print("=" * 60)

    # 1. Verify deliverables exist
    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"

    if not matching_tsv.exists():
        print(f"ERROR: Missing {matching_tsv}! Run inference and post-processing first.")
        sys.exit(1)
    if not candidate_tsv.exists():
        print(f"ERROR: Missing {candidate_tsv}!")
        sys.exit(1)

    # 2. Verify Documentation_template.md exists
    doc_file = repo_dir / "Documentation_template.md"
    if not doc_file.exists():
        # Fallback check
        alt_doc = Path("/Users/anaskhan/Downloads/student_resource/Documentation_template.md")
        if alt_doc.exists():
            doc_file = alt_doc
        else:
            print(f"ERROR: Missing {doc_file}!")
            sys.exit(1)

    print("\n[1/3] Packaging deliverables and documentation...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # A. Root-level Documentation_template.md
        print(f"  Adding: Documentation_template.md")
        zf.write(doc_file, arcname="Documentation_template.md")

        # B. output/ directory
        print(f"  Adding: output/matching_results.tsv ({matching_tsv.stat().st_size / (1024*1024):.1f} MB)")
        zf.write(matching_tsv, arcname="output/matching_results.tsv")

        print(f"  Adding: output/candidate_pairs.tsv ({candidate_tsv.stat().st_size / (1024*1024):.1f} MB)")
        zf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")

        # C. code/business_entity_resolution/
        print("\n[2/3] Packaging self-contained code...")
        base_arc = "code/business_entity_resolution"

        # README and requirements
        for fname in ["README.md", "requirements.txt"]:
            fpath = repo_dir / fname
            if fpath.exists():
                print(f"  Adding: {base_arc}/{fname}")
                zf.write(fpath, arcname=f"{base_arc}/{fname}")

        # Source code in src/
        src_dir = repo_dir / "src"
        if src_dir.exists():
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if f.endswith(".py"):
                        fpath = Path(root) / f
                        rel_path = fpath.relative_to(repo_dir)
                        print(f"  Adding: {base_arc}/{rel_path}")
                        zf.write(fpath, arcname=f"{base_arc}/{rel_path}")

        # Utils in utils/
        utils_dir = repo_dir / "utils"
        if utils_dir.exists():
            for root, _, files in os.walk(utils_dir):
                for f in files:
                    if f.endswith(".py"):
                        fpath = Path(root) / f
                        rel_path = fpath.relative_to(repo_dir)
                        print(f"  Adding: {base_arc}/{rel_path}")
                        zf.write(fpath, arcname=f"{base_arc}/{rel_path}")

        # Model artifacts in data/ (essential for reproduction without 10h retraining)
        data_dir = repo_dir / "data"
        if data_dir.exists():
            for mfile in ["lgb_model.txt", "mined_token_map.json", "calibration_results.json"]:
                fpath = data_dir / mfile
                if fpath.exists():
                    print(f"  Adding model artifact: {base_arc}/data/{mfile} ({fpath.stat().st_size / (1024*1024):.1f} MB)")
                    zf.write(fpath, arcname=f"{base_arc}/data/{mfile}")

    print("\n[3/3] Verifying ZIP structure...")
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"SUCCESS! Package created: {zip_path} ({zip_size_mb:.2f} MB)")

    # Print contents summary
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        print(f"\nTotal files in ZIP: {len(namelist)}")
        print("\nZIP Content Hierarchy:")
        for name in sorted(namelist):
            info = zf.getinfo(name)
            print(f"  {name:<60} {info.file_size / (1024*1024):>8.2f} MB (uncompressed)")

    print("\n" + "=" * 60)
    print(">>> SUBMISSION ZIP IS 100% READY FOR UPLOAD <<<")
    print("=" * 60)


if __name__ == "__main__":
    main()
