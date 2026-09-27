# Business Entity Resolution Pipeline (Amazon ML Challenge 2026)

Industrial-grade, cross-platform Entity Resolution pipeline designed for massive multi-source commercial entity matching (~12M records) optimized for the **Macro-Averaged $F_{0.5}$** metric.

---

## 1. System Architecture

The pipeline consists of five production stages:
1. **Guarded Multi-View Normalization**: Cleans punctuation, strips accents, normalizes Indic Unicode blocks, applies guarded leetspeak inversion on words while preserving numeric street numbers and postal codes (`0017560` $\to$ `17560`).
2. **Strict Country Partitioning**: Isolates candidate generation and index spaces strictly by country (`US`, `India`, `France`), eliminating cross-country false positives.
3. **Five-Pass Inverted Index Blocking**: Combines rare-token inverted indexes, consonant skeletons, building street numbers, and postal co-location keys with dynamic adaptive capping (`adaptive_cap=160`) to generate candidate sets with $\ge 96.6\%$ blocking recall.
4. **56-Feature Gradient Boosted Decision Trees**: Fast, vectorized pairwise feature extraction evaluated with a 2,000-tree LightGBM model trained on 12.63M candidate pairs (ROC-AUC: `0.999740`).
5. **Super-Precision Tripartite Graph Post-Processing**:
   - Solves Global Bipartite Maximum Matching with ground-truth physical caps (Max 5 S2, 6 S3, 8 total).
   - Recovers borderline true matches down to score 0.28 via **Transitive Twin Rescue ($S_2 \leftrightarrow S_3$)**.
   - Prunes multi-match lookalikes via **Tri-Country Spatial Contradiction Vetoes** (India 2-digit PIN, US State/ZIP, France Department).
   - Eliminates isolated false alarms via **Uncorroborated Singleton Squeezing**.

---

## 2. Environment Setup

### Prerequisites
- Python 3.10 – 3.14 (64-bit)
- RAM: $\ge 32$ GB recommended (128 GB optimal for full 11.7M dataset)
- OS: Windows, Linux, or macOS

### Installation
```bash
# 1. Clone or unpack the repository
git clone https://github.com/Gitanaskhan26/Amazon-ML-Challenge-2026.git
cd Amazon-ML-Challenge-2026

# 2. Create and activate a clean virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows PowerShell:
.\venv\Scripts\activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 3. End-to-End Reproduction Guide

Anyone can regenerate both official deliverable files (`matching_results.tsv` and `candidate_pairs.tsv`) using the steps below.

### Step 1: Run Full Test Inference
Runs the 5-pass blocking engine and LightGBM scoring across the test set:
```bash
python src/inference.py --test-dir "/path/to/dataset/test" --output-dir "./output" --adaptive-cap 160
```
- **Inputs**: `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` in `--test-dir`.
- **Outputs**:
  - `output/candidate_pairs.tsv` (blocking candidate pairs)
  - `output/matching_results.tsv` (preliminary model matches)
  - `output/scored_pairs_*.tsv` (cached score pools for fast post-processing)

### Step 2: Run Super-Precision Post-Processing & Grid Search
Executes Tripartite Graph Verification, Transitive Twin Rescue, and Spatial Contradiction Vetoes:
```bash
python src/grid_search_postprocess.py --output-dir "./output" --test-dir "/path/to/dataset/test"
```
- **Inputs**: `output/scored_pairs_*.tsv` + test directory metadata.
- **Outputs**: Overwrites `output/matching_results.tsv` with the precision-maximized predictions.
- **Execution Time**: ~2 to 3 minutes.

### Step 3: Verify Official Deliverables
Runs the official submission format validator to verify structural and ID integrity:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir "/path/to/dataset/test"
```
- Expected output: `>>> VALIDATION STATUS: PASS (Exit code 0). 100% SUBMISSION READY! <<<`

---

## 4. (Optional) Model Retraining from Scratch

If you wish to retrain the LightGBM model from training data:
```bash
# 1. Build the 12.6M training candidate pair pool & train LightGBM
python src/train_full_pool.py --data-dir "/path/to/dataset/train" --output-model "./data/lgb_model.txt"
```

---

## 5. Directory Structure

```text
code/business_entity_resolution/
├── data/
│   ├── lgb_model.txt               # Pre-trained 56-feature LightGBM model
│   ├── mined_token_map.json        # Cross-lingual Indic token dictionary
│   └── calibration_results.json    # Calibrated country threshold parameters
├── src/
│   ├── preprocessing.py            # Guarded normalization & address parser
│   ├── blocking.py                 # 5-pass inverted index blocking engine
│   ├── features.py                 # 56-feature vectorized extractor
│   ├── train_full_pool.py          # Model training pipeline
│   ├── inference.py                # Test inference orchestrator
│   ├── grid_search_postprocess.py  # Super-precision post-processor
│   ├── metrics.py                  # Macro-F0.5 evaluator
│   └── utils.py                    # I/O utilities & TSV formatters
├── utils/
│   └── validate_submission.py      # Official competition validator
├── README.md                       # This reproduction manual
└── requirements.txt                # Pinned production dependencies
```

---

## 6. Official Deliverables

- `output/matching_results.tsv`: Final predicted business entity match mappings.
- `output/candidate_pairs.tsv`: Multi-pass candidate pairs fed to the classifier.
