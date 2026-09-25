# Business Entity Resolution Pipeline (Amazon ML Challenge 2026)

High-performance, cross-platform Entity Resolution pipeline designed for massive multi-source commercial data (~12M entities) evaluated on **Macro-Averaged $F_{0.5}$**.

---

## 1. Architecture Highlights

1. **Guarded Multi-View Normalization**: Reverses synthetic noise and leetspeak (`1` $\to$ `l`, `0` $\to$ `o`) on words while strictly protecting street numbers and postal codes (`17560` is never corrupted).
2. **100% Strict Country Partitioning**: Isolates candidate generation by country (`US`, `India`, and zero-shot `France`). Verified on 7.64M ground truth pairs (0 cross-country matches).
3. **Five-Pass High-Recall Blocking**: Combines rare-token inverted indexes, street-number keys, character 3-grams, and address-only keys (Pass D & E) to achieve $\ge 99.2\%$ true match recall.
4. **Global 1-to-At-Most-1 Invariant Enforcement**: Solves greedy bipartite matching to ensure no Source 2 or Source 3 record is mistakenly assigned to multiple reference entities.
5. **Per-Entity Expected-$F_{0.5}$ Prefix Selection**: Evaluates probability distributions per entity to maximize expected $F_{0.5}$ and accurately identify singletons.

---

## 2. Windows Environment Setup

On your Windows PC (Intel i9-12900K + 128 GB RAM + T1000 GPU):

```powershell
# 1. Clone repository
git clone https://github.com/Gitanaskhan26/Amazon-ML-Challenge-2026.git
cd Amazon-ML-Challenge-2026

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install optimized dependencies
pip install -r requirements.txt
```

---

## 3. Execution Commands

### Step 1: Create Stratified Validation Split & LOCO Probes
Generates a representative 100k-entity validation set + Leave-One-Country-Out (LOCO) probes:
```powershell
python src/create_val_split.py --data-dir "C:\path\to\dataset\train" --output-dir "./data/val" --val-size 100000
```

### Step 2: Audit 5-Pass Blocking Recall
Runs blocking on the validation split and logs pair-level recall and per-pass marginal gain:
```powershell
python src/blocking.py --val-dir "./data/val" --adaptive-cap 30
```

### Step 3: Run Full Test Inference & Generate Deliverables
Runs full inference on the 11.7M test set and outputs both submission files:
```powershell
python src/inference.py --test-dir "C:\path\to\dataset\test" --output-dir "./output" --adaptive-cap 25
```

### Step 4: Validate Deliverables Locally
Runs the official submission format validator:
```powershell
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir "C:\path\to\dataset\test"
```

---

## 4. Deliverables Produced

* `output/matching_results.tsv`: Final entity matches (uploaded to the leaderboard).
* `output/candidate_pairs.tsv`: Candidate set from blocking stage.
