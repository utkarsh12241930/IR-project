# Quick Start: Baseline vs HANS Comparison

## Current Setup ✅

- **Dataset**: MS MARCO 20K subset (16K train, 2K val, 2K test)
- **Location**: `data/mmarco/`
- **Goal**: Compare Baseline DPR vs HANS to showcase novelty

## Running the Comparison

### Step 1: Run Baseline Training

```bash
python simpletransformers/examples/retrieval/train_baseline.py
```

**What it does:**
- Trains DPR without HANS
- Saves to: `trained_models/baseline_dpr/`
- Time: ~30-60 minutes

### Step 2: Run HANS Training

```bash
python simpletransformers/examples/retrieval/train_hans.py
```

**What it does:**
- Trains DPR with HANS enabled
- Saves to: `trained_models/hans_dpr/`
- Time: ~35-70 minutes

### Step 3: Compare Results

Check evaluation metrics in:
- Baseline: `trained_models/baseline_dpr/eval_results.txt`
- HANS: `trained_models/hans_dpr/eval_results.txt`

**Key metrics to compare:**
- **MRR@10** (Mean Reciprocal Rank)
- **Recall@100**
- **Loss** (lower is better)

## Expected Results

HANS should show **improvements** over baseline:
- Higher MRR@10
- Higher Recall@100
- Better handling of hard queries

## Monitoring

Check training status:
```bash
python simpletransformers/examples/retrieval/monitor_training.py
```

## Files Created

After training completes:
- `trained_models/baseline_dpr/eval_results.txt` - Baseline metrics
- `trained_models/hans_dpr/eval_results.txt` - HANS metrics
- Model checkpoints in respective directories

Compare these to showcase HANS improvements!

