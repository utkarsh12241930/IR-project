# How to Train and Compare Baseline vs HANS

## Step-by-Step Guide (Layman Terms)

### Step 1: Download the MS MARCO Dataset
First, you need to get the data. Run this command in your terminal:

```bash
cd simpletransformers/examples/retrieval
python download_msmarco.py
```

This will:
- Download MS MARCO training data
- Save it to `data/msmarco/msmarco-train.tsv`
- Take a few minutes depending on your internet speed

**What this does:** Gets the real-world question-answer data that we'll train on.

---

### Step 2: Train the Baseline Model (Without HANS)

This trains the normal DPR model - the "before" version.

```bash
python train_baseline.py
```

**What happens:**
- The script loads the MS MARCO data
- Trains a DPR model for 2 epochs (you can increase this later)
- Saves the model to `trained_models/baseline_dpr/`
- Shows evaluation metrics during training

**Time:** Depends on your GPU, but expect 30 minutes to a few hours for a small subset.

**What to look for:** The script will print metrics like MRR@10, Recall@10, etc. Write these down!

---

### Step 3: Train the HANS Model (With HANS)

This trains the same model but WITH your novelty - the "after" version.

```bash
python train_hans.py
```

**What happens:**
- Same as baseline, BUT with `use_hans = True`
- The model now adapts negative sampling based on query hardness
- Saves to `trained_models/hans_dpr/`
- Shows the same metrics during training

**Time:** Similar to baseline (maybe slightly longer due to hardness computation)

**What to look for:** Compare the metrics to baseline - HANS should show improvements!

---

### Step 4: Compare the Results

After both trainings complete, you'll have two sets of metrics:

**Baseline Results** (from `train_baseline.py`):
- MRR@10: X.XX
- Recall@10: X.XX
- etc.

**HANS Results** (from `train_hans.py`):
- MRR@10: Y.YY
- Recall@10: Y.YY
- etc.

**Create a simple table:**

| Metric | Baseline | HANS | Improvement |
|--------|----------|------|-------------|
| MRR@10 | X.XX     | Y.YY | +Z.ZZ%      |
| Recall@10 | X.XX  | Y.YY | +Z.ZZ%      |

This table shows your novelty works!

---

## Quick Start (For Testing on Small Data)

If you want to test quickly before running full training:

1. **Modify the scripts** to use a smaller dataset:
   - In both `train_baseline.py` and `train_hans.py`, add this after loading the data:
   ```python
   # Use only first 1000 examples for quick testing
   train_data = train_data.head(1000)
   ```

2. **Reduce training time:**
   - Set `num_train_epochs = 1` (instead of 2)
   - Set `train_batch_size = 4` (instead of 8)

3. **Run both scripts** - should complete in 10-20 minutes

---

## Troubleshooting

**Problem:** "File not found: data/msmarco/msmarco-train.tsv"
- **Solution:** Run `download_msmarco.py` first (Step 1)

**Problem:** "CUDA out of memory"
- **Solution:** Reduce `train_batch_size` to 4 or 2

**Problem:** Training is too slow
- **Solution:** Use a smaller subset of data (see Quick Start above)

**Problem:** Metrics look the same
- **Solution:** This is normal for small datasets. Try:
  - More epochs (3-5)
  - Full dataset
  - Tune HANS parameters (margin, temperature ranges)

---

## What Files to Run

1. **`download_msmarco.py`** - Download data (run once)
2. **`train_baseline.py`** - Train without HANS
3. **`train_hans.py`** - Train with HANS
4. Compare the results!

---

## Understanding the Output

When training runs, you'll see:
- **Loss values** - Should decrease over time
- **MRR@10** - Mean Reciprocal Rank (higher is better)
- **Recall@10** - How many relevant docs found in top 10 (higher is better)

**Goal:** HANS should have higher MRR and Recall than baseline!

