# Training Time Estimates for mMARCO

## Dataset Information
- **Training examples**: 16,000
- **Validation examples**: 2,000
- **Test examples**: 2,000
- **Epochs**: 2
- **Batch size**: 8

## Time Estimates

### Baseline (LANS) Training
- **Initialization**: 2-5 minutes
  - Downloading DPR models from HuggingFace (~500 MB)
  - Loading and preprocessing dataset
  - Setting up training infrastructure

- **Training per epoch**: 15-30 minutes
  - Depends on CPU/GPU speed
  - With CPU: ~25-30 minutes per epoch
  - With GPU: ~10-15 minutes per epoch

- **Total Baseline Training**: **30-60 minutes**
  - 2 epochs × 15-30 min = 30-60 minutes
  - Plus initialization time

### HANS Training
- **Initialization**: 2-5 minutes
  - Same as baseline, plus HANS component initialization
  - Hardness predictor setup

- **Training per epoch**: 18-35 minutes
  - Slightly longer than baseline due to:
    - Hardness score computation
    - Adaptive parameter calculations
    - Additional loss terms

- **Total HANS Training**: **35-70 minutes**
  - 2 epochs × 18-35 min = 36-70 minutes
  - Plus initialization time

## Total Comparison Time

**Sequential (one after another)**: 65-130 minutes (1-2 hours)
**Parallel (if you have resources)**: 35-70 minutes (longest of the two)

## What Happens During Training

1. **Initialization (2-5 min)**
   - Downloads models
   - Loads dataset
   - Preprocesses data
   - Creates output directories

2. **Training Loop**
   - Processes batches
   - Computes loss
   - Updates model weights
   - Evaluates every 1000 steps

3. **Evaluation**
   - Runs on validation set
   - Computes metrics (MRR@10, Recall@100, etc.)
   - Saves results

## Monitoring Progress

Check training status with:
```bash
python simpletransformers/examples/retrieval/monitor_training.py
```

Or check the output directories:
- Baseline: `trained_models/baseline_dpr/`
- HANS: `trained_models/hans_dpr/`

## Expected Output Files

After training completes, you'll find:
- `eval_results.txt` - Final evaluation metrics
- `train_results.txt` - Training progress
- `pytorch_model.bin` - Trained model weights
- `model_config.json` - Model configuration
- `hardness_predictor.pt` - HANS predictor (HANS only)

## Comparison Metrics

Compare these metrics between baseline and HANS:
- **MRR@10** (Mean Reciprocal Rank at 10)
- **Recall@100** (Recall at 100)
- **Loss** (Training and validation loss)

HANS should show improvements, especially on harder queries!

