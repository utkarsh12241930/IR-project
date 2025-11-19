# HANS Training Guide - Where to Find Your Model

## 📍 Model Save Location

Your trained HANS model will be saved to:

```
C:\Users\utkar\OneDrive\Desktop\mharo\trained_models\hans_dpr
```

## 🗂️ How to Access the Folder

### Method 1: File Explorer (Easiest)
1. Open **File Explorer** (Windows key + E)
2. Copy and paste this path into the address bar:
   ```
   C:\Users\utkar\OneDrive\Desktop\mharo\trained_models\hans_dpr
   ```
3. Press Enter

### Method 2: Command Line
Open PowerShell or Command Prompt and run:
```powershell
cd C:\Users\utkar\OneDrive\Desktop\mharo\trained_models\hans_dpr
explorer .
```

### Method 3: From Project Root
If you're in the project root (`mharo` folder), you can navigate to:
```
trained_models/hans_dpr
```

## 📦 What Files Will Be Saved

When training completes, you'll find:

### Essential Files:
- **`model_config.json`** - Model configuration
- **`pytorch_model.bin`** or **`model.safetensors`** - Model weights (large file, ~400-500 MB)
- **`training_args.bin`** - Training arguments and settings
- **`hardness_predictor.pt`** - HANS hardness predictor model

### Results Files:
- **`train_results.txt`** - Training loss and metrics over time
- **`eval_results.txt`** - Evaluation metrics (MRR@K, Recall@K, etc.)
- **`trainer_state.json`** - Training state and checkpoints

### Checkpoints (if saved):
- **`checkpoint-XXXX/`** - Model checkpoints saved during training

## ⏱️ Training Status

### How to Check if Training is Complete:

1. **Run the status checker:**
   ```bash
   python simpletransformers/examples/retrieval/check_training_status.py
   ```

2. **Check for completion indicators:**
   - Look for `eval_results.txt` - contains final evaluation metrics
   - Check `train_results.txt` - should have entries for all training steps
   - Verify `pytorch_model.bin` or `model.safetensors` exists (this is the trained model)

3. **Expected Training Time:**
   - With 16K training examples, 2 epochs, batch size 8
   - Estimated time: **30-60 minutes** (depends on your GPU/CPU)
   - Training will evaluate every 1000 steps

## 📊 What to Look For

### Training Complete Indicators:
✅ `eval_results.txt` exists and has final metrics  
✅ `pytorch_model.bin` or `model.safetensors` exists (large file)  
✅ `hardness_predictor.pt` exists (HANS-specific)  
✅ `train_results.txt` shows all training steps completed  

### Evaluation Metrics to Check:
Open `eval_results.txt` and look for:
- **MRR@10** (Mean Reciprocal Rank at 10) - Higher is better
- **Recall@100** - Higher is better
- **Loss** - Lower is better

## 🔄 If Training Didn't Start

If the `trained_models/hans_dpr` folder doesn't exist, training may not have started. To start training:

```bash
cd C:\Users\utkar\OneDrive\Desktop\mharo
python simpletransformers/examples/retrieval/train_hans.py
```

## 📝 Quick Status Check Command

Run this anytime to check training status:
```bash
python simpletransformers/examples/retrieval/check_training_status.py
```

## 🎯 After Training Completes

1. **Compare with Baseline:**
   - Run `train_baseline.py` to get baseline results
   - Compare metrics in `eval_results.txt` from both runs

2. **Test the Model:**
   - Use the saved model for inference
   - Test on the test set (`msmarco-test.tsv`)

3. **View Results:**
   - Open `eval_results.txt` to see final metrics
   - Check `train_results.txt` for training progress

---

**Full Path:** `C:\Users\utkar\OneDrive\Desktop\mharo\trained_models\hans_dpr`

