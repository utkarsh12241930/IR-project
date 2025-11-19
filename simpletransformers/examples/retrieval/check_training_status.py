"""
Check HANS Training Status
This script checks if training is complete and shows where the model is saved.
"""
import os
from pathlib import Path
import time

print("=" * 60)
print("HANS Training Status Checker")
print("=" * 60)

# Model save location
output_dir = Path("trained_models/hans_dpr")
output_dir_abs = output_dir.resolve()

print(f"\n[Model Save Location]")
print(f"  Relative path: {output_dir}")
print(f"  Absolute path: {output_dir_abs}")
print(f"\n[How to Access]")
print(f"  1. Open File Explorer")
print(f"  2. Navigate to: {output_dir_abs}")
print(f"  3. Or copy this path: {output_dir_abs}")

# Check if directory exists
if output_dir.exists():
    print(f"\n[Status] Model directory EXISTS")
    print(f"  Location: {output_dir_abs}")
    
    # List files
    files = list(output_dir.rglob("*"))
    if files:
        print(f"\n[Files Found] {len(files)} files/directories")
        print("\n  Key files to look for:")
        print("    - model_config.json (model configuration)")
        print("    - pytorch_model.bin or model.safetensors (model weights)")
        print("    - training_args.bin (training arguments)")
        print("    - eval_results.txt (evaluation results)")
        print("    - train_results.txt (training results)")
        print("    - hardness_predictor.pt (HANS hardness predictor)")
        
        # Check for specific important files
        important_files = {
            "model_config.json": "Model configuration",
            "pytorch_model.bin": "Model weights (PyTorch format)",
            "model.safetensors": "Model weights (SafeTensors format)",
            "training_args.bin": "Training arguments",
            "eval_results.txt": "Evaluation results",
            "train_results.txt": "Training results",
            "hardness_predictor.pt": "HANS hardness predictor"
        }
        
        print("\n  [Found Files]:")
        for filename, description in important_files.items():
            found = list(output_dir.rglob(filename))
            if found:
                for f in found:
                    size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
                    mod_time = time.ctime(f.stat().st_mtime)
                    print(f"    [OK] {f.name} ({size_mb:.2f} MB) - {description}")
                    print(f"         Modified: {mod_time}")
            else:
                print(f"    [ ] {filename} - {description} (not found yet)")
        
        # Check for checkpoints
        checkpoints = list(output_dir.rglob("checkpoint-*"))
        if checkpoints:
            print(f"\n  [Checkpoints Found]: {len(checkpoints)} checkpoint directories")
            for cp in sorted(checkpoints)[-3:]:  # Show last 3
                print(f"    - {cp.name}")
    else:
        print("\n[Status] Directory exists but is empty (training may have just started)")
else:
    print(f"\n[Status] Model directory does NOT exist yet")
    print("  Training may still be running or hasn't started yet.")
    print("  The directory will be created when training begins.")

# Check for training logs
log_files = [
    Path("trained_models/hans_dpr/train_results.txt"),
    Path("trained_models/hans_dpr/eval_results.txt"),
]

print(f"\n[Training Logs]")
for log_file in log_files:
    if log_file.exists():
        print(f"  [OK] {log_file.name} exists")
        print(f"       Size: {log_file.stat().st_size / 1024:.2f} KB")
        print(f"       Modified: {time.ctime(log_file.stat().st_mtime)}")
        
        # Show last few lines if it's a text file
        if log_file.suffix == ".txt":
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"       Last 3 lines:")
                        for line in lines[-3:]:
                            print(f"         {line.strip()}")
            except:
                pass
    else:
        print(f"  [ ] {log_file.name} (not created yet)")

print("\n" + "=" * 60)
print("To check training progress:")
print("  1. Run this script again: python check_training_status.py")
print("  2. Check the output directory for new files")
print("  3. Look for 'eval_results.txt' for evaluation metrics")
print("=" * 60)

