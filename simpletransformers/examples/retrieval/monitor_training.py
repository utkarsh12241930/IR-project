"""
Monitor HANS Training Progress
Run this script to check if training is still running and see progress.
"""
import os
import time
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("HANS Training Monitor")
print("=" * 60)
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check if training directory exists
output_dir = Path("trained_models/hans_dpr")
output_dir_abs = output_dir.resolve()

print(f"[Model Location]")
print(f"  {output_dir_abs}")
print()

# Check for training files
if output_dir.exists():
    print("[Status] Training directory EXISTS - Training has started!")
    print()
    
    # Check for key files
    files_found = {}
    important_files = {
        "model_config.json": "Model config",
        "pytorch_model.bin": "Model weights (PyTorch)",
        "model.safetensors": "Model weights (SafeTensors)",
        "training_args.bin": "Training args",
        "hardness_predictor.pt": "HANS predictor",
        "train_results.txt": "Training results",
        "eval_results.txt": "Evaluation results",
        "trainer_state.json": "Training state"
    }
    
    print("[Files Status]")
    for filename, description in important_files.items():
        found = list(output_dir.rglob(filename))
        if found:
            f = found[0]
            size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
            mod_time = datetime.fromtimestamp(f.stat().st_mtime)
            age = datetime.now() - mod_time
            print(f"  [OK] {filename:30s} - {size_mb:6.2f} MB - Modified {age.seconds//60} min ago")
            files_found[filename] = f
        else:
            print(f"  [ ] {filename:30s} - Not created yet")
    
    # Check for checkpoints
    checkpoints = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
    if checkpoints:
        print(f"\n[Checkpoints] {len(checkpoints)} found")
        for cp in checkpoints[-3:]:  # Show last 3
            mod_time = datetime.fromtimestamp(cp.stat().st_mtime)
            age = datetime.now() - mod_time
            print(f"  - {cp.name} (modified {age.seconds//60} min ago)")
    
    # Check training state
    trainer_state = output_dir / "trainer_state.json"
    if trainer_state.exists():
        print("\n[Training Progress]")
        try:
            import json
            with open(trainer_state, 'r') as f:
                state = json.load(f)
                if 'epoch' in state:
                    print(f"  Current epoch: {state.get('epoch', 'N/A')}")
                if 'global_step' in state:
                    print(f"  Global step: {state.get('global_step', 'N/A')}")
                if 'max_steps' in state:
                    max_steps = state.get('max_steps')
                    current_step = state.get('global_step', 0)
                    if max_steps:
                        progress = (current_step / max_steps) * 100
                        print(f"  Progress: {progress:.1f}% ({current_step}/{max_steps} steps)")
        except Exception as e:
            print(f"  Could not read training state: {e}")
    
    # Check eval results
    eval_file = output_dir / "eval_results.txt"
    if eval_file.exists():
        print("\n[Latest Evaluation Results]")
        try:
            with open(eval_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Show last evaluation
                    for line in lines[-10:]:
                        if line.strip():
                            print(f"  {line.strip()}")
        except Exception as e:
            print(f"  Could not read eval results: {e}")
    
    # Check if training might be complete
    has_model = (output_dir / "pytorch_model.bin").exists() or (output_dir / "model.safetensors").exists()
    has_eval = (output_dir / "eval_results.txt").exists()
    
    if has_model and has_eval:
        print("\n" + "=" * 60)
        print("[!] Training appears to be COMPLETE!")
        print("=" * 60)
        print("Check eval_results.txt for final metrics.")
    else:
        print("\n[Status] Training is IN PROGRESS")
        print("  Run this script again in a few minutes to check progress.")
        
else:
    print("[Status] Training directory does NOT exist yet")
    print("  Training may still be initializing...")
    print("  Wait a minute and run this script again.")

print("\n" + "=" * 60)
print("To check again, run: python simpletransformers/examples/retrieval/monitor_training.py")
print("=" * 60)

