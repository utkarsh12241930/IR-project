"""
Automatically run HANS evaluation when training completes.
This script checks if HANS training is complete and runs evaluation + presentation.
"""
import os
import sys
import subprocess
import time

script_dir = os.path.dirname(os.path.abspath(__file__))

def check_training_complete():
    """Check if HANS training has completed."""
    log_file = os.path.join(script_dir, "hans_medium_training_new.log")
    
    if not os.path.exists(log_file):
        return False, "Training log not found"
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Check for completion indicators
    if "HANS TRAINING COMPLETE" in content or "Training completed" in content or "Model saved to" in content:
        # Also check if model directory exists
        model_dir = os.path.join(script_dir, "trained_models", "hans_dpr_medium")
        if os.path.exists(model_dir):
            return True, "Training complete"
    
    return False, "Training still in progress"

def run_evaluation():
    """Run HANS evaluation."""
    print("="*80)
    print("Running HANS Evaluation...")
    print("="*80)
    
    eval_script = os.path.join(script_dir, "evaluate_hans_medium.py")
    result = subprocess.run([sys.executable, eval_script], cwd=script_dir)
    
    if result.returncode == 0:
        print("\nEvaluation completed successfully!")
        return True
    else:
        print("\nEvaluation failed!")
        return False

def create_presentation():
    """Create presentation from evaluation results."""
    print("="*80)
    print("Creating Presentation...")
    print("="*80)
    
    pres_script = os.path.join(script_dir, "create_hans_presentation_medium.py")
    result = subprocess.run([sys.executable, pres_script], cwd=script_dir)
    
    if result.returncode == 0:
        print("\nPresentation created successfully!")
        return True
    else:
        print("\nPresentation creation failed!")
        return False

if __name__ == "__main__":
    print("="*80)
    print("HANS Evaluation Automation Script")
    print("="*80)
    print("\nChecking training status...")
    
    is_complete, message = check_training_complete()
    
    if is_complete:
        print(f"Status: {message}")
        print("\nProceeding with evaluation...")
        
        # Run evaluation
        if run_evaluation():
            # Create presentation
            create_presentation()
            
            print("\n" + "="*80)
            print("ALL DONE! HANS evaluation and presentation complete.")
            print("="*80)
            print("\nGenerated files:")
            print("  - hans_medium_evaluation_results.json")
            print("  - hans_medium_evaluation_summary.txt")
            print("  - HANS_MEDIUM_RESULTS_PRESENTATION.md")
            print("  - hans_medium_results_latex.tex")
            print("  - hans_medium_results_table.csv")
        else:
            print("\nEvaluation failed. Please check the error messages above.")
    else:
        print(f"Status: {message}")
        print("\nTraining is still in progress.")
        print("Please wait for training to complete, then run this script again.")
        print("\nTo check training status manually, run:")
        print("  python -c \"import os; content = open('hans_medium_training_new.log', 'r', encoding='utf-8', errors='ignore').read(); print('COMPLETE' if 'HANS TRAINING COMPLETE' in content else 'RUNNING')\"")

