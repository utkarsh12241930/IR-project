"""
Create a smaller subset of mMARCO dataset for 6-hour HANS training
Target: ~2500 training examples, ~250 validation, ~250 test
"""
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data", "mmarco")

# Define subset sizes for 6-hour training window
SUBSET_SIZE_TRAIN = 2000  # Reduced to fit in 6 hours (~5.7 hours estimated)
SUBSET_SIZE_VAL = 200
SUBSET_SIZE_TEST = 200

def create_6hour_subset():
    """Create smaller subsets of mMARCO dataset for 6-hour training."""
    print("="*80)
    print("Creating 6-Hour mMARCO Subset for HANS Training")
    print("="*80)
    print(f"\nTarget sizes:")
    print(f"  Training: {SUBSET_SIZE_TRAIN} examples")
    print(f"  Validation: {SUBSET_SIZE_VAL} examples")
    print(f"  Test: {SUBSET_SIZE_TEST} examples")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if medium dataset exists (we can use it as source)
    source_file = os.path.join(data_dir, "mmarco-train-medium.tsv")
    if not os.path.exists(source_file):
        # Fall back to full dataset
        source_file = os.path.join(data_dir, "mmarco-train.tsv")
        if not os.path.exists(source_file):
            print(f"\n[ERROR] Source dataset not found at {source_file}")
            print("[INFO] Please run create_medium_subset.py first to create the base dataset.")
            return
    
    # Load source dataset
    print(f"\n[INFO] Loading dataset from {source_file}...")
    full_df = pd.read_csv(source_file, sep="\t")
    print(f"[OK] Loaded {len(full_df)} examples")
    
    # Remove duplicates
    full_df = full_df.drop_duplicates(subset=['query_text', 'gold_passage'])
    print(f"[OK] After removing duplicates: {len(full_df)} examples")
    
    # Shuffle
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create subsets
    total_needed = SUBSET_SIZE_TRAIN + SUBSET_SIZE_VAL + SUBSET_SIZE_TEST
    
    # Adjust sizes if dataset is smaller than needed
    actual_train_size = min(SUBSET_SIZE_TRAIN, len(full_df))
    actual_val_size = min(SUBSET_SIZE_VAL, len(full_df) - actual_train_size)
    actual_test_size = min(SUBSET_SIZE_TEST, len(full_df) - actual_train_size - actual_val_size)
    
    if len(full_df) < total_needed:
        print(f"[WARNING] Dataset has only {len(full_df)} examples, but need {total_needed}")
        print(f"[INFO] Using all available examples and adjusting proportions...")
        # Use 80/10/10 split if not enough data
        actual_train_size = int(len(full_df) * 0.8)
        actual_val_size = int(len(full_df) * 0.1)
        actual_test_size = len(full_df) - actual_train_size - actual_val_size
    
    # Split dataset
    train_df = full_df[:actual_train_size].copy()
    val_df = full_df[actual_train_size:actual_train_size + actual_val_size].copy()
    test_df = full_df[actual_train_size + actual_val_size:actual_train_size + actual_val_size + actual_test_size].copy()
    
    # Save subsets with "6hour" suffix
    train_output = os.path.join(data_dir, "mmarco-train-6hour.tsv")
    val_output = os.path.join(data_dir, "mmarco-validation-6hour.tsv")
    test_output = os.path.join(data_dir, "mmarco-test-6hour.tsv")
    
    train_df.to_csv(train_output, sep="\t", index=False)
    val_df.to_csv(val_output, sep="\t", index=False)
    test_df.to_csv(test_output, sep="\t", index=False)
    
    print("\n" + "="*80)
    print("6-HOUR SUBSET CREATION COMPLETE")
    print("="*80)
    print(f"\nTraining set: {len(train_df)} examples -> {train_output}")
    print(f"Validation set: {len(val_df)} examples -> {val_output}")
    print(f"Test set: {len(test_df)} examples -> {test_output}")
    print("\n[OK] Ready for 6-hour HANS training!")
    print("\nEstimated training time:")
    print(f"  - With batch size 16: ~{len(train_df) // 16} steps")
    print(f"  - At ~163 seconds/step: ~{((len(train_df) // 16) * 163) / 3600:.1f} hours")

if __name__ == "__main__":
    create_6hour_subset()

