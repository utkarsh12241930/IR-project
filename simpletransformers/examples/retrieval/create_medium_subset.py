"""
Create a medium-sized subset of mMARCO dataset for 3-hour training
Target: ~5000 training examples, ~500 validation, ~500 test
"""
import os
import pandas as pd
from datasets import load_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data", "mmarco")

# Define subset sizes for 3-hour training window
SUBSET_SIZE_TRAIN = 5000
SUBSET_SIZE_VAL = 500
SUBSET_SIZE_TEST = 500

def create_medium_subset():
    """Create medium-sized subsets of mMARCO dataset."""
    print("="*80)
    print("Creating Medium-Sized mMARCO Subset (for 3-hour training)")
    print("="*80)
    print(f"\nTarget sizes:")
    print(f"  Training: {SUBSET_SIZE_TRAIN} examples")
    print(f"  Validation: {SUBSET_SIZE_VAL} examples")
    print(f"  Test: {SUBSET_SIZE_TEST} examples")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if full dataset exists
    train_file = os.path.join(data_dir, "mmarco-train.tsv")
    val_file = os.path.join(data_dir, "mmarco-validation.tsv")
    
    if not os.path.exists(train_file):
        print("\n[INFO] Full dataset not found. Attempting to download...")
        try:
            # Try to load from HuggingFace
            print("Loading mMARCO dataset from HuggingFace...")
            dataset = load_dataset("unicamp-dl/mmarco", "english", split="train")
            
            # Convert to DataFrame
            df = pd.DataFrame(dataset)
            
            # Extract query_text and gold_passage
            if 'query' in df.columns and 'passage' in df.columns:
                train_df = pd.DataFrame({
                    'query_text': df['query'],
                    'gold_passage': df['passage']
                })
            elif 'query_text' in df.columns and 'gold_passage' in df.columns:
                train_df = df[['query_text', 'gold_passage']].copy()
            else:
                print("[ERROR] Could not find expected columns in dataset")
                return
            
            # Save full dataset
            train_df.to_csv(train_file, sep="\t", index=False)
            print(f"[OK] Saved full dataset to {train_file}")
            
        except Exception as e:
            print(f"[ERROR] Could not load mMARCO: {e}")
            print("[INFO] Falling back to MS MARCO (English only)...")
            try:
                dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")
                df = pd.DataFrame(dataset)
                
                if 'query' in df.columns and 'passages' in df.columns:
                    # Extract first passage as gold
                    train_df = pd.DataFrame({
                        'query_text': df['query'],
                        'gold_passage': df['passages'].apply(lambda x: x['passage_text'][0] if isinstance(x, dict) and 'passage_text' in x else str(x)[:500])
                    })
                else:
                    print("[ERROR] Could not parse MS MARCO dataset")
                    return
                
                train_df.to_csv(train_file, sep="\t", index=False)
                print(f"[OK] Saved MS MARCO dataset to {train_file}")
            except Exception as e2:
                print(f"[ERROR] Could not load MS MARCO either: {e2}")
                return
    
    # Load full dataset
    print(f"\n[INFO] Loading dataset from {train_file}...")
    full_df = pd.read_csv(train_file, sep="\t")
    print(f"[OK] Loaded {len(full_df)} examples")
    
    # Remove duplicates
    full_df = full_df.drop_duplicates(subset=['query_text', 'gold_passage'])
    print(f"[OK] After removing duplicates: {len(full_df)} examples")
    
    # Shuffle
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create subsets
    total_needed = SUBSET_SIZE_TRAIN + SUBSET_SIZE_VAL + SUBSET_SIZE_TEST
    
    # Adjust sizes if dataset is smaller than needed
    actual_train_size = SUBSET_SIZE_TRAIN
    actual_val_size = SUBSET_SIZE_VAL
    actual_test_size = SUBSET_SIZE_TEST
    
    if len(full_df) < total_needed:
        print(f"[WARNING] Dataset has only {len(full_df)} examples, but need {total_needed}")
        print(f"[INFO] Using all available examples and adjusting proportions...")
        actual_train_size = min(SUBSET_SIZE_TRAIN, int(len(full_df) * 0.8))
        actual_val_size = min(SUBSET_SIZE_VAL, int(len(full_df) * 0.1))
        actual_test_size = len(full_df) - actual_train_size - actual_val_size
    
    # Split dataset
    train_df = full_df[:actual_train_size].copy()
    val_df = full_df[actual_train_size:actual_train_size + actual_val_size].copy()
    test_df = full_df[actual_train_size + actual_val_size:actual_train_size + actual_val_size + actual_test_size].copy()
    
    # Save subsets
    train_output = os.path.join(data_dir, "mmarco-train-medium.tsv")
    val_output = os.path.join(data_dir, "mmarco-validation-medium.tsv")
    test_output = os.path.join(data_dir, "mmarco-test-medium.tsv")
    
    train_df.to_csv(train_output, sep="\t", index=False)
    val_df.to_csv(val_output, sep="\t", index=False)
    test_df.to_csv(test_output, sep="\t", index=False)
    
    print("\n" + "="*80)
    print("SUBSET CREATION COMPLETE")
    print("="*80)
    print(f"\nTraining set: {len(train_df)} examples -> {train_output}")
    print(f"Validation set: {len(val_df)} examples -> {val_output}")
    print(f"Test set: {len(test_df)} examples -> {test_output}")
    print("\n[OK] Ready for training!")

if __name__ == "__main__":
    create_medium_subset()

