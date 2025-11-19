"""
Add Hard Negatives to MS MARCO Dataset
This script adds hard negative passages to the MS MARCO dataset for HANS training.
Hard negatives are randomly sampled from other passages in the dataset.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("Adding Hard Negatives to MS MARCO Dataset")
print("=" * 60)

# Get script directory
script_dir = Path(__file__).parent
data_dir = script_dir / "data" / "mmarco"  # Changed to mmarco

# Load datasets
print("\n[Step 1] Loading datasets...")
train_path = data_dir / "mmarco-train.tsv"
val_path = data_dir / "mmarco-validation.tsv"
test_path = data_dir / "mmarco-test.tsv"

train_df = pd.read_csv(train_path, sep="\t")
val_df = pd.read_csv(val_path, sep="\t")
test_df = pd.read_csv(test_path, sep="\t")

print(f"  Train: {len(train_df)} examples")
print(f"  Validation: {len(val_df)} examples")
print(f"  Test: {len(test_df)} examples")

# Collect all passages for negative sampling
print("\n[Step 2] Collecting all passages for negative sampling...")
all_passages = pd.concat([
    train_df["gold_passage"],
    val_df["gold_passage"],
    test_df["gold_passage"]
]).unique().tolist()
print(f"  Total unique passages: {len(all_passages)}")

def add_hard_negatives(df, num_negatives=2, seed=42):
    """Add hard negative passages to a dataframe - OPTIMIZED VERSION"""
    np.random.seed(seed)
    
    df = df.copy()
    
    # Add hard negative columns
    for i in range(num_negatives):
        df[f"hard_negative_{i}"] = None
    
    # Pre-compute candidate negatives for each row (much faster)
    print(f"  Adding {num_negatives} hard negatives per example...")
    print(f"  Processing {len(df)} examples...")
    
    # Vectorized approach - much faster
    all_passages_list = list(all_passages)
    hard_negatives_list = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1}/{len(df)} examples...")
        
        # Get all passages except the current gold passage
        candidate_negatives = [p for p in all_passages_list if p != row["gold_passage"]]
        
        # Randomly sample hard negatives
        if len(candidate_negatives) >= num_negatives:
            sampled_negatives = np.random.choice(
                candidate_negatives, 
                size=num_negatives, 
                replace=False
            ).tolist()
        else:
            # If not enough candidates, sample with replacement
            sampled_negatives = np.random.choice(
                candidate_negatives,
                size=num_negatives,
                replace=True
            ).tolist()
        
        hard_negatives_list.append(sampled_negatives)
    
    # Assign all at once (faster than row-by-row)
    for i in range(num_negatives):
        df[f"hard_negative_{i}"] = [neg[i] for neg in hard_negatives_list]
    
    return df

# Add hard negatives to each dataset
print("\n[Step 3] Adding hard negatives to training set...")
train_df_with_negatives = add_hard_negatives(train_df, num_negatives=2, seed=42)

print("\n[Step 4] Adding hard negatives to validation set...")
val_df_with_negatives = add_hard_negatives(val_df, num_negatives=2, seed=42)

print("\n[Step 5] Adding hard negatives to test set...")
test_df_with_negatives = add_hard_negatives(test_df, num_negatives=2, seed=42)

# Save updated datasets
print("\n[Step 6] Saving updated datasets...")

# Save with hard negatives
train_df_with_negatives.to_csv(data_dir / "mmarco-train-with-negatives.tsv", sep="\t", index=False)
val_df_with_negatives.to_csv(data_dir / "mmarco-validation-with-negatives.tsv", sep="\t", index=False)
test_df_with_negatives.to_csv(data_dir / "mmarco-test-with-negatives.tsv", sep="\t", index=False)

print(f"  [OK] Saved: {data_dir / 'mmarco-train-with-negatives.tsv'}")
print(f"  [OK] Saved: {data_dir / 'mmarco-validation-with-negatives.tsv'}")
print(f"  [OK] Saved: {data_dir / 'mmarco-test-with-negatives.tsv'}")

# Show sample
print("\n[Step 7] Sample data with hard negatives:")
print("\nTraining set sample:")
sample = train_df_with_negatives.head(1)
print(f"  Query: {sample['query_text'].iloc[0][:80]}...")
print(f"  Gold passage: {sample['gold_passage'].iloc[0][:80]}...")
print(f"  Hard negative 0: {sample['hard_negative_0'].iloc[0][:80]}...")
print(f"  Hard negative 1: {sample['hard_negative_1'].iloc[0][:80]}...")

print("\n" + "=" * 60)
print("SUCCESS! Hard negatives added to all datasets!")
print("=" * 60)
print("\nNew files created:")
print("  - mmarco-train-with-negatives.tsv")
print("  - mmarco-validation-with-negatives.tsv")
print("  - mmarco-test-with-negatives.tsv")
print("\nYou can now use these files for HANS training!")
print("=" * 60)

