"""
Download and Prepare Multilingual MS MARCO (mMARCO) Dataset
This dataset is better suited for showcasing HANS as it contains multiple languages
with varying linguistic hardness (code-mixing, ambiguity, morphology, etc.)
"""
import os
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("Multilingual MS MARCO (mMARCO) Dataset Download")
print("=" * 60)
print("\nThis dataset contains multiple languages and is ideal for showcasing HANS!")
print("=" * 60)

# Create output directory (will be created in script directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data", "mmarco")
os.makedirs(data_dir, exist_ok=True)

# Step 1: Load mMARCO dataset - TRUE MULTILINGUAL VERSION
print("\n[Step 1] Downloading mMARCO (Multilingual MS MARCO) dataset from HuggingFace...")
print("This may take several minutes...")
print("mMARCO contains 14 languages: English, Chinese, French, German, Indonesian,")
print("Italian, Portuguese, Russian, Spanish, Arabic, Dutch, Hindi, Japanese, Vietnamese")

dataset = None
dataset_name = None

# Try to load true mMARCO - it's available as separate language splits
try:
    # mMARCO is available per language, let's combine multiple languages
    print("\nLoading multilingual data from multiple language splits...")
    
    # Load a few languages to create a multilingual dataset
    languages_to_load = ["english", "french", "spanish", "german"]  # Start with 4 languages
    
    all_examples = []
    for lang in languages_to_load:
        try:
            print(f"  Loading {lang}...")
            lang_dataset = load_dataset("unicamp-dl/mmarco", lang, split="train")
            # Take a subset from each language to balance
            lang_subset = lang_dataset.select(range(min(5000, len(lang_dataset))))
            all_examples.append(lang_subset)
            print(f"    [OK] Loaded {len(lang_subset)} {lang} examples")
        except Exception as e:
            print(f"    [WARNING] Could not load {lang}: {e}")
            continue
    
    if all_examples:
        # Combine all languages
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(all_examples)
        dataset_name = f"mMARCO Multilingual ({len(all_examples)} languages)"
        print(f"[OK] Combined dataset: {len(dataset)} examples from {len(all_examples)} languages")
    else:
        raise Exception("Could not load any language splits")
        
except Exception as e:
    print(f"\n[WARNING] Could not load true mMARCO: {e}")
    print("Falling back to MS MARCO (English) - note: this is NOT multilingual")
    print("For true multilingual, you may need to download mMARCO manually")
    try:
        dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")
        dataset_name = "MS MARCO v1.1 (English only - NOT multilingual)"
        print(f"[OK] Loaded {dataset_name} as fallback")
    except Exception as e2:
        print(f"Error: {e2}")
        raise

# Step 2: Convert to DataFrame
print("\n[Step 2] Converting to DataFrame...")
df = dataset.to_pandas()
print(f"[OK] Converted! Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# Step 3: Extract query and passage
print("\n[Step 3] Extracting query and passage data...")

if "query" in df.columns:
    df["query_text"] = df["query"]
elif "query_text" not in df.columns:
    print("Warning: 'query' or 'query_text' column not found!")
    print(f"Available columns: {df.columns.tolist()}")

# Handle passages
if "passages" in df.columns:
    print("  Extracting gold passages...")
    def extract_gold_passage(row):
        try:
            if pd.isna(row):
                return ""
            if isinstance(row, dict):
                if "passage_text" in row and "is_selected" in row:
                    passages = row["passage_text"]
                    is_selected = row["is_selected"]
                    for i, selected in enumerate(is_selected):
                        if selected == 1 and i < len(passages):
                            return passages[i]
            elif isinstance(row, list) and len(row) > 0:
                if isinstance(row[0], dict):
                    for passage_dict in row:
                        if passage_dict.get("is_selected", 0) == 1:
                            return passage_dict.get("passage_text", "")
            return ""
        except:
            return ""
    
    df["gold_passage"] = df["passages"].apply(extract_gold_passage)
    print(f"[OK] Extracted gold passages")
    print(f"  Non-empty passages: {df['gold_passage'].notna().sum()} / {len(df)}")

# Step 4: Clean data
print("\n[Step 4] Cleaning data...")
initial_size = len(df)
df = df[df["query_text"].notna() & (df["query_text"] != "")]
if "gold_passage" in df.columns:
    df = df[df["gold_passage"].notna() & (df["gold_passage"] != "")]
print(f"[OK] Cleaned! Removed {initial_size - len(df)} rows")
print(f"  Remaining: {len(df)} examples")

# Step 5: Create 20K subset (for manageable training)
print("\n[Step 5] Creating 20K subset...")
subset_size = min(20000, len(df))
df_subset = df.head(subset_size).copy()
print(f"[OK] Created subset! Size: {len(df_subset):,} examples")

# Step 6: Split into train/val/test
print("\n[Step 6] Splitting dataset (80/10/10)...")
train_df, temp_df = train_test_split(df_subset, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

print(f"[OK] Split complete!")
print(f"  Train:      {len(train_df):,} examples")
print(f"  Validation: {len(val_df):,} examples")
print(f"  Test:       {len(test_df):,} examples")

# Step 7: Save datasets
print("\n[Step 7] Saving datasets...")
output_columns = ["query_text", "gold_passage"]

# Get script directory for proper path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data", "mmarco")
os.makedirs(data_dir, exist_ok=True)

train_df[output_columns].to_csv(os.path.join(data_dir, "mmarco-train.tsv"), sep="\t", index=False)
val_df[output_columns].to_csv(os.path.join(data_dir, "mmarco-validation.tsv"), sep="\t", index=False)
test_df[output_columns].to_csv(os.path.join(data_dir, "mmarco-test.tsv"), sep="\t", index=False)

print(f"[OK] Saved all datasets to data/mmarco/")

print("\n" + "=" * 60)
print("SUCCESS! Dataset ready for HANS training!")
print("=" * 60)
print("\nFiles created:")
print("  - data/mmarco/mmarco-train.tsv")
print("  - data/mmarco/mmarco-validation.tsv")
print("  - data/mmarco/mmarco-test.tsv")
print("\nNext step: Run add_hard_negatives.py to add hard negatives")
print("=" * 60)
