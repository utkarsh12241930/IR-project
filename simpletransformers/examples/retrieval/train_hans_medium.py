"""
HANS DPR training on medium-sized dataset (for 3-hour training window)
"""
import os
import sys

# Add simpletransformers to path
script_dir = os.path.dirname(os.path.abspath(__file__))
simpletransformers_dir = os.path.join(script_dir, "..", "..")
sys.path.insert(0, os.path.abspath(simpletransformers_dir))

import logging
import pandas as pd
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

# Configuring logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Specifying the path to the training data (medium subset)
train_data_path = os.path.join(script_dir, "data/mmarco/mmarco-train-medium.tsv")
eval_data_path = os.path.join(script_dir, "data/mmarco/mmarco-validation-medium.tsv")

# Loading the training data
if train_data_path.endswith(".tsv"):
    train_data = pd.read_csv(train_data_path, sep="\t")
else:
    train_data = train_data_path

# Loading the validation data
if eval_data_path.endswith(".tsv"):
    eval_data = pd.read_csv(eval_data_path, sep="\t")
else:
    eval_data = eval_data_path

print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(eval_data)}")

# Configuring the model arguments - HANS ENABLED
model_args = RetrievalArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.use_cached_eval_features = False
model_args.include_title = False if "mmarco" in train_data_path else True
model_args.max_seq_length = 128  # Reduced sequence length for faster training
model_args.num_train_epochs = 1  # Single epoch for 3-hour window
model_args.train_batch_size = 16   # Smaller batch for larger dataset
model_args.eval_batch_size = 16
model_args.use_hf_datasets = True
model_args.learning_rate = 2e-6
model_args.warmup_steps = 100  # More warmup for larger dataset
model_args.save_steps = 5000
model_args.evaluate_during_training = False  # Disable to save time
model_args.save_model_every_epoch = True
model_args.wandb_project = None  # Disable wandb
model_args.n_gpu = 1
model_args.data_format = "msmarco"
model_args.output_dir = "trained_models/hans_dpr_medium"

# Check for hard negatives and use them if available
train_with_negatives = train_data_path.replace(".tsv", "-with-negatives.tsv")
if os.path.exists(train_with_negatives):
    model_args.hard_negatives = True
    model_args.n_hard_negatives = 2
    train_data_path = train_with_negatives
    eval_data_path = eval_data_path.replace(".tsv", "-with-negatives.tsv")
    print(f"[INFO] Using dataset with hard negatives: {train_data_path}")
else:
    model_args.hard_negatives = False
    model_args.n_hard_negatives = 0
    print(f"[INFO] Using dataset without hard negatives (will use in-batch negatives)")

# IMPORTANT: HANS is ON
model_args.use_hans = True

# HANS-specific parameters
model_args.hans_min_negatives = 1
model_args.hans_max_negatives = 3
model_args.hans_contrastive_weight_min = 0.5
model_args.hans_contrastive_weight_max = 1.5
model_args.hans_margin_min = 0.1
model_args.hans_margin_max = 0.5
model_args.hans_temperature_min = 0.5
model_args.hans_temperature_max = 1.0
model_args.hans_hardness_loss_weight = 0.1

# Defining the model type and names
model_type = "dpr"
model_name = None
context_name = "facebook/dpr-ctx_encoder-single-nq-base"
question_name = "facebook/dpr-question_encoder-single-nq-base"

# Main execution
if __name__ == "__main__":
    print("=" * 50)
    print("TRAINING HANS DPR (MEDIUM DATASET)")
    print("=" * 50)
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(eval_data)}")
    print(f"HANS: ENABLED")
    print(f"Estimated training time: ~2-3 hours (CPU)")
    print("=" * 50)
    
    # Creating the model
    model = RetrievalModel(
        model_type,
        model_name,
        context_name,
        question_name,
        args=model_args,
        use_cuda=False,  # Set to False for CPU-only training
    )

    # Training the model
    model.train_model(
        train_data,
        eval_data=eval_data,
        eval_set="dev",
    )
    
    print("=" * 50)
    print("HANS TRAINING COMPLETE!")
    print(f"Model saved to: {model_args.output_dir}")
    print("=" * 50)

