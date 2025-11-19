"""
HANS DPR training (with Hardness-Adaptive Negative Sampling)
Run this after baseline to compare results.
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

# Specifying the path to the training and validation data
# Option 1: Use MS MARCO (English) - works but limited multilingual showcase
# Option 2: Use mMARCO (Multilingual) - better for showcasing HANS
# Uncomment the mMARCO paths if you've downloaded the multilingual version

# Using mMARCO (Multilingual) paths:
train_data_path = os.path.join(script_dir, "data/mmarco/mmarco-train.tsv")
eval_data_path = os.path.join(script_dir, "data/mmarco/mmarco-validation.tsv")

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

# Configuring the model arguments - HANS VERSION
model_args = RetrievalArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.use_cached_eval_features = False
model_args.include_title = False if "mmarco" in train_data_path or "msmarco" in train_data_path else True
model_args.max_seq_length = 256
model_args.num_train_epochs = 2  # Reduced for quick testing
model_args.train_batch_size = 8   # Reduced for quick testing
model_args.use_hf_datasets = True
model_args.learning_rate = 1e-6
model_args.warmup_steps = 500
model_args.save_steps = 10000
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 1000
model_args.save_model_every_epoch = True
model_args.wandb_project = None  # Disable wandb (no API key needed)
# Enable hard negatives if using dataset with hard negatives
# Check if hard negatives file exists
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
model_args.n_gpu = 1
model_args.data_format = "msmarco"  # Same format works for mmarco
model_args.output_dir = "trained_models/hans_dpr"

# IMPORTANT: HANS is ON - this is the key difference!
model_args.use_hans = True
model_args.hans_min_negatives = 1
model_args.hans_max_negatives = 2
model_args.hans_contrastive_weight_min = 0.6
model_args.hans_contrastive_weight_max = 1.4
model_args.hans_margin_min = 0.05
model_args.hans_margin_max = 0.4
model_args.hans_temperature_min = 0.8
model_args.hans_temperature_max = 1.4
model_args.hans_hardness_loss_weight = 0.1

# Defining the model type and names
model_type = "dpr"
model_name = None
context_name = "facebook/dpr-ctx_encoder-single-nq-base"
question_name = "facebook/dpr-question_encoder-single-nq-base"

# Main execution
if __name__ == "__main__":
    print("=" * 50)
    print("TRAINING HANS DPR (HARDNESS-ADAPTIVE NEGATIVE SAMPLING)")
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

