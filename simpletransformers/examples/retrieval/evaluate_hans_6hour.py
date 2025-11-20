"""
Evaluate HANS DPR Model (6-Hour Dataset) with F2 Score
This script evaluates the HANS model trained on 6-hour dataset.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add simpletransformers to path
script_dir = os.path.dirname(os.path.abspath(__file__))
simpletransformers_dir = os.path.join(script_dir, "..", "..")
sys.path.insert(0, os.path.abspath(simpletransformers_dir))

import logging
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs
from simpletransformers.retrieval.retrieval_utils import get_output_embeddings

# Configure logging
logging.basicConfig(level=logging.WARNING)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

def calculate_f2_score(precision, recall):
    """Calculate F2 score: F2 = 5 * (precision * recall) / (4 * precision + recall)"""
    if precision == 0 and recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall + 1e-10)

def calculate_mrr_at_k(ranks, k):
    """Calculate MRR@k given a list of ranks."""
    reciprocal_ranks = []
    for rank in ranks:
        if rank <= k and rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def calculate_top_k_accuracy(ranks, k):
    """Calculate top-k accuracy given a list of ranks."""
    hits = sum(1 for rank in ranks if rank <= k and rank > 0)
    return hits / len(ranks) if ranks else 0.0

def calculate_precision_at_k(ranks, k):
    """Calculate precision@k: relevant retrieved / k"""
    hits = sum(1 for rank in ranks if rank <= k and rank > 0)
    return hits / (len(ranks) * k) if ranks else 0.0

def calculate_recall_at_k(ranks, k):
    """Calculate recall@k: relevant retrieved / total relevant (1 per query)"""
    hits = sum(1 for rank in ranks if rank <= k and rank > 0)
    return hits / len(ranks) if ranks else 0.0

def get_embeddings_batch(model, texts, is_query=False, batch_size=32):
    """Get embeddings for a batch of texts."""
    embeddings_list = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {'queries' if is_query else 'passages'}"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        if is_query:
            tokenizer = model.query_tokenizer
            encoder = model.query_encoder
        else:
            tokenizer = model.context_tokenizer
            encoder = model.context_encoder
        
        # Tokenize batch
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=model.args.max_seq_length,
            return_tensors="pt"
        )
        
        # Move to device
        device = model.device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get embeddings
        encoder.eval()
        with torch.no_grad():
            outputs = encoder(**encoded)
            embeddings = get_output_embeddings(
                outputs,
                concatenate_embeddings=model.args.larger_representations and model.args.model_type == "custom",
                n_cls_tokens=(1 + model.args.extra_cls_token_count),
                use_pooler_output=model.args.use_pooler_output,
                args=model.args,
                query_embeddings=is_query,
            )
            embeddings = embeddings.cpu().numpy()
            embeddings_list.append(embeddings)
    
    return np.vstack(embeddings_list)

def evaluate_hans_6hour(model_path, eval_data_path, top_k_values=[1, 5, 10, 20, 50, 100]):
    """Evaluate HANS model with F2 score and other metrics."""
    print("="*80)
    print("HANS DPR MODEL EVALUATION (6-Hour Dataset)")
    print("="*80)
    
    # Load evaluation data
    eval_data = pd.read_csv(eval_data_path, sep="\t")
    print(f"\nLoaded {len(eval_data)} evaluation examples")
    
    # Configure model args
    model_args = RetrievalArgs()
    model_args.use_hf_datasets = True
    model_args.max_seq_length = 128
    model_args.include_title = False if "mmarco" in eval_data_path else True
    model_args.data_format = "msmarco"
    model_args.use_cuda = False
    model_args.use_hans = True  # HANS model
    
    print("\nLoading model...")
    model = RetrievalModel(
        model_type="dpr",
        model_name=model_path,
        context_encoder_name=None,
        query_encoder_name=None,
        args=model_args,
        use_cuda=False,
    )
    print("Model loaded successfully!")
    
    # Get all unique passages from eval data
    all_passages = eval_data['gold_passage'].unique().tolist()
    print(f"\nTotal unique passages in evaluation set: {len(all_passages)}")
    
    # Embed all passages
    print("\nEmbedding passages...")
    passage_embeddings = get_embeddings_batch(model, all_passages, is_query=False, batch_size=16)
    passage_embeddings = torch.tensor(passage_embeddings)
    
    # Normalize embeddings for cosine similarity
    passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=1)
    
    # Create passage to index mapping
    passage_to_idx = {passage: idx for idx, passage in enumerate(all_passages)}
    
    # Get all queries
    queries = eval_data['query_text'].tolist()
    gold_passages = eval_data['gold_passage'].tolist()
    
    # Embed all queries
    print("\nEmbedding queries...")
    query_embeddings = get_embeddings_batch(model, queries, is_query=True, batch_size=16)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    
    # Evaluate each query
    print("\nEvaluating queries...")
    all_ranks = []
    
    for i in tqdm(range(len(queries)), desc="Computing similarities"):
        query_emb = query_embeddings[i:i+1]  # Keep batch dimension
        
        # Calculate cosine similarities
        similarities = torch.mm(query_emb, passage_embeddings.t())
        similarities = similarities.squeeze(0).numpy()
        
        # Get rank of gold passage
        gold_passage = gold_passages[i]
        gold_idx = passage_to_idx[gold_passage]
        sorted_indices = np.argsort(similarities)[::-1]  # Descending order
        rank = np.where(sorted_indices == gold_idx)[0][0] + 1  # 1-indexed
        all_ranks.append(rank)
    
    print(f"\nEvaluation complete!")
    print("="*80)
    print("EVALUATION RESULTS - HANS DPR (6-HOUR DATASET)")
    print("="*80)
    print("\nStandard DPR Evaluation Metrics (as used in research papers):")
    print("-" * 80)
    
    # Calculate metrics
    results = {}
    
    # MRR@k - Most important metric in DPR papers
    print("\nMRR@k (Mean Reciprocal Rank):")
    print(f"{'k':<10} {'MRR@k':<15}")
    print("-" * 25)
    for k in top_k_values:
        mrr = calculate_mrr_at_k(all_ranks, k)
        results[f'mrr_at_{k}'] = float(mrr)
        print(f"{k:<10} {mrr:.6f}")
    
    # Recall@k - Standard metric in DPR papers
    print("\nRecall@k:")
    print(f"{'k':<10} {'Recall@k':<15}")
    print("-" * 25)
    for k in top_k_values:
        recall = calculate_recall_at_k(all_ranks, k)
        results[f'recall_at_{k}'] = float(recall)
        print(f"{k:<10} {recall:.6f}")
    
    # Top-k Accuracy - Also commonly reported
    print("\nTop-k Accuracy:")
    print(f"{'k':<10} {'Top-k Acc':<15}")
    print("-" * 25)
    for k in top_k_values:
        acc = calculate_top_k_accuracy(all_ranks, k)
        results[f'top_{k}_accuracy'] = float(acc)
        print(f"{k:<10} {acc:.6f}")
    
    # F2 Score (as requested by user)
    print("\nF2 Score (as requested):")
    print(f"{'k':<10} {'F2@k':<15}")
    print("-" * 25)
    for k in top_k_values:
        precision = calculate_precision_at_k(all_ranks, k)
        recall = results[f'recall_at_{k}']
        f2 = calculate_f2_score(precision, recall)
        results[f'f2_at_{k}'] = float(f2)
        results[f'precision_at_{k}'] = float(precision)
        print(f"{k:<10} {f2:.6f}")
    
    # Additional stats
    results['mean_rank'] = float(np.mean(all_ranks))
    results['median_rank'] = float(np.median(all_ranks))
    results['min_rank'] = int(np.min(all_ranks))
    results['max_rank'] = int(np.max(all_ranks))
    
    print("\nRank Statistics:")
    print("-" * 80)
    print(f"Mean Rank: {results['mean_rank']:.2f}")
    print(f"Median Rank: {results['median_rank']:.2f}")
    print(f"Min Rank: {results['min_rank']}")
    print(f"Max Rank: {results['max_rank']}")
    
    # Save results
    output_file = os.path.join(script_dir, "hans_6hour_evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Save summary in research paper format
    summary_file = os.path.join(script_dir, "hans_6hour_evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HANS DPR MODEL EVALUATION RESULTS (6-HOUR DATASET)\n")
        f.write("(Standard DPR Evaluation Metrics as used in research papers)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Evaluation Data: {eval_data_path}\n")
        f.write(f"Number of Examples: {len(eval_data)}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("="*80 + "\n\n")
        
        # Format similar to research paper tables
        f.write("MRR@k (Mean Reciprocal Rank):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'k':<10} {'MRR@k':<15}\n")
        f.write("-" * 25 + "\n")
        for k in top_k_values:
            f.write(f"{k:<10} {results[f'mrr_at_{k}']:.6f}\n")
        
        f.write("\nRecall@k:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'k':<10} {'Recall@k':<15}\n")
        f.write("-" * 25 + "\n")
        for k in top_k_values:
            f.write(f"{k:<10} {results[f'recall_at_{k}']:.6f}\n")
        
        f.write("\nTop-k Accuracy:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'k':<10} {'Top-k Acc':<15}\n")
        f.write("-" * 25 + "\n")
        for k in top_k_values:
            f.write(f"{k:<10} {results[f'top_{k}_accuracy']:.6f}\n")
        
        f.write("\nF2 Score:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'k':<10} {'F2@k':<15}\n")
        f.write("-" * 25 + "\n")
        for k in top_k_values:
            f.write(f"{k:<10} {results[f'f2_at_{k}']:.6f}\n")
        
        f.write("\nRank Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Rank: {results['mean_rank']:.2f}\n")
        f.write(f"Median Rank: {results['median_rank']:.2f}\n")
        f.write(f"Min Rank: {results['min_rank']}\n")
        f.write(f"Max Rank: {results['max_rank']}\n")
    
    print(f"Summary saved to: {summary_file}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    model_path = os.path.join(script_dir, "trained_models", "hans_dpr_6hour")
    eval_data_path = os.path.join(script_dir, "data", "mmarco", "mmarco-validation-6hour.tsv")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please wait for training to complete.")
        sys.exit(1)
    
    if not os.path.exists(eval_data_path):
        print(f"ERROR: Evaluation data not found at {eval_data_path}")
        sys.exit(1)
    
    evaluate_hans_6hour(model_path, eval_data_path)

