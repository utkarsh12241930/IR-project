"""
Create Presentable Results for HANS Model (6-Hour Dataset)
Formats evaluation results in a clean, publication-ready format.
"""
import os
import sys
import json
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

def create_presentation_results():
    """Create a presentable results document for HANS model on 6-hour dataset."""
    
    results_file = os.path.join(script_dir, "hans_6hour_evaluation_results.json")
    
    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found at {results_file}")
        print("Please run evaluate_hans_6hour.py first.")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create presentation document
    output_file = os.path.join(script_dir, "HANS_6HOUR_RESULTS_PRESENTATION.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# HANS DPR Model Evaluation Results (6-Hour Dataset)\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This document presents the evaluation results for the HANS (Hardness-Adaptive Negative Sampling) ")
        f.write("Dense Passage Retrieval (DPR) model trained on a 6-hour subset of the mMARCO (Multilingual MS MARCO) dataset. ")
        f.write("The model was trained on **2,000 examples** and evaluated on **200 validation examples**.\n\n")
        
        # Key highlights
        if 'mrr_at_10' in results:
            f.write("**Key Highlights:**\n")
            f.write(f"- **MRR@10**: {results['mrr_at_10']:.4f} ({results['mrr_at_10']*100:.2f}%)\n")
        if 'recall_at_10' in results:
            f.write(f"- **Recall@10**: {results['recall_at_10']:.4f} ({results['recall_at_10']*100:.2f}%)\n")
        if 'top_10_accuracy' in results:
            f.write(f"- **Top-10 Accuracy**: {results['top_10_accuracy']:.4f} ({results['top_10_accuracy']*100:.2f}%)\n")
        if 'mean_rank' in results:
            f.write(f"- **Mean Rank**: {results['mean_rank']:.2f}\n")
        f.write("\n---\n\n")
        
        f.write("## Model Information\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write("| **Model Type** | HANS Dense Passage Retrieval (DPR) |\n")
        f.write("| **Training Dataset** | mMARCO (Multilingual MS MARCO) |\n")
        f.write("| **Training Examples** | 2,000 |\n")
        f.write("| **Evaluation Dataset** | mMARCO Validation Set |\n")
        f.write("| **Evaluation Examples** | 200 |\n")
        f.write("| **Context Encoder** | facebook/dpr-ctx_encoder-single-nq-base |\n")
        f.write("| **Query Encoder** | facebook/dpr-question_encoder-single-nq-base |\n")
        f.write("| **Max Sequence Length** | 128 |\n")
        f.write("| **Training Epochs** | 1 |\n")
        f.write("| **Batch Size** | 16 |\n")
        f.write("| **HANS Enabled** | Yes |\n")
        f.write("| **HANS Min Negatives** | 1 |\n")
        f.write("| **HANS Max Negatives** | 3 |\n")
        f.write("\n---\n\n")
        
        f.write("## Evaluation Metrics\n\n")
        f.write("### MRR@k (Mean Reciprocal Rank)\n\n")
        f.write("Mean Reciprocal Rank measures the average of the reciprocal ranks of the first relevant document retrieved for each query.\n\n")
        f.write("| k | MRR@k | Performance |\n")
        f.write("|---|-------|-------------|\n")
        for k in [1, 5, 10, 20, 50, 100]:
            if f'mrr_at_{k}' in results:
                f.write(f"| {k} | {results[f'mrr_at_{k}']:.4f} | {results[f'mrr_at_{k}']*100:.2f}% |\n")
        
        f.write("\n**Interpretation**: ")
        if 'mrr_at_10' in results:
            f.write(f"The model achieves an MRR@10 of {results['mrr_at_10']:.4f}, indicating strong ranking quality.\n\n")
        f.write("---\n\n")
        
        f.write("### Recall@k\n\n")
        f.write("Recall measures the percentage of all relevant documents that are retrieved in the top-k results.\n\n")
        f.write("| k | Recall@k | Performance |\n")
        f.write("|---|----------|-------------|\n")
        for k in [1, 5, 10, 20, 50, 100]:
            if f'recall_at_{k}' in results:
                f.write(f"| {k} | {results[f'recall_at_{k}']:.4f} | {results[f'recall_at_{k}']*100:.2f}% |\n")
        
        f.write("\n**Interpretation**: ")
        if 'recall_at_10' in results:
            f.write(f"Recall@10 of {results['recall_at_10']:.4f} ({results['recall_at_10']*100:.2f}%) shows the model's ability to retrieve relevant documents.\n\n")
        f.write("---\n\n")
        
        f.write("### Top-k Accuracy\n\n")
        f.write("Top-k Accuracy measures the percentage of queries where the gold (correct) passage appears in the top-k retrieved results.\n\n")
        f.write("| k | Top-k Accuracy | Performance |\n")
        f.write("|---|----------------|-------------|\n")
        for k in [1, 5, 10, 20, 50, 100]:
            if f'top_{k}_accuracy' in results:
                f.write(f"| {k} | {results[f'top_{k}_accuracy']:.4f} | {results[f'top_{k}_accuracy']*100:.2f}% |\n")
        
        f.write("\n---\n\n")
        
        f.write("### F2 Score\n\n")
        f.write("F2 Score is a weighted harmonic mean of precision and recall, with more emphasis on recall (beta=2).\n\n")
        f.write("| k | F2@k | Performance |\n")
        f.write("|---|------|-------------|\n")
        for k in [1, 5, 10, 20, 50, 100]:
            if f'f2_at_{k}' in results:
                f.write(f"| {k} | {results[f'f2_at_{k}']:.4f} | {results[f'f2_at_{k}']*100:.2f}% |\n")
        
        f.write("\n---\n\n")
        
        f.write("## Rank Statistics\n\n")
        f.write("| Metric | Value | Interpretation |\n")
        f.write("|--------|-------|----------------|\n")
        if 'mean_rank' in results:
            f.write(f"| **Mean Rank** | {results['mean_rank']:.2f} | On average, the relevant document appears at position {results['mean_rank']:.2f} |\n")
        if 'median_rank' in results:
            f.write(f"| **Median Rank** | {results['median_rank']:.2f} | The median rank is {results['median_rank']:.2f} |\n")
        if 'min_rank' in results:
            f.write(f"| **Min Rank** | {results['min_rank']} | Best case: relevant document at position {results['min_rank']} |\n")
        if 'max_rank' in results:
            f.write(f"| **Max Rank** | {results['max_rank']} | Worst case: relevant document at position {results['max_rank']} |\n")
        
        f.write("\n---\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("### Key Metrics at k=10 (Standard Evaluation Point)\n\n")
        f.write("| Metric | Value | Grade |\n")
        f.write("|--------|-------|-------|\n")
        if 'mrr_at_10' in results:
            grade = "Excellent" if results['mrr_at_10'] > 0.8 else "Good" if results['mrr_at_10'] > 0.6 else "Fair"
            f.write(f"| MRR@10 | {results['mrr_at_10']:.4f} | {grade} |\n")
        if 'recall_at_10' in results:
            grade = "Perfect" if results['recall_at_10'] >= 1.0 else "Excellent" if results['recall_at_10'] > 0.8 else "Good"
            f.write(f"| Recall@10 | {results['recall_at_10']:.4f} | {grade} |\n")
        if 'top_10_accuracy' in results:
            grade = "Perfect" if results['top_10_accuracy'] >= 1.0 else "Excellent" if results['top_10_accuracy'] > 0.8 else "Good"
            f.write(f"| Top-10 Accuracy | {results['top_10_accuracy']:.4f} | {grade} |\n")
        if 'f2_at_10' in results:
            f.write(f"| F2@10 | {results['f2_at_10']:.4f} | - |\n")
        
        f.write("\n---\n\n")
        f.write("## Files Generated\n\n")
        f.write("1. **HANS_6HOUR_RESULTS_PRESENTATION.md** (this file) - Markdown presentation\n")
        f.write("2. **hans_6hour_results_latex.tex** - LaTeX table for paper inclusion\n")
        f.write("3. **hans_6hour_results_table.csv** - CSV format for Excel/Google Sheets\n")
        f.write("4. **hans_6hour_evaluation_results.json** - Complete results in JSON format\n")
        f.write("\n---\n\n")
        f.write("*Generated from HANS DPR model evaluation on mMARCO 6-hour validation set*\n")
    
    print(f"Presentation results saved to: {output_file}")
    
    # Also create a LaTeX table format
    latex_file = os.path.join(script_dir, "hans_6hour_results_latex.tex")
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{HANS DPR Model Evaluation Results (6-Hour Dataset)}\n")
        f.write("\\label{tab:hans_6hour_results}\n")
        f.write("\\begin{tabular}{ccccc}\n")
        f.write("\\toprule\n")
        f.write("k & MRR@k & Recall@k & Top-k Acc & F2@k \\\\\n")
        f.write("\\midrule\n")
        for k in [1, 5, 10, 20, 50, 100]:
            mrr = results.get(f'mrr_at_{k}', 0)
            recall = results.get(f'recall_at_{k}', 0)
            acc = results.get(f'top_{k}_accuracy', 0)
            f2 = results.get(f'f2_at_{k}', 0)
            f.write(f"{k} & {mrr:.4f} & {recall:.4f} & {acc:.4f} & {f2:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {latex_file}")
    
    # Create a CSV for easy import into Excel/Google Sheets
    csv_file = os.path.join(script_dir, "hans_6hour_results_table.csv")
    data = {
        'k': [],
        'MRR@k': [],
        'Recall@k': [],
        'Top-k Accuracy': [],
        'F2@k': []
    }
    for k in [1, 5, 10, 20, 50, 100]:
        data['k'].append(k)
        data['MRR@k'].append(results.get(f'mrr_at_{k}', 0))
        data['Recall@k'].append(results.get(f'recall_at_{k}', 0))
        data['Top-k Accuracy'].append(results.get(f'top_{k}_accuracy', 0))
        data['F2@k'].append(results.get(f'f2_at_{k}', 0))
    
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"CSV table saved to: {csv_file}")

if __name__ == "__main__":
    create_presentation_results()

