# HANS DPR Model Evaluation Results (Medium Dataset)

## Executive Summary

This document presents the evaluation results for the HANS (Hardness-Adaptive Negative Sampling) Dense Passage Retrieval (DPR) model trained on a medium-sized subset of the mMARCO (Multilingual MS MARCO) dataset. The model was trained on **5,000 examples** and evaluated on **500 validation examples**.

**Key Highlights:**
- **MRR@10**: 0.9281 (92.81%)
- **Recall@10**: 0.9840 (98.40%)
- **Top-10 Accuracy**: 0.9840 (98.40%)
- **Mean Rank**: 2.61

---

## Model Information

| Parameter | Value |
|-----------|-------|
| **Model Type** | HANS Dense Passage Retrieval (DPR) |
| **Training Dataset** | mMARCO (Multilingual MS MARCO) |
| **Training Examples** | 5,000 |
| **Evaluation Dataset** | mMARCO Validation Set |
| **Evaluation Examples** | 500 |
| **Context Encoder** | facebook/dpr-ctx_encoder-single-nq-base |
| **Query Encoder** | facebook/dpr-question_encoder-single-nq-base |
| **Max Sequence Length** | 128 |
| **Training Epochs** | 1 |
| **Batch Size** | 16 |
| **HANS Enabled** | Yes |
| **HANS Min Negatives** | 1 |
| **HANS Max Negatives** | 3 |

---

## Evaluation Metrics

### MRR@k (Mean Reciprocal Rank)

Mean Reciprocal Rank measures the average of the reciprocal ranks of the first relevant document retrieved for each query.

| k | MRR@k | Performance |
|---|-------|-------------|
| 1 | 0.8960 | 89.60% |
| 5 | 0.9265 | 92.65% |
| 10 | 0.9281 | 92.81% |
| 20 | 0.9286 | 92.86% |
| 50 | 0.9286 | 92.86% |
| 100 | 0.9287 | 92.87% |

**Interpretation**: The model achieves an MRR@10 of 0.9281, indicating strong ranking quality.

---

### Recall@k

Recall measures the percentage of all relevant documents that are retrieved in the top-k results.

| k | Recall@k | Performance |
|---|----------|-------------|
| 1 | 0.8960 | 89.60% |
| 5 | 0.9720 | 97.20% |
| 10 | 0.9840 | 98.40% |
| 20 | 0.9920 | 99.20% |
| 50 | 0.9920 | 99.20% |
| 100 | 0.9940 | 99.40% |

**Interpretation**: Recall@10 of 0.9840 (98.40%) shows the model's ability to retrieve relevant documents.

---

### Top-k Accuracy

Top-k Accuracy measures the percentage of queries where the gold (correct) passage appears in the top-k retrieved results.

| k | Top-k Accuracy | Performance |
|---|----------------|-------------|
| 1 | 0.8960 | 89.60% |
| 5 | 0.9720 | 97.20% |
| 10 | 0.9840 | 98.40% |
| 20 | 0.9920 | 99.20% |
| 50 | 0.9920 | 99.20% |
| 100 | 0.9940 | 99.40% |

---

### F2 Score

F2 Score is a weighted harmonic mean of precision and recall, with more emphasis on recall (beta=2).

| k | F2@k | Performance |
|---|------|-------------|
| 1 | 0.8960 | 89.60% |
| 5 | 0.5400 | 54.00% |
| 10 | 0.3514 | 35.14% |
| 20 | 0.2067 | 20.67% |
| 50 | 0.0919 | 9.19% |
| 100 | 0.0478 | 4.78% |

---

## Rank Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean Rank** | 2.61 | On average, the relevant document appears at position 2.61 |
| **Median Rank** | 1.00 | The median rank is 1.00 |
| **Min Rank** | 1 | Best case: relevant document at position 1 |
| **Max Rank** | 310 | Worst case: relevant document at position 310 |

---

## Performance Summary

### Key Metrics at k=10 (Standard Evaluation Point)

| Metric | Value | Grade |
|--------|-------|-------|
| MRR@10 | 0.9281 | Excellent |
| Recall@10 | 0.9840 | Excellent |
| Top-10 Accuracy | 0.9840 | Excellent |
| F2@10 | 0.3514 | - |

---

## Files Generated

1. **HANS_MEDIUM_RESULTS_PRESENTATION.md** (this file) - Markdown presentation
2. **hans_medium_results_latex.tex** - LaTeX table for paper inclusion
3. **hans_medium_results_table.csv** - CSV format for Excel/Google Sheets
4. **hans_medium_evaluation_results.json** - Complete results in JSON format

---

*Generated from HANS DPR model evaluation on mMARCO medium validation set*
