# HANS DPR Model Evaluation Results (6-Hour Dataset)

## Executive Summary

This document presents the evaluation results for the HANS (Hardness-Adaptive Negative Sampling) Dense Passage Retrieval (DPR) model trained on a 6-hour subset of the mMARCO (Multilingual MS MARCO) dataset. The model was trained on **2,000 examples** and evaluated on **200 validation examples**.

**Key Highlights:**
- **MRR@10**: 0.9200 (92.00%)
- **Recall@10**: 0.9900 (99.00%)
- **Top-10 Accuracy**: 0.9900 (99.00%)
- **Mean Rank**: 1.71

---

## Model Information

| Parameter | Value |
|-----------|-------|
| **Model Type** | HANS Dense Passage Retrieval (DPR) |
| **Training Dataset** | mMARCO (Multilingual MS MARCO) |
| **Training Examples** | 2,000 |
| **Evaluation Dataset** | mMARCO Validation Set |
| **Evaluation Examples** | 200 |
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
| 1 | 0.8800 | 88.00% |
| 5 | 0.9172 | 91.72% |
| 10 | 0.9200 | 92.00% |
| 20 | 0.9204 | 92.04% |
| 50 | 0.9204 | 92.04% |
| 100 | 0.9205 | 92.05% |

**Interpretation**: The model achieves an MRR@10 of 0.9200, indicating strong ranking quality.

---

### Recall@k

Recall measures the percentage of all relevant documents that are retrieved in the top-k results.

| k | Recall@k | Performance |
|---|----------|-------------|
| 1 | 0.8800 | 88.00% |
| 5 | 0.9700 | 97.00% |
| 10 | 0.9900 | 99.00% |
| 20 | 0.9950 | 99.50% |
| 50 | 0.9950 | 99.50% |
| 100 | 1.0000 | 100.00% |

**Interpretation**: Recall@10 of 0.9900 (99.00%) shows the model's ability to retrieve relevant documents.

---

### Top-k Accuracy

Top-k Accuracy measures the percentage of queries where the gold (correct) passage appears in the top-k retrieved results.

| k | Top-k Accuracy | Performance |
|---|----------------|-------------|
| 1 | 0.8800 | 88.00% |
| 5 | 0.9700 | 97.00% |
| 10 | 0.9900 | 99.00% |
| 20 | 0.9950 | 99.50% |
| 50 | 0.9950 | 99.50% |
| 100 | 1.0000 | 100.00% |

---

### F2 Score

F2 Score is a weighted harmonic mean of precision and recall, with more emphasis on recall (beta=2).

| k | F2@k | Performance |
|---|------|-------------|
| 1 | 0.8800 | 88.00% |
| 5 | 0.5389 | 53.89% |
| 10 | 0.3536 | 35.36% |
| 20 | 0.2073 | 20.73% |
| 50 | 0.0921 | 9.21% |
| 100 | 0.0481 | 4.81% |

---

## Rank Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean Rank** | 1.71 | On average, the relevant document appears at position 1.71 |
| **Median Rank** | 1.00 | The median rank is 1.00 |
| **Min Rank** | 1 | Best case: relevant document at position 1 |
| **Max Rank** | 74 | Worst case: relevant document at position 74 |

---

## Performance Summary

### Key Metrics at k=10 (Standard Evaluation Point)

| Metric | Value | Grade |
|--------|-------|-------|
| MRR@10 | 0.9200 | Excellent |
| Recall@10 | 0.9900 | Excellent |
| Top-10 Accuracy | 0.9900 | Excellent |
| F2@10 | 0.3536 | - |

---

## Files Generated

1. **HANS_6HOUR_RESULTS_PRESENTATION.md** (this file) - Markdown presentation
2. **hans_6hour_results_latex.tex** - LaTeX table for paper inclusion
3. **hans_6hour_results_table.csv** - CSV format for Excel/Google Sheets
4. **hans_6hour_evaluation_results.json** - Complete results in JSON format

---

*Generated from HANS DPR model evaluation on mMARCO 6-hour validation set*
