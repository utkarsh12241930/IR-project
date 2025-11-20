# Baseline DPR Model Evaluation Results (Medium Dataset)

## Executive Summary

This document presents the evaluation results for the Baseline Dense Passage Retrieval (DPR) model trained on a medium-sized subset of the mMARCO (Multilingual MS MARCO) dataset. The model was trained on **5,000 examples** and evaluated on **500 validation examples**.

**Key Highlights:**
- **MRR@10**: 0.9313 (93.13%)
- **Recall@10**: 0.9820 (98.20%)
- **Top-10 Accuracy**: 0.9820 (98.20%)
- **Mean Rank**: 2.63

---

## Model Information

| Parameter | Value |
|-----------|-------|
| **Model Type** | Dense Passage Retrieval (DPR) |
| **Training Dataset** | mMARCO (Multilingual MS MARCO) |
| **Training Examples** | 5,000 |
| **Evaluation Dataset** | mMARCO Validation Set |
| **Evaluation Examples** | 500 |
| **Context Encoder** | facebook/dpr-ctx_encoder-single-nq-base |
| **Query Encoder** | facebook/dpr-question_encoder-single-nq-base |
| **Max Sequence Length** | 128 |
| **Training Epochs** | 1 |
| **Batch Size** | 16 |

---

## Evaluation Metrics

### MRR@k (Mean Reciprocal Rank)

Mean Reciprocal Rank measures the average of the reciprocal ranks of the first relevant document retrieved for each query.

| k | MRR@k | Performance |
|---|-------|-------------|
| 1 | 0.9000 | 90.00% |
| 5 | 0.9300 | 93.00% |
| 10 | 0.9313 | 93.13% |
| 20 | 0.9319 | 93.19% |
| 50 | 0.9319 | 93.19% |
| 100 | 0.9319 | 93.19% |

**Interpretation**: The model achieves an MRR@10 of 0.9313, indicating strong ranking quality.

---

### Recall@k

Recall measures the percentage of all relevant documents that are retrieved in the top-k results.

| k | Recall@k | Performance |
|---|----------|-------------|
| 1 | 0.9000 | 90.00% |
| 5 | 0.9740 | 97.40% |
| 10 | 0.9820 | 98.20% |
| 20 | 0.9920 | 99.20% |
| 50 | 0.9920 | 99.20% |
| 100 | 0.9960 | 99.60% |

**Interpretation**: Recall@10 of 0.9820 (98.20%) shows the model's ability to retrieve relevant documents.

---

### Top-k Accuracy

Top-k Accuracy measures the percentage of queries where the gold (correct) passage appears in the top-k retrieved results.

| k | Top-k Accuracy | Performance |
|---|----------------|-------------|
| 1 | 0.9000 | 90.00% |
| 5 | 0.9740 | 97.40% |
| 10 | 0.9820 | 98.20% |
| 20 | 0.9920 | 99.20% |
| 50 | 0.9920 | 99.20% |
| 100 | 0.9960 | 99.60% |

---

### F2 Score

F2 Score is a weighted harmonic mean of precision and recall, with more emphasis on recall (beta=2).

| k | F2@k | Performance |
|---|------|-------------|
| 1 | 0.9000 | 90.00% |
| 5 | 0.5411 | 54.11% |
| 10 | 0.3507 | 35.07% |
| 20 | 0.2067 | 20.67% |
| 50 | 0.0919 | 9.19% |
| 100 | 0.0479 | 4.79% |

---

## Rank Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean Rank** | 2.63 | On average, the relevant document appears at position 2.63 |
| **Median Rank** | 1.00 | The median rank is 1.00 |
| **Min Rank** | 1 | Best case: relevant document at position 1 |
| **Max Rank** | 319 | Worst case: relevant document at position 319 |

---

## Performance Summary

### Key Metrics at k=10 (Standard Evaluation Point)

| Metric | Value | Grade |
|--------|-------|-------|
| MRR@10 | 0.9313 | Excellent |
| Recall@10 | 0.9820 | Excellent |
| Top-10 Accuracy | 0.9820 | Excellent |
| F2@10 | 0.3507 | - |

---

## Files Generated

1. **BASELINE_MEDIUM_RESULTS_PRESENTATION.md** (this file) - Markdown presentation
2. **baseline_medium_results_latex.tex** - LaTeX table for paper inclusion
3. **baseline_medium_results_table.csv** - CSV format for Excel/Google Sheets
4. **baseline_medium_evaluation_results.json** - Complete results in JSON format

---

*Generated from baseline DPR model evaluation on mMARCO medium validation set*
