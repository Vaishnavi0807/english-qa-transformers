# English QA: Transformer-Based Extractive Question Answering on SQuAD v2

## Abstract
This report presents a comparative analysis of transformer architectures (BERT, RoBERTa, DeBERTa) for extractive Question Answering on SQuAD v2, augmented with semantic evaluation (BERTScore) and a generative comparison (Flan-T5). We emphasize unanswerable question handling through calibrated null thresholding and structured error taxonomy.

## 1. Introduction
Question Answering (QA) evaluates a model’s ability to comprehend text and extract relevant information. SQuAD v2 introduces unanswerable queries necessitating reliable abstention mechanisms. We investigate model architecture impacts on span extraction fidelity and unanswerable detection.

## 2. Related Work
- SQuAD (Rajpurkar et al. 2016, 2018): Benchmark evolution adding unanswerable questions.
- BERT (Devlin et al. 2019): Contextual embeddings revolutionizing QA baselines.
- RoBERTa (Liu et al. 2019): Enhanced training via dynamic masking and larger batches.
- DeBERTa (He et al. 2021): Disentangled attention + absolute/relative position improvements.
- T5 / Flan (Raffel et al. 2020; Chung et al. 2022): Unified text-to-text paradigm, instruction tuning.
- BERTScore (Zhang et al. 2020): Semantic similarity metric complementing lexical measures.

## 3. Research Questions
1. Does architecture materially affect the precision-recall balance for answerable detection?
2. Does BERTScore reorder model ranking compared to EM/F1?
3. How does a generative model’s abstention strategy compare to extractive null scoring?

## 4. Dataset
We use a stratified subset (9,000 train / 1,000 validation) preserving answerable/unanswerable ratio. Each instance includes context, question, answers (possibly empty), and `is_impossible`.

## 5. Methodology
### 5.1 Preprocessing
- Tokenization with model-specific fast tokenizers, truncating context only (max length 384).
- Overflow handling with stride 128 to improve coverage.
- Offset mapping for character-to-token span alignment.
- CLS-based null answer representation.

### 5.2 Training
- Hugging Face Trainer, fp16 mixed precision, warmup 10% steps.
- Weight decay 0.01 for regularization.
- Model-specific learning rates (DeBERTa lower).
- Early stopping optional (not enabled by default due to small epochs).

### 5.3 Evaluation
- Raw predictions -> post-processing -> best non-null vs null score comparison.
- Null threshold tuned by sweeping score_diff values to maximize F1.
- Metrics computed: EM, F1, Answerability Accuracy, BERTScore-P/R/F1.
- Qualitative error categorization.

### 5.4 Generative Model Handling
Flan-T5 prompted with explicit abstention token (“unanswerable”). Predictions normalized via lowercase, punctuation removal, article stripping.

## 6. Metrics & Threshold Calibration
We define:
- Score difference = best_non_null_score − null_score
- If score_diff < threshold → abstain (predict empty string).
Grid search across observed diffs yields tuned threshold. Provide both raw and tuned metrics.

## 7. Error Taxonomy
Categories:
1. Exact
2. Partial_Overlap
3. Wrong_Entity
4. False_Positive_NonNull
5. False_Negative_Null
6. Correct_NoAnswer

## 8. Results (Placeholder – fill after execution)
| Model | EM | F1 | Answerability Acc | BERTScore F1 | Tuned Threshold | Tuned F1 |
|-------|----|----|-------------------|--------------|-----------------|----------|
| BERT  | XX | XX | XX                | XX           | t=...           | XX       |
| RoBERTa | XX | XX | XX              | XX           | t=...           | XX       |
| DeBERTa | XX | XX | XX              | XX           | t=...           | XX       |
| Flan-T5 | XX | XX | XX              | XX           | (prompt)        | XX       |

Observations (fill after training):
- DeBERTa expected to outperform on F1 due to improved representation.
- RoBERTa may yield stronger EM due to training regime.
- Flan-T5 may have lower EM but competitive BERTScore.

## 9. Analysis
Discuss boundary errors, entity confusions, and abstention reliability. Highlight cases where high semantic similarity (BERTScore) still fails EM due to paraphrasing.

## 10. Limitations
- Subset sampling may under-represent rare linguistic phenomena.
- Threshold tuning on validation may overfit small subset.
- Generative evaluation lacks beam search / calibration exploration.

## 11. Future Work
- Multi-dataset transfer (Natural Questions).
- Confidence calibration (reliability diagrams).
- Knowledge distillation for lightweight deployment.
- Retrieval augmentation for long context handling.

## 12. Conclusion
Multi-model comparison surfaces trade-offs between precision, semantic richness, and abstention reliability. Null threshold tuning is critical for balanced performance under SQuAD v2 constraints.

## 13. References
Add full citations (placeholder):
- Devlin et al. (2019) BERT
- Liu et al. (2019) RoBERTa
- He et al. (2021) DeBERTa
- Rajpurkar et al. (2016, 2018) SQuAD
- Raffel et al. (2020) T5
- Zhang et al. (2020) BERTScore
- Chung et al. (2022) Flan
