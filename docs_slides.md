# English QA: Transformer-Based Extractive Question Answering (SQuAD v2)

---

## Slide 1: Title
- English QA: Transformer-Based Extractive QA on SQuAD v2
- Team: Adi | Amrutha | Gayatri
- Course: EDS 6352 – NLP

---

## Slide 2: Motivation
- Need accurate comprehension + abstention for unanswerable queries.
- SQuAD v2 challenges naive answer extraction.
- Comparing architectures reveals robustness characteristics.

---

## Slide 3: Research Questions
1. Architecture vs unanswerable detection?
2. Semantic vs lexical metric divergence?
3. Extractive vs generative trade-offs?

---

## Slide 4: Dataset
- SQuAD v2 (answerable + unanswerable)
- Stratified subset (9K train / 1K validation)
- Preserved ratio ~ original distribution
- Context length variability

---

## Slide 5: Models
- BERT-base (baseline)
- RoBERTa-base
- DeBERTa-base
- Optional: Flan-T5 (generative contrast)

---

## Slide 6: Pipeline Diagram
[Data] → Tokenization → Training → Logits → Post-processing → Threshold Calibration → Metrics → Error Analysis

---

## Slide 7: Metrics
- EM, F1
- Answerability Accuracy
- BERTScore (semantic)
- Null threshold optimization

---

## Slide 8: Null Threshold
- Score_diff = best_non_null − null
- Sweep threshold to maximize F1
- Balances abstention vs over-answering

---

## Slide 9: Results (Placeholder)
Table with EM/F1/BERTScore/Answerability
Bar chart: F1 comparison

---

## Slide 10: Error Analysis
Categories:
- Exact
- Partial Overlap
- Wrong Entity
- False Positive
- False Negative
- Correct NoAnswer

---

## Slide 11: Qualitative Examples
Case 1: DeBERTa abstains correctly; RoBERTa hallucinates.
Case 2: Flan-T5 paraphrase vs exact span mismatch.

---

## Slide 12: Insights
- DeBERTa stronger on nuanced boundaries.
- RoBERTa robust lexical matching.
- Semantic score inflates generative outputs but not EM.

---

## Slide 13: Limitations
- Subset size
- Threshold overfitting risk
- Generative evaluation simplistic prompting

---

## Slide 14: Future Work
- Multi-dataset transfer
- Calibration curves
- Distillation
- Retrieval augmentation

---

## Slide 15: Conclusion
Balanced extractive approach + threshold tuning crucial for SQuAD v2.
Generative models offer semantic fluency but struggle with exactness.

---

## Slide 16: Questions
Thank you!