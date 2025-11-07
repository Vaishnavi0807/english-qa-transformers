# English QA: Transformer-Based Extractive Question Answering on SQuAD v2

## 1. Overview
This project builds and compares multiple transformer models (BERT, DeBERTa, RoBERTa) for **extractive question answering** on the SQuAD v2 dataset, which includes answerable and unanswerable questions. An optional **generative comparison** using Flan-T5 is included. The focus is on:
- Accurate span extraction
- Robust handling of unanswerable questions
- Rigorous evaluation beyond EM/F1 (semantic metrics, answerability accuracy)
- Error analysis and threshold calibration

## 2. Research Questions
1. How do model architectures differ in detecting unanswerable questions?
2. Does semantic-aware evaluation (BERTScore) alter model ranking compared to EM/F1?
3. What are trade-offs between generative (Flan-T5) and extractive approaches?

## 3. Dataset Strategy
We use a stratified subset of 9,000 training + 1,000 validation examples preserving the answerable/unanswerable ratio. Optionally, you can evaluate on the full validation set for final reporting.
- Source: SQuAD v2 via Hugging Face (`squad_v2`)
- Reproducible sampling with a fixed seed (42)

## 4. Models & Hyperparameters
| Model        | Checkpoint                | Max Len | LR     | Epochs | Train Batch | Eval Batch | Warmup | FP16 |
|--------------|---------------------------|---------|--------|--------|-------------|------------|--------|------|
| BERT-base    | bert-base-uncased         | 384     | 3e-5   | 3–4    | 16          | 16         | 10%    | Yes  |
| RoBERTa-base | roberta-base              | 384     | 3e-5   | 3–4    | 16          | 16         | 10%    | Yes  |
| DeBERTa-base | microsoft/deberta-base    | 384     | 2e-5   | 3–4    | 8–12        | 16         | 10%    | Yes  |
| Flan-T5-base | google/flan-t5-base       | 512     | 1e-4   | 3      | 8           | 8          | 5%     | Yes  |

Use gradient accumulation if GPU memory is limited.

## 5. Metrics
- Exact Match (EM)
- Token-level F1
- Answerability accuracy (binary correctness of no-answer vs answerable)
- BERTScore (semantic similarity)
- Optional: Length-stratified performance and calibration curves

## 6. Project Structure
```
.
├── config.yaml
├── requirements.txt
├── Makefile
├── run.sh
├── src/
│   ├── utils.py
│   ├── run_subset.py
│   ├── postprocess.py
│   ├── metrics.py
│   ├── train_bert.py
│   ├── train_roberta.py
│   ├── train_deberta.py
│   ├── train_flan_t5.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   ├── inference.py
├── notebooks/
│   ├── 01_data_preparation.ipynb (optional skeleton)
│   ├── 02_train_models.ipynb
│   ├── 03_evaluation_analysis.ipynb
├── results/
│   ├── predictions/
│   ├── metrics/
│   ├── checkpoints/
├── docs/
│   ├── report.md
│   ├── slides.md
```

## 7. Reproducibility
- Fixed random seeds (42) for NumPy, Python `random`, PyTorch.
- Environment versions pinned in `requirements.txt`.
- Logging hyperparameters and metrics to CSV/JSON.
- Null answer threshold tuned via validation sweep.

## 8. Setup & Execution

### (A) Install
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### (B) Create Subset
```
python src/run_subset.py --output_path data/subset_train.json --valid_path data/subset_valid.json
```

### (C) Train (Example: BERT)
```
python src/train_bert.py \
  --train_path data/subset_train.json \
  --valid_path data/subset_valid.json \
  --output_dir results/checkpoints/bert
```

Repeat for other models:
```
python src/train_roberta.py ...
python src/train_deberta.py ...
python src/train_flan_t5.py ...
```

### (D) Evaluate
```
python src/evaluate.py \
  --model_dir results/checkpoints/bert \
  --valid_path data/subset_valid.json \
  --out_dir results/metrics/bert
```

### (E) Threshold Tuning
```
python src/evaluate.py \
  --model_dir results/checkpoints/bert \
  --valid_path data/subset_valid.json \
  --out_dir results/metrics/bert \
  --tune_null_threshold True
```

### (F) Error Analysis
```
python src/error_analysis.py \
  --predictions results/metrics/bert/predictions.json \
  --truth data/subset_valid.json \
  --out_dir results/metrics/bert/errors
```

### (G) Inference Demo
```
python src/inference.py \
  --model_dir results/checkpoints/roberta \
  --question "Who founded the company?" \
  --context_file demo_context.txt
```

## 9. Demo Guidance
Show:
1. Single question inference with predicted span + confidence.
2. Comparison table EM/F1/BERTScore across models.
3. Threshold visualization (null score distribution).
4. One qualitative example where a model correctly abstains vs another hallucinating.

## 10. Risk Mitigation
| Risk | Mitigation |
|------|------------|
| OOM | Reduce batch size + gradient accumulation |
| Poor null threshold | Tune on validation using F1 maximizing approach |
| Unbalanced subset | Enforced stratified sampling |
| Slow training | Start earliest with baseline; parallel team roles |
| Generative model poor EM | Pre-normalize answers; enforce unanswerable token |

## 11. Academic Integrity & Citations
Cite:
- Original SQuAD papers (Rajpurkar et al. 2016, 2018).
- BERT (Devlin et al. 2019), RoBERTa (Liu et al. 2019), DeBERTa (He et al. 2021), T5 (Raffel et al. 2020).
- BERTScore (Zhang et al. 2020).

## 12. Next Steps
Optional extensions: Distillation, Natural Questions transfer, confidence calibration charts, retrieval augmentation.

## 13. Team Responsibilities
- Adi: Baseline BERT + data subset scripts
- Amrutha: DeBERTa tuning + threshold calibration
- Gayatri: RoBERTa + visualization + report integration
- Joint: Generative comparison & error taxonomy

## 14. Quick Make Targets
```
make subset
make train_bert
make evaluate_bert
make all_models
```

## 15. License
Add MIT or Apache-2.0 if desired.

---

For full technical implementation details refer to `docs/report.md`. Slides overview: see `docs/slides.md`.

Let me know if you want a converted `.pptx` script or Weights & Biases integration.