import argparse, json, os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import Dataset
import torch
from src.utils import read_json, load_config, ensure_dir, log
from src.postprocess import squad_postprocess
from src.metrics import evaluate_predictions, save_metrics, threshold_sweep

def load_features(examples, tokenizer, max_length, doc_stride):
    def preprocess(batch):
        tokenized = tokenizer(
            batch["question"],
            batch["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []
        for i, mapping in enumerate(sample_mapping):
            tokenized["example_id"].append(batch["id"][mapping])
        # Clean offset mapping for non-context tokens
        new_offsets = []
        for i, offsets in enumerate(tokenized["offset_mapping"]):
            seq_ids = tokenized.sequence_ids(i)
            filtered = []
            for k, o in enumerate(offsets):
                if seq_ids[k] == 1:
                    filtered.append(o)
                else:
                    filtered.append(None)
            new_offsets.append(filtered)
        tokenized["offset_mapping"] = new_offsets
        return tokenized
    ds = Dataset.from_list(examples)
    features = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    return features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--valid_path", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tune_null_threshold", default="False")
    ap.add_argument("--use_bert_score", default="True")
    args = ap.parse_args()
    cfg = load_config(args.config)
    valid_path = args.valid_path or cfg["paths"]["subset_valid"]
    examples = read_json(valid_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    model.eval()

    features = load_features(examples, tokenizer,
                             cfg["training_defaults"]["max_length"],
                             cfg["training_defaults"]["doc_stride"])

    # Convert features to torch batches
    start_logits_list, end_logits_list = [], []
    batch_size = 16
    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        input_ids = torch.tensor(batch["input_ids"])
        attention_mask = torch.tensor(batch["attention_mask"])
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits_list.append(out.start_logits.cpu().numpy())
        end_logits_list.append(out.end_logits.cpu().numpy())
    import numpy as np
    start_logits = np.concatenate(start_logits_list, axis=0)
    end_logits = np.concatenate(end_logits_list, axis=0)

    raw_preds, meta = squad_postprocess(
        examples, features, (start_logits, end_logits),
        n_best_size=cfg["evaluation"]["n_best_size"],
        max_answer_length=cfg["evaluation"]["max_answer_length"],
        tokenizer=tokenizer,
        null_score_diff_threshold=0.0
    )

    metrics = evaluate_predictions(examples, raw_preds, answerability_meta=meta,
                                   use_bert_score=args.use_bert_score.lower() == "true")

    # Threshold tuning
    if args.tune_null_threshold.lower() == "true":
        best = threshold_sweep(examples, meta, raw_preds,
                               [ex["answers"]["text"][0] if len(ex["answers"]["text"]) > 0 else "" for ex in examples])
        tuned_preds, tuned_meta = squad_postprocess(
            examples, features, (start_logits, end_logits),
            n_best_size=cfg["evaluation"]["n_best_size"],
            max_answer_length=cfg["evaluation"]["max_answer_length"],
            tokenizer=tokenizer,
            null_score_diff_threshold=best["threshold"]
        )
        tuned_metrics = evaluate_predictions(examples, tuned_preds, answerability_meta=tuned_meta,
                                             use_bert_score=args.use_bert_score.lower() == "true")
        metrics["Tuned_Threshold"] = best["threshold"]
        metrics["Tuned_F1"] = tuned_metrics["F1"]
        metrics["Tuned_EM"] = tuned_metrics["EM"]

    out_dir = args.out_dir or f"results/metrics/{os.path.basename(args.model_dir)}"
    ensure_dir(out_dir)
    with open(f"{out_dir}/predictions.json", "w") as f:
        json.dump(raw_preds, f, indent=2)
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    save_metrics(metrics, f"{out_dir}/metrics.json")
    log(f"Evaluation complete. Metrics saved to {out_dir}/metrics.json")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()