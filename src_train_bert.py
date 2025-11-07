import argparse, os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from src.utils import set_seed, load_config, read_json, ensure_dir, log
import numpy as np

def prepare_features(examples, tokenizer, max_length, doc_stride):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized["offset_mapping"]
    start_positions = []
    end_positions = []
    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]
        if len(answers["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        sequence_ids = tokenized.sequence_ids(i)
        if sequence_ids.count(1) == 0:
            start_positions.append(0); end_positions.append(0); continue
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - list(reversed(sequence_ids)).index(1)
        if not (offsets[context_start][0] <= start_char and offsets[context_end - 1][1] >= end_char):
            start_positions.append(0); end_positions.append(0)
            continue
        start_token = end_token = None
        for j in range(context_start, context_end):
            if offsets[j][0] <= start_char < offsets[j][1]:
                start_token = j
            if offsets[j][0] < end_char <= offsets[j][1]:
                end_token = j
        if start_token is None or end_token is None:
            start_positions.append(0); end_positions.append(0)
        else:
            start_positions.append(start_token); end_positions.append(end_token)
    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train_path", default=None)
    ap.add_argument("--valid_path", default=None)
    ap.add_argument("--output_dir", default="results/checkpoints/bert")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    model_name = cfg["models"]["bert"]
    train_path = args.train_path or cfg["paths"]["subset_train"]
    valid_path = args.valid_path or cfg["paths"]["subset_valid"]

    train_data = read_json(train_path)
    valid_data = read_json(valid_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    train_ds = Dataset.from_list(train_data)
    valid_ds = Dataset.from_list(valid_data)

    cols = train_ds.column_names
    train_feats = train_ds.map(
        lambda x: prepare_features(x, tokenizer, cfg["training_defaults"]["max_length"], cfg["training_defaults"]["doc_stride"]),
        batched=True,
        remove_columns=cols
    )
    valid_feats = valid_ds.map(
        lambda x: prepare_features(x, tokenizer, cfg["training_defaults"]["max_length"], cfg["training_defaults"]["doc_stride"]),
        batched=True,
        remove_columns=cols
    )

    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=cfg["training_defaults"]["learning_rate"],
        per_device_train_batch_size=cfg["training_defaults"]["batch_size"],
        per_device_eval_batch_size=cfg["training_defaults"]["batch_size"],
        num_train_epochs=cfg["training_defaults"]["epochs"],
        warmup_ratio=cfg["training_defaults"]["warmup_ratio"],
        weight_decay=cfg["training_defaults"]["weight_decay"],
        fp16=cfg["training_defaults"]["fp16"],
        logging_steps=50,
        save_strategy="epoch",
        report_to=[]
    )

    def compute_metrics(_): return {}
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_feats,
        eval_dataset=valid_feats,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    ensure_dir(args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log("Training complete for BERT.")

if __name__ == "__main__":
    main()