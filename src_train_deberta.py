import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from src.utils import set_seed, load_config, read_json, ensure_dir, log
from src.train_bert import prepare_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train_path", default=None)
    ap.add_argument("--valid_path", default=None)
    ap.add_argument("--output_dir", default="results/checkpoints/deberta")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    model_name = cfg["models"]["deberta"]

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

    lr = 2e-5  # Slightly different recommended LR for DeBERTa
    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        num_train_epochs=cfg["training_defaults"]["epochs"],
        warmup_ratio=cfg["training_defaults"]["warmup_ratio"],
        weight_decay=cfg["training_defaults"]["weight_decay"],
        fp16=cfg["training_defaults"]["fp16"],
        logging_steps=50,
        save_strategy="epoch",
        report_to=[]
    )

    trainer = Trainer(model=model, args=args_tr,
                      train_dataset=train_feats, eval_dataset=valid_feats,
                      tokenizer=tokenizer)
    trainer.train()
    ensure_dir(args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log("Training complete for DeBERTa.")

if __name__ == "__main__":
    main()