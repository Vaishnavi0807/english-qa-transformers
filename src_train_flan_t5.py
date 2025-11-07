import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from src.utils import set_seed, load_config, read_json, ensure_dir, log

PROMPT_TEMPLATE = "Answer the question based on the context. If not answerable, output 'unanswerable'.\nContext: {context}\nQuestion: {question}\nAnswer:"

def format_examples(records):
    inputs, targets = [], []
    for r in records:
        prompt = PROMPT_TEMPLATE.format(context=r["context"], question=r["question"])
        if r["is_impossible"] or len(r["answers"]["text"]) == 0:
            target = "unanswerable"
        else:
            target = r["answers"]["text"][0]
        inputs.append(prompt)
        targets.append(target)
    return inputs, targets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train_path", default=None)
    ap.add_argument("--valid_path", default=None)
    ap.add_argument("--output_dir", default="results/checkpoints/flan_t5")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    model_name = cfg["models"]["flan_t5"]
    train_path = args.train_path or cfg["paths"]["subset_train"]
    valid_path = args.valid_path or cfg["paths"]["subset_valid"]
    train_data = read_json(train_path)
    valid_data = read_json(valid_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_inputs, train_targets = format_examples(train_data)
    valid_inputs, valid_targets = format_examples(valid_data)

    train_ds = Dataset.from_dict({"input_text": train_inputs, "labels_text": train_targets})
    valid_ds = Dataset.from_dict({"input_text": valid_inputs, "labels_text": valid_targets})

    def preprocess(examples):
        model_inputs = tokenizer(examples["input_text"], max_length=512, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["labels_text"], max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_enc = train_ds.map(preprocess, batched=True)
    valid_enc = valid_ds.map(preprocess, batched=True)

    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to=[]
    )

    trainer = Trainer(model=model, args=args_tr,
                      train_dataset=train_enc, eval_dataset=valid_enc,
                      tokenizer=tokenizer)
    trainer.train()
    ensure_dir(args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log("Training complete for Flan-T5.")

if __name__ == "__main__":
    main()