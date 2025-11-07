import argparse, json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from src.postprocess import squad_postprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--context_file", required=True)
    ap.add_argument("--threshold", type=float, default=0.0)
    args = ap.parse_args()

    with open(args.context_file, "r") as f:
        context = f.read()

    example = {
        "id": "demo-1",
        "context": context,
        "question": args.question,
        "answers": {"text": [""], "answer_start": []},
        "is_impossible": False
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    model.eval()

    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = ["demo-1"] * len(sample_mapping)

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

    with torch.no_grad():
        out = model(
            input_ids=torch.tensor(tokenized["input_ids"]),
            attention_mask=torch.tensor(tokenized["attention_mask"])
        )
    start_logits = out.start_logits.cpu().numpy()
    end_logits = out.end_logits.cpu().numpy()
    features = []
    for i in range(len(tokenized["input_ids"])):
        features.append({
            "example_id": tokenized["example_id"][i],
            "offset_mapping": tokenized["offset_mapping"][i],
            "input_ids": tokenized["input_ids"][i]
        })

    preds, meta = squad_postprocess(
        [example], features, (start_logits, end_logits),
        n_best_size=20, max_answer_length=30,
        tokenizer=tokenizer,
        null_score_diff_threshold=args.threshold
    )
    answer = preds["demo-1"]
    print(json.dumps({
        "question": args.question,
        "answer": answer,
        "meta": meta["demo-1"]
    }, indent=2))

if __name__ == "__main__":
    main()