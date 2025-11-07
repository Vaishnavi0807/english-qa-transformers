import argparse, json, os
from src.utils import read_json, ensure_dir, log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def categorize(example, pred):
    gold_answers = example["answers"]["text"]
    gold = gold_answers[0] if len(gold_answers) > 0 else ""
    is_unanswerable = example["is_impossible"]
    if is_unanswerable and pred == "":
        return "Correct_NoAnswer"
    if is_unanswerable and pred != "":
        return "False_Positive_NonNull"
    if not is_unanswerable and pred == "":
        return "False_Negative_Null"
    # For answerable:
    if pred.strip() == gold.strip():
        return "Exact"
    # Partial overlap:
    gold_tokens = set(gold.lower().split())
    pred_tokens = set(pred.lower().split())
    if len(gold_tokens & pred_tokens) > 0:
        return "Partial_Overlap"
    return "Wrong_Entity"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    preds = read_json(args.predictions)
    truth = read_json(args.truth)
    ensure_dir(args.out_dir)

    rows = []
    for ex in truth:
        pid = ex["id"]
        pred = preds.get(pid, "")
        cat = categorize(ex, pred)
        rows.append({
            "id": pid,
            "question": ex["question"],
            "gold": ex["answers"]["text"][0] if len(ex["answers"]["text"]) > 0 else "",
            "pred": pred,
            "is_unanswerable": ex["is_impossible"],
            "category": cat
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "error_analysis.csv"), index=False)

    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x="category")
    plt.xticks(rotation=30)
    plt.title("Error Category Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "error_distribution.png"))

    log("Error analysis complete.")

if __name__ == "__main__":
    main()