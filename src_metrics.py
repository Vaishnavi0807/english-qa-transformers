import re
import string
from collections import Counter
from bert_score import score as bert_score
import numpy as np
import json
from src.utils import save_json

def normalize_answer(s):
    def lower(text): return text.lower()
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def white_space_fix(text): return ' '.join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_em(a_pred, a_gold):
    return int(normalize_answer(a_pred) == normalize_answer(a_gold))

def compute_f1(a_pred, a_gold):
    pred_tokens = normalize_answer(a_pred).split()
    gold_tokens = normalize_answer(a_gold).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate_predictions(examples, predictions, answerability_meta=None, use_bert_score=False):
    ems, f1s = [], []
    answerable_acc = []
    gold_texts, pred_texts = [], []
    for ex in examples:
        gold_answers = ex["answers"]["text"]
        gold = gold_answers[0] if len(gold_answers) > 0 else ""
        pred = predictions.get(ex["id"], "")
        ems.append(compute_em(pred, gold))
        f1s.append(compute_f1(pred, gold))
        is_impossible = ex["is_impossible"]
        pred_is_empty = (pred.strip() == "")
        answerable_acc.append(int((is_impossible and pred_is_empty) or (not is_impossible and not pred_is_empty)))
        gold_texts.append(gold)
        pred_texts.append(pred)

    metrics = {
        "EM": float(np.mean(ems)),
        "F1": float(np.mean(f1s)),
        "Answerability_Accuracy": float(np.mean(answerable_acc)),
    }
    if use_bert_score:
        P, R, F1b = bert_score(pred_texts, gold_texts, lang="en", rescale_with_baseline=True)
        metrics["BERTScore_P"] = float(P.mean().item())
        metrics["BERTScore_R"] = float(R.mean().item())
        metrics["BERTScore_F1"] = float(F1b.mean().item())
    if answerability_meta:
        metrics["Null_Score_Mean"] = float(np.mean([m["null_score"] for m in answerability_meta.values()]))
    return metrics

def save_metrics(metrics, path):
    save_json(metrics, path)

def threshold_sweep(examples, meta, predictions_raw, gold_answers):
    # meta contains score_diff; we vary threshold
    diffs = [meta[e["id"]]["score_diff"] for e in examples]
    thresholds = sorted(set(diffs))
    best = {"threshold": 0.0, "F1": -1.0, "EM": -1.0}
    for t in thresholds:
        new_preds = {}
        for ex in examples:
            m = meta[ex["id"]]
            if m["score_diff"] < t:
                new_preds[ex["id"]] = ""
            else:
                new_preds[ex["id"]] = predictions_raw[ex["id"]]
        metrics = evaluate_predictions(examples, new_preds)
        if metrics["F1"] > best["F1"]:
            best = {"threshold": t, "F1": metrics["F1"], "EM": metrics["EM"]}
    return best