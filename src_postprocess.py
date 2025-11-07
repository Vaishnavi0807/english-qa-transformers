import collections
import numpy as np

def squad_postprocess(examples, features, raw_predictions, n_best_size=20,
                      max_answer_length=30, tokenizer=None, null_score_diff_threshold=0.0):
    """
    examples: original dataset list of dicts
    features: tokenized feature dataset (list of dicts with offset_mapping, input_ids, etc.)
    raw_predictions: (start_logits, end_logits)
    Returns: predictions dict {id: best_answer}, and per-example metadata.
    """
    start_logits, end_logits = raw_predictions
    example_id_to_features = collections.defaultdict(list)
    for i, feat in enumerate(features):
        example_id_to_features[feat["example_id"]].append(i)

    predictions = {}
    metadata = {}
    for example in examples:
        example_id = example["id"]
        feature_indices = example_id_to_features[example_id]
        valid_answers = []
        min_null_score = None
        for idx in feature_indices:
            sl = start_logits[idx]
            el = end_logits[idx]
            offsets = features[idx]["offset_mapping"]
            cls_index = features[idx]["input_ids"].index(tokenizer.cls_token_id) if tokenizer.cls_token_id in features[idx]["input_ids"] else 0
            null_score = sl[cls_index] + el[cls_index]
            if min_null_score is None or null_score < min_null_score:
                min_null_score = null_score
                null_answer = {"text": "", "score": null_score}

            start_indices = np.argsort(sl)[-n_best_size:]
            end_indices = np.argsort(el)[-n_best_size:]
            for s in start_indices:
                for e in end_indices:
                    if s == cls_index and e == cls_index:
                        continue
                    if e < s or (e - s + 1) > max_answer_length:
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    start_char, end_char = offsets[s][0], offsets[e][1]
                    answer_text = example["context"][start_char:end_char]
                    score = sl[s] + el[e]
                    valid_answers.append({"text": answer_text, "score": score})

        if len(valid_answers) == 0:
            best_non_null = {"text": "", "score": -1e9}
        else:
            best_non_null = max(valid_answers, key=lambda x: x["score"])

        score_diff = best_non_null["score"] - null_answer["score"]
        if score_diff < null_score_diff_threshold:
            final_answer = ""
            chosen = null_answer
        else:
            final_answer = best_non_null["text"]
            chosen = best_non_null

        predictions[example_id] = final_answer
        metadata[example_id] = {
            "null_score": null_answer["score"],
            "best_non_null_score": best_non_null["score"],
            "score_diff": score_diff,
            "final": final_answer,
            "chosen_type": "null" if final_answer == "" else "non_null"
        }
    return predictions, metadata