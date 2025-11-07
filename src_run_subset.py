import argparse
from datasets import load_dataset
import numpy as np
from src.utils import set_seed, load_config, save_json, log

def create_subset(train, valid, cfg, seed):
    set_seed(seed)
    is_imp = np.array(train["is_impossible"])
    ans_idx = np.where(~is_imp)[0]
    unans_idx = np.where(is_imp)[0]

    def sample(arr, n): return list(np.random.choice(arr, size=n, replace=False))
    ans_sel = sample(ans_idx, cfg["subset"]["train_size_answerable"])
    unans_sel = sample(unans_idx, cfg["subset"]["train_size_unanswerable"])
    subset_train = train.select(ans_sel + unans_sel)

    if cfg["subset"]["valid_size"] < len(valid):
        valid_subset = valid.select(range(cfg["subset"]["valid_size"]))
    else:
        valid_subset = valid

    train_records = []
    for ex in subset_train:
        train_records.append({
            "id": ex["id"],
            "context": ex["context"],
            "question": ex["question"],
            "answers": ex["answers"],
            "is_impossible": ex["is_impossible"]
        })
    valid_records = []
    for ex in valid_subset:
        valid_records.append({
            "id": ex["id"],
            "context": ex["context"],
            "question": ex["question"],
            "answers": ex["answers"],
            "is_impossible": ex["is_impossible"]
        })

    return train_records, valid_records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output_path", default=None)
    ap.add_argument("--valid_path", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    dataset = load_dataset("squad_v2")
    train, valid = dataset["train"], dataset["validation"]
    train_rec, valid_rec = create_subset(train, valid, cfg, seed)

    out_train = args.output_path or cfg["paths"]["subset_train"]
    out_valid = args.valid_path or cfg["paths"]["subset_valid"]
    save_json(train_rec, out_train)
    save_json(valid_rec, out_valid)
    log(f"Saved subset train={len(train_rec)} valid={len(valid_rec)}")

if __name__ == "__main__":
    main()