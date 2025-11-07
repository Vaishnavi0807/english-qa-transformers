#!/usr/bin/env bash
set -e

python src/run_subset.py --config config.yaml
python src/train_bert.py --config config.yaml
python src/train_roberta.py --config config.yaml
python src/train_deberta.py --config config.yaml
python src/evaluate.py --model_dir results/checkpoints/bert --config config.yaml
python src/evaluate.py --model_dir results/checkpoints/roberta --config config.yaml
python src/evaluate.py --model_dir results/checkpoints/deberta --config config.yaml