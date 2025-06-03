#!/usr/bin/env python3
"""
Add learned reward scores to each entry in a JSONL file using a trained reward model.

For each sample:
  - fully_guided_predictions ⇒ fully_guided_learnt_reward
  - partial_guided_predictions ⇒ partial_guided_learnt_reward

Example usage:
python add_learnt_rewards.py \
  --input_path collected_data/all_train_data.jsonl \
  --reward_model_path ./reward_model \
  --output_path collected_data/all_train_data_with_learnt_rewards.jsonl
"""
import argparse
import os
import sys
import torch
from tqdm import tqdm

# allow imports from this directory
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir)
from utils import read_jsonl, write_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to JSONL file containing samples to score",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        required=True,
        help="Directory or model identifier of the trained reward model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output JSONL path for samples with added learned rewards",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max tokens for tokenizer truncation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for scoring predictions"
    )
    return parser.parse_args()


def score_texts(texts, tokenizer, model, device, max_length, batch_size):
    all_scores = []
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Scoring batches", leave=False
    ):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().tolist()
        all_scores.extend(scores)
    return all_scores


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading reward model from {args.reward_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path
    ).to(device)
    model.eval()

    print(f"Reading input samples from {args.input_path}")
    data = read_jsonl(args.input_path)
    updated = []
    for rec in tqdm(data, desc="Samples"):
        prompt = rec.get("prompt", "")
        # Score fully guided predictions
        fg = rec.get("fully_guided_predictions", [])
        if fg:
            texts = [prompt + "\n" + p for p in fg]
            rec["fully_guided_learnt_reward"] = score_texts(
                texts, tokenizer, model, device, args.max_length, args.batch_size
            )
        else:
            rec["fully_guided_learnt_reward"] = []
        # Score partial guided predictions
        pg = rec.get("partial_guided_predictions", [])
        if pg:
            texts = [prompt + "\n" + p for p in pg]
            rec["partial_guided_learnt_reward"] = score_texts(
                texts, tokenizer, model, device, args.max_length, args.batch_size
            )
        else:
            rec["partial_guided_learnt_reward"] = []
        updated.append(rec)

    print(f"Writing updated samples to {args.output_path}")
    write_jsonl(updated, args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
