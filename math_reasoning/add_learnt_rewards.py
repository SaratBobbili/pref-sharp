#!/usr/bin/env python
import argparse
import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# allow imports of classifier and utils from the math_reasoning root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
math_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, math_root)
# also keep local script directory for any relative imports
sys.path.insert(0, script_dir)
from utils import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Add scalar rewards for partial-guided responses using a trained reward model"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to all_train_pref_data_binary.jsonl",
    )
    parser.add_argument(
        "--reward_model_dir",
        type=str,
        required=True,
        help="Directory of the trained reward model checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for model inference"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to write output JSONL (overwrites input if not set)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length for input sequences",
    )
    args = parser.parse_args()

    # load data
    print(f"Reading data from {args.input_path}")
    data = read_jsonl(args.input_path)

    # load tokenizer and model
    print(f"Loading tokenizer from {args.reward_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load reward model
    print(f"Loading reward model from {args.reward_model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_dir,
        num_labels=1,
    )
    model.to(device).eval()
    if device.type == "cuda":
        model.half()

    # build flat index of (record_idx, item_idx) using training keys
    mapping = []
    for rec_idx, rec in enumerate(data):
        n = len(rec.get("partial_guided_predictions", []))
        for j in range(n):
            mapping.append((rec_idx, j))

    # prepare container for new reward values
    new_rewards = [
        [0.0] * len(rec.get("partial_guided_predictions", [])) for rec in data
    ]

    # batched inference using on-the-fly tokenization
    bs = args.batch_size
    for i in tqdm(range(0, len(mapping), bs), desc="Inferring rewards"):
        batch = mapping[i : i + bs]
        # prepare texts and tokenize
        texts = [
            data[rec_idx]["prompt"]
            + "\n"
            + data[rec_idx]["partial_guided_predictions"][j]
            for rec_idx, j in batch
        ]
        encoding = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attn_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits.float().squeeze(-1)
            vals = logits.cpu().tolist()

        for k, (rec_idx, j) in enumerate(batch):
            new_rewards[rec_idx][j] = vals[k]

    # assign new rewards for partially guided predictions
    for idx, rec in enumerate(data):
        rec["partial_guided_predictions_reward"] = new_rewards[idx]

    # compute rewards for fully guided predictions
    mapping_fully = []
    for rec_idx, rec in enumerate(data):
        n_full = len(rec.get("fully_guided_predictions", []))
        for j in range(n_full):
            mapping_fully.append((rec_idx, j))

    new_rewards_fully = [
        [0.0] * len(rec.get("fully_guided_predictions", [])) for rec in data
    ]

    for i in tqdm(
        range(0, len(mapping_fully), bs), desc="Inferring rewards (fully-guided)"
    ):
        batch = mapping_fully[i : i + bs]
        # prepare texts and tokenize for fully guided predictions
        texts_full = [
            data[rec_idx]["prompt"]
            + "\n"
            + data[rec_idx]["fully_guided_predictions"][j]
            for rec_idx, j in batch
        ]
        encoding_full = tokenizer(
            texts_full,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids_full = encoding_full["input_ids"].to(device)
        attn_mask_full = encoding_full["attention_mask"].to(device)

        with torch.no_grad():
            outputs_full = model(
                input_ids=input_ids_full, attention_mask=attn_mask_full
            )
            logits_full = outputs_full.logits.float().squeeze(-1)
            vals_full = logits_full.cpu().tolist()

        for k, (rec_idx, j) in enumerate(batch):
            new_rewards_fully[rec_idx][j] = vals_full[k]

    # assign new rewards for fully guided predictions
    for idx, rec in enumerate(data):
        rec["fully_guided_predictions_reward"] = new_rewards_fully[idx]

    # write out
    out_path = args.output_path or args.input_path
    print(f"Writing updated data to {out_path}")
    write_jsonl(data, out_path)


if __name__ == "__main__":
    main()
