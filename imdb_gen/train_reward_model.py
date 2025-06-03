#!/usr/bin/env python3
"""
Train a Llama reward model with BCE loss on preference data.

Input  = prompt + partial_guided_prediction
Target = partial_guided_vs_fully_pref  (0 / 1)

Example usage:
python train_reward_model.py \
  --data_path collected_data/all_train_data.jsonl \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --output_dir ./reward_model
"""
import argparse
import logging

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="JSONL with preference data"
    )
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument("--output_dir", type=str, default="./reward_model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--eval_holdout_frac",
        type=float,
        default=0.02,
        help="Fraction for validation split (0 → no eval)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

    # 1. Load and flatten JSONL manually
    import json
    from datasets import Dataset

    logger.info("Reading JSONL and flattening examples from %s", args.data_path)
    records = []
    with open(args.data_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    texts, labels = [], []
    for rec in records:
        prompt = rec.get("prompt", "")
        preds = rec.get("partial_guided_predictions", [])
        prefs = rec.get("partial_guided_vs_fully_pref", [])
        for pred, p in zip(preds, prefs):
            # skip examples without a valid preference label
            if p is None:
                continue
            texts.append(prompt + "\n" + pred)
            labels.append(float(p))
    logger.info("Total examples after flattening: %d", len(texts))
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    # 2. Tokenizer + padding will work on `dataset`

    # 3. Tokenizer + padding
    logger.info("Loading tokenizer and model %s", args.model_name)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def preprocess(batch):
        enc = tok(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        enc["labels"] = batch["label"]
        return enc

    proc_ds = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["text", "label"],
        desc="Tokenising",
    )

    # Optional train/val split
    if 0 < args.eval_holdout_frac < 1:
        split = proc_ds.train_test_split(
            test_size=args.eval_holdout_frac, seed=args.seed
        )
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = proc_ds, None

    # 4. Model initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=1
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    model.resize_token_embeddings(len(tok))

    class RewardTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.view(-1)
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred: EvalPrediction):
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.reshape(-1)
        labels = eval_pred.label_ids.reshape(-1)
        preds = (logits > 0).astype(int)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    # 5. TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=500 if eval_ds is not None else None,
        load_best_model_at_end=True if eval_ds is not None else False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb",
        fp16=False,
        bf16=torch.cuda.is_available(),
    )

    # 6. Trainer and run
    logger.info("Initialising Trainer")
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
