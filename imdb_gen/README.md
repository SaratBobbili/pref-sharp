# IMDb Preference Generation Experiments

This directory contains scripts to collect, combine, train, and evaluate preference‐guided IMDb text generation models.

## Requirements

Install the following packages:

1. torch 2.4.1
2. transformers 4.45.2
3. datasets 3.0.2
4. numpy 1.26.4
5. tqdm 4.66.5
6. wandb 0.18.5
7. pandas 2.2.3

## Pipeline Overview

1. Collect training data using classifier‐guided generation
2. Combine per‐sample JSON outputs into a single JSONL
3. Train a PITA/Q#/Q#HF classifier
4. Evaluate guided generation on 5k IMDb test prompts

## Usage

### 1. Collect training data

Generate preference‐guided IMDb examples (positive/negative reward) by sampling with classifier guidance:

```bash
python collect_training_data.py \
  --start_index 0 --end_index -1 \
  --is_first_round 1 \
  --ref_model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --classifier_model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --classifier_type V \
  --inference_mode expectation \
  --loss_type bce \
  --data_path stanfordnlp/imdb \
  --output_dir collected_data/tinyllama_imdb
```

### 2. Combine training data

Aggregate per‐sample JSONs into a single JSONL:

```bash
python combine_training_data.py \
  --data_template_path collected_data/tinyllama_imdb
```

Output: `collected_data/all_train_data.jsonl`

### 3. Train PITA classifier

Train the Custom Llama classifier (PITA) on IMDb data:

```bash
python train_classifier.py \
  --ref_model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --classifier_model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --init_mode reuse \
  --inference_mode expectation \
  --loss_type bce \
  --use_bias 1 \
  --dataset_type imdb \
  --data_paths collected_data/all_train_data.jsonl \
  --batch_size 8 --num_epochs 5 \
  --lr 2e-5 --warmup_step -1 \
  --eta 1 --output_dir checkpoints/tinyllama_imdb
```



### 4. Evaluate guided generation

Run a single‐pass guided generation on 5k IMDb test prompts, compute sentiment and trajectory KL:

```bash
python eval_ckpt.py \
  --classifier_ckpt_path checkpoints/tinyllama_imdb/ckpt_25000/ \
  --data_path stanfordnlp/imdb \
  --batch_size 8 --max_new_tokens 60 \
  --temperature 1.0 --top_p 0.9 \
  --num_samples 1 \
  --output_dir results_pita
```

Outputs:

- `imdb_eval_results.jsonl`: per‐prompt generation, sentiment_score, traj_kl
- `imdb_eval_stats.json`: average sentiment and average trajectory KL


### 5. Train Q# classifier

Switch reward key to Q# in line 352 of `train_classifier.py` and run again:

```bash
python train_classifier.py \
  --ref_model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --classifier_model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --init_mode reuse \
  --inference_mode expectation \
  --loss_type bce \
  --use_bias 1 \
  --dataset_type imdb \
  --data_paths collected_data/all_train_data.jsonl \
  --batch_size 8 --num_epochs 5 \
  --lr 2e-5 --warmup_step -1 \
  --eta 1 --output_dir checkpoints/tinyllama_imdb_qsharp
```


### 6. Evaluate Q# classifier

```bash
python eval_ckpt.py \
  --classifier_ckpt_path checkpoints/tinyllama_imdb_qsharp/ckpt_25000/ \
  --data_path stanfordnlp/imdb \
  --batch_size 8 --max_new_tokens 60 \
  --temperature 1.0 --top_p 0.9 \
```

### 7. Train Q#HF classifier

Train reward function from preference data:

```bash
python train_reward_model.py \
  --data_path collected_data/all_train_data.jsonl \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --output_dir ./reward_model
```


Add Q#HF rewards to `collected_data/all_train_data.jsonl`:

```bash
python add_learnt_rewards.py \
  --input_path collected_data/all_train_data.jsonl \
  --reward_model_path ./reward_model \
  --output_path collected_data/all_train_data_with_learnt_rewards.jsonl
```

Switch reward key to Q#HF in line 353 of `train_classifier.py` and run again:

```bash
python train_classifier.py \
  --ref_model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --classifier_model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --init_mode reuse \
  --inference_mode expectation \
  --loss_type mle \
  --use_bias 1 \
  --dataset_type imdb \
  --data_paths collected_data/all_train_data.jsonl \
  --batch_size 8 --num_epochs 5 \
  --lr 2e-5 --warmup_step -1 \
  --eta 1 --output_dir checkpoints/tinyllama_imdb_qsharphf
```

### 8. Evaluate Q#HF classifier

```bash
python eval_ckpt.py \
  --classifier_ckpt_path checkpoints/tinyllama_imdb_qsharphf/ckpt_25000/ \
  --data_path stanfordnlp/imdb \
  --batch_size 8 --max_new_tokens 60 \
  --temperature 1.0 --top_p 0.9 \
```
