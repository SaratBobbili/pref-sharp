# Math reasoning experiments

## Requirements
The following packages are needed to run the code:
1. *torch* 2.4.1
2. *transformers* 4.45.2
3. *datasets* 3.0.2
4. *numpy* 1.26.4
5. *tqdm* 4.66.5
6. *wandb* 0.18.5
7. *pandas* 2.2.3
8. *math_verify* 0.7.0

## Usage

To create training data on GSM8K for Llama 3 8B Instruct:
```bash
python collect_training_data.py --start_index 0 --end_index -1 --is_first_round 1 --ref_model_id meta-llama/Meta-Llama-3-8B-Instruct \
--classifier_model_id meta-llama/Llama-3.2-1B-Instruct --classifier_type V --inference_mode bernoulli --loss_type bce --use_bias 0 --data_path dataset/gsm8k_train.jsonl \
--train_eval_save_path dataset/gsm8k_train_eval.json --use_chat_template 1 --eta 0 --temperature 0.8 --top_p 0.9 --max_new_tokens 1024 \
 --dtype bfloat16 --match_fn_type symbolic --output_dir collected_data/llama_3_8b_instruct_gsm8k/
```
This will create a dataset in `collected_data/llama_3_8b_instruct_gsm8k/` that can be used for training. To speed up dataset creation, you can run the command in parallel with different `--start_index` and `--end_index` values.

To combine the training data with the GSM8K training data:
```bash
python combine_training_data.py --data_template_path collected_data/llama_3_8b_instruct_gsm8k/ --data_path dataset/gsm8k_train.jsonl --train_eval_save_path dataset/gsm8k_train_eval.json
```

To train the Q# model on the collected data:
```bash
python train_classifier.py --ref_model_id meta-llama/Meta-Llama-3-8B-Instruct --classifier_model_id meta-llama/Llama-3.2-1B-Instruct \
--original_problems_path dataset/gsm8k_train.jsonl --train_eval_save_path dataset/gsm8k_train_eval.json --init_mode reuse --inference_mode bernoulli \
--loss_type bce --dataset_type gsm8k --data_paths collected_data/llama_3_8b_instruct_gsm8k/all_train_data.jsonl --drop_no_variation 1 --eta 1 --output_dir checkpoints/llama_3_8b_instruct_gsm8k/ --num_epochs 5
```

To evaluate the Q# model by guiding the reference model on the GSM8K test set:
```bash
python eval_ckpt.py --classifier_ckpt_path checkpoints/llama_3_8b_instruct_gsm8k/ckpt_15000/ --eta 10 --data_path dataset/gsm8k_test.jsonl --train_eval_save_path dataset/gsm8k_test_eval.json
```
The `--eta` parameter controls the strength of the guidance. A higher value will result in more guidance.

The above three steps can be run for multiple times to iteratively improve the Q# model.

We also provide examples of collected data and checkpoints for the Q# model trained in [box link](https://cornell.box.com/s/quqoizybla1hnlk0uuyoxrqh2sna2qtc).
