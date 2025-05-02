#!/bin/bash

TOP_K=10
python evaluate_ckpts.py --model gpt2 --dataset graph --deg 2 --path 5 --num_nodes 50 \
    --batch_size 256 --seed 1337 --top_k $TOP_K --temperature 1.0 \
    --pretrained_ckpt checkpoints/pretrain/deg_2_path_5.pt \
    --classifier_ckpt checkpoints/classifier/deg_2_path_5.pt \
    --reinforce_ckpt checkpoints/reinforce/deg_2_path_5.pt \
    --dpo_ckpt checkpoints/dpo/deg_2_path_5.pt \
    --rpo_ckpt checkpoints/rpo/deg_2_path_5.pt

python evaluate_ckpts.py --model gpt2 --dataset graph --deg 5 --path 5 --num_nodes 50 \
    --batch_size 256 --seed 1337 --top_k $TOP_K --temperature 1.0 \
    --pretrained_ckpt checkpoints/pretrain/deg_5_path_5.pt \
    --classifier_ckpt checkpoints/classifier/deg_5_path_5.pt \
    --reinforce_ckpt checkpoints/reinforce/deg_5_path_5.pt \
    --dpo_ckpt checkpoints/dpo/deg_5_path_5.pt \
    --rpo_ckpt checkpoints/rpo/deg_5_path_5.pt

python evaluate_ckpts.py --model gpt2-medium --dataset graph --deg 3 --path 8 --num_nodes 50 \
    --batch_size 256 --seed 1337 --top_k $TOP_K --temperature 1.0 \
    --pretrained_ckpt checkpoints/pretrain/deg_3_path_8.pt \
    --classifier_ckpt checkpoints/classifier/deg_3_path_8.pt \
    --reinforce_ckpt checkpoints/reinforce/deg_3_path_8.pt \
    --dpo_ckpt checkpoints/dpo/deg_3_path_8.pt \
    --rpo_ckpt checkpoints/rpo/deg_3_path_8.pt
