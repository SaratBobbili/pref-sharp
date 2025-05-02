import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from contextlib import nullcontext
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb
import json
import time
from collections import defaultdict
from tokenizing import get_tokenizer
from utils.training_utils import get_lr, get_run_name, AverageMeter, set_seed
from evaluate import evaluate, evaluate_forced
from models import get_model
from data import get_dataset

# Parse arguments
parser = argparse.ArgumentParser(description="Next-token prediction")
# Data
parser.add_argument(
        "--model", type=str, default='gpt', help="Learning rate",
    )
parser.add_argument(
        "--n_layer", type=int, default=6, help="Number of layers",
    )
parser.add_argument(
        "--n_embd", type=int, default=240, help="Embedding size",
    )
parser.add_argument(
        "--n_head", type=int, default=6, help="Number of heads",
    )
parser.add_argument(
    "--dataset", default='graph', type=str, help="Choice of dataset"
    )
parser.add_argument(
    "--n_test", default=10000, type=int, help="Number of test samples"
    )
parser.add_argument(
    "--num_nodes", default=50, type=int, help="Number of node values in graph"
    )
parser.add_argument(
    "--deg", default=2, type=int, help="Degree of starting node"
    )
parser.add_argument(
    "--path_len", default=5, type=int, help="Path length in star graph"
    )
parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size",
    )
parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate",
    )
parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Strength of weight decay",
    )
parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs",
    )
parser.add_argument(
        "--save_every", type=int, default=5000, help="Interval (in steps) at which to save model",
    )
parser.add_argument(
        "--teacherless", action=argparse.BooleanOptionalAction, default=False, help="Standard or teacherless training",
    )
parser.add_argument(
        "--reverse", action=argparse.BooleanOptionalAction, default=False, help="Standard format or reverse targets",
    )
parser.add_argument(
        "--eval_every", type=int, default=5000, help="Interval (in steps) to evaluate the model on test",
    )
parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=False, help="Whether to use wandb",
    )
parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Wandb username",
    )
parser.add_argument(
        "--piref_ckpt", type=str, required=True, help="piref checkpoint path",
    )
parser.add_argument(
        "--seed", type=int, default=42, help="Random seed",
    )
parser.add_argument(
        "--beta", type=float, default=0.1, help="Beta for DPO",
    )
parser.add_argument(
        '--compile', action=argparse.BooleanOptionalAction, default=False, help='Whether to compile the model'
    )
parser.add_argument(
        '--use_rpo', type=int, default=0, help='Whether to use RPO'
    )

args = parser.parse_args()
set_seed(args.seed)

# System stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb_entity = args.wandb_entity
wandb_log = args.use_wandb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Model stuff
top_k = 1

dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
decay_lr = True
args.use_flash = True if device == 'cuda' else False

run_name = get_run_name(args)
assert args.model.startswith('gpt2'), "Only gpt2 models are supported for now."
run_name = run_name + '_bs_' + str(args.batch_size) + '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + '_' + args.model
if args.use_rpo:
    path = 'checkpoints/rpo/' + run_name + '.pt'
else:
    path = 'checkpoints/dpo/' + run_name + '.pt'
if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))
# Get tokenizer and de-tokenizer
tokenizer = get_tokenizer(args)
# load original datasets
pretrain_data_path = 'data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + str(args.num_nodes) + '_200k_pretrain.txt'
posttrain_data_path = 'data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + str(args.num_nodes) + '_200k_posttrain.txt'
test_data_path = 'data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + str(args.num_nodes) + '_20k_test.txt'
pretrain_data = get_dataset(args, tokenizer, device, pretrain_data_path, n_sample=20000)  # only for train_eval purposes
posttrain_data = get_dataset(args, tokenizer, device, posttrain_data_path, n_sample=20000)  # only for train_eval purposes
test_data = get_dataset(args, tokenizer, device, test_data_path)
pretrain_loader = DataLoader(pretrain_data, batch_size=args.batch_size, shuffle=False)
posttrain_loader = DataLoader(posttrain_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# also load pairwise data for dpo training
def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(l) for l in f]

classifier_test_data_path = f'data/datasets/gpt2_graphs_rollouts/deg_{args.deg}_path_{args.path_len}_nodes_{args.num_nodes}_20k_test.jsonl'
classifier_test_data = load_jsonl(classifier_test_data_path)
classifier_train_data_path = f'data/datasets/gpt2_graphs_rollouts/deg_{args.deg}_path_{args.path_len}_nodes_{args.num_nodes}_200k_posttrain.jsonl'
classifier_train_data = load_jsonl(classifier_train_data_path)
num_prefix_tokens = len(classifier_train_data[0]['x'])
num_target_tokens = len(classifier_train_data[0]['y_pred_chosen']) - num_prefix_tokens

# sometimes generated data may be invalid...
def filter_data(data):
    filtered_data = []
    invalid = 0
    for datum in data:
        if len(datum['y_pred_reject']) == (num_prefix_tokens + num_target_tokens):
            filtered_data.append(datum)
        else:
            invalid += 1
    print(f"Filtered out {invalid} invalid samples.")
    return filtered_data

classifier_train_data = filter_data(classifier_train_data)
classifier_test_data = filter_data(classifier_test_data)

args.block_size = num_prefix_tokens + num_target_tokens
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None
del tokenizer

model = get_model(args)
piref_model = get_model(args)
def load_from_ckpt(model, ckpt_path):
    sd = torch.load(ckpt_path)
    # TODO: remove this line when we save the model without the "_orig_mod." prefix
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        new_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        sd = new_sd
    model.load_state_dict(sd)
    print(f"Successfully loaded from {ckpt_path}!")

load_from_ckpt(model, args.piref_ckpt)
load_from_ckpt(piref_model, args.piref_ckpt)
if args.compile:
    print("compiling the model... (takes a ~minute)")
    piref_model = torch.compile(piref_model)
    model = torch.compile(model)

model.to(device)
model.train()
piref_model.to(device)
piref_model.eval()

ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

def collate_fn(batch):
    return {k: torch.tensor([d[k] for d in batch]) for k in batch[0].keys()}

classifier_train_loader = DataLoader(classifier_train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, pin_memory=True, num_workers=6)
classifier_test_loader = DataLoader(classifier_test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

# warmup 10% of the iters, and decay to 10% of max_lr.
optimizer = model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=args.lr, betas=(0.9, 0.999), device_type=device)
max_iters = len(classifier_train_loader) * args.epochs
lr_decay_iters = max_iters
warmup_iters = max(100, 0.1 * max_iters)
min_lr = 0.1 * args.lr

# Setup wandb logging
if wandb_log:
    wandb.init(project='next-token-failures-dpo', entity=wandb_entity, config=args.__dict__, tags=['dpo'])
    wandb.run.name = run_name

num_iters = 0
for ep in range(1, args.epochs+1):
    if args.save_every > 0 and ep % args.save_every == 0 and ep > 0:
        sd = model._orig_mod.state_dict() if args.compile else model.state_dict()
        torch.save(sd, path + "_epoch_" + str(ep))

    train_bar = tqdm(classifier_train_loader)
    total_loss = AverageMeter()
    avg_meter_map = defaultdict(AverageMeter)
    start_time = time.time()
    num_tokens = 0
    for d in train_bar:
        lr = get_lr(num_iters, args.lr, warmup_iters, lr_decay_iters, min_lr) if decay_lr else args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        bs = d['x'].shape[0]
        y_pred_chosen = d['y_pred_chosen'].to(device, non_blocking=True)
        y_pred_reject = d['y_pred_reject'].to(device, non_blocking=True)
        y_pred_chosen_targets = y_pred_chosen[:, 1:].clone()
        y_pred_chosen_targets[:, :-num_target_tokens] = -1
        y_pred_reject_targets = y_pred_reject[:, 1:].clone()
        y_pred_reject_targets[:, :-num_target_tokens] = -1

        stacked_inputs = torch.cat([y_pred_chosen, y_pred_reject], dim=0)
        stacked_targets = torch.cat([y_pred_chosen_targets, y_pred_reject_targets], dim=0)

        with ctx:
            with torch.no_grad():
                _, neg_piref_stacked_logps, _ = piref_model(stacked_inputs[:, :-1], stacked_targets, reduce_loss=False)
            _, neg_model_stacked_logps, _ = model(stacked_inputs[:, :-1], stacked_targets, reduce_loss=False)

            piref_stacked_logps = -1 * neg_piref_stacked_logps[:, -num_target_tokens:].float().sum(-1)  # [2bs]
            model_stacked_logps = -1 * neg_model_stacked_logps[:, -num_target_tokens:].float().sum(-1)

            piref_chosen_logps = piref_stacked_logps[:bs]  # [bs]
            piref_reject_logps = piref_stacked_logps[bs:]
            model_chosen_logps = model_stacked_logps[:bs]
            model_reject_logps = model_stacked_logps[bs:]

            delta_piref = piref_chosen_logps - piref_reject_logps
            delta_model = model_chosen_logps - model_reject_logps
            delta = delta_model - delta_piref
            loss = -torch.log(torch.sigmoid(args.beta * delta)).mean()
            if args.use_rpo:
                loss -= model_chosen_logps.mean()

        total_loss.update(loss.item(), bs)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        num_iters += 1
        avg_meter_map['grad_norm'].update(grad_norm, 1)
        num_tokens += bs * y_pred_chosen.shape[1]
        toks_per_sec = round(num_tokens / (time.time() - start_time))
        train_bar.set_description(
            f'Epoch: [{ep}/{args.epochs}] Loss: {total_loss.get():.4f} Toks per sec: {toks_per_sec}'
        )

        with torch.no_grad():
            avg_meter_map['piref_chosen_logps'].update(piref_chosen_logps.mean().item(), bs)
            avg_meter_map['piref_reject_logps'].update(piref_reject_logps.mean().item(), bs)
            avg_meter_map['model_chosen_logps'].update(model_chosen_logps.mean().item(), bs)
            avg_meter_map['model_reject_logps'].update(model_reject_logps.mean().item(), bs)
            avg_meter_map['delta_piref'].update(delta_piref.mean().item(), bs)
            avg_meter_map['delta_model'].update(delta_model.mean().item(), bs)
            avg_meter_map['delta'].update(delta.mean().item(), bs)

    cur_time = time.time()
    toks_per_sec = round(num_tokens / (cur_time - start_time))
    print(f"Epoch {ep} took {cur_time - start_time:.2f} seconds. {toks_per_sec} toks/sec")
    if wandb_log:
        wandb.log({
            'toks_per_sec': toks_per_sec,
            'loss': total_loss.get(),
            'lr': lr,
            **{k: v.get() for k, v in avg_meter_map.items()},
        }, step=ep)

    if args.eval_every > 0 and ep % args.eval_every == 0:
        # Generate sequences and check accuracies
        results = {}
        results = evaluate(model, pretrain_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='pretrain')
        results = evaluate_forced(model, pretrain_loader, ctx=ctx, results=results, mode='pretrain')
        results = evaluate(model, posttrain_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='posttrain')
        results = evaluate_forced(model, posttrain_loader, ctx=ctx, results=results, mode='posttrain')
        results = evaluate(model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='test')
        results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='test')
        if wandb_log:
            wandb.log(results, step=ep)

    torch.cuda.empty_cache()