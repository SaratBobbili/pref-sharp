import argparse
from collections import defaultdict
from contextlib import nullcontext
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb
import json
import time
import os

from tokenizing import get_tokenizer
from utils.training_utils import get_lr, get_run_name, AverageMeter, set_seed
from data import get_dataset
import evaluate
from models import get_model




# Parse arguments
parser = argparse.ArgumentParser(description="Next-token prediction")
# Data
parser.add_argument(
        "--piref_ckpt", type=str, default='', help="Path to piref model."
    )
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
        "--seed", type=int, default=42, help="Random seed",
    )
parser.add_argument(
    "--dropout", type=float, default=0.1, help="Dropout rate",
    )
parser.add_argument(
    "--mlp_expansion_factor", type=int, default=4, help="MLP Expansion factor",
    )
parser.add_argument(
    '--compile', action=argparse.BooleanOptionalAction, default=False, help='Whether to compile the model'
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

dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
decay_lr = True
args.use_flash = True if device == 'cuda' else False

run_name = get_run_name(args)
assert args.model.startswith('gpt2'), "Only gpt2 models are supported for now."
run_name = run_name + '_bs_' + str(args.batch_size) + '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + '_' + args.model
path = 'checkpoints/classifier/' + run_name + '.pt'
if not os.path.exists('checkpoints/classifier'):
    os.makedirs('checkpoints/classifier')
# Get tokenizer and de-tokenizer
tokenizer = get_tokenizer(args)

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
model = get_model(args, is_classifier=True)

if args.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

model.to(device)
model.train()
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

def collate_fn(batch):
    return {k: torch.tensor([d[k] for d in batch]) for k in batch[0].keys()}

classifier_train_loader = DataLoader(classifier_train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, pin_memory=True, num_workers=6)
classifier_test_loader = DataLoader(classifier_test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=6)

# warmup 10% of the iters, and decay to 10% of max_lr.
optimizer = model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=args.lr, betas=(0.9, 0.999), device_type=device)
max_iters = len(classifier_train_loader) * args.epochs
lr_decay_iters = max_iters
warmup_iters = max(100, 0.1 * max_iters)
min_lr = 0.1 * args.lr

# Setup wandb logging
if wandb_log:
    wandb.init(project='next-token-failures', entity=wandb_entity, config=args.__dict__, tags=['classifier'])
    wandb.run.name = run_name

num_iters = 0
for ep in range(1, args.epochs+1):
    train_bar = tqdm(classifier_train_loader)
    avg_meter_map = defaultdict(AverageMeter)
    total_loss, total_acc = AverageMeter(), AverageMeter()
    start_time = time.time()
    num_tokens = 0
    for d in train_bar:
        # determine and set the learning rate for this iteration
        lr = get_lr(num_iters, args.lr, warmup_iters, lr_decay_iters, min_lr) if decay_lr else args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        input_ids = torch.cat([d['y_pred_reject'], d['y_pred_chosen']], dim=0).to(device, non_blocking=True)
        labels = torch.cat([d['y_pred_reject_completely_correct'], d['y_pred_chosen_completely_correct']], dim=0).to(device, non_blocking=True)
        target = labels.unsqueeze(1).expand(-1, num_target_tokens).float()
        with ctx:
            logits = model(input_ids, num_target_tokens)
            token_loss = torch.nn.functional.binary_cross_entropy_with_logits(input=logits, target=target, reduction='none')
            avg_token_loss = token_loss.mean()

        avg_token_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        num_iters += 1

        with torch.no_grad():
            for idx in range(1, num_target_tokens):
                batch_size = input_ids.shape[0]
                if idx % 2 == 1:
                    continue
                i = idx // 2  # because separated by comma

                avg_meter_map[f'token_{i}/bce_loss'].update(token_loss[:, idx].mean().item(), batch_size)
                first_token_correct = torch.abs(labels - 1) < 1e-3
                n_correct = first_token_correct.sum().item()
                n_incorrect = batch_size - n_correct
                avg_meter_map[f'token_{i}/cond_1st_correct/pred'].update(torch.sigmoid(logits[:, idx])[first_token_correct].mean().item(), n_correct)
                avg_meter_map[f'token_{i}/cond_1st_correct/target'].update(labels[first_token_correct].mean().item(), n_correct)
                avg_meter_map[f'token_{i}/cond_1st_correct/logit'].update(logits[:, idx][first_token_correct].mean().item(), n_correct)
                avg_meter_map[f'token_{i}/cond_1st_incorrect/pred'].update(torch.sigmoid(logits[:, idx])[~first_token_correct].mean().item(), n_incorrect)
                avg_meter_map[f'token_{i}/cond_1st_incorrect/target'].update(labels[~first_token_correct].mean().item(), n_incorrect)
                avg_meter_map[f'token_{i}/cond_1st_incorrect/logit'].update(logits[:, idx][~first_token_correct].mean().item(), n_incorrect)

        total_loss.update(avg_token_loss.item(), input_ids.shape[0] * num_target_tokens)
        avg_meter_map['grad_norm'].update(grad_norm, 1)
        num_tokens += input_ids.shape[0] * input_ids.shape[1]
        toks_per_sec = round(num_tokens / (time.time() - start_time))
        train_bar.set_description(
            f'Epoch: [{ep}/{args.epochs}] BCE: {avg_token_loss.item():.4f}, Loss: {total_loss.get():.4f}, Toks per sec: {toks_per_sec}'
        )

    cur_time = time.time()
    toks_per_sec = round(num_tokens / (cur_time - start_time))
    print(f"Epoch {ep} took {cur_time - start_time} seconds. {toks_per_sec} toks/sec")
    if wandb_log:
        wandb.log({
            'toks_per_sec': toks_per_sec,
            'loss': total_loss.get(),
            'lr': lr,
            **{k: v.get() for k, v in avg_meter_map.items()},
        }, step=ep)

    # evaluate the loss on train/val sets and write checkpoints
    # perform evaluation with
    if ep % args.eval_every == 0:
        # first evaluate bce loss on test set
        results = evaluate.evaluate_bce_loss(args.model, model, classifier_test_loader, num_target_tokens, ctx, prefix="test/")
        if wandb_log:
            wandb.log(results, step=ep)

    if ep % args.save_every == 0 and ep > 0:
        sd = model._orig_mod.state_dict() if args.compile else model.state_dict()
        torch.save(sd, path + "_epoch_" + str(ep))