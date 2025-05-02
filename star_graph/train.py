import time
import argparse
from contextlib import nullcontext
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb
import os

from tokenizing import get_tokenizer
from utils.training_utils import get_lr, get_run_name, AverageMeter, set_seed
from data import get_dataset
from evaluate import evaluate, evaluate_forced
from models import get_model


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
        "--eval_train", action=argparse.BooleanOptionalAction, default=False, help="Eval for training set",
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
        "--dtype", type=str, default="bfloat16", help="data type",
    )
parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate",
    )
parser.add_argument(
    "--mlp_expansion_factor", type=int, default=4, help="MLP Expansion factor",
    )
parser.add_argument(
    '--n_train', type=int, default=-1, help='Number of training samples'
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

dtype = args.dtype
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
beta1 = 0.9
beta2 = 0.999
decay_lr = True
args.compile = True if device == 'cuda' else False
args.use_flash = True if device == 'cuda' else False

run_name = get_run_name(args)
assert args.model.startswith('gpt2'), "Only GPT2 models are supported"
run_name = run_name + '_bs_' + str(args.batch_size) + '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + '_n_' + str(args.n_train) + '_' + args.model
path = 'checkpoints/pretrain/' + run_name + '.pt'
if not os.path.exists('checkpoints/pretrain'):
    os.makedirs('checkpoints/pretrain')
# Get tokenizer and de-tokenizer
tokenizer = get_tokenizer(args)
pretrain_data_path = 'data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + str(args.num_nodes) + '_200k_pretrain.txt'
test_data_path = 'data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + str(args.num_nodes) + '_20k_test.txt'
train_data = get_dataset(args, tokenizer, device, pretrain_data_path, n_sample=args.n_train if args.n_train > 0 else None)
test_data = get_dataset(args, tokenizer, device, test_data_path)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

args.block_size = train_data.num_tokens
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None
model = get_model(args)

if args.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

model.to(device)
model.train()
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

optimizer = model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=args.lr, betas=(beta1, beta2), device_type=device)
max_iters = len(train_loader) * args.epochs
warmup_iters = max(100, 0.01 * max_iters)
lr_decay_iters = max_iters
min_lr = 0.1 * args.lr

# Setup wandb logging
if wandb_log:
    wandb.init(project='next-token-failures-sft', entity=wandb_entity, config=args.__dict__, tags=['sft'])
    wandb.run.name = run_name

num_iters = 0
for ep in range(1, args.epochs + 1):
    train_bar = tqdm(train_loader)
    total_loss, total_acc = AverageMeter(), AverageMeter()
    grad_norms = AverageMeter()
    start_time = time.time()
    num_tokens = 0
    for x, y in train_bar:
        # determine and set the learning rate for this iteration
        lr = get_lr(num_iters, args.lr, warmup_iters, lr_decay_iters, min_lr) if decay_lr else args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            _, loss, accs = model(x, y)

        total_loss.update(loss.item(), x.shape[0] * train_data.num_target_tokens)
        total_acc.update(accs['acc'], x.shape[0])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        num_iters += 1
        grad_norms.update(grad_norm, 1)
        num_tokens += x.shape[0] * x.shape[1]
        toks_per_sec = round(num_tokens / (time.time() - start_time))
        train_bar.set_description(
            f'Epoch: [{ep}/{args.epochs}] Loss: {total_loss.get():.4f} Acc: {total_acc.get(percentage=True):.2f}% GradNorm: {grad_norms.get():.2f} Toks per sec: {toks_per_sec}'
        )

    cur_time = time.time()
    toks_per_sec = round(num_tokens / (cur_time - start_time))
    print(f"Epoch {ep} took {cur_time - start_time:.2f} seconds. {toks_per_sec} toks/sec")
    if wandb_log:
        wandb.log({
            'toks_per_sec': toks_per_sec,
            'loss': total_loss.get(),
            'accuracy': total_acc.get(percentage=True),
            'lr': lr,
            'grad_norm': grad_norms.get(),
        }, step=ep)

    # evaluate the loss on train/val sets and write checkpoints
    if ep % args.eval_every == 0:
        # Generate sequences and check accuracies
        results = {}
        if args.eval_train:
            results = evaluate(model, train_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='train', eval_ratio=0.1)
            results = evaluate_forced(model, train_loader, ctx=ctx, results=results, mode='train', eval_ratio=0.1)

        results = evaluate(model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='test')
        results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='test')

        if wandb_log:
            wandb.log(results, step=ep)

    if args.save_every > 0 and ep % args.save_every == 0 and ep > 0:
        sd = model._orig_mod.state_dict() if args.compile else model.state_dict()
        torch.save(sd, path + "_epoch_" + str(ep))
