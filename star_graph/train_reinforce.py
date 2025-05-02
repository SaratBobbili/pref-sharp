import argparse
from contextlib import nullcontext
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb
import time
from collections import defaultdict
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
    "--piref_ckpt", type=str, required=True, help="warmstart from pretrained model",
)
parser.add_argument(
    "--online", action=argparse.BooleanOptionalAction, default=False, help="Whether to use online data",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed",
)
parser.add_argument(
    '--compile', action=argparse.BooleanOptionalAction, default=False, help='Whether to compile the model'
)
parser.add_argument(
    '--baseline', type=int, default=0, help="Whether to use baseline."
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
beta1 = 0.9
beta2 = 0.999
decay_lr = True
args.use_flash = True if device == 'cuda' else False

run_name = get_run_name(args)
assert args.model.startswith('gpt2')
run_name = run_name + '_bs_' + str(args.batch_size) + '_lr_' + str(args.lr) + '_seed_' + str(
    args.seed) + '_' + args.model
if args.baseline:
    run_name = run_name + '_baseline'
path = 'checkpoints/reinforce/' + run_name + '.pt'
if not os.path.exists('checkpoints/reinforce'):
    os.makedirs('checkpoints/reinforce')
# Get tokenizer and de-tokenizer
tokenizer = get_tokenizer(args)
pretrain_data_path = 'data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + str(
    args.num_nodes) + '_200k_pretrain.txt'
posttrain_data_path = 'data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + str(
    args.num_nodes) + '_200k_posttrain.txt'
test_data_path = 'data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + str(
    args.num_nodes) + '_20k_test.txt'
pretrain_data = get_dataset(args, tokenizer, device, pretrain_data_path, n_sample=20000)  # for eval only
posttrain_data = get_dataset(args, tokenizer, device, posttrain_data_path)
test_data = get_dataset(args, tokenizer, device, test_data_path)
pretrain_loader = DataLoader(pretrain_data, batch_size=args.batch_size, shuffle=False)
posttrain_loader = DataLoader(posttrain_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

num_prefix_tokens = posttrain_data.num_prefix_tokens
num_target_tokens = posttrain_data.num_target_tokens
args.block_size = pretrain_data.num_tokens
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None
model = get_model(args)


def load_from_ckpt(model, ckpt_path):
    sd = torch.load(ckpt_path)
    # TODO: remove this line when we save the model without the "_orig_mod." prefix
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        new_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        sd = new_sd
    model.load_state_dict(sd)
    print(f"Successfully loaded from {ckpt_path}!")


load_from_ckpt(model, args.piref_ckpt)
if args.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

model.to(device)
model.train()
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

optimizer = model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=args.lr, betas=(0.9, 0.999),
                                       device_type=device)
max_iters = len(posttrain_loader) * args.epochs
lr_decay_iters = max_iters
warmup_iters = max(100, 0.1 * max_iters)
min_lr = 0.1 * args.lr

# Setup wandb logging
if wandb_log:
    wandb.init(project='next-token-failures-reinforce', entity=wandb_entity, config=args.__dict__, tags=['reinforce'])
    wandb.run.name = run_name

num_iters = 0
for ep in range(1, args.epochs + 1):
    if args.save_every > 0 and ep % args.save_every == 0 and ep > 0:
        sd = model._orig_mod.state_dict() if args.compile else model.state_dict()
        torch.save(sd, path + "_epoch_" + str(ep))

    posttrain_loader.dataset.eval()
    train_bar = tqdm(posttrain_loader)
    total_loss = AverageMeter()
    avg_meter_map = defaultdict(AverageMeter)
    start_time = time.time()
    num_tokens = 0
    for x in train_bar:
        lr = get_lr(num_iters, args.lr, warmup_iters, lr_decay_iters, min_lr) if decay_lr else args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        y = x[:, num_prefix_tokens:].clone()
        x = x[:, :num_prefix_tokens].clone()
        with torch.no_grad():
            with ctx:
                y_pred = model.generate(x, num_target_tokens, temperature=0.8, top_k=top_k)

        correct = y.eq(y_pred[:, -num_target_tokens:]).float()  # [bs, num_target_tokens]
        completely_correct = correct.sum(dim=1).eq(num_target_tokens).float()  # [bs]
        input_x = y_pred[:, :-1].clone()
        input_y = y_pred[:, 1:].clone()
        # only use the last num_target_tokens
        input_y[:, :-num_target_tokens] = -1
        with ctx:
            _, neg_logps, _ = model(input_x, input_y, reduce_loss=False)
            # only use the last num_target_tokens
            neg_logps = neg_logps[:, -num_target_tokens:]  # [bs, num_target_tokens]
            neg_logps = neg_logps.sum(1)  # [bs]
            if args.baseline:
                traj_reward = completely_correct - completely_correct.mean()  # [bs]
            else:
                traj_reward = 1.1 * completely_correct - 0.1  # [bs]
            loss = (neg_logps * traj_reward).mean()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        num_iters += 1

        total_loss.update(loss.item(), x.shape[0])
        avg_meter_map['grad_norm'].update(grad_norm, 1)
        num_tokens += x.shape[0] * x.shape[1]
        toks_per_sec = round(num_tokens / (time.time() - start_time))
        train_bar.set_description(
            f"Epoch: [{ep}/{args.epochs}] Loss: {total_loss.get():.4f} GradNorm: {avg_meter_map['grad_norm'].get():.2f} Toks per sec: {toks_per_sec}"
        )

    cur_time = time.time()
    toks_per_sec = round(num_tokens / (cur_time - start_time))
    print(f"Epoch {ep} took {cur_time - start_time:.2f} seconds. {toks_per_sec} toks/sec")
    wandb.log({
        'toks_per_sec': toks_per_sec,
        'loss': total_loss.get(),
        'lr': lr,
        **{k: v.get() for k, v in avg_meter_map.items()},
    }, step=ep)

    if args.eval_every > 0 and ep % args.eval_every == 0:
        results = {}
        results = evaluate(model, pretrain_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results,
                           mode='pretrain')
        results = evaluate_forced(model, pretrain_loader, ctx=ctx, results=results, mode='pretrain')

        results = evaluate(model, posttrain_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results,
                           mode='posttrain', eval_ratio=0.1)
        results = evaluate_forced(model, posttrain_loader, ctx=ctx, results=results, mode='posttrain', eval_ratio=0.1)

        results = evaluate(model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='test')
        results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='test')

        if wandb_log:
            wandb.log(results, step=ep)

    torch.cuda.empty_cache()
