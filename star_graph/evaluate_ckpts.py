import time
import argparse
from contextlib import nullcontext
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from tokenizing import get_tokenizer
from utils.training_utils import set_seed
from data import get_dataset
import evaluate
from models import get_model


# Parse arguments
parser = argparse.ArgumentParser(description="Next-token prediction")
# Data
parser.add_argument(
    "--model", type=str, default='gpt', help="Learning rate",
    )
parser.add_argument(
    "--pretrained_ckpt", type=str, required=True, help="Path to pretrained checkpoint",
    )
parser.add_argument(
    "--classifier_ckpt", type=str, required=True, help="Path to classifier checkpoint",
    )
parser.add_argument(
    "--reinforce_ckpt", type=str, required=True, help="Path to classifier checkpoint",
    )
parser.add_argument(
    "--dpo_ckpt", type=str, required=True, help="Path to classifier checkpoint",
    )
parser.add_argument(
    "--rpo_ckpt", type=str, required=True, help="Path to classifier checkpoint",
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
    "--seed", type=int, default=42, help="Random seed",
    )
parser.add_argument(
    "--dtype", type=str, default="bfloat16", help="data type",
    )
parser.add_argument(
    "--top_k", type=int, default=1, help="top_k for sampling",
    )
parser.add_argument(
    "--temperature", type=float, default=0.8, help="temperature for sampling",
    )
parser.add_argument(
    "--teacherless", action=argparse.BooleanOptionalAction, default=False, help="Standard or teacherless training",
    )
parser.add_argument(
    "--reverse", action=argparse.BooleanOptionalAction, default=False, help="Standard format or reverse targets",
    )


args = parser.parse_args()
set_seed(args.seed)

# System stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

dtype = args.dtype
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
args.use_flash = True if device == 'cuda' else False

tokenizer = get_tokenizer(args)
test_data_path = 'data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + str(args.num_nodes) + '_20k_test.txt'
test_data = get_dataset(args, tokenizer, device, test_data_path)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

args.block_size = test_data.num_tokens
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None
model = get_model(args)

def load_from_ckpt(model, ckpt_path):
    sd = torch.load(ckpt_path)
    # # TODO: remove this line when we save the model without the "_orig_mod." prefix
    # if any(k.startswith("_orig_mod.") for k in sd.keys()):
    #     new_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    #     sd = new_sd
    model.load_state_dict(sd)

load_from_ckpt(model, args.pretrained_ckpt)

classifier_model = get_model(args, is_classifier=True)
load_from_ckpt(classifier_model, args.classifier_ckpt)
classifier_model = torch.compile(classifier_model)

model.to(device)
model.eval()
classifier_model.to(device)
classifier_model.eval()
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

all_results = {}
print(f"Evaluating pretrained model on G({args.deg}, {args.path_len}):")
results = evaluate.evaluate(model, test_loader, temperature=args.temperature, ctx=ctx, top_k=args.top_k, results={}, mode='pretrain')
all_results['pretrain'] = results['pretrain/accuracy']
print(results)

for eta in [1e-1]:
    print('-'*50)
    print(f"Evaluating Q# with eta={eta}, top_k={args.top_k} on G({args.deg}, {args.path_len}):")
    results = evaluate.evaluate(model, test_loader, temperature=args.temperature, ctx=ctx, top_k=args.top_k, results={}, mode=f'q# ({eta=})',
                                classifier=classifier_model, eta=eta)
    all_results[f'q# ({eta})'] = results[f'q# ({eta=})/accuracy']
    print(results)

    print(f"Evaluating CD with eta={eta}, top_k={args.top_k} on G({args.deg}, {args.path_len}):")
    results = evaluate.evaluate(model, test_loader, temperature=args.temperature, ctx=ctx, top_k=args.top_k, results={}, mode=f'cd ({eta=})',
                                classifier=classifier_model, eta=eta, guide_with_cd=True)
    all_results[f'cd ({eta})'] = results[f'cd ({eta=})/accuracy']
    print(results)
del classifier_model
torch.cuda.empty_cache()

# reinforce
load_from_ckpt(model, args.reinforce_ckpt)
print(f"Evaluating reinforce model on G({args.deg}, {args.path_len}):")
results = evaluate.evaluate(model, test_loader, temperature=args.temperature, ctx=ctx, top_k=args.top_k, results={}, mode='reinforce')
all_results['reinforce'] = results['reinforce/accuracy']
print(results)

# dpo
load_from_ckpt(model, args.dpo_ckpt)
print(f"Evaluating dpo model on G({args.deg}, {args.path_len}):")
results = evaluate.evaluate(model, test_loader, temperature=args.temperature, ctx=ctx, top_k=args.top_k, results={}, mode='dpo')
all_results['dpo'] = results['dpo/accuracy']
print(results)

# rpo
load_from_ckpt(model, args.rpo_ckpt)
print(f"Evaluating rpo model on G({args.deg}, {args.path_len}):")
results = evaluate.evaluate(model, test_loader, temperature=args.temperature, ctx=ctx, top_k=args.top_k, results={}, mode='rpo')
all_results['rpo'] = results['rpo/accuracy']
print(results)

print("ALL RESULTS: ")
print(all_results)

# save results
import json
with open(f'results_deg_{args.deg}_path_{args.path_len}.json', 'w') as f:
    json.dump(all_results, f)