import argparse
from contextlib import nullcontext
from tqdm import tqdm
import torch
import os, json
from collections import OrderedDict
from tokenizing import get_tokenizer
from data import get_dataset
from models import get_model
from utils.training_utils import AverageMeter


# Parse arguments
parser = argparse.ArgumentParser(description="Next-token prediction")
# Data
parser.add_argument(
        "--max_trajs_per_prompt", type=int, default=1000, help="Number of trajectories per prompt",
    )
parser.add_argument(
        "--model_ckpt", type=str, required=True, help="path to model ckpt",
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
        "--teacherless", action=argparse.BooleanOptionalAction, default=False, help="Standard or teacherless training",
    )
parser.add_argument(
        "--reverse", action=argparse.BooleanOptionalAction, default=False, help="Standard format or reverse targets",
    )
parser.add_argument(
        "--greedy_decoding", action=argparse.BooleanOptionalAction, default=False, help="Use greedy decoding",
    )
parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate",
    )
parser.add_argument(
    "--mlp_expansion_factor", type=int, default=4, help="MLP Expansion factor",
    )
parser.add_argument(
    "--start_index_ratio", type=float, default=0, help="Start index",
)
parser.add_argument(
    "--end_index_ratio", type=float, default=1, help="End index",
)

args = parser.parse_args()


# System stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Model stuff
top_k = None
if args.greedy_decoding:
    top_k = 1

# Optimiser
dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
args.compile = True if device == 'cuda' else False
args.use_flash = True if device == 'cuda' else False

# Get tokenizer and de-tokenizer
tokenizer = get_tokenizer(args)
test_data_path = 'data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + str(args.num_nodes) + '_20k_test.txt'
test_data = get_dataset(args, tokenizer, device, test_data_path)

args.block_size = test_data.num_tokens
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None
model = get_model(args)
state_dict = torch.load(args.model_ckpt)
model.load_state_dict(state_dict)

if args.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

model.to(device)
model.eval()

# initialize a GradScaler. If enabled=False scaler is a no-op
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

@torch.no_grad()
def collect_rollouts(model, dataset, ctx, temperature, top_k, batch_size, max_trajs_per_prompt=1024):
    print(f"EVALUATING WITH TEMPERATURE: {temperature}, TOP_K: {top_k}")
    total_completely_correct = AverageMeter()
    num_prefix_tokens = dataset.num_prefix_tokens
    num_target_tokens = dataset.num_target_tokens

    # Switch dataset and model to "eval" mode
    dataset.eval()
    model.eval()
    bar = tqdm(total=len(dataset))
    dataset_iter = iter(dataset)
    cur_global_idx = 0

    failed_outputs = {}
    done_outputs = {}
    current_outputs = OrderedDict()
    current_idx = []
    current_batch = []
    while True:
        # fill the current batch
        while len(current_idx) < batch_size and cur_global_idx < len(dataset):
            current_idx.append(cur_global_idx)
            current_batch.append(next(dataset_iter))
            bar.update(1)
            cur_global_idx += 1

        # perform inference on batch
        stacked_batch = torch.stack(current_batch)
        x = stacked_batch[:, :num_prefix_tokens]
        y = stacked_batch[:, num_prefix_tokens:]
        with ctx:
            y_pred = model.generate(x, num_target_tokens, temperature=temperature, top_k=top_k)
        correct = y.eq(y_pred[:, -num_target_tokens:]).float()
        completely_correct = correct.sum(dim=1).eq(num_target_tokens).to(torch.float)

        # find results that form a pair
        done_idx = set()
        for i, global_i in enumerate(current_idx):
            if global_i not in current_outputs:
                current_outputs[global_i] = {
                    'x': x[i].cpu().tolist(),
                    'y_pred_chosen': x[i].cpu().tolist() + y[i].cpu().tolist(),
                    'y_pred_chosen_correct': torch.ones_like(correct[i]).cpu().tolist(),
                    'y_pred_chosen_completely_correct': torch.ones_like(completely_correct[i]).item(),
                    'n_try': 0,
                }

            if completely_correct[i].item() == 0:
                current_outputs[global_i]['y_pred_reject'] = y_pred[i].cpu().tolist()
                current_outputs[global_i]['y_pred_reject_correct'] = correct[i].cpu().tolist()
                current_outputs[global_i]['y_pred_reject_completely_correct'] = completely_correct[i].item()
                # since we know the correct traj is y, we can move this to done
                done_outputs[global_i] = current_outputs.pop(global_i)
                done_idx.add(i)
            else:
                current_outputs[global_i]['n_try'] += 1
                if current_outputs[global_i]['n_try'] >= max_trajs_per_prompt:
                    failed_outputs[global_i] = current_outputs.pop(global_i)
                    done_idx.add(i)

        current_idx = [v for i, v in enumerate(current_idx) if i not in done_idx]
        current_batch = [v for i, v in enumerate(current_batch) if i not in done_idx]
        total_completely_correct.update(completely_correct.mean().item(), x.shape[0])
        bar.set_description(f"Done: {len(done_outputs)}, Failed: {len(failed_outputs)}, Cur_BSZ: {x.shape[0]}, Correct: {total_completely_correct.get(percentage=True):.2f} ")

        if cur_global_idx == len(dataset) and len(current_idx) == 0:
            break

    # Switch back to train mode
    dataset.train()
    model.train()
    print(f"Done with {len(done_outputs)} rollouts, failed {len(failed_outputs)}")
    return {'done': done_outputs, 'fail': failed_outputs}


# Generate pairwise rollouts and save the token ids
for suffix in ['20k_test', '200k_posttrain']:
    graphs_data_path = f'data/datasets/graphs/deg_{args.deg}_path_{args.path_len}_nodes_{args.num_nodes}_{suffix}.txt'
    graphs_data = get_dataset(args, tokenizer, device, graphs_data_path)
    rollouts_data_path = f'data/datasets/gpt2_graphs_rollouts/deg_{args.deg}_path_{args.path_len}_nodes_{args.num_nodes}_{suffix}.jsonl'
    if not os.path.exists(os.path.dirname(rollouts_data_path)):
        os.makedirs(os.path.dirname(rollouts_data_path))
    if not os.path.exists(rollouts_data_path):
        outputs = collect_rollouts(model, graphs_data, temperature=1.0, ctx=ctx, top_k=top_k, batch_size=args.batch_size, max_trajs_per_prompt=args.max_trajs_per_prompt)
        # only save the dones
        done_outputs = outputs['done']
        with open(rollouts_data_path, 'w') as f:
            for global_idx, output in done_outputs.items():
                output['global_idx'] = global_idx
                f.write(json.dumps(output) + '\n')
