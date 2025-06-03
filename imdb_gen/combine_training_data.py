import glob
import json
import os
from tqdm import tqdm
from utils import write_jsonl
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--data_template_path",
    default="collected_data/tinyllama_imdb",
    type=str,
    help="directory containing per-sample JSON outputs (default: collected_data/tinyllama_imdb)",
)
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)

data_template_path = args.data_template_path
data_template_paths = [os.path.join(data_template_path, "*.json")]
output_path = os.path.join(os.path.dirname(data_template_path), "all_train_data.jsonl")
print("data_template_paths", data_template_paths)
print("output_path", output_path)
# define keys to aggregate per prompt
additional_keys = [
    "fully_guided_predictions",
    "fully_guided_rewards",
    "partial_guided_prompts",
    "partial_guided_prompts_tokenized",
    "num_response_tokens_in_partial_guided_prompts",
    "partial_guided_responses_tokenized",
    "partial_guided_predictions",
    "partial_guided_rewards",
    "partial_guided_vs_fully_pref",
]
# aggregate per-sample JSONs into one entry per idx
aggregated = {}
for pattern in data_template_paths:
    for path in tqdm(glob.glob(pattern)):
        with open(path, "r") as f:
            data = json.load(f)
        idx = data.get("idx")
        if idx is None:
            continue
        if idx not in aggregated:
            # initialize base entry
            entry = {
                "idx": idx,
                "prompt": data.get("prompt"),
                "full_text": data.get("full_text"),
            }
            # include label if present
            if "label" in data:
                entry["label"] = data.get("label")
            # initialize lists for each key
            for key in additional_keys:
                entry[key] = list(data.get(key, []))
            aggregated[idx] = entry
        else:
            entry = aggregated[idx]
            # extend lists for each key
            for key in additional_keys:
                if key in data:
                    entry[key].extend(data.get(key, []))
# produce sorted list of entries
problem_data = [aggregated[k] for k in sorted(aggregated.keys())]
write_jsonl(problem_data, output_path)
print("done")
