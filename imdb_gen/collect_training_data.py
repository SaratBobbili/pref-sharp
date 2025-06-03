import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification,
)
from classifier import (
    CustomLlamaForSequenceClassification,
    CustomValueGuidedLogitProcessor,
)
from accuracy_utils import (
    sample_match_strict,
    numeric_or_symbolic_correctness,
    quick_evaluate_single,
    evaluate_preference,
)
from utils import (
    read_jsonl,
    tokenize_with_chat_template,
    generate_with_classifier_guidance,
    get_parent_directory,
    resolve_dict_value,
    get_output_indices,
)
import json
import os
import math
from tqdm import tqdm, trange
import glob
import copy
from datasets import load_dataset  # added for IMDB dataset loading

parser = argparse.ArgumentParser(description="")
parser.add_argument("--start_index", default=0, type=int, help="start index for data")
parser.add_argument(
    "--end_index", default=-1, type=int, help="end index for data, -1 means all data"
)
parser.add_argument(
    "--eval_ratio", default=0.1, type=float, help="ratio of data for evaluation"
)
parser.add_argument(
    "--is_first_round",
    required=True,
    type=int,
    help="whether this is the first round of collecting data, will zero init classifier and also do a split of train eval data if needed",
)

parser.add_argument(
    "--ref_model_id",
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    type=str,
    help="reference model id meta-llama/Llama-3.1-8B-Instruct",
)
parser.add_argument(
    "--classifier_model_id",
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    type=str,
    help="classifier model id (for tokenizer, reuse weights)",
)
parser.add_argument(
    "--classifier_ckpt_path", default=None, type=str, help="a ckpt path"
)
parser.add_argument(
    "--classifier_type", default=None, type=str, help="classifier type Q or V"
)
parser.add_argument(
    "--inference_mode",
    default=None,
    type=str,
    help="inference mode supported by the classifier. First round does not matter",
)
parser.add_argument(
    "--loss_type",
    default=None,
    type=str,
    help="loss type for the classifier, unused for evaluation",
)
parser.add_argument(
    "--use_bias",
    default=None,
    type=int,
    help="whether to use bias for the classification layer, llama 3 does not have bias",
)
parser.add_argument(
    "--data_path",
    default=None,
    type=str,
    help="path to the data dataset/gsm8k_train.jsonl",
)
parser.add_argument(
    "--train_eval_save_path",
    default=None,
    type=str,
    help="train eval split dataset/gsm8k_train_eval.json",
)
parser.add_argument("--batch_size", default=16, type=int, help="batch size")
parser.add_argument(
    "--num_samples", default=16, type=int, help="number of samples per problem"
)
parser.add_argument(
    "--use_chat_template",
    default=0,
    type=int,
    help="whether to use chat template for generation",
)
parser.add_argument(
    "--eta",
    default=None,
    type=float,
    help="eta for the classifier, larger it is, less KL regularization. Unused for expectation inference mode",
)
parser.add_argument(
    "--top_k", type=int, default=20, help="top k logits to modify, -1 means all logits"
)
parser.add_argument(
    "--temperature", default=None, type=float, help="temperature for sampling 0.8"
)
parser.add_argument("--top_p", default=None, type=float, help="top p for sampling 0.9")
parser.add_argument(
    "--max_new_tokens", default=None, type=int, help="max tokens for sampling 1024"
)
parser.add_argument(
    "--dtype", default=None, type=str, help="data type for the model bfloat16"
)
parser.add_argument(
    "--match_fn_type",
    default=None,
    type=str,
    help="matching function type for evaluation, symbolic or strict; symbolic",
)
parser.add_argument(
    "--output_dir", default=None, type=str, help="default use classifier_ckpt_path"
)
parser.add_argument(
    "--force", default=0, type=int, help="force overwrite existing files"
)
parser.add_argument("--seed", default=47, type=int, help="seed for reproduction")
parser.add_argument(
    "--num_atoms", default=11, type=int, help="number of atoms for mle classifier"
)
parser.add_argument(
    "--V_min", default=0, type=float, help="V_min for histogram learning"
)
parser.add_argument(
    "--V_max", default=1, type=float, help="V_max for histogram learning"
)

args = parser.parse_args()
args_dict = vars(args)
print(args_dict)

start_index = args.start_index
end_index = args.end_index
is_first_round = bool(args.is_first_round)
eval_ratio = args.eval_ratio

if is_first_round:
    training_args_dict = {}
else:
    with open(
        os.path.join(get_parent_directory(args.classifier_ckpt_path), "args.json"), "r"
    ) as f:
        training_args_dict = json.load(f)
    print(training_args_dict)

ref_model_id = resolve_dict_value(args_dict, training_args_dict, "ref_model_id")
classifier_model_id = resolve_dict_value(
    args_dict, training_args_dict, "classifier_model_id"
)
classifier_ckpt_path = args.classifier_ckpt_path
inference_mode = resolve_dict_value(args_dict, training_args_dict, "inference_mode")
loss_type = resolve_dict_value(args_dict, training_args_dict, "loss_type")
use_bias = bool(resolve_dict_value(args_dict, training_args_dict, "use_bias"))
data_path = resolve_dict_value(
    args_dict, training_args_dict, "data_path", "original_problems_path"
)
train_eval_save_path = args.train_eval_save_path
classifier_type = resolve_dict_value(args_dict, training_args_dict, "classifier_type")
batch_size = args.batch_size
num_samples = args.num_samples
use_chat_template = args.use_chat_template
eta = resolve_dict_value(args_dict, training_args_dict, "eta")
top_k = resolve_dict_value(args_dict, training_args_dict, "top_k")
assert eta >= 0
temperature = resolve_dict_value(args_dict, training_args_dict, "temperature")
top_p = resolve_dict_value(args_dict, training_args_dict, "top_p")
max_new_tokens = resolve_dict_value(args_dict, training_args_dict, "max_new_tokens")
# dtype and match_fn_type come directly from CLI for IMDB
dtype = args.dtype
match_fn_type = args.match_fn_type
output_dir = args.output_dir
force = args.force
seed = args.seed
num_atoms = resolve_dict_value(args_dict, training_args_dict, "num_atoms")
V_min = resolve_dict_value(args_dict, training_args_dict, "V_min")
V_max = resolve_dict_value(args_dict, training_args_dict, "V_max")

if classifier_ckpt_path is None:
    classifier_ckpt_path = classifier_model_id

os.makedirs(output_dir, exist_ok=True)

set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(ref_model_id)
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_id)
# ensure a distinct pad token (not same as eos)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    classifier_tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.padding_side = "left"  # for inference
print("tokenizer padding side:", tokenizer.padding_side)
# ensure vocab sizes match after adding special tokens
assert len(tokenizer) == len(classifier_tokenizer), "tokenizer vocab size mismatch"
vocab_size = len(tokenizer)
if temperature == 0:
    do_sample = False
    temperature = 1.0
else:
    do_sample = True
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Load IMDB dataset and build prompts
if data_path == "stanfordnlp/imdb":
    dataset_type = "IMDB"
    # load and shuffle to mix positive/negative examples
    dataset = load_dataset(data_path, split="train").shuffle(seed)
    train_data = []
    for idx, item in enumerate(dataset):
        full_text = item["text"]
        # tokenize and truncate to first 60 tokens to avoid long-sequence warnings
        tokenized = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=60
        )
        prompt_ids = tokenized["input_ids"][0]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        # include label for tracking balance
        train_data.append(
            {
                "prompt": prompt_text,
                "full_text": full_text,
                "idx": idx,
                "label": item.get("label"),
            }
        )
else:
    raise ValueError(f"Unknown dataset name: {data_path}")
if end_index == -1:
    end_index = len(train_data)

# Disable chat template for IMDB generation
use_chat_template = 0

# Sentiment reward model (DistilBERT IMDB)
reward_name = "lvwerra/distilbert-imdb"
rank_model, reward_tokenizer = AutoModelForSequenceClassification.from_pretrained(
    reward_name
), AutoTokenizer.from_pretrained(reward_name)
rank_model.eval()
# move reward model to device to match input tensors
rank_model = rank_model.to(device)

# prepare loading kwargs (e.g. bfloat16)
model_loading_kwargs = {}
if dtype == "bfloat16":
    model_loading_kwargs["torch_dtype"] = torch.bfloat16
    print("loading model with bfloat16")

ref_model = AutoModelForCausalLM.from_pretrained(
    ref_model_id, **model_loading_kwargs, device_map=device
)
# resize embeddings so new pad token is recognized
ref_model.resize_token_embeddings(len(tokenizer))
classifier_model = CustomLlamaForSequenceClassification.from_pretrained(
    classifier_ckpt_path,
    **model_loading_kwargs,
    num_labels=vocab_size,
    loss_type=loss_type,
    use_bias=use_bias,
    classifier_type=classifier_type,
    device_map=device,
    num_atoms=num_atoms,
    V_min=V_min,
    V_max=V_max,
)
# resize classifier embeddings as well
classifier_model.resize_token_embeddings(len(tokenizer))

ref_model.eval()
classifier_model.eval()
torch.set_grad_enabled(False)  # disable gradients globally
if is_first_round:
    classifier_model.zero_init_classifier()

logit_processor = CustomValueGuidedLogitProcessor(
    eta=eta,
    ref_model=ref_model,
    ref_model_tokenizer=tokenizer,
    value_classifier=classifier_model,
    inference_mode=inference_mode,
    top_k=top_k,
    use_cache=True,
)
logit_processor_disabled = CustomValueGuidedLogitProcessor(
    eta=eta,
    ref_model=ref_model,
    ref_model_tokenizer=tokenizer,
    value_classifier=classifier_model,
    inference_mode="disabled",
    top_k=top_k,
    use_cache=True,
)

# generation parameters for guided sampling
generate_kwargs = {
    "temperature": temperature,
    "top_p": top_p,
    "do_sample": do_sample,
    # limit output to same length as input (60 tokens)
    "max_new_tokens": 60,
    "top_k": 0,
}

# Multi-phase generation with resume, fully guided and partial guided steps
for repeat_index in trange(num_samples, desc="Repeats"):
    current_seed = seed + 50 * repeat_index
    set_seed(current_seed)
    print(f"repeat {repeat_index}")
    # resume logic: skip already-processed indices
    if not force:
        existing = glob.glob(os.path.join(output_dir, f"*_r{repeat_index}.json"))
        existing_indices = [int(os.path.basename(p).split("_")[0]) for p in existing]
    else:
        existing_indices = []
    # select data to infer
    train_data_to_infer = []
    for j in range(start_index, end_index):
        if train_data[j]["idx"] not in existing_indices:
            train_data_to_infer.append(copy.deepcopy(train_data[j]))
    print(
        f"total number of problems to infer for repeat {repeat_index}: {len(train_data_to_infer)}"
    )
    num_batches = math.ceil(len(train_data_to_infer) / batch_size)
    for b in tqdm(range(num_batches), desc=f"Repeat {repeat_index}"):
        set_seed(current_seed)
        batch_start = b * batch_size
        batch_samples = train_data_to_infer[batch_start : batch_start + batch_size]
        # tokenize prompts
        prompts = [s["prompt"] for s in batch_samples]
        inputs, formatted_prompts = tokenize_with_chat_template(
            tokenizer, prompts, use_chat_template, device
        )
        # Fully guided generation
        outputs = generate_with_classifier_guidance(
            ref_model, tokenizer, logit_processor, inputs, generate_kwargs, True, False
        )
        outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # sentiment scoring for fully guided
        full_scores = []
        for k, text in enumerate(outputs_text):
            reward_in = reward_tokenizer(
                formatted_prompts[k],
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(device)
            with torch.no_grad():
                logits = rank_model(**reward_in).logits
            full_scores.append(logits[0][1].cpu().item())
        # store fully guided outputs and scores
        for k, sample in enumerate(batch_samples):
            sample.setdefault("fully_guided_predictions", []).append(outputs_text[k])
            # store absolute reward (positive sentiment logit) for fully guided output
            sample.setdefault("fully_guided_rewards", []).append(full_scores[k])
        # Partial guided generation: single variant
        end_idxs = get_output_indices(outputs, tokenizer.eos_token_id)
        lengths = end_idxs + 1
        cut_locs = torch.floor(
            torch.rand(lengths.size()).to(lengths.device) * lengths
        ).int()
        skip_flags = [cut_locs[i] + 1 == lengths[i] for i in range(len(lengths))]
        skip_idxs = [i for i, f in enumerate(skip_flags) if f]
        # build prefixes
        queries = [
            inputs["input_ids"][i].masked_select(
                inputs["attention_mask"][i].to(torch.bool)
            )
            for i in range(len(inputs["input_ids"]))
        ]
        partial_resps = [outputs[i][: cut_locs[i] + 1] for i in range(len(outputs))]
        # prepare inputs for disabled-classifier generation
        concat_ids = [torch.cat([q, r]) for q, r in zip(queries, partial_resps)]
        coll_in = data_collator(
            [
                {
                    "input_ids": concat_ids[i],
                    "attention_mask": torch.ones_like(concat_ids[i]),
                }
                for i in range(len(concat_ids))
                if i not in skip_idxs
            ]
        )
        coll_in.pop("labels", None)
        coll_in = coll_in.to(device)
        # generate one partial guided continuation
        partial_out = generate_with_classifier_guidance(
            ref_model,
            tokenizer,
            logit_processor_disabled,
            coll_in,
            generate_kwargs,
            True,
            False,
        )
        partial_ei = get_output_indices(partial_out, tokenizer.eos_token_id)
        partial_len = partial_ei + 1
        # map back to full token lists
        partial_ids = []
        cnt = 0
        for i in range(len(queries)):
            if i in skip_idxs:
                partial_ids.append(None)
            else:
                cont = partial_out[cnt][: partial_len[cnt]].tolist()
                ids = partial_resps[i].tolist() + cont
                partial_ids.append(ids)
                cnt += 1
        # decode and score partial outputs
        part_preds = []
        part_rewards = []
        for i, ids in enumerate(partial_ids):
            if ids is None:
                part_preds.append(None)
                part_rewards.append(None)
            else:
                text = tokenizer.decode(ids, skip_special_tokens=True)
                part_preds.append(text)
                base = tokenizer.decode(concat_ids[i].tolist())
                inp = reward_tokenizer(
                    base, text, return_tensors="pt", truncation=True, padding=True
                ).to(device)
                with torch.no_grad():
                    s = rank_model(**inp).logits[0][1].cpu().item()
                part_rewards.append(s)
        # compute preference vs fully guided
        part_vs_full_pref = []
        for i, pr in enumerate(part_rewards):
            if pr is None:
                part_vs_full_pref.append(None)
            else:
                part_vs_full_pref.append(1 if pr > full_scores[i] else 0)
        # record partial-guided data
        for k, sample in enumerate(batch_samples):
            sample.setdefault("partial_guided_prompts_tokenized", []).append(
                concat_ids[k].tolist()
            )
            sample.setdefault("partial_guided_prompts", []).append(
                tokenizer.decode(concat_ids[k].tolist())
            )
            sample.setdefault(
                "num_response_tokens_in_partial_guided_prompts", []
            ).append(cut_locs[k].item() + 1)
            sample.setdefault("partial_guided_responses_tokenized", []).append(
                partial_ids[k] or []
            )
            sample.setdefault("partial_guided_predictions", []).append(
                part_preds[k] or ""
            )
            sample.setdefault("partial_guided_rewards", []).append(part_rewards[k])
            sample.setdefault("partial_guided_vs_fully_pref", []).append(
                part_vs_full_pref[k]
            )
        # save output
        for sample in batch_samples:
            out_file = os.path.join(output_dir, f"{sample['idx']}_r{repeat_index}.json")
            assert not os.path.exists(out_file), f"output exists: {out_file}"
            with open(out_file, "w") as f:
                json.dump(sample, f, indent=4)
print("done")
