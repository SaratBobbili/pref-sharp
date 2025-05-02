import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, DataCollatorForLanguageModeling, AutoModelForSequenceClassification
from classifier import CustomLlamaForSequenceClassification, CustomValueGuidedLogitProcessor
from accuracy_utils import sample_match_strict, numeric_or_symbolic_correctness, quick_evaluate_single, evaluate_preference
from utils import read_jsonl, tokenize_with_chat_template, generate_with_classifier_guidance, get_parent_directory, resolve_dict_value, get_output_indices
import json
import os
import math
from tqdm import tqdm
import glob
import copy

parser = argparse.ArgumentParser(description='')
parser.add_argument('--start_index', default=0, type=int, help='start index for data')
parser.add_argument('--end_index', default=-1, type=int, help='end index for data, -1 means all data')
parser.add_argument('--eval_ratio', default=0.1, type=float, help='ratio of data for evaluation')
parser.add_argument('--is_first_round', required=True, type=int, help='whether this is the first round of collecting data, will zero init classifier and also do a split of train eval data if needed')

parser.add_argument('--ref_model_id', default=None, type=str,
                    help='reference model id meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--classifier_model_id', default=None, type=str, help='classifier model id (for tokenizer, reuse weights)')
parser.add_argument('--classifier_ckpt_path', default=None, type=str, help='a ckpt path')
parser.add_argument('--classifier_type', default=None, type=str, help='classifier type Q or V')
parser.add_argument('--inference_mode', default=None, type=str,
                    help='inference mode supported by the classifier. First round does not matter')
parser.add_argument('--loss_type', default=None, type=str, help='loss type for the classifier, unused for evaluation')
parser.add_argument('--use_bias', default=None, type=int,
                    help='whether to use bias for the classification layer, llama 3 does not have bias')
parser.add_argument('--data_path', default=None, type=str, help='path to the data dataset/gsm8k_train.jsonl')
parser.add_argument('--train_eval_save_path', default=None, type=str,
                    help='train eval split dataset/gsm8k_train_eval.json')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--num_samples', default=16, type=int, help='number of samples per problem')
parser.add_argument('--use_chat_template', default=None, type=int, help='whether to use chat template for generation')
parser.add_argument('--eta', default=None, type=float,
                    help='eta for the classifier, larger it is, less KL regularization. Unused for expectation inference mode')
parser.add_argument('--top_k', type=int, default=20, help='top k logits to modify, -1 means all logits')
parser.add_argument('--temperature', default=None, type=float, help='temperature for sampling 0.8')
parser.add_argument('--top_p', default=None, type=float, help='top p for sampling 0.9')
parser.add_argument('--max_new_tokens', default=None, type=int, help='max tokens for sampling 1024')
parser.add_argument('--dtype', default=None, type=str, help='data type for the model bfloat16')
parser.add_argument('--match_fn_type', default=None, type=str,
                    help='matching function type for evaluation, symbolic or strict; symbolic')
parser.add_argument('--output_dir', default=None, type=str,
                    help='default use classifier_ckpt_path')
parser.add_argument('--force', default=0, type=int, help='force overwrite existing files')
parser.add_argument('--seed', default=47, type=int, help='seed for reproduction')
parser.add_argument('--num_atoms', default=11, type=int, help='number of atoms for mle classifier')
parser.add_argument('--V_min', default=0, type=float, help='V_min for histogram learning')
parser.add_argument('--V_max', default=1, type=float, help='V_max for histogram learning')

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
    with open(os.path.join(get_parent_directory(args.classifier_ckpt_path), 'args.json'), 'r') as f:
        training_args_dict = json.load(f)
    print(training_args_dict)

ref_model_id = resolve_dict_value(args_dict, training_args_dict, 'ref_model_id')
classifier_model_id = resolve_dict_value(args_dict, training_args_dict, 'classifier_model_id')
classifier_ckpt_path = args.classifier_ckpt_path
inference_mode = resolve_dict_value(args_dict, training_args_dict, 'inference_mode')
loss_type = resolve_dict_value(args_dict, training_args_dict, 'loss_type')
use_bias = bool(resolve_dict_value(args_dict, training_args_dict, 'use_bias'))
data_path = resolve_dict_value(args_dict, training_args_dict, 'data_path', 'original_problems_path')
train_eval_save_path = resolve_dict_value(args_dict, training_args_dict, 'train_eval_save_path')
classifier_type = resolve_dict_value(args_dict, training_args_dict, 'classifier_type')
batch_size = args.batch_size
num_samples = args.num_samples
use_chat_template = resolve_dict_value(args_dict, training_args_dict, 'use_chat_template')
eta = resolve_dict_value(args_dict, training_args_dict, 'eta')
top_k = resolve_dict_value(args_dict, training_args_dict, 'top_k')
assert eta >= 0
temperature = resolve_dict_value(args_dict, training_args_dict, 'temperature')
top_p = resolve_dict_value(args_dict, training_args_dict, 'top_p')
max_new_tokens = resolve_dict_value(args_dict, training_args_dict, 'max_new_tokens')
dtype = resolve_dict_value(args_dict, training_args_dict, 'dtype')
match_fn_type = resolve_dict_value(args_dict, training_args_dict, 'match_fn_type')
output_dir = args.output_dir
force = args.force
seed = args.seed
num_atoms = resolve_dict_value(args_dict, training_args_dict, 'num_atoms')
V_min = resolve_dict_value(args_dict, training_args_dict, 'V_min')
V_max = resolve_dict_value(args_dict, training_args_dict, 'V_max')

if classifier_ckpt_path is None:
    classifier_ckpt_path = classifier_model_id

os.makedirs(output_dir, exist_ok=True)

set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(ref_model_id)
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_id)
assert len(tokenizer) == len(classifier_tokenizer), "tokenizer vocab size mismatch"
vocab_size = len(tokenizer)
if tokenizer.pad_token is None:
    assert 'Llama-3' in ref_model_id
    tokenizer.pad_token = tokenizer.added_tokens_decoder[128002].content  # reserved special token 0
tokenizer.padding_side = "left"  # for inference
print('tokenizer padding side:', tokenizer.padding_side)
if temperature == 0:
    do_sample = False
    temperature = 1.0
else:
    do_sample = True
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

reward_model = None
math_datasets = ["GSM8K", "MATH"]
if 'gsm8k_' in data_path:
    dataset_type = 'GSM8K'
    answer_key = 'answer'
elif 'math_' in data_path:
    dataset_type = 'MATH'
    answer_key = 'solution'  # require additional processing
else:
    raise ValueError('Unknown dataset name: %s' % data_path)

if match_fn_type == 'strict':
    match_fn = sample_match_strict
elif match_fn_type == 'symbolic':
    match_fn = numeric_or_symbolic_correctness
else:
    raise ValueError('Unknown match function type: %s' % match_fn_type)
train_data = read_jsonl(data_path)
if end_index == -1:
    end_index = len(train_data)
with open(train_eval_save_path, 'r') as f:
    train_eval_problems_d = json.load(f)
generate_kwargs = {'temperature': temperature, 'top_p': top_p, 'do_sample': do_sample, 'max_new_tokens': max_new_tokens, "top_k": 0}
model_loading_kwargs = {}
if dtype == 'bfloat16':
    model_loading_kwargs['torch_dtype'] = torch.bfloat16
    print('loading model with bfloat16')
ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id, **model_loading_kwargs, device_map=device)
classifier_model = CustomLlamaForSequenceClassification.from_pretrained(classifier_ckpt_path, **model_loading_kwargs,
                                                                        num_labels=vocab_size,
                                                                        loss_type=loss_type, use_bias=use_bias, classifier_type=classifier_type,
                                                                        device_map=device, num_atoms=num_atoms,
                                                                        V_min=V_min, V_max=V_max)

reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model, reward_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
#rank_model = rank_model.bfloat16().to(f"cuda:{local_rank}")

rank_model.eval()
ref_model.eval()
classifier_model.eval()
torch.set_grad_enabled(False)  # disable gradients globally
if is_first_round:
    classifier_model.zero_init_classifier()

logit_processor = CustomValueGuidedLogitProcessor(eta=eta, ref_model=ref_model, ref_model_tokenizer=tokenizer,
                                                  value_classifier=classifier_model, inference_mode=inference_mode, top_k=top_k,
                                                  use_cache=True)
logit_processor_disabled = CustomValueGuidedLogitProcessor(eta=eta, ref_model=ref_model, ref_model_tokenizer=tokenizer,
                                                           value_classifier=classifier_model, inference_mode='disabled', top_k=top_k,
                                                           use_cache=True)
for i in range(num_samples):
    repeat_index = i
    current_seed = seed + 50 * repeat_index
    #set_seed(current_seed)
    print('repeat {0}'.format(repeat_index))
    if not force:
        existing_data_paths = glob.glob(os.path.join(output_dir, '*_r{0}.json'.format(repeat_index)))
        existing_indices = [int(os.path.basename(path).split('_')[0]) for path in existing_data_paths]
    else:
        existing_indices = []
    train_data_to_infer = []
    for j in range(start_index, end_index):
        if j not in existing_indices:
            train_data_to_infer.append(copy.deepcopy(train_data[j]))
    print('total number of problems to infer for repeat {0}:'.format(repeat_index), len(train_data_to_infer))
    num_batches = math.ceil(len(train_data_to_infer) / batch_size)
    for j in tqdm(range(num_batches)):
        set_seed(current_seed)
        batch_start_index = j * batch_size
        batch_end_index = min((j + 1) * batch_size, len(train_data_to_infer))
        batch_indices = list(range(batch_start_index, batch_end_index))
        current_prompts = [train_data_to_infer[k]['prompt'] for k in range(batch_start_index, batch_end_index)]
        current_inputs, current_formatted_prompts = tokenize_with_chat_template(tokenizer, current_prompts, use_chat_template, device)
        current_outputs = generate_with_classifier_guidance(ref_model, tokenizer, logit_processor, current_inputs, generate_kwargs, True, False)
        current_outputs_text = tokenizer.batch_decode(current_outputs, skip_special_tokens=True)

        current_batch_predictions_correctness = []
        for k in range(len(current_outputs_text)):
            solution_or_answer = str(train_data_to_infer[batch_indices[k]][answer_key])
            prediction_correctness = quick_evaluate_single(dataset_type, solution_or_answer, None,
                                                           True, match_fn, current_outputs_text[k])
            current_batch_predictions_correctness.append(prediction_correctness)

        for k in range(len(current_outputs_text)):
            if 'fully_guided_predictions' not in train_data_to_infer[batch_indices[k]]:
                train_data_to_infer[batch_indices[k]]['fully_guided_predictions'] = []
            train_data_to_infer[batch_indices[k]]['fully_guided_predictions'].append(current_outputs_text[k])
            if 'fully_guided_predictions_correctness' not in train_data_to_infer[batch_indices[k]]:
                train_data_to_infer[batch_indices[k]]['fully_guided_predictions_correctness'] = []
            train_data_to_infer[batch_indices[k]]['fully_guided_predictions_correctness'].append(current_batch_predictions_correctness[k])

        # start randomly cutting responses
        # outputs_end_indices = (current_outputs == tokenizer.eos_token_id).nonzero(as_tuple=True)[1]
        outputs_end_indices = get_output_indices(current_outputs, tokenizer.eos_token_id)
        outputs_lengths = outputs_end_indices + 1
        random_cut_locations = torch.floor(torch.rand(outputs_lengths.size()).to(outputs_lengths.device) * outputs_lengths).int()
        skip_inference_flags = [random_cut_locations[k] + 1 == outputs_lengths[k] for k in range(len(outputs_lengths))]
        skip_inference_indices = [k for k in range(len(outputs_lengths)) if skip_inference_flags[k]]

        # queries is a list of unpadded input_ids
        queries = [current_inputs['input_ids'][k].masked_select(current_inputs['attention_mask'][k].to(torch.bool)) for k in range(len(current_inputs['input_ids']))]
        partial_responses = [current_outputs[k][:random_cut_locations[k] + 1] for k in range(len(current_outputs))]
        prompt_partial_response_input_ids = [torch.cat([q, r]) for q, r in zip(queries, partial_responses)]
        prompt_partial_response_input_data_exclude_skip = data_collator([{'input_ids': prompt_partial_response_input_ids[k], 'attention_mask': torch.ones_like(prompt_partial_response_input_ids[k])} for k in range(len(prompt_partial_response_input_ids)) if k not in skip_inference_indices])
        prompt_partial_response_input_data_exclude_skip.pop('labels', None)
        prompt_partial_response_input_data_exclude_skip = prompt_partial_response_input_data_exclude_skip.to(device)

        #generate first pair of responses with classifier guidance
        first_current_outputs_disabled_classifier = generate_with_classifier_guidance(ref_model, tokenizer, logit_processor_disabled, prompt_partial_response_input_data_exclude_skip, generate_kwargs, True, False)
        # current_outputs_disabled_classifier_text = tokenizer.batch_decode(current_outputs_disabled_classifier, skip_special_tokens=True)
        first_current_outputs_disabled_classifier_end_indices = get_output_indices(first_current_outputs_disabled_classifier, tokenizer.eos_token_id)
        first_current_outputs_disabled_classifier_lengths = first_current_outputs_disabled_classifier_end_indices + 1
        
        #generate second pair of responses with classifier guidance
        set_seed(current_seed+1) # changing the seed to get different responses
        second_current_outputs_disabled_classifier = generate_with_classifier_guidance(ref_model, tokenizer, logit_processor_disabled, prompt_partial_response_input_data_exclude_skip, generate_kwargs, True, False)
        # current_outputs_disabled_classifier_text = tokenizer.batch_decode(current_outputs_disabled_classifier, skip_special_tokens=True)
        second_current_outputs_disabled_classifier_end_indices = get_output_indices(second_current_outputs_disabled_classifier, tokenizer.eos_token_id)
        second_current_outputs_disabled_classifier_lengths = second_current_outputs_disabled_classifier_end_indices + 1
        
        
        partial_guided_response_pair_tokenized = []


        start_counter = 0
        for k in range(len(queries)):
            if k in skip_inference_indices:
                partial_guided_response_pair_tokenized.append([])
            else:
                current_first_partial_guided_response_tokenized = first_current_outputs_disabled_classifier[start_counter][:first_current_outputs_disabled_classifier_lengths[start_counter]].tolist()
                current_second_partial_guided_response_tokenized = second_current_outputs_disabled_classifier[start_counter][:second_current_outputs_disabled_classifier_lengths[start_counter]].tolist()
                partial_guided_response_pair_tokenized.append([current_first_partial_guided_response_tokenized, current_second_partial_guided_response_tokenized])
                start_counter += 1
        assert start_counter == len(first_current_outputs_disabled_classifier)
        assert start_counter == len(second_current_outputs_disabled_classifier)

        partial_guided_predictions = []
        for k in range(len(queries)):
            if k in skip_inference_indices:
                print('skipping inference for index:', k)
                partial_guided_predictions.append(current_outputs_text[k])
                continue
            partial_guided_first_prediction_tokenized = partial_responses[k].tolist() + partial_guided_response_pair_tokenized[k][0]
            partial_guided_second_prediction_tokenized = partial_responses[k].tolist() + partial_guided_response_pair_tokenized[k][1]
            partial_guided_first_prediction = tokenizer.decode(partial_guided_first_prediction_tokenized, skip_special_tokens=True)
            partial_guided_second_prediction = tokenizer.decode(partial_guided_second_prediction_tokenized, skip_special_tokens=True)
            partial_guided_predictions.append([partial_guided_first_prediction, partial_guided_second_prediction])

        current_batch_partial_guided_pred_preferences = []
        for k in range(len(queries)):
            solution_or_answer = str(train_data_to_infer[batch_indices[k]][answer_key])
            question = tokenizer.decode(prompt_partial_response_input_ids[k].tolist())
            inputs_response1 = reward_tokenizer(question,  partial_guided_predictions[k][0], return_tensors='pt')
            score1 = rank_model(**inputs_response1).logits[0].cpu().item()
            inputs_response2 = reward_tokenizer(question, partial_guided_predictions[k][1], return_tensors='pt')
            score2 = rank_model(**inputs_response2).logits[0].cpu().item()
            soft_pref = [1, 0] if score1 > score2 else [0, 1]
            prediction_preferences = evaluate_preference(dataset_type, solution_or_answer, None, True, match_fn, partial_guided_predictions[k][0],partial_guided_predictions[k][1],soft_pref)
            current_batch_partial_guided_pred_preferences.append(prediction_preferences)

        for k in range(len(queries)):
            if 'partial_guided_prompts_tokenized' not in train_data_to_infer[batch_indices[k]]:
                train_data_to_infer[batch_indices[k]]['partial_guided_prompts_tokenized'] = []
            train_data_to_infer[batch_indices[k]]['partial_guided_prompts_tokenized'].append(prompt_partial_response_input_ids[k].tolist())
            if 'partial_guided_prompts' not in train_data_to_infer[batch_indices[k]]:
                train_data_to_infer[batch_indices[k]]['partial_guided_prompts'] = []
            train_data_to_infer[batch_indices[k]]['partial_guided_prompts'].append(tokenizer.decode(prompt_partial_response_input_ids[k].tolist()))
            if 'num_response_tokens_in_partial_guided_prompts' not in train_data_to_infer[batch_indices[k]]:
                train_data_to_infer[batch_indices[k]]['num_response_tokens_in_partial_guided_prompts'] = []
            train_data_to_infer[batch_indices[k]]['num_response_tokens_in_partial_guided_prompts'].append(random_cut_locations[k].item() + 1)

            # from here on the data should contain pairs of responses and their preferences.
            if 'partial_guided_responses_tokenized' not in train_data_to_infer[batch_indices[k]]:
                train_data_to_infer[batch_indices[k]]['partial_guided_responses_tokenized'] = []
            train_data_to_infer[batch_indices[k]]['partial_guided_responses_tokenized'].append(partial_guided_response_pair_tokenized[k])
            if 'partial_guided_predictions' not in train_data_to_infer[batch_indices[k]]:
                train_data_to_infer[batch_indices[k]]['partial_guided_predictions'] = []
            train_data_to_infer[batch_indices[k]]['partial_guided_predictions'].append(partial_guided_predictions[k])
            # sanity check
            if k in skip_inference_indices:
                assert partial_guided_predictions[k] == current_outputs_text[k], f"skip inference prediction mismatch: {partial_guided_predictions} vs {current_outputs_text[k]}"

            if 'partial_guided_predictions_correctness' not in train_data_to_infer[batch_indices[k]]:
                train_data_to_infer[batch_indices[k]]['partial_guided_predictions_correctness'] = []
            train_data_to_infer[batch_indices[k]]['partial_guided_predictions_correctness'].append(current_batch_partial_guided_pred_preferences[k])
        # save individual problems in a batch
        for k in range(len(queries)):
            problem_id = train_eval_problems_d[train_data_to_infer[batch_indices[k]]['problem']]['id']
            current_problem_output_path = os.path.join(output_dir, f'{problem_id}_r{repeat_index}.json')
            assert not os.path.exists(current_problem_output_path), f"problem {problem_id} data already exists for repeat {repeat_index}"
            with open(current_problem_output_path, 'w') as f:
                json.dump(train_data_to_infer[batch_indices[k]], f, indent=4)
print('done')
