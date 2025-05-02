import torch
from tqdm import tqdm
from collections import defaultdict
from utils.training_utils import AverageMeter
from functools import partial

# Function to evaluate performance when generating
@torch.no_grad()
def evaluate(model, loader, ctx, temperature, top_k, results=None, mode='test', eval_ratio=1.0, classifier=None, eta=0.1, guide_with_cd: bool=False):
    """
    Generates sequences (without teacher-forcing) and calculates accuracies
    """
    num_prefix_tokens = loader.dataset.num_prefix_tokens
    num_target_tokens = loader.dataset.num_target_tokens

    # Switch dataset and model to "eval" mode
    loader.dataset.eval()
    model.eval()
    total_acc = AverageMeter()
    tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
    bar = tqdm(loader)

    max_num_iters = round(len(loader) * eval_ratio)
    num_iters = 0

    generate_fn = model.generate if classifier is None else partial(model.generate_with_classifier, classifier, eta, guide_with_cd)

    #model.set_cache(loader.dataset.device)
    for x in bar:
        y = x[:, num_prefix_tokens:].clone()
        x = x[:, :num_prefix_tokens].clone()

        with ctx:
            y_pred = generate_fn(x, num_target_tokens, temperature=temperature, top_k=top_k)
        #model.reset_cache()

        # Check how many tokens we get right and how many predictions are completely correct
        correct = y.eq(y_pred[:, -num_target_tokens:]).float()

        # Completely correct
        completely_correct = torch.mean(correct.sum(dim=1).eq(num_target_tokens).to(torch.float))
        total_acc.update(completely_correct.item(), x.shape[0])

        # Individual token accuracy
        per_token_acc = correct.mean(dim=0)
        for i in range(num_target_tokens):
            tokens_corr[i].update(per_token_acc[i].item(), x.shape[0])

        bar.set_description(f'{mode} accuracy: {total_acc.get(percentage=True):.2f}')
        num_iters += 1
        if num_iters >= max_num_iters:
            break

    #model.empty_cache()

    # Switch back to train mode
    loader.dataset.train()
    model.train()

    if results is not None:
        results[mode + '/accuracy'] = total_acc.get(percentage=True)
        for i in range(num_target_tokens):
            results[mode + '/token_' + str(i + 1)] = tokens_corr[i].get(percentage=True)
    return results


# Function to evaluate performance when applying teacher forcing
@torch.no_grad()
def evaluate_forced(model, loader, ctx, results=None, mode='test', eval_ratio = 1.0):
    """
    Generates sequences with teacher-forcing and calculates accuracies
    """
    num_target_tokens = loader.dataset.num_target_tokens
    total_acc, total_loss = AverageMeter(), AverageMeter()
    tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
    bar = tqdm(loader)

    max_num_iters = round(len(loader) * eval_ratio)
    num_iters = 0

    for x, y in bar:
        # Produce logits with teacher-forcing (i.e. like during training)
        with ctx:
            logits, loss, accs = model(x, y)

        total_acc.update(val=accs['acc'], num=x.shape[0])
        total_loss.update(val=loss, num=x.shape[0])
        for i in range(num_target_tokens):
            tokens_corr[i].update(accs['token_acc'][i], x.shape[0])

        bar.set_description('Forced Loss: {:.4f} Forced Acc: {:.2f}'.format(total_loss.get(),
                                                              total_acc.get(percentage=True)))
        num_iters += 1
        if num_iters >= max_num_iters:
            break

    if results is not None:
        results[mode + '/forced/loss'] = total_loss.get()
        results[mode + '/forced/accuracy'] = total_acc.get(percentage=True)
        for i in range(num_target_tokens):
            results[mode + '/forced/token_' + str(i + 1)] = tokens_corr[i].get(percentage=True)

    return results


def add_prefix_to_dict(d, prefix):
    return {prefix + k: v for k, v in d.items()}


@torch.no_grad()
def evaluate_bce_loss(model_name, model, loader, num_target_tokens, ctx, prefix=""):
    model.eval()

    device = next(model.parameters()).device
    bar = tqdm(loader)
    total_loss = AverageMeter()
    avg_meter_map = defaultdict(AverageMeter)

    for d in bar:
        input_ids = torch.cat([d['y_pred_reject'], d['y_pred_chosen']], dim=0).to(device, non_blocking=True)
        labels = torch.cat([d['y_pred_reject_completely_correct'], d['y_pred_chosen_completely_correct']], dim=0).to(device, non_blocking=True)
        target = labels.unsqueeze(1).expand(-1, num_target_tokens).float()
        with ctx:
            logits = model(input_ids, num_target_tokens)
            token_loss = torch.nn.functional.binary_cross_entropy_with_logits(input=logits, target=target, reduction='none')
            avg_loss = token_loss.mean()

        for idx in range(num_target_tokens):
            batch_size = input_ids.shape[0]
            if model_name == "gpt":
                i = idx
            elif model_name.startswith("gpt2"):
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

        total_loss.update(avg_loss.item(), input_ids.shape[0] * num_target_tokens)
        bar.set_description('(Evaluation) BCE: {:.4f}, Loss: {:.4f}'.format(avg_loss.item(), total_loss.get()))

    model.train()
    return add_prefix_to_dict({'loss': total_loss.get(), **{k: v.get() for k, v in avg_meter_map.items()}}, prefix)


def decode(x, tokenizer):
    if isinstance(x, torch.Tensor):
        x = x[x != -1]
        decoded_list = tokenizer.decode(x.tolist())
    else:
        decoded_list = tokenizer.decode(x)
    decoded_list = [str(i) for i in decoded_list]
    return " ".join(decoded_list)
