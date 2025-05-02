import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.cache import Cache
from utils.training_utils import accuracy


class Transformer(nn.Module):
    def __init__(self, config, block):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        # Positional encoding has to be overwritten in __init__ of subclass
        self.pos_encoding = lambda x: 0

        self.layers = nn.ModuleList(
            [block(config, layer_idx) for layer_idx in range(config.n_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.n_embd)

        if config.cache:
            # Instantiated but not occupying memory yet
            self.cache = Cache(config)
        else:
            self.cache = None

        # Initialize weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('mlp.projection.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        all_params, non_emb_params = self.get_num_params()
        print("Number of parameters: %.2fM" % (all_params/1e6,),
              " Number of non-embedding parameters: %.2fM" % (non_emb_params/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        all_params = sum(p.numel() for p in self.parameters())
        non_emb_params = all_params

        if non_embedding:
            # Count the parameters of the embedding and head if not tied
            if self.embed_tokens != self.lm_head:
                non_emb_params -= self.embed_tokens.weight.numel()
                non_emb_params -= self.lm_head.weight.numel()
            else:
                non_emb_params -= self.embed_tokens.weight.numel()
            # Subtract positional embeddings if used
            if self.pos_encoding(torch.tensor([1, 2, 3])) != 0:
                non_emb_params -= self.pos_encoding.weight

        return all_params, non_emb_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        import inspect
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer


    def forward(self, idx, targets=None, reduce_loss=True):
        device = idx.device
        bsz, seq_len = idx.size()
        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len}, block size is only " \
                                                  f"{self.config.block_size}"
        tok_emb = self.embed_tokens(idx)
        start_pos = 0 if self.cache is None or not self.cache.use_caching else self.cache.cur_seq_len[0]
        pos = torch.arange(start_pos, seq_len + start_pos, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_encoding(pos)
        x = tok_emb + pos_emb

        for block in self.layers:
            x = block(x, self.cache)

        x = self.final_layernorm(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # Calculate loss with ignore_index=-1, meaning we skip the gradient contributions from those tokens
            # which is basically the prefix tokens
            if reduce_loss:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                # transpose since input should have (N, C, d1, d2, ...) and target is (N, d1, d2, ...)
                loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-1, reduction='none')
            acc, token_acc = accuracy(logits, targets)
            accs = {"acc": acc, "token_acc": token_acc}
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss, accs = None, None

        return logits, loss, accs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        bsz, prefix_len = idx.shape
        seq_len = prefix_len + max_new_tokens - 1
        device = idx.device

        # Decode in parallel if teacherless
        if self.config.teacherless_token is not None:
            idx_next = torch.tensor(self.config.teacherless_token) * torch.ones((bsz, max_new_tokens - 1)).long()
            idx_next = idx_next.to(device)
            idx = torch.cat((idx, idx_next), dim=1)
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond, targets=idx_cond)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            out = torch.multinomial(probs.reshape((bsz * seq_len, -1)), num_samples=1).reshape((bsz, seq_len))

            return out

        out = idx.clone()
        idx_next = idx.clone()
        if self.cache is not None:
            self.set_cache(device=device)

        for i in range(max_new_tokens):
            if self.cache is not None and self.cache.use_caching:
                # If we're caching, only propagate the last token
                idx = idx_next
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((out, idx_next), dim=1)
            out = idx.clone()

        if self.cache is not None:
            self.empty_cache()

        return out

    @torch.no_grad()
    def generate_with_classifier(self, classifier, eta, guide_with_cd, idx, max_new_tokens, temperature=1.0, top_k=None):
        assert self.config.teacherless_token is None, "Cannot use classifier with teacherless token"
        assert eta > 0
        out = idx.clone()
        for i in range(max_new_tokens):
            logits, _, _ = self(idx)
            logits = logits[:, -1, :] / temperature  # [bs, vocab_size]
            v, topk_idxs = torch.topk(logits, min(top_k, logits.size(-1)))  # [bs, top_k]
            # logits are log pi_refs
            logits[logits < v[:, [-1]]] = -float('Inf')
            # inefficient for-loop impl of classifier
            classifier_logits = []
            for i in range(top_k):
                classifier_input = torch.cat([idx, topk_idxs[:, [i]]], dim=1)
                classifier_out = classifier(classifier_input, num_target_tokens=1)
                classifier_logits.append(classifier_out)
            classifier_logits = torch.cat(classifier_logits, dim=-1)  # [bs, top_k]
            bernoulli_probs = torch.sigmoid(classifier_logits)
            if guide_with_cd:
                # log (p/eta)
                qs = torch.log(bernoulli_probs / eta)  # [bs, top_k]
            else:
                # log (p*exp(1/eta) + 1-p)
                # qs = torch.log(bernoulli_probs * (math.e ** (1/eta)) + (1-bernoulli_probs))
                qs = torch.logaddexp(torch.log(bernoulli_probs) + 1/eta, torch.log(1-bernoulli_probs))  # [bs, top_k]
            logits = torch.scatter_add(logits, 1, topk_idxs, qs)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((out, idx_next), dim=1)
            out = idx.clone()
        return out

    def set_cache(self, device=None, mode=True):
        """
        Activates caching. After set_cache() memory is allocated and cache is ready to be populated
        """
        self.cache.use_caching = mode
        if mode and self.cache.key_cache is None:
            # Allocate memory for caching
            self.cache.build(device)

    def empty_cache(self):
        """
        Free memory by removing cache.
        """
        self.set_cache(mode=False)
        self.cache.delete()

    def reset_cache(self):
        """
        Set cache back to zero entries
        """
        self.cache.empty()
