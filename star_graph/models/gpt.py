import torch
import torch.nn as nn

from models.config import GPTConfig
from models.lib import Attention, MLP, LayerNorm
from models.base_model import Transformer
from utils.load import load_gpt


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Attention(config, layer_idx, rotary=False)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln_1(x), cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(Transformer):
    def __init__(self, config):
        super().__init__(config, block=Block)
        # Add positional encoding
        self.pos_encoding = nn.Embedding(config.block_size, config.n_embd)
        # Tie weights
        # TODO: should this be flipped? (i.e. self.lm_head.weight = self.embed_tokens.weight)
        self.embed_tokens.weight = self.lm_head.weight
        # added init pos_encoding
        self._init_weights(self.pos_encoding)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.pos_encoding.weight = nn.Parameter(self.pos_encoding.weight[:block_size])
        for block in self.layers:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, teacherless_token=None, max_bsz=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # only dropout can be overridden see more notes below
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layers=12, n_heads=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layers=24, n_heads=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layers=36, n_heads=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layers=48, n_heads=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        config_args['teacherless_token'] = teacherless_token
        config_args['max_bsz'] = max_bsz

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Match the two checkpoints
        sd = load_gpt(sd, sd_hf)
        model.load_state_dict(sd, strict=True)

        return model


class GPTClassifier(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = torch.nn.Linear(config.n_embd, 1)
        self._init_weights(self.lm_head)

    @classmethod
    def from_gpt_model(cls, gpt_model):
        print("Constructing Classifier from GPT model")
        config = gpt_model.config
        classifier = GPTClassifier(config)
        state_dict = gpt_model.state_dict()
        for key in list(state_dict.keys()):
            if key.startswith("lm_head"):
                state_dict.pop(key)
        classifier.load_state_dict(state_dict, strict=False)
        classifier.lm_head = torch.nn.Linear(config.n_embd, 1)
        classifier._init_weights(classifier.lm_head)
        return classifier

    def forward(self, idx, num_target_tokens):
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

        x = self.final_layernorm(x)  # [bs, seq_len, hidden_dim]
        x = x[:, -num_target_tokens:]  # [bs, num_target_tokens, hidden_dim]
        logits = self.lm_head(x)  # [bs, num_target_tokens, 1]
        return logits.squeeze(-1)  # [bs, num_target_tokens]


if __name__ == "__main__":
    import types
    from tokenizing import get_tokenizer

    args = types.SimpleNamespace()
    args.model = 'gpt2'
    tokenizer = get_tokenizer(args)

    model = GPT.from_pretrained(model_type='gpt2')
    model.eval()
    text = "Hello my name is"
    idx = torch.tensor(tokenizer.encode(text), dtype=torch.int32).unsqueeze(0)
    #model.set_cache(device='cpu')
    out = model.generate(idx, max_new_tokens=54, top_k=1)
    print(tokenizer.decode(out.numpy().squeeze()))
