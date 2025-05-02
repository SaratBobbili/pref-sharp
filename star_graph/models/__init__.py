from models.gpt import GPT, GPTClassifier
from models.pythia import Pythia
from models.config import GPTConfig


def get_model(args, is_classifier=False):
    if args.model == 'gpt':
        config = GPTConfig(n_layers=args.n_layer, n_heads=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                           bias=True, vocab_size=args.vocab_size, dropout=args.dropout, use_flash=args.use_flash,
                           teacherless_token=args.teacherless_token, max_bsz=args.batch_size, mlp_expansion_factor=args.mlp_expansion_factor)
        model = GPT(config)
        if is_classifier:
            model = GPTClassifier.from_gpt_model(model)

    elif args.model.startswith('gpt2'):
        model = GPT.from_pretrained(args.model, teacherless_token=args.teacherless_token, max_bsz=args.batch_size)
        if args.block_size < 1024:
            model.crop_block_size(args.block_size)

        if is_classifier:
            model = GPTClassifier.from_gpt_model(model)

    elif args.model.startswith('pythia'):
        model = Pythia.from_pretrained(args.model, teacherless_token=args.teacherless_token)

    return model
