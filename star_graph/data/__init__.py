from data.graphs import Graphs
from data.chess import Chess


def get_dataset(args, tokenizer, device, data_path, n_sample=None):
    if args.teacherless and tokenizer.name == 'numeral':
        teacherless_token = tokenizer.encode('$')[0]
    elif args.teacherless:
        teacherless_token = tokenizer.encode('$')[0]
    else:
        teacherless_token = None
    return Graphs(tokenizer=tokenizer, data_path=data_path, device=device,
                  teacherless_token=teacherless_token, reverse=args.reverse, n_sample=n_sample)
