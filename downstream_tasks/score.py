from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import score
import argparse
import random
import numpy as np
import torch
import os
import pdb
from data_loaders import load_examples


def get_model(model_name, key_file):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float32).eval()
    encoder = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
    name = model_name
    return model, encoder, name

def get_examples(dataset_name, split, stem, n_shot, variant, data_seed):
    examples = load_examples(dataset_name, n_shot, data_seed)
    closed_label_space = True
    return examples, closed_label_space


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--model', type=str, default='xl')
    parser.add_argument('--n-shot', type=int, default=0)
    parser.add_argument('--variant', type=int, default=None)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('--key', type=str, default='api.key')
    parser.add_argument('--output', type=str, default=ModuleNotFoundError)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(args)

    if args.debug:
        pdb.set_trace()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, encoder, name = get_model(args.model, args.key)
    if args.dataset.endswith('-rev'):
        stem = f'data/{args.dataset[:-4]}/'
    else:
        stem = f'data/{args.dataset}/'
    examples, closed_label_space = get_examples(args.dataset, args.split, stem, args.n_shot, args.variant, args.data_seed)

    accs = score(model, args.model, encoder, examples, stem, args.split, args.batch, args.dataset, args.output)

    # print results
    print(f'{name} gets {accs}% on {args.dataset}')
    print(f"{accs['domain_cond']} & {accs['lm']} & {accs['tok_mean']} & {accs['pmi']} & {accs['dcpmi']}")
    print(f"{accs['domain_cond']}, {accs['lm']}, {accs['tok_mean']}, {accs['dcpmi']}")
