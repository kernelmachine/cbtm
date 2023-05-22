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


def get_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float32).eval()
    encoder = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
    name = model_name
    return model, encoder, name

def get_examples(dataset_name, split, n_shot, data_seed):
    examples = load_examples(dataset_name, split, n_shot, data_seed)
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str) #just here so the slurm script doesn't complain
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--mixture-folder', type=str, required=True)
    parser.add_argument('--n-shot', type=int, default=0)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)

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

    model, encoder, name = get_model(args.model_path)
    examples = load_examples(args.data_dir, args.n_shot, args.seed)

    accs = score(model, args.model_path, encoder, examples, args.batch, args.mixture_folder)

    # print results
    print(f'{name} gets {accs}% on {args.dataset}')
    print(f"{accs['domain_cond']} & {accs['lm']} & {accs['tok_mean']} & {accs['pmi']} & {accs['dcpmi']}")
    print(f"{accs['domain_cond']}, {accs['lm']}, {accs['tok_mean']}, {accs['dcpmi']}")