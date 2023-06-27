import argparse
import json
import math
import numpy as np
import os
import pandas as pd
import pdb
import random
import torch

from collections import defaultdict, Counter
from scipy.special import softmax
from pathlib import Path


'''
acc
'''
def main(args):
    seeds = [int(s) for s in args.seeds]
    if len(args.topk) == 1 and args.topk[0] == 'all':
        exponent = int(math.log(args.num_clusters, 2))
        topk_list = [int(2**j) for j in range(exponent + 1)]
    else:
        topk_list = [int(t) if (int(t) != -1) else args.num_clusters for t in args.topk]
    print(f"{args.dataset}")
    for topk in topk_list:
        collect_accs(args, topk, seeds)


def collect_accs(args, topk, seeds):
    accs = []
    for seed in seeds:
        save_dir = os.path.join(args.mixture_folder, f'{args.n_shot}shot_seed{seed}', args.expert_outputs_dir)
        accs_path = f'{save_dir}/ensemble/{args.method}/top{topk}/accs'
        if args.num_clusters == 1:
            for root, dirs, files in os.walk(save_dir):
                if 'finetune' in root and 'accs' in files:
                    accs_path = f'{root}/accs'
        with open(accs_path, 'r') as f:
            acc_dict = json.loads(f.read().strip())
            accs.append(acc_dict['lm'])
    avg_acc = sum(accs) / len(accs)
    print(f'top {topk} : {accs} : {avg_acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--expert-outputs-dir', type=str)
    parser.add_argument('--mixture-folder', type=str, required=True)
    parser.add_argument('--topk', nargs='+', default=[-1])
    parser.add_argument('--n-shot', type=int, default=0)

    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--seeds', nargs='+', default=[0])
    parser.add_argument('--num-clusters', type=int, default=0)
    parser.add_argument('--method', type=str, default="standard", choices=["standard", "random"])
    parser.add_argument('--dev_path', type=str, default="standard")

    args = parser.parse_args()
    print(args)
    main(args)



