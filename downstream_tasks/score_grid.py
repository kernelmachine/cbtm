#!/usr/bin/env python
"""
Launch streaming language model fine-tuning on all environments.

"""
import os
import re

from metaseq.fb_sweep.sweep import (
    hyperparam,
    main as fb_sweep_main,
)

DEFAULT_RANDOM_SEED = 1234

# have to do this at the module level, unfortunately; unable to use args.<env>
def get_grid(args):
    grid = []

    def H(*args, **kwargs):
        grid.append(hyperparam(*args, **kwargs))
    H("--dataset", args.dataset)
    H("--data-dir", args.data_dir)
    H("--n-shot", args.n_shot)
    H("--split", args.split)
    H("--batch", args.batch)
    H("--seed", args.seed)
    H("--mixture-folder", args.mixture_folder)

    models = []
    for name, _folders, files in os.walk(args.models_parent_folder):
        # regex = re.compile(re_string) if re_string else None
        if args.model_file_name not in files:
            continue #no model file found
        models.append(name)

    H("--model-path", models, save_dir_key=lambda x: os.path.basename(x))
    return grid


def postprocess_hyperparams(args, config):
    pass


def add_args(parser):
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--models-parent-folder', type=str, required=True)
    parser.add_argument('--model-file-name', type=str)
    parser.add_argument('--n-shot', type=int, default=0)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch', type=int, default=10)
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--key', type=str, default='api.key')
    parser.add_argument('--mixture-folder', type=str, default='output')
    # parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--script', default='score.py')
    return parser.parse_args()

if __name__ == "__main__":
    fb_sweep_main(get_grid, postprocess_hyperparams, add_extra_options_func=add_args)
