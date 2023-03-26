#!/usr/bin/env python
"""
Launch streaming language model fine-tuning on all environments.

"""
import os

from metaseq.fb_sweep.sweep import (
    hyperparam,
    get_env_from_args,
    main as fb_sweep_main,
)
from metaseq.constants import MODEL_SIZES, DATA_LOCATIONS, ComputeEnvs, VALID_SUBSETS
from metaseq.cbtm_constants import VOCAB_DIR, PATH_TO_1_3B_MODEL, PATH_TO_6_7B_MODEL, DATA_DIR

DEFAULT_RANDOM_SEED = 1234

# have to do this at the module level, unfortunately; unable to use args.<env>
for _cluster, _folder in DATA_LOCATIONS.items():
    if os.path.exists(_folder):
        
        
        from metaseq.fb_sweep.dependency_checks import *  # noqa
        break

PRETRAIN_MODEL_LOCATIONS = {
    ComputeEnvs.FAIR: {
        "1.3b": PATH_TO_1_3B_MODEL,
        "6.7b": PATH_TO_6_7B_MODEL
    }
}


def get_grid(args):
    grid = []
    cluster_env = get_env_from_args(args)
    DATA_ROOT = DATA_LOCATIONS[cluster_env]

    FINE_TUNE_DATA_CONFIGS = {
        "ft_data": {
            "path": args.data_dir,
            "valid_subsets": [args.valid_subset],
            "train_subsets": [args.train_subset]
        }
    }

    def H(*args, **kwargs):
        grid.append(hyperparam(*args, **kwargs))
    if args.data is None:
        if args.data_type is None:
            raise Exception(
                f"Either args.data or args.data_type arguments must be set. Available data_type(s): FINE_TUNE_DATA_CONFIGS.keys()"
            )
        assert args.data_type in FINE_TUNE_DATA_CONFIGS
        args.data = args.data_dir
        # check if given valid subsets exist otherwise select all valid subsets
        if args.valid_subset:
            avail_valid_subsets = args.valid_subset.split(',')
        else:
            avail_valid_subsets = [
                f.name for f in os.scandir(args.data) if f.is_dir() and "valid" in f.name
            ]
        if args.valid_subset not in avail_valid_subsets:
            args.valid_subset = ",".join(avail_valid_subsets)
        if args.train_subset:
            avail_train_subsets = args.train_subset.split(',')
        else:
            avail_train_subsets = [
                f.name for f in os.scandir(args.data) if f.is_dir() and "train" in f.name
            ]
        if args.train_subset not in avail_train_subsets:
            args.train_subset = ",".join(avail_train_subsets)

    size = MODEL_SIZES[args.model_size]
    if args.finetune_from_model is None and args.restore_file is None:
        args.finetune_from_model = PRETRAIN_MODEL_LOCATIONS[cluster_env][
            args.model_size
        ]

    if args.restore_file:
        H("--restore-file", args.restore_file, save_dir_key=lambda _: args.model_size)
    elif args.finetune_from_model:
        H(
            "--finetune-from-model",
            args.finetune_from_model,
            save_dir_key=lambda _: args.model_size,
        )

    grid += [
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--no-best-checkpoints"),
        hyperparam("--save-interval", args.save_interval_updates),
        hyperparam("--save-interval-updates", args.save_interval_updates),
        hyperparam("--keep-interval-updates", args.keep_interval_updates),

        hyperparam("--validate-interval-updates", args.interval),
    ]
    # hyperparam("--no-save-optimizer-state"),
    if args.validate_at_beginning:
        grid += [hyperparam("--validate-at-beginning")]
    if args.no_save:
        H("--no-save")
    else:
        H("--best-checkpoint-metric", "loss")

    if args.label_loss:
        H("--task", "streaming_instruction_finetune_language_modeling")
    else:
        H("--task", "streaming_finetune_language_modeling")
    if args.path_to_clusters_dir != "None" and args.path_to_clusters_dir is not None and args.random_clusters:
        raise ValueError("cannot set path to clusters dir and random clusters! choose one")
    if args.path_to_clusters_dir != "None" and args.path_to_clusters_dir is not None:
        H("--num-clusters", args.num_clusters, save_dir_key=lambda val: f"{args.cluster_tag}.numclusters{val}" if args.cluster_tag is not None else f"numclusters{val}")
        H("--path-to-clusters-dir", args.path_to_clusters_dir)
        H("--train-cluster", [int(x) for x in args.train_cluster.split(',')], save_dir_key=lambda val: f"cluster{val}")
    else:
        H("--num-clusters", 1, save_dir_key=lambda val: f"{args.cluster_tag}.numclusters{val}" if args.cluster_tag is not None else f"numclusters{val}")
        H("--train-cluster", 0, save_dir_key=lambda val: f"cluster{val}")
    if args.random_clusters:
        H("--random-clusters")
        H("--num-clusters", args.num_clusters, save_dir_key=lambda val: f"randomclusters.numclusters{val}")
        H("--train-cluster", [int(x) for x in args.train_cluster.split(',')], save_dir_key=lambda val: f"cluster{val}")
    
    if args.add_cluster_token:
        H("--add-cluster-token")
        H("--num-clusters", args.num_clusters, save_dir_key=lambda val: f"randomclusters.numclusters{val}")
        H("--train-cluster", [int(x) for x in args.train_cluster.split(',')], save_dir_key=lambda val: f"cluster{val}")
    

    path_to_vocab = VOCAB_DIR
    if not path_to_vocab:
        raise ValueError("VOCAB_DIR must be set in metaseq-internal.cbtm_constants")
    H("--vocab-filename", f"{path_to_vocab}/gpt2-vocab.json")# , save_dir_key=lambda _: "gpt2")
    H(
        "--merges-filename",
        f"{path_to_vocab}/gpt2-merges.txt",
    )
    H("--sample-break-mode", args.sbm) # , save_dir_key=lambda val: f"sbm_{val}")


    if args.valid_subset == "valid":
        H(
            "--combine-valid-subsets"
        )  # this by default assumes the split name as valid (in metaseq/main)
    else:
        H(
            "--valid-subset", args.valid_subset
        )  # valid sets are separated by comma and given as a string
    
    
    H(
        "--train-subset", args.train_subset
    )  # train sets are separated by comma and given as a string

    assert (
        args.tps == 2048
    ), "Fix required to allow loading learned positional embeddings with different ws"
    H("--tensor-parallel-init-model-on-gpu")
    H("--model-parallel-size", size.model_parallel)
    H("--criterion", "vocab_parallel_cross_entropy")
    H("--distribute-checkpointed-activations")
    H("--arch", "transformer_lm_megatron")
    H("--activation-fn", "relu")
    H("--decoder-learned-pos")
    H("--share-decoder-input-output-embed")
    if not args.embdr:
        H("--no-emb-dropout", save_dir_key=lambda _: "0edr")
    if args.min_loss_scale > 0:
        H("--min-loss-scale", args.min_loss_scale)
    # Add document attention seperator to efficiently finetune under streaming setting.
    if args.self_attn_doc_sep:
        H("--self-attn-doc-sep", 2, save_dir_key=lambda val: f"docsep_{val}")
    H("--checkpoint-activations", binary_flag=True) #, save_dir_key=lambda _: "ckpt")
    # this model requires checkpoint activations to load
    H("--use-sharded-state")
    H("--decoder-learned-pos")
    H("--gradient-predivide-factor", 32.0)
    H("--no-scale-embedding")
    H("--full-megatron-init")
    H("--megatron-init-sigma", 0.006)

    H("--tokens-per-sample", args.tps) #, save_dir_key=lambda val: f"tps_{val}")
    H("--ddp-backend", "fully_sharded")
    H("--save-async")
    H("--quiet")

    if args.max_valid_steps > 0:
        H("--max-valid-steps", args.max_valid_steps)

    grid.extend(
        [
            hyperparam("--decoder-layers", size.n_layers),
            hyperparam("--decoder-embed-dim", size.emb_size),
            hyperparam("--decoder-ffn-embed-dim", size.ffn_size),
            hyperparam("--decoder-attention-heads", size.n_heads),
            hyperparam("--share-decoder-input-output-embed"),
        ]
    )

    grid += [
        hyperparam("--max-update", args.max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--total-num-update", args.max_update),
        hyperparam("--warmup-updates", args.warmup_update, save_dir_key=lambda val: f"wu{val}"),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--batch-size", args.bs, save_dir_key=lambda val: f"bsz{val}"),
        # Use a fixed batch size for valid. Since we limit the max valid steps,
        # the number of valid examples should be consistent across different hyperparam
        hyperparam("--batch-size-valid", 4),
        hyperparam("--update-freq", args.uf, save_dir_key=lambda val: f"uf{val}"),
    ]

    # regularization
    dropout = args.dropout
    grid += [
        hyperparam("--dropout", dropout), #, save_dir_key=lambda val: f"dr{val}"),
        # --attention-dropout will be set to mirror --dropout in postprocess_args
        hyperparam(
            "--attention-dropout", dropout), #, save_dir_key=lambda val: f"atdr{val}"),
    ]
    if args.wd > 0:
        H("--weight-decay", args.wd, save_dir_key=lambda val: f"wd{val}")
    is_175B = args.model_size == "175b"
    H("--adam-betas", "(0.9, 0.95)")
    H("--adam-eps", 1e-6)
    H("--clip-norm", args.clip_norm, save_dir_key=lambda val: f"clip{val}" if args.clip_norm < 1.0 else "")
    if not args.no_fp16_adam:
        # H("--fp16-adam-stats")
        H("--optimizer", "adam", save_dir_key=lambda val: "fp16adam")
    else:
        H("--optimizer", "adam", save_dir_key=lambda val: "fp32adam")

    # random seed
    grid += [
        hyperparam("--seed", args.random_seed, save_dir_key=lambda val: f"rs{val}")
    ]

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", args.lr, save_dir_key=lambda val: f"lr{val}"),
    ]
    H("--end-learning-rate", args.end_learning_rate, save_dir_key=lambda val: f"endlr{val:.3g}" if args.end_learning_rate !=0 else "")

    if args.bf16:
        H("--fp16")  # this need to be set for bf16
        H("--bf16", save_dir_key=lambda _: "bf16")
    else:
        H("--fp16")

    # Below settings are not needed if using `finetune-from-model` args
    # If restore-file is set, then anyway we don't need the reset of meters
    # such that we can continue training

    # H("--reset-meters")
    # H("--reset-dataloader")
    # H("--reset-optimizer")
    H("--fp16-init-scale", 128)

    # data loading settings
    H("--num-workers", args.nw)
    H("--num-workers-valid", args.nw)

    # logging settings
    H("--log-format", "json")
    H("--log-interval", 10)
    if args.no_zero3:
        H("--no-reshard-after-forward")
    H("--patience", args.patience, save_dir_key=lambda val: f"pat_{val}")
    if args.wandb_project is not None:
        H("--wandb-project", args.wandb_project)

    return grid


def postprocess_hyperparams(args, config):
    pass


def add_args(parser):
    parser.add_argument("--model-size", choices=MODEL_SIZES.keys(), required=True)
    parser.add_argument(
        "--finetune-from-model",
        help="load an existing checkpoint for initial fine-tuning",
    )
    parser.add_argument(
        "--restore-file", help="load an existing checkpoint for continuing training"
    )
    parser.add_argument("--data-type", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--cheat", action="store_true")
    parser.add_argument(
        "--random-seed", type=int, nargs="+", default=[DEFAULT_RANDOM_SEED]
    )
    parser.add_argument("--right-trunc", action="store_true")
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-5])
    parser.add_argument("--no-fp16-adam", action="store_true")
    parser.add_argument("--valid-subset", type=str, default="valid")
    parser.add_argument("--train-subset", type=str, default="train")
    

    parser.add_argument("--max-update", "--mu", type=int, default=None)
    parser.add_argument("--tps", "--seq-len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--end-learning-rate", type=float, default=0.0)
    parser.add_argument("--uf", type=int, default=1)
    parser.add_argument("--bs", type=int, nargs="+", default=[8])
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--warmup-update", type=int, default=60)
    parser.add_argument("--interval", type=int, default=10000000)
    parser.add_argument("--save-interval-updates", type=int, default=10000000)
    parser.add_argument("--keep-interval-updates", type=int, default=1)

    parser.add_argument("--validate-at-beginning", action="store_true")
    parser.add_argument("--no-zero3", action="store_true")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min-loss-scale", type=float, default=-1)
    parser.add_argument("--sbm", type=str, default="none")
    parser.add_argument("--nw", type=int, default=0)
    parser.add_argument("--label-loss", action="store_true")
    parser.add_argument("--embdr", action="store_true")
    parser.add_argument("--eps", type=int, nargs="+", default=[-1])
    parser.add_argument("--self-attn-doc-sep", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-valid-steps", type=int, default=-1)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--pretrain-data-sampling-prob", type=float, default=0.0)
    parser.add_argument("--path-to-clusters-dir", type=str, default=None)
    parser.add_argument("--train-cluster", type=str, default=None)
    parser.add_argument("--random-clusters", action="store_true")
    parser.add_argument("--add-cluster-token", action="store_true")

    parser.add_argument("--num-clusters", type=int)
    parser.add_argument("--cluster-tag", type=str)

if __name__ == "__main__":
    fb_sweep_main(get_grid, postprocess_hyperparams, add_extra_options_func=add_args)
