#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import json
import uuid
import logging
import math
import os
import random
import sys
import time
import numpy as np
from argparse import Namespace
import argparse
from pathlib import Path
import torch
import submitit
from omegaconf import DictConfig

from metaseq import checkpoint_utils, distributed_utils, options, tasks, utils
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap
from metaseq.sequence_scorer_btm import SequenceScorerBTM

import numpy as np
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from metaseq.cbtm_constants import VOCAB_DIR, DEFAULT_SLURM_ACCOUNT, DEFAULT_SLURM_PARTITION, DEFAULT_SLURM_CONSTRAINT

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("metaseq_cli.eval_lm")


def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


def validate_btm(itr, cfg, models, scorer, precomputed_prior=None, topk=-1, target_dictionary=None):
    score_sum = 0
    count = 0
    device = torch.distributed.get_rank()
    all_reduce = torch.distributed.is_initialized()
    if precomputed_prior is not None:
        assert len(precomputed_prior) == torch.distributed.get_world_size()
        priors = np.array(precomputed_prior)
    else:
        priors = np.array([1/torch.distributed.get_world_size()] * torch.distributed.get_world_size())

    ppls_all = []
    pbar = tqdm(enumerate(itr))
    for k, sample in pbar:
        
        if (
            cfg.dataset.max_valid_steps is not None
            and k > cfg.dataset.max_valid_steps
        ):
            break
        sample_cuda = utils.move_to_cuda(sample, device=device % 8)
        sample_cuda['net_input'] = sample_cuda['net_input']['src_tokens']
        hypos = scorer.generate(models, sample_cuda, ensemble=True, prior=priors, ensemble_weighted_average=True, all_reduce=all_reduce, topk=topk)
        
        expert_probs = []
        for i, hypos_i in enumerate(hypos):
            hypo = hypos_i[0]
            tokens = hypo["tokens"]
            pos_scores = hypo["positional_scores"].float()
            inf_scores = pos_scores.eq(float("inf")) | pos_scores.eq(float("-inf"))
            if inf_scores.any():
                if target_dictionary:
                    logger.info(
                        "skipping tokens with inf scores:",
                        target_dictionary.string(tokens[inf_scores.nonzero()]),
                    )
                pos_scores = pos_scores[(~inf_scores).nonzero()]
            score_sum += pos_scores.sum().cpu()
            count += pos_scores.numel()
            expert_ps = hypo['expert_probs'].mean(1).unsqueeze(0).cpu().numpy() if hypo['expert_probs'] is not None else []
            expert_probs.append(expert_ps)
        expert_probs = np.concatenate(expert_probs, 0)
        avg_nll_loss = get_aggregated_loss(score_sum, count)  # convert to base 2
        ppl = 2 ** avg_nll_loss
        ppls_all.append(ppl)
        if torch.distributed.get_rank() == 0:
            pbar.set_description(f"ppl: {ppl}")
    avg_nll_loss = get_aggregated_loss(score_sum, count) 
    ppl = 2 ** avg_nll_loss
    return ppl, None


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(cfg: DictConfig, cur_shard_str, output_dir, path_to_clusterer, random_clusters, temperature, cluster_ratio, average, topk, ensemble_type, output_prefix, **unused_kwargs):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    seed_everything(42)
    if cfg.eval_lm.context_window > 0:
        # reduce tokens per sample by the required context window size
        cfg.task.tokens_per_sample -= cfg.eval_lm.context_window

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    load_sharded = False
    rank = distributed_utils.get_data_parallel_rank()

    if load_sharded:
        cfg.checkpoint.checkpoint_suffix += f"-shard{rank}"

    # Pack multiple documents into one batch, but don't split documents across batches.
    cfg.task.sample_break_mode = "eos_pad_8"

    # Initialize the task using the current *cfg*
    task = tasks.setup_task(cfg.task)

    def _build_fn(train_cfg):
        # cfg.distributed_training must be passed in (using args like ddp-backend and --use-sharded-state.
        # It is not inferred from the checkpoint
        extra = {
            "use_sharded_state": load_sharded,
        }
        with fsdp_enable_wrap(cfg.distributed_training, **extra):
            model = fsdp_wrap(task.build_model(train_cfg.model))

        return model

    # Load ensemble
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        build_model_hook = lambda cfg, _: _build_fn(cfg)
    else:
        build_model_hook = None

    model_overrides = {}

    
    model_overrides["batch_size_valid"] = cfg.dataset.batch_size
    if average:
        load_models = utils.split_paths(cfg.common_eval.path)
    else:
        load_models = [utils.split_paths(cfg.common_eval.path)[torch.distributed.get_rank()]]
    models, _, task = checkpoint_utils.load_model_ensemble_and_task(
        load_models,
        arg_overrides=model_overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
        task=task,
        build_model_hook=build_model_hook,    )
        
    if not average:
        for model in models:
            model.to(torch.distributed.get_rank() % 8)

    for model in models:
        if cfg.common.fp16:
            model.half()
    for model in models:
        model.prepare_for_inference_(cfg)
        model.eval()
        logger.info(
            "num. model params: {:,}".format(sum(p.numel() for p in model.parameters()))
        )

    # Load dataset split
    task.load_dataset(cfg.dataset.valid_subset, combine=False, cur_shard_str=cur_shard_str)
    eval_split = cfg.dataset.valid_subset

    scorer = SequenceScorerBTM(task.target_dictionary,
                                tokenizer=task.tokenizer,
                                temperature=temperature,
                                ensemble_type=ensemble_type,
                                num_clusters=torch.distributed.get_world_size() if not average else len(models),
                                path_to_clusterer=path_to_clusterer,
                                cluster_ratio=cluster_ratio,
                                average=average,
                                random_clusters=random_clusters)
    
    itr = task.get_batch_iterator(
            dataset=task.dataset(eval_split),
            max_tokens=cfg.dataset.max_tokens_valid,
            max_sentences=cfg.dataset.batch_size_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                model.max_positions(),
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=1,
            shard_id=0,
            num_workers=cfg.dataset.num_workers_valid,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=cfg.dataset.data_buffer_size,
            disable_iterator_cache=True,
            skip_remainder_batch=False,
        ).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )

    ppl, final_prior = validate_btm(itr, cfg, models, scorer, topk=topk, target_dictionary=task.target_dictionary)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{output_prefix}.json", "w+") as f:
        json.dump({"dataset": eval_split, f"{output_prefix}_ppl": ppl.item(), f'{output_prefix}_prior': final_prior}, f)    

    
def get_aggregated_loss(score_sum, count):
    return (
        -score_sum / count / math.log(2) if count > 0 else 0
    )


def build_eval_config(
    model_paths,
    data_dir,
    data_subset,
    vocab_dir,
    metrics="",
    savedir="",
    num_gpus_per_node=8,
    max_valid_steps=None,
    path_to_clusters_dir=None,
    distributed_port=56000,
    eval_cluster=None,
):
    model_overrides = {
        "bpe": "hf_byte_bpe",
        "bpe_merges": os.path.join(vocab_dir, "gpt2-merges.txt"),
        "merges_filename": os.path.join(vocab_dir, "gpt2-merges.txt"),
        "bpe_vocab": os.path.join(vocab_dir, "gpt2-vocab.json"),
        "vocab_filename": os.path.join(vocab_dir, "gpt2-vocab.json"),
        "bpe_add_prefix_space": False,
        "specify_arch": True,
        "tensor_parallel_init_model_on_gpu": True,
    },

    eval_lm_input_args = (
        [data_dir]
        + ["--memory-efficient-fp16"]
        + ["--fp16"]
        + ["--bpe"] + ["hf_byte_bpe"]
        + ["--bpe-merges"] + [os.path.join(vocab_dir, "gpt2-merges.txt")]
        + ["--merges-filename"] + [os.path.join(vocab_dir, "gpt2-merges.txt")]
        + ["--bpe-vocab"] + [os.path.join(vocab_dir, "gpt2-vocab.json")]
        + ["--vocab-filename"] + [os.path.join(vocab_dir, "gpt2-vocab.json")]
        + ["--distributed-world-size"] + [str(num_gpus_per_node)]
        + ["--distributed-port"] + [str(distributed_port)]
        + ["--all-gather-list-size"] + ["200000000"]
        + ["--ddp-backend"] + ["c10d"] 
        
    )

    if path_to_clusters_dir is not None:
        eval_lm_input_args = (eval_lm_input_args
                                + ["--path-to-clusters-dir"] + [str(path_to_clusters_dir)])

    parser = options.get_eval_lm_parser(default_task="streaming_finetune_language_modeling")
    args = options.parse_args_and_arch(parser, eval_lm_input_args)
    metaseq_cfg = convert_namespace_to_omegaconf(args)

    metaseq_cfg.common_eval.path = ":".join(model_paths)
    metaseq_cfg.task.data = data_dir
    metaseq_cfg.dataset.valid_subset = data_subset
    metaseq_cfg.dataset.train_subset = "train"

    if path_to_clusters_dir is not None:
        metaseq_cfg.task.path_to_clusters_dir = path_to_clusters_dir
    if eval_cluster is not None:
        metaseq_cfg.task.train_cluster = eval_cluster
    metaseq_cfg.common_eval.model_overrides = model_overrides
    metaseq_cfg.common.fp16 = True
    metaseq_cfg.common.seed = 42
    # Various settings to keep metaseq happy
    metaseq_cfg.dataset.batch_size = 1
    metaseq_cfg.task.tokens_per_sample = 2048
    metaseq_cfg.dataset.batch_size_valid = 1
    metaseq_cfg.dataset.required_batch_size_multiple = 1
    metaseq_cfg.dataset.max_valid_steps = max_valid_steps

    metaseq_cfg.common.log_format = "json"
    metaseq_cfg.task.compute_data_pruning_metrics = metrics
    metaseq_cfg.task.compute_data_pruning_metrics_savedir = savedir

    metaseq_cfg.common.model_parallel_size = 1

    return metaseq_cfg


def call_main(cfg, main, **kwargs):
    cfg.distributed_training.distributed_port = 56000
    if cfg.distributed_training.distributed_init_method is None:
        distributed_utils.infer_init_method(cfg.distributed_training)

    if cfg.distributed_training.distributed_init_method is not None:
        # distributed training
        if not cfg.distributed_training.distributed_no_spawn:
            start_rank = cfg.distributed_training.distributed_rank
            cfg.distributed_training.distributed_rank = None  # assign automatically
            kwargs["start_rank"] = start_rank
            return distributed_utils._spawn_helper(main, cfg, kwargs)
        else:
            return distributed_utils.distributed_main(
                cfg.distributed_training.device_id, main, cfg, kwargs
            )
    else:
        # single GPU main
        return main(cfg, **kwargs)

def add_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-paths", 
        type=str,
    )
    parser.add_argument(
        "--data-dir", 
        type=str,
    )
    parser.add_argument(
        "--data-subset", 
        type=str,
        nargs='+',
        default="valid_small/c4",
    )
    parser.add_argument(
        "--eval-cluster",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--shard", 
        type=str,
        default="00",
    )
    parser.add_argument(
        "--temperature", 
        type=float,
        default=0.1,
    )
    
    parser.add_argument(
        "--cluster-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--ensemble-type", 
        choices=['bayes', 'clustering', 'product_of_experts'], 
        default='clustering'
    )
    parser.add_argument(
        "--max-valid-steps", 
        type=int,
        default=None,
    )
    parser.add_argument(
        "--path-to-clusterer", 
        type=str,
        default=None,
    )
    parser.add_argument(
        "--path-to-clusters-dir", 
        type=str,
        default=None,
    )
    parser.add_argument(
        "--job-dir", 
        type=str,
    )
    parser.add_argument(
        "--output-prefix", 
        type=str,
        default="result",
    )
    parser.add_argument(
        "--average", 
        action='store_true',
    )
    parser.add_argument(
        "--use-random-port",
        action='store_true',
    )
    parser.add_argument(
        "--random-clusters", 
        action='store_true',
    )
    parser.add_argument(
        "--submitit", 
        action='store_true',
    )
    parser.add_argument(
        "--slurm-partition", 
        type=str,
        default=DEFAULT_SLURM_PARTITION
    )
    parser.add_argument(
        "--slurm-account", 
        type=str,
        default=DEFAULT_SLURM_ACCOUNT
    )
    parser.add_argument(
        "--slurm-constraint", 
        type=str,
        default=DEFAULT_SLURM_CONSTRAINT
    )
    parser.add_argument(
        "--debug", 
        action='store_true',
    )

    cmd_args = parser.parse_args()
    cmd_args.model_paths = cmd_args.model_paths.split(',')
    return cmd_args

def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/tmp/").is_dir():
        p = Path(f"/tmp/")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file



def builder(cmd_args, dataset, port, eval_cluster, temperature):
    metaseq_cfg = build_eval_config(
        cmd_args.model_paths,
        cmd_args.data_dir,
        dataset,
        VOCAB_DIR,
        max_valid_steps=cmd_args.max_valid_steps,
        path_to_clusters_dir=cmd_args.path_to_clusters_dir,
        eval_cluster=eval_cluster
    )
    precomputed_prior = None
    cmd_args.output_dir = cmd_args.job_dir
    call_main(metaseq_cfg,
                main,
                cur_shard_str=cmd_args.shard,
                precomputed_prior=precomputed_prior,
                dataset=dataset,
                path_to_clusterer=cmd_args.path_to_clusterer,
                output_dir=cmd_args.job_dir,
                port=port,
                ensemble_type=cmd_args.ensemble_type,
                output_prefix=cmd_args.output_prefix,
                eval_cluster=eval_cluster,
                temperature=temperature,
                cluster_ratio=cmd_args.cluster_ratio,
                random_clusters=cmd_args.random_clusters,
                average=cmd_args.average,
                use_random_port=cmd_args.use_random_port,
                topk=cmd_args.topk)
    

if __name__ == "__main__":
    cmd_args = add_args()

    cmd_args.eval_cluster = [int(x) for x in cmd_args.eval_cluster.split(",")] if cmd_args.eval_cluster is not None else None

    if cmd_args.debug:
        cmd_args.data_subset = [cmd_args.data_subset[0]]

    if cmd_args.eval_cluster is not None and len(cmd_args.eval_cluster) > 1:
        func = lambda x: builder(cmd_args, cmd_args.data_subset[0], x[1], x[0], cmd_args.temperature)
    else:
        func = lambda x: builder(cmd_args, x[0], x[1], None, cmd_args.temperature)


    if cmd_args.eval_cluster is not None and len(cmd_args.eval_cluster) > 1:
        random_ports = np.random.randint(1024, 56000, len(cmd_args.eval_cluster))
    else:
        random_ports = np.random.randint(1024, 56000, len(cmd_args.data_subset))

    if cmd_args.submitit:
        executor = submitit.AutoExecutor(folder=cmd_args.job_dir, slurm_max_num_timeout=30)

        num_gpus_per_node = 8 if len(cmd_args.model_paths) > 8 else len(cmd_args.model_paths)
        num_tasks_per_node = 8 if len(cmd_args.model_paths) > 8 else len(cmd_args.model_paths)
        nodes = len(cmd_args.model_paths) // 8 if len(cmd_args.model_paths) > 8 else 1
        timeout_min = 30

        kwargs = {}

        executor.update_parameters(
            mem_gb=100,
            gpus_per_node=num_gpus_per_node,
            tasks_per_node=num_tasks_per_node,  # one task per GPU
            cpus_per_task=4,
            nodes=nodes,
            timeout_min=timeout_min,  # max is 60 * 72
            # Below are cluster dependent parameters
            slurm_partition=cmd_args.slurm_partition,
            slurm_constraint=cmd_args.slurm_constraint,
            slurm_account=cmd_args.slurm_account,
            slurm_array_parallelism=(len(cmd_args.eval_cluster) 
                                     if cmd_args.eval_cluster is not None and len(cmd_args.eval_cluster) > 1 else len(cmd_args.data_subset)),
            **kwargs
        )

        executor.update_parameters(name="eval")


        cmd_args.dist_url = get_init_file().as_uri()

        if cmd_args.eval_cluster is not None and len(cmd_args.eval_cluster) > 1:
            print(f"launching {len(cmd_args.eval_cluster)} jobs.")
        else:
            print(f"launching {len(cmd_args.data_subset)} jobs.")

        if cmd_args.eval_cluster is not None and len(cmd_args.eval_cluster) > 1:
            args = zip(cmd_args.eval_cluster, random_ports)
        else:
            args = zip(cmd_args.data_subset, random_ports)
        job = executor.map_array(func, args)

    else:
        if cmd_args.eval_cluster is not None and len(cmd_args.eval_cluster) > 1:
            for x in zip(cmd_args.eval_cluster, random_ports):
                func(x)
        else:
            for x in zip(cmd_args.data_subset, random_ports):
                func(x)

