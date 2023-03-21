import argparse
import os
import subprocess

from metaseq.cbtm_constants import (DEFAULT_SLURM_ACCOUNT,
                                             DEFAULT_SLURM_CONSTRAINT,
                                             DEFAULT_SLURM_PARTITION,
                                             PATH_TO_CBTM)

# using getlogin() returning username
# username = os.getlogin()

LEARNING_RATES = {
    "1.3b": 2e-5,
    "6.7b": 1.2e-5
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=['6.7b','1.3b', '125m', '350m'])
    parser.add_argument("--run", choices=['slurm', 'local'])
    parser.add_argument("--path-to-clusters-dir")
    parser.add_argument("--data-name", type=str)
    parser.add_argument("--path-to-data", type=str)
    parser.add_argument("--valid-subset", type=str, default='valid')
    parser.add_argument("--train-subset", type=str, default='train')

    parser.add_argument("--num-clusters", type=int)
    parser.add_argument("--cluster-tag", type=str)
    parser.add_argument("--train-cluster", type=str, default=None)
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000)

    parser.add_argument("--learning-rate", "-lr", type=float, default=None)
    parser.add_argument("--batch-size", "-bs", type=int, default=8)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--use-random-clusters", type=str, choices=['true', 'false'], default='false')
    parser.add_argument("--add-cluster-token", type=str, choices=['true', 'false'], default='false')

    parser.add_argument("--instruction-finetuning", type=str, choices=['true', 'false'], default='false')
    parser.add_argument('--subset', default=None)
    parser.add_argument('--slurm-partition',  type=str, default=DEFAULT_SLURM_PARTITION)
    parser.add_argument('--slurm-account', type=str, default=DEFAULT_SLURM_ACCOUNT)
    parser.add_argument('--slurm-constraint',  type=str, default=DEFAULT_SLURM_CONSTRAINT)

    args = parser.parse_args()

    if args.train_cluster is not None:
        train_cluster = args.train_cluster
    else:
        train_cluster = "None"
    world_size = args.num_gpus * args.num_nodes

    learning_rate = LEARNING_RATES[args.model_size] if args.learning_rate is None else args.learning_rate
    if not PATH_TO_CBTM:
        raise ValueError("PATH_TO_CBTM must be set in metaseq/cbtm_constants.py.")

    command = f"bash {PATH_TO_CBTM}/metaseq/scripts/train_cbtm.sh \
                {args.num_nodes} \
                {args.num_gpus} \
                {args.model_size} \
                {args.path_to_clusters_dir} \
                {args.num_clusters} \
                {train_cluster} \
                {args.run} \
                {learning_rate} \
                {args.update_freq} \
                {args.batch_size} \
                {args.use_random_clusters} \
                {args.cluster_tag} \
                {args.data_name} \
                {args.instruction_finetuning} \
                {args.subset} \
                {args.train_subset} \
                {args.valid_subset} \
                {args.path_to_data} \
                {PATH_TO_CBTM} \
                {args.slurm_partition} \
                {args.slurm_account} \
                {args.slurm_constraint} \
                {args.max_steps} \
                {args.add_cluster_token} \
                "
    subprocess.run(command.split(), check=True, text=True)