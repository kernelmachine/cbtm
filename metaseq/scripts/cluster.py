import argparse
import os
import numpy as np
import pandas as pd
import pickle
import submitit
import torch
import uuid
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from kmeans_pytorch import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from metaseq.data import JsonlDataset
from metaseq.cbtm_constants import DEFAULT_SLURM_ACCOUNT, DEFAULT_SLURM_CONSTRAINT, DEFAULT_SLURM_PARTITION

def get_shard_str(epoch, data_dir, split):
    shards = {}
    for shard_id in os.listdir(os.path.join(data_dir, split)):
        assert (
            int(shard_id) not in shards
        ), f"shard id: {shard_id} not in shards: {shards}"
        shards[int(shard_id)] = shard_id
    assert min(shards.keys()) == 0
    assert max(shards.keys()) == len(shards) - 1
    cur_shard_str = shards[(epoch - 1) % len(shards)]
    return cur_shard_str

def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


def _collate_fn(items):
    return items


def example_in_cluster(text, vectorizer, kmeans, random_clusters=False):
    if random_clusters:       
        clusters = np.random.choice(range(kmeans.n_clusters), len(text))
    else:
        clusters = kmeans.predict(torch.from_numpy(vectorizer.transform(text)))
    return list(clusters)
    

def cluster_file(file, data_dir, split, cur_shard_str, tfidf, kmeans, num_clusters, output_prefix):
    path = os.path.join(data_dir, split, cur_shard_str, file.split('/')[-1])
    output_dir = os.path.join(output_prefix, split, cur_shard_str)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output = output_dir + f"/{file.split('/')[-1]}"
    dataset = JsonlDataset(
                path=path,
                tokenizer=None,
                include_path_infos_in_jsonl_dataset=True
            )
    dataloader = DataLoader(dataset, batch_size=10000, num_workers=0, collate_fn=_collate_fn)
    zs = []
    counter = 0
    for batch in tqdm(dataloader):
        text = [x['item']['text'] for x in batch]
        ids = [x['sp_id'] for x in batch]
        cluster = example_in_cluster(text,  tfidf, kmeans, random_clusters=False)
        zs.extend([{"sp_id": x, "cluster": y.item()} for x,y in zip(ids, cluster)])
        counter += 1
    df = pd.DataFrame(zs)
    df.to_json(output, lines=True, orient='records')


def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out


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


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--path-to-clusterer")
    parser.add_argument("--num-clusters")
    parser.add_argument("--output-prefix")
    parser.add_argument("--split")
    parser.add_argument('--slurm-partition', type=str, default=DEFAULT_SLURM_PARTITION)
    parser.add_argument('--slurm-account', type=str, default=DEFAULT_SLURM_ACCOUNT)
    parser.add_argument('--slurm-constraint', type=str, default=DEFAULT_SLURM_CONSTRAINT)

    parser.add_argument('--run', type=str, default='slurm', choices=['slurm', 'local'])

    cmd_args = parser.parse_args()
    executor = submitit.AutoExecutor(folder=cmd_args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = 1
    nodes = 1
    timeout_min = 60

    kwargs = {}
    
    path_to_clusterer = Path(cmd_args.path_to_clusterer)
    kmeans = load_model(path_to_clusterer / "kmeans.pkl")
    tfidf = load_model(path_to_clusterer / "tfidf.pkl")

    def flatten(ls):
        return [y for x in ls for y in x]
    num_shards = len(os.listdir(os.path.join(cmd_args.data_dir,cmd_args.split)))
    cur_shard_strs = [get_shard_str(i, cmd_args.data_dir, cmd_args.split) for i in range(num_shards)]
    cur_shard_strs = ["00000"]
    files = {cur_shard_str: list(map(lambda x: os.path.join(cmd_args.data_dir, cmd_args.split, cur_shard_str) + "/" +  x,
                os.listdir(os.path.join(cmd_args.data_dir, cmd_args.split, cur_shard_str)))) for cur_shard_str in cur_shard_strs}
    files_ = {x: [z for z in y if 'npy' not in z] for x,y in files.items()}
    files = {}
    files_already_done = {}
    for x,y in files_.items():
        fs = [z.replace(cmd_args.data_dir, cmd_args.output_prefix) for z in y]
        files[x] = [p for z,p in zip(fs, y) if not Path(z).exists()]
        files_already_done[x] = [p for z,p in zip(fs, y) if Path(z).exists()]
    num_files = sum([len(y) for x,y in files.items()])
    num_files_already_done = sum([len(y) for x,y in files_already_done.items()])
    print(f"found {num_files} files to cluster.")
    print(f"found {num_files_already_done} files already clustered.")
    if cmd_args.run == 'slurm':
        if num_files > 0:
            
            executor.update_parameters(
                mem_gb=40 * num_gpus_per_node,
                gpus_per_node=num_gpus_per_node,
                tasks_per_node=num_gpus_per_node,  # one task per GPU
                cpus_per_task=10,
                nodes=nodes,
                timeout_min=timeout_min,  # max is 60 * 72
                # Below are cluster dependent parameters
                slurm_partition=cmd_args.slurm_partition,
                slurm_account=cmd_args.slurm_account,
                slurm_constraint=cmd_args.slurm_constraint,
                slurm_array_parallelism=len(files),
                **kwargs
            )

            executor.update_parameters(name="cluster")
            
            cmd_args.dist_url = get_init_file().as_uri()

            args = []

            for x,y in files.items():
                for ys in y:
                    ps = os.path.join(cmd_args.output_prefix, cmd_args.split, x, ys.split('/')[-1])
                    if not Path(ps).exists():
                        args.append((x, ys))

            print(f"launching {len(args)} jobs.")

            func = lambda x: cluster_file(x[1], cmd_args.data_dir, cmd_args.split, x[0], tfidf, kmeans, cmd_args.num_clusters, cmd_args.output_prefix)

            batch_size = 100
            batches = batch(args, n=batch_size)

            for batch in tqdm(batches, total=len(args) // batch_size):
                job = executor.map_array(func, batch)
    else:
        args = []
        for x,y in files.items():
            for ys in y:
                ps = os.path.join(cmd_args.output_prefix, cmd_args.split, x, ys.split('/')[-1])
                if not Path(ps).exists():
                    args.append((x, ys))
        for x in args:
            cluster_file(x[1], cmd_args.data_dir, cmd_args.split, x[0], tfidf, kmeans, cmd_args.num_clusters, cmd_args.output_prefix)
