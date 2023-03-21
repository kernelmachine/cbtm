# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import mmap
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm
import pickle


logger = logging.getLogger(__name__)





class JsonlDataset(torch.utils.data.Dataset):
    """
    For loading JSONL data and encoding on-the-fly with a given tokenizer.

    JSONL format is expected to roughly follow that of The Pile.
    One-line-per-document of the form:
    ```
    {
        "text": "text goes here, with newlines",
        "meta": {"pile_set_name": "name of corpus", "other": "metadata"}
    }
    ```

    Note that only the "text" key is used.
    """

    def __init__(self, path: str, tokenizer_func: Optional[Callable] = None, tokenizer=None, include_path_infos_in_jsonl_dataset: bool = False, recache=False, path_to_clusters=None, train_cluster=None, num_clusters=None, random_clusters=False):
        self.path = path
        self.tokenizer_func = tokenizer_func
        self.tokenizer = tokenizer
        self.include_path_infos_in_jsonl_dataset = include_path_infos_in_jsonl_dataset
        self.clusters = {}
        self.random_clusters = random_clusters
        self.num_clusters = num_clusters
        if not self.random_clusters and path_to_clusters is not None:
            for file in os.listdir(path_to_clusters):
                df = pd.read_json(os.path.join(path_to_clusters,  file), lines=True)
                self.clusters = {**self.clusters, **dict(zip(df.sp_id, df.cluster))}
                # with open(os.path.join(path_to_clusters,  file), 'r') as f:
                    
                #     for line in tqdm(f):
                #         l = json.loads(line)
                #         self.clusters[l['sp_id']] = int(l['cluster'])
        self.train_cluster = train_cluster
        self.threadlocal = threading.local()
        # TODO(susan): Fix this fairseq reference. _build_index fails otherwise.
        self.cache = Path(f"{path}.fairseq.idx.npy")
        if self.cache.exists() and not recache:
            self.offsets = np.load(self.cache)
        else:
            self.offsets = self._build_index(path)
            np.save(self.cache, self.offsets)
        # print(f'n offsets: {len(self.offsets)}')

    def _get_mmap(self):
        if not hasattr(self.threadlocal, "handles"):
            f = open(self.path, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.threadlocal.handles = [f, mm]
            if (
                self.path.endswith(".gz")
                or self.path.endswith(".bz")
                or self.path.endswith(".bz2")
            ):
                raise NotImplementedError(
                    "Compressed files are not supported because .seek() would require "
                    "rereading the entire file, making performance too slow."
                )
        return self.threadlocal.handles[-1]

    def get_cluster(self, item):
        if self.random_clusters:       
            return np.random.choice(2, 1)
        else:
            return self.clusters[item['sp_id']]
        

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError
        f = self._get_mmap()
        f.seek(self.offsets[idx])
        item = f.readline().decode("utf-8")
        item = json.loads(item)
        if self.clusters or self.random_clusters:
            if not self.random_clusters:
                cluster = self.clusters.get(f"{self.path}|{idx}", -1)
            else:
                cluster = np.random.choice(self.num_clusters, 1)
            
            if self.train_cluster is not None and cluster != self.train_cluster:
                
                return {"item": None, "cluster": cluster}
        else:
            cluster = -1
        if self.tokenizer_func is not None:
            item = self.tokenizer_func(item)
        # v = self.context_clusterer(item.tolist())
        if self.include_path_infos_in_jsonl_dataset:
            return {
                "item": item,
                "sp_id": f"{self.path}|{idx}",
                "cluster": cluster
            }
        else:
            return {
                "item": item,
                "cluster": cluster
            }

    def __len__(self):
        return len(self.offsets)

    def _build_index(self, path: str):
        """Build index of start positions of each line."""
        logger.info(f"Building index for file: {path}")
        f = self._get_mmap()
        f.seek(0)
        offsets = []
        cur = 0
        while True:
            line = f.readline()
            if line == b"":
                break
            offsets.append(cur)
            cur += len(line)
        return offsets

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != "threadlocal":
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, "handles"):
            # cleanup files we opened on initialization
            while self.threadlocal.handles:
                self.threadlocal.handles.pop().close()

    @staticmethod
    def exists(path):
        return os.path.exists(path)


if __name__ == "__main__":
    """Usage:
    python metaseq/data/jsonl_dataset.py "flan_streaming/valid/00/*.jsonl"
    """
    parser = argparse.ArgumentParser(
        description="Precompute index file from JSONL files"
    )
    parser.add_argument(
        "pattern", help="glob to jsonl files, e.g. flan_streaming/valid/00/*.jsonl"
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    from glob import glob

    from tqdm import tqdm

    for f in tqdm(list(glob(args.pattern))):
        JsonlDataset(f, recache=True)
