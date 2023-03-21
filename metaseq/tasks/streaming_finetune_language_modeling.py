# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Streaming Language Modeling task that loads corpora in src-tgt format and performs
on-the-fly tokenization.
"""

import logging
import os
from typing import Any, Dict, List

import torch
from tqdm import trange
from metaseq.data import (
    JsonlDataset,
    StreamingShuffleDataset,
    StreamingTokenBlockDataset,
    data_utils,
)
from metaseq.tasks.streaming_language_modeling import (
    StreamingLanguageModelingTask,
    StreamingLanguageModelingConfig,
)
from metaseq.tasks import register_task
from tqdm.auto import tqdm
import json
import pickle

logger = logging.getLogger(__name__)
def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out

@register_task(
    "streaming_finetune_language_modeling", dataclass=StreamingLanguageModelingConfig
)
class StreamingFinetuneLanguageModelingTask(StreamingLanguageModelingTask):
    def _tokenize_src_tgt_json(self, json):
        src = json["text"].rstrip(" ")
        full_tokens = torch.LongTensor(
            self.tokenizer.encode(src).ids + [self.eod]
        )
        # src_tokens_len = len(self.tokenizer.encode(src).ids)
        # tgt_tokens = torch.clone(full_tokens)
        # tgt_tokens[:src_tokens_len] = self.dictionary.pad_index
        return full_tokens
    

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        The folder structure is assumed to look like:

            /path/to/data/train/00/foo.jsonl
            /path/to/data/train/00/bar.jsonl
            /path/to/data/train/01/foo.jsonl
            /path/to/data/train/01/bar.jsonl
            /path/to/data/valid/00/foo.jsonl
            /path/to/data/valid/00/bar.jsonl

        In this example, we have two "shards" of training data, which will be
        iterated over in epochs 1 and 2, respectively. Subsequent epochs will
        cycle back over the same data. We also have two different data sources
        in each shard (foo and bar), which will be combined and shuffled.

        Each jsonl entry is a dict with "src" and "tgt" keys. Loss is computed
        only on the tgt tokens.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        # This function reads a bunch of jsonl files, concats them together,
        # shuffles them, then chunks them into blocks of tokens (e.g., 2048).
        # determine number of shards for this split shards = {}
        if "valid" in split:
            cur_shard_str = self.get_shard_str(1, split)
        else:
            cur_shard_str = self.get_shard_str(epoch, split)

        # concatenate any jsonl files that are part of the shard
        datasets, corpora = [], []
        # if 'train' in split and self.args.path_to_clusters is not None:
        #     clusters = {}
        #     with open(self.args.path_to_clusters, 'r') as f:
        #         for line in tqdm(f):
        #             l = json.loads(line)
        #             clusters[l['sp_id']] = l['cluster']
        # else:
        #     clusters = None
        #     self.clusters = pd.read_json(path_to_clusters, lines=True)
        #     self.clusters = dict(zip(self.clusters.sp_id, self.clusters.cluster))
        # if self.args.path_to_clusterer is not None:
        #     kmeans = load_model(self.args.path_to_clusterer + '/kmeans.pkl')
        #     vectorizer = load_model(self.args.path_to_clusterer + '/tfidf.pkl')
        for file in tqdm(sorted(
            os.listdir(os.path.join(self.args.data, split, cur_shard_str))
        )):
            if not file.endswith(".jsonl"):
                continue
            if self.args.path_to_clusters_dir is not None:
                path_to_clusters = os.path.join(self.args.path_to_clusters_dir, split, cur_shard_str)
            else:
                path_to_clusters = None
            datasets.append(
                JsonlDataset(
                    path=os.path.join(self.args.data, split, cur_shard_str, file),
                    tokenizer_func=self._tokenize_src_tgt_json,
                    tokenizer=self.tokenizer,
                    train_cluster=self.args.train_cluster,
                    path_to_clusters=path_to_clusters,
                    random_clusters=self.args.random_clusters,
                    num_clusters=self.args.num_clusters
                )
            )
            corpora.append(os.path.splitext(file)[0])
        assert len(datasets) > 0

        # if (
        #     self.args.multicorpus_sampling_alpha != 1
        #     or self.args.multicorpus_sampling_maximum > 0
        # ):
        #     datasets = self._alpha_sampling(datasets, corpora, epoch)

        dataset = torch.utils.data.ConcatDataset(datasets)

        # shuffle order across epochs
        # dataset = StreamingShuffleDataset(dataset, seed=self.args.seed)

        self.datasets[split] = StreamingTokenBlockDataset(
            dataset,
            # We generate blocks with one extra token, so that we have a target
            # for the final input token. This results in slight data loss.
            block_size=self.args.tokens_per_sample + 1,
            break_mode=self.args.sample_break_mode,
            # we drop the remainder block during training
            drop_last=(split == "train"),
            padding_idx=self.source_dictionary.pad(),
            # 1284 is a randomly-generated offset to decouple the seed used here
            # from the seed used above in StreamingShuffleDataset
            # TODO: Track this seed to avoid collisions. See issue #65
            seed=1284 + self.args.seed,
        )

    def _collate_fn(self, items: List[Dict[str, Any]]):
        # StreamingTokenBlockDataset returns None as filler
        if len([x for x in items if x is not None]) == 0:
            return {}
        
        src_tokens = data_utils.collate_tokens(
            [x["block"] for x in items if x is not None],
            pad_idx=self.source_dictionary.pad(),
            pad_to_bsz=self.args.batch_size,
        )
        # tgt_tokens = data_utils.collate_tokens(
        #     [x["tgt_block"] for x in items if x is not None],
        #     pad_idx=self.source_dictionary.pad(),
        #     pad_to_bsz=self.args.batch_size,
        # )

        # generate inputs and targets
        input = src_tokens[:, :-1].contiguous()
        target = src_tokens[:, 1:].contiguous()
        ids = torch.cat([x["ids"] for x in items if x is not None])
        clusters = torch.cat([x['clusters'] for x in items if x is not None])
        if ids.numel() != torch.unique(ids).numel():
            n_duplicate = ids.numel() - torch.unique(ids).numel()
            logger.error(
                f"found {n_duplicate}/{ids.numel()} duplicate document IDs in the same batch!"
            )
        # metaseq expects batches to have the following structure
        return {
            "id": ids,
            "clusters": clusters,
            "net_input": {
                "src_tokens": input,
            },
            "target": target,
            "nsentences": input.size(0),
            "ntokens": input.ne(self.dictionary.pad()).sum(),
            "ntokens_target": target.ne(self.dictionary.pad()).sum(),
        }
