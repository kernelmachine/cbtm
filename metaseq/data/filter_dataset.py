# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import pandas as pd
import json

from metaseq.file_io import PathManager

from metaseq.data import data_utils
from . import BaseWrapperDataset

logger = logging.getLogger(__name__)

class FilterDataset(BaseWrapperDataset):
    """Filters dataset by excluding examples that are easy/hard to learn.
    
    Hard/easy to learn are defined by a metric between 0.0 and 1.0, with
    0.0 meaning easy to learn, and 1.0 meaning hard to learn.

    During initialization, this class expects:
    - `frac_data`: how much of the original dataset we want to keep 
    during training. It calculates the closest `frac_data` we can keep
    (keep the hard examples, throw away easy examples, 
    as per https://arxiv.org/abs/2206.14486.
    - `metric_file`: path to jsonl file, where each line should be

        {
            "name": dataset_name, 
            "index": index in dataset jsonl file, 
            "metric": metric value
        }
    where `metric` should be between 0.0 and 1.0 as described above

    """

    def __init__(self, dataset, frac_data, metric_data, dataset_name_to_index):
        super().__init__(dataset)
        assert 0.0 <= frac_data <= 1.0
        self.frac_data = frac_data
        self.concat_dataset = dataset
        self.metric_data = metric_data


        # We only ever include stuff that is in metric_data. If our actual training set is a superset - we don't care.
        limit = int(np.ceil(len(self.metric_data) * self.frac_data))

        self.metric_data.sort_values('metric', inplace=True, ascending=False)
        logger.info(f"Concat dataset length: {len(self.concat_dataset)}")

        # If there are a subset of data points in the csv file, then just train on those data points
        # otherwise, take the limit defined by `frac_data`
        self.metric_data = self.metric_data[:limit]

        logger.info(f"New df length after adding frac_data limit: {len(self.metric_data )}")


        self.dataset_name_to_index = dataset_name_to_index


        # We use a single file to store metrics for all shards / datasets, but our
        # indexing logic depends on the index having data points only from the currently processed shard.
        # So as a final pre-processing step, only include the examples from the current shard
        curent_shard_keys = list(self.dataset_name_to_index.keys())
        logger.info(f"Filtering metric files for shards: {curent_shard_keys}")
        self.metric_data = self.metric_data[self.metric_data['name'].isin(curent_shard_keys)]
        logger.info(f"Metric df length after filtering for shard: {len(self.metric_data )}")

        self.length = len(self.metric_data)


    @staticmethod
    def retrieve_metric_df(metric_file, dataset_name_to_index):
        assert PathManager.exists(metric_file), "Error! Provided `metric_file` is not a valid filepath"
        assert PathManager.isfile(metric_file), "Error! Provided `metric_file` is not a valid file"
        assert metric_file.endswith(".jsonl") or metric_file.endswith(".csv"), "Error! `metric_file` must be a `jsonl` file"

        if metric_file.endswith(".jsonl"):
            with open(metric_file, "r") as f:
                lines = f.read().splitlines()

            df = pd.DataFrame(lines)
            df.columns = ['temp']
            df['temp'].apply(json.loads)
            df = pd.json_normalize(df['temp'].apply(json.loads))

        elif metric_file.endswith(".csv"):
            df = pd.read_csv(metric_file)

        logger.info(f"Raw metric file length: {len(df)}")

        return df

    def __getitem__(self, index):
        assert 0 <= index < self.length

        metadata = self.metric_data.iloc[index]
        dataset_name = str(metadata["name"])
        sample_idx = int(metadata["index"])

        assert dataset_name in self.dataset_name_to_index, f"Error: dataset path {dataset_name} not in dataset_index. Keys: {list(self.dataset_name_to_index.keys())}"
        dataset_index = self.dataset_name_to_index[str(metadata["name"])]

        
        return self.concat_dataset.datasets[dataset_index][sample_idx]

    def __len__(self):
        return self.length