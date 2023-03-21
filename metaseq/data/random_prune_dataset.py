# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import json

from metaseq.file_io import PathManager

from metaseq.data import data_utils
from . import BaseWrapperDataset


class RandomPruneDataset(BaseWrapperDataset):
    """Randomly prunes examples in the dataset
    """

    def __init__(self, dataset, seed, frac_data):
        super().__init__(dataset)
        
        N = len(dataset)
        assert N > 0

        self.seed = seed
        self.frac_data = frac_data
        self.length = int(np.ceil(N * self.frac_data))

        self.indices = None

        with data_utils.numpy_seed(3):
            self.indices = np.random.permutation(N)

        self.dataset = dataset


    def __getitem__(self, index):
        assert 0 <= index <= self.length
        return self.dataset[self.indices[index]]

    def __len__(self):
        return self.length