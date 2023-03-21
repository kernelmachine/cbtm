import os

from dataclasses import dataclass
from enum import Enum


@dataclass
class Size:
    n_layers: int
    emb_size: int
    n_heads: int
    d_head: int
    batch_size: int
    lr: float
    model_parallel: int

    @property
    def ffn_size(self):
        return 4 * self.emb_size


# from appendix b of https://arxiv.org/pdf/2005.14165.pdf
# see table 2.1 in https://arxiv.org/pdf/2005.14165.pdf

# assert all sizes make sense, as the gpt-3 paper contains typos

TOTAL_TRAIN_TOKENS = 300e9
TOTAL_WARMUP_TOKENS = 375e6
M = 1024 * 1024  # 1 million
MODEL_SIZES = {
    "8m": Size(4, 128, 2, 64, int(0.5 * M), 2.0e-3, 2),  # tiny
    "25m": Size(6, 256, 4, 64, int(0.5 * M), 1.0e-3, 2),  # 25m
    "50m": Size(8, 512, 8, 64, int(0.5 * M), 9.0e-4, 2),  # 50m
    "125m": Size(12, 768, 12, 64, int(0.5 * M), 6.0e-4, 2),  # small
    "350m": Size(24, 1024, 16, 64, int(0.5 * M), 3.0e-4, 2),  # medium
    "760m": Size(24, 1536, 16, 96, int(0.5 * M), 2.5e-4, 2),  # large
    "1.3b": Size(24, 2048, 32, 64, int(1.0 * M), 2.0e-4, 2),  # xl
    "2.7b": Size(32, 2560, 32, 80, int(1.0 * M), 1.6e-4, 4),  # 2.7b
    "6.7b": Size(32, 4096, 32, 128, int(2.0 * M), 1.2e-4, 2),  # 6.7b
    "13b": Size(40, 5120, 40, 128, int(4.0 * M), 1.0e-4, 2),  # 13b
    "30b": Size(48, 7168, 56, 128, int(4.0 * M), 1.0e-4, 2),
    "66b": Size(64, 9216, 72, 128, int(1.0 * M), 1.0e-4, 4),  # 66b on 512 GPUs in RSC
    "175b": Size(96, 12288, 96, 128, int(0.25 * M), 3e-5, 8),  # GPTZ/GPT-3
}

# from appendix b of https://arxiv.org/pdf/2005.14165.pdf
# see table 2.1 in https://arxiv.org/pdf/2005.14165.pdf

for name, size in MODEL_SIZES.items():
    assert size.n_heads * size.d_head == size.emb_size, name


class ComputeEnvs(Enum):
    AZURE = "azure"
    AWS = "aws"
    RSC = "rsc"
    FAIR = "fair"


DATA_LOCATIONS = {
    ComputeEnvs.FAIR: "/large_experiments/xlmg/data/gptz",
}


VALID_SUBSETS = [
    "valid/C4",
    "valid/C4_small",
]

