# SLURM variables
DEFAULT_SLURM_ACCOUNT="zlab"
DEFAULT_SLURM_CONSTRAINT="[rtx6k|a40|a100]"
DEFAULT_SLURM_PARTITION="ckpt"

# path to data directory
DATA_DIR="/gscratch/zlab/sg01/data/"

# where models will be saved (we will add them under a folder called `opt_ft` in this directory)
SERIALIZATION_DIR="/gscratch/zlab/sg01/experiments/"

# path to vocabulary (gpt2-merges.txt and gpt2-encoder.json)
VOCAB_PATH="/gscratch/zlab/sg01/vocab/"

# path to pretrained models
PRETRAINED_MODELS_DIR="/gscratch/zlab/sg01/opt/"

# path to 1.3B parameter OPT checkpoint
PATH_TO_1_3B_MODEL="/gscratch/zlab/sg01/opt/1.3b/checkpoint_last.pt"
# path to 6.7B parameter OPT checkpoint
PATH_TO_6_7B_MODEL="/gscratch/zlab/sg01/opt/6.7b/checkpoint_last.pt"

# path to metaseq libraries
PATH_TO_METASEQ="/gscratch/zlab/sg01/metaseq/"

