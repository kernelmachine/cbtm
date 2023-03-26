# SLURM variables
DEFAULT_SLURM_ACCOUNT="slurm_account_name"
DEFAULT_SLURM_CONSTRAINT=""
DEFAULT_SLURM_PARTITION="slurm_partition_name"

# top level CBTM folder
CBTM_DIR = f"/home/cbtm/"

# path to cbtm code
PATH_TO_CBTM=f"/home/cbtm/"

# path to data directory
DATA_DIR = f"{CBTM_DIR}/data"

# where models will be saved (we will add them under a folder called `opt_ft` in this directory)
SERIALIZATION_DIR = f"{CBTM_DIR}/experiments"

# where clusterers and clusters will be saved
KMEANS_DIR = f"{CBTM_DIR}/clusterers"
CLUSTERS_DIR = f"{CBTM_DIR}/clusters"

# path to vocabulary (gpt2-merges.txt and gpt2-vocab.json)
VOCAB_DIR = f"{CBTM_DIR}/vocab"

# paths to pretrained models
PRETRAINED_MODELS_DIR = f"{CBTM_DIR}/pretrained_models"
# path to 1.3B parameter OPT checkpoint
PATH_TO_1_3B_MODEL = f"{PRETRAINED_MODELS_DIR}/opt/1.3b/checkpoint_last.pt"
# path to 6.7B parameter OPT checkpoint
PATH_TO_6_7B_MODEL=f"{PRETRAINED_MODELS_DIR}/opt/6.7b/checkpoint_last.pt"

