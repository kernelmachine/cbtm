# Cluster-Branch-Train-Merge (c-BTM)

Code for the paper *Scaling Expert Language Models with Unsupervised Domain Discovery*

This repository is a fork of [metaseq](https://github.com/facebookresearch/metaseq/).

## Citation

If you use this code, please consider citing our work:

```
@article{cbtm,
 author = {Suchin Gururangan and Margaret Li and Mike Lewis and Weijia Shi and Tim Althoff and Noah A. Smith and Luke Zettlemoyer},
 title = {Scaling Expert Language Models with Unsupervised Domain Discovery},
 year = {2023}
}
```

## Create a new conda env (recommended)

We supply an `environment.yml` file; this will create a conda environment with python 3.9 and a variety of dependencies. This will take a few minutes.

```bash
conda env create -f environment.yml
conda activate cbtm
```

### Install PyTorch

We tested this code with torch compiled with Cuda 11.3.

```bash
pip3 install torch==1.10.1+cu113  -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Install Megatron

Make sure you have a GPU and CUDA visible for this step.

```bash
git clone --branch fairseq_v2 https://github.com/ngoyal2707/Megatron-LM.git
cd Megatron-LM
pip3 install six regex
pip3 install -e .
```

### Install fairscale

```bash
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
git checkout prefetch_fsdp_params_simple
pip3 install -e .
```

### Install balanced-kmeans

```bash
git clone https://github.com/kernelmachine/balanced-kmeans.git
cd balanced-kmeans
pip3 install -e .
```


### (Optional) Install Apex

Apex may not be compatible with all GPUs. In particular, if you're seeing that CUDA doesn't support your model during the forward pass, you might want to try uninstalling Apex and trying again.

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout e2083df5eb96643c61613b9df48dd4eea6b07690
```

Depending on your hardware, you may need to comment out lines 101-107 in setup.py before running the next pip install.

```bash
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```

### Install c-BTM library

Build the c-BTM library. This won't really do anything if you've used the `environment.yml` file to build your conda environment.

```bash
cd /path/to/cbtm
pip3 install -e .
```


# c-BTM Training and Evaluation

## Step 0: Set up data, models, and directories

We'll use the following environment variables in this tutorial, for simplicity. You can set these to whatever you want.

```bash
export CBTM_DIR=$PWD; 
export DATA_DIR=${CBTM_DIR}/data;
export SERIALIZATION_DIR=${CBTM_DIR}/experiments;
export KMEANS_DIR=${CBTM_DIR}/clusterers;
export CLUSTERS_DIR=${CBTM_DIR}/clusters;
export VOCAB_DIR=$CBTM_DIR/vocab
export PRETRAINED_MODELS_DIR=${CBTM_DIR}/pretrained_models;
mkdir -p ${CBTM_DIR} ${DATA_DIR} ${SERIALIZATION_DIR} ${KMEANS_DIR} ${CLUSTERS_DIR} ${VOCAB_DIR} ${PRETRAINED_MODELS_DIR};
```


### Configure cbtm_constants.py


Next, the constants necessary to make this repo work are at `metaseq/cbtm_constants.py`. Modify these to suit your local environment. 

Make sure the variables in `metaseq/cbtm_constants.py` are consistent with the paths you set as environment variables above. 

### Download vocab files and seed models

We use the GPT-2 vocabulary:

```bash
mkdir -p $VOCAB_DIR
wget -O $VOCAB_DIR/gpt2-vocab.json http://s3.wasabisys.com/c4-example/vocab/gpt2-vocab.json
wget -O $VOCAB_DIR/gpt2-merges.txt http://s3.wasabisys.com/c4-example/vocab/gpt2-merges.txt
```

Download the OPT-1.3B and OPT-6.7B checkpoints, which we use as our seed models:

```bash
mkdir -p $PRETRAINED_MODELS_DIR
mkdir -p ${PRETRAINED_MODELS_DIR}/opt/1.3b/
mkdir -p ${PRETRAINED_MODELS_DIR}/opt/6.7b/
wget -qO- dl.fbaipublicfiles.com/cbtm/opt_models/1.3B/sharded_for_ddp.tgz | tar xvz  --strip-components 6 -C ${PRETRAINED_MODELS_DIR}/opt/1.3b/
wget -qO- dl.fbaipublicfiles.com/cbtm/opt_models/6.7B/sharded_for_ddp_part_0.tgz | tar xvz  --strip-components 6 -C ${PRETRAINED_MODELS_DIR}/opt/6.7b/
wget -qO- dl.fbaipublicfiles.com/cbtm/opt_models/6.7B/sharded_for_ddp_part_1.tgz | tar xvz --strip-components 6 -C ${PRETRAINED_MODELS_DIR}/opt/6.7b/

```



### Download data

We provide some sample C4 data to get you started. Our model only expects (sharded) line-separated jsonl files, split into train and validation data. If you'd like to train on your own data, just follow the overall data layout in the example. 


```bash
mkdir -p ${DATA_DIR}/c4_example/
wget -qO- http://s3.wasabisys.com/c4-example/c4_example.tar.gz | tar xvz -C ${DATA_DIR}/c4_example/
```

This example dataset is a single shard of C4 and a small sample of data from the validation dataset. 

You can download the full C4 dataset from Huggingface datasets at the following link: https://huggingface.co/datasets/c4. Keep in mind that the dataset is very large, and comes as `json.gz` files. Our code expects raw jsonl files in the structure from the example directory, so make sure you have enough space (in total, it's about 1.4 terabytes of data uncompressed).

Metaseq expects the data to be in this format:

```
{"text": "this is a document", "id": 0}
{"text": "this is another document", "id": 1}
```


## Step 1: Train Clusterer

This command trains a balanced k-means clusterer on a single shard of the C4 training data. Here we use k=8, and give as an argument a folder which contains a file called `C4.jsonl`, as described above. 

Make sure you have access to a GPU here, to speed up training!

```bash
NUM_CLUSTERS=8;
DATASET=c4_example;
python -m metaseq.scripts.train_clusterer \
--data-dir ${DATA_DIR}/${DATASET}/train/00000 \
--num-clusters ${NUM_CLUSTERS} \
--balanced \
--output-dir ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/
```

This will create `tfidf.pkl` (a tf-idf embedder) and `kmeans.pkl` (a kmeans clusterer) pickle files at `${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/`.

## Step 2: Cluster data

This code uses your trained clusterer to cluster the dataset's documents. This is _substantially_ faster if you can parallelize it as slurm jobs. 

We will do this automatically for you via `submitit`, just provide your slurm account and partition (either as a flag to the program, or in `metaseq/cbtm_constants.py`)

If you don't have access to slurm, you can cluster your data locally with the flag `--run local`, but it might take some time!

```bash
DATASET=c4_example;
NUM_CLUSTERS=8;

# Cluster train data
python -m metaseq.scripts.cluster \
--job-dir ${CBTM_DIR}/cluster_logs \
--data-dir ${DATA_DIR}/${DATASET} \
--path-to-clusterer ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
--num-clusters ${NUM_CLUSTERS} \
--output-prefix ${CLUSTERS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
--split train \
--run slurm; 

# Cluster validation data
python -m metaseq.scripts.cluster \
--job-dir ${CBTM_DIR}/cluster_logs \
--data-dir ${DATA_DIR}/${DATASET} \
--path-to-clusterer ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
--num-clusters ${NUM_CLUSTERS} \
--output-prefix ${CLUSTERS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
--split valid/C4_small \
--run slurm;
```

Logs for these clustering jobs appear in `${CBTM_DIR}/cluster_logs`.

After these jobs complete, open files in ${CLUSTERS_DIR}, e.g., `${CLUSTERS_DIR}/${DATASET}/${NUM_CLUSTERS}/train/00000/C4.jsonl`. You should see lines like the following:

```
{"sp_id":"\/gscratch\/zlab\/sg01\/data\/c4_example\/train\/00000\/C4.jsonl|0","cluster":5}
{"sp_id":"\/gscratch\/zlab\/sg01\/data\/c4_example\/train\/00000\/C4.jsonl|1","cluster":3}
{"sp_id":"\/gscratch\/zlab\/sg01\/data\/c4_example\/train\/00000\/C4.jsonl|2","cluster":2}
```

The field `sp_id` indicates a line (i.e., document) within a file, and the field `cluster` indicates its predicted cluster.

## Step 3: Train Models

Now we'll use the clustered data to train experts. You'll need at least 4 GPUs simultaneously to train each model.

This tutorial uses our `train_cbtm` script, which interfaces with SLURM. 

We have also provided an example sbatch script, if desired, in `metaseq/scripts/example_sbatch.sh`. You may need to edit this example sbatch command to include any additional slurm arguments you might need to get it working on your system.


### Train experts


The following command will train 8 expert models with 4 GPUs each for 50 steps (increase to 10000 steps to replicate our paper).


```bash
NUM_CLUSTERS=8;
DATASET=c4_example;
python -m metaseq.scripts.train_cbtm \
   --model-size 1.3b \
   --run slurm   \
   --path-to-clusters-dir $CLUSTERS_DIR/${DATASET}/$NUM_CLUSTERS/ \
   --num-clusters $NUM_CLUSTERS  \
   --num-nodes 1 \
   --num-gpus 4 \
   --data-name ${DATASET}  \
   --path-to-data $DATA_DIR/${DATASET} \
   --learning-rate 2e-4 \
   --max-steps 50 \
   --valid-subset valid/C4_small \
   --train-subset train
```

To train on a specific cluster(s), you can add the flag `--train-cluster 1,3,5`

To debug locally, change the `run` flag to `--run local`.

This command will output checkpoints and logs to `${SERIALIZATION_DIR}/8_clusters/`.


### Dense training

The following command will train a dense model with 4 GPUs for 50 steps (increase to 10000 steps to replicate our paper).

```bash
DATASET=c4_example;
python -m metaseq.scripts.train_cbtm \
   --num-clusters 1 \
   --model-size 1.3b \
   --run slurm \
   --data-name $DATASET  \
   --num-nodes 1 \
   --num-gpus 4 \
   --data-name ${DATASET}  \
   --path-to-data $DATA_DIR/$DATASET  \
   --learning-rate 2e-4 \
   --max-steps 50 \
   --valid-subset valid/C4_small \
   --train-subset train
```

To debug locally, change the `run` flag to `--run local`.

This command will output checkpoints to `${SERIALIZATION_DIR}/1_clusters/`.



## Evaluation

To evaluate your models, first consolidate your shards into a single checkpoint file.

The following script depends on the [`gnu-parallel`](https://www.gnu.org/software/parallel/) package.


```bash
NUM_CLUSTERS=8;
bash metaseq/scripts/consolidate_fsdp_shards.sh ${SERIALIZATION_DIR}/${NUM_CLUSTERS}_clusters/ "*ngpu4"
```

This will create a `consolidated.pt` checkpoint in each model's folder. 

Now the checkpoints are ready for eval. To launch on slurm:

```bash
export NUM_CLUSTERS=8;
# we want as many GPUs as we have clusters
export NUM_GPUS=${NUM_CLUSTERS};
export DATASET=c4_example;
export EVAL_DIR=${SERIALIZATION_DIR}/${NUM_CLUSTERS}_clusters/eval

mkdir -p ${EVAL_DIR};

# get model checkpoints
CONSOLIDATED_MODEL_PATHS=;
# this function gets all model checkpoint directories and sorts them by the cluster ID
# modify the folder pattern to match the directories names of your models, if you need.
FOLDER_PATTERN="cbtm\.c4_example\.*ngpu4"
mapfile -t MODEL_FOLDERS < <(find ${SERIALIZATION_DIR}/${NUM_CLUSTERS}_clusters/ -type d -name $FOLDER_PATTERN  -name "*\.cluster*" -printf "%f|%p\n" | sort -t "|" -k1,1 -t "|" -k2,2 | cut -d "|" -f 2)
for folder in "${MODEL_FOLDERS[@]}"; do
    # check if there are any consolidated.pt files in the model folder 
    if test -f "${folder}/consolidated.pt"; then
        CONSOLIDATED_MODEL_PATHS+="${folder}/consolidated.pt ";
    fi;
done
# function to join checkpoints into comma separated string
function join { local IFS=","; echo "$*"; }

# these model paths should be ordered by cluster ID!
JOINED_MODEL_PATHS=$(join ${CONSOLIDATED_MODEL_PATHS[@]})

python -m metaseq_cli.eval_cbtm \
    --data-dir ${DATA_DIR}/${DATASET} \
    --data-subset valid/C4_small \
    --path-to-clusterer ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
    --model-paths $(join ${CONSOLIDATED_MODEL_PATHS[@]}) \
    --job-dir ${EVAL_DIR} \
    --temperature 0.1 \
    --max-valid-steps 200 \
    --ensemble-type clustering \
    --submitit
```

You can check out logs for your slurm job at `${EVAL_DIR}`.

To launch locally, remove the flag `--submitit` in the command above. Make sure you have `$NUM_CLUSTERS` GPUs visible though!

This will output perplexity results to `${EVAL_DIR}/result.json`.

Use the same command as above to evaluate your dense models, just change the environment variable `NUM_CLUSTERS=1`.

On our machine we get `ppl: 17.86` for the 8-cluster model from this tutorial, and `ppl: 18.56` for the 1-cluster model. (Note that these perplexities are only for reproducibility purposes; increase the step size to 10000 to replicate our runs from the paper).

## MoE baseline training via sparse upcycling

See our [fairseq fork](https://github.com/kernelmachine/moe-fairseq) for instructions and code to train the  sparse upcycling MoE baseline.

## Open-sourced pretrained models

### Downloading clusterers and embedders

To get a pretrained clusterer and embedder from the paper, you can run:

```bash 
# c4 or s2orc
export DATASET=c4
# 2, 4, 8, 16, 32, 64, 128
export NUM_CLUSTERS=4
mkdir -p ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/
wget -O ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/tfidf.pkl https://dl.fbaipublicfiles.com/cbtm/clusterers/${DATASET}/${NUM_CLUSTERS}/tfidf.pkl
wget -O ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/kmeans.pkl https://dl.fbaipublicfiles.com/cbtm/clusterers/${DATASET}/${NUM_CLUSTERS}/kmeans.pkl
```

### Downloading language models

To access the language models we trained in our paper, you can run the following command:

```bash
# c4 or s2orc
export DATASET=c4
# opt1.3b or opt6.7b
export MODEL_ARCH=opt1.3b
# 4, 8, 16, 32, 64, 128, 256, 512, 1024
export GPUS_PER_EXPERT=4
# 2, 4, 8, 16, 32, 64, 128
export NUM_CLUSTERS=64
# anything from 0 to $NUM_CLUSTERS-1
export CLUSTER_NUMBER=28
FOLDER=${DATASET}.${MODEL_ARCH}.${NUM_CLUSTERS}_clusters.cluster${CLUSTER_NUMBER}.ngpu${GPUS_PER_EXPERT}/
mkdir -p ${PRETRAINED_MODELS_DIR}/cbtm/$FOLDER
wget -O ${PRETRAINED_MODELS_DIR}/cbtm/$FOLDER/consolidated.pt https://dl.fbaipublicfiles.com/cbtm/cbtm_models/$DATASET/${MODEL_ARCH}/${NUM_CLUSTERS}_clusters/ngpu${GPUS_PER_EXPERT}/${CLUSTER_NUMBER}/consolidated.pt
```

To get all n experts for an n-cluster c-BTM model, you can do a for loop:

```bash
# c4 or s2orc
export DATASET=c4
# opt1.3b or opt6.7b
export MODEL_ARCH=opt1.3b
# 4, 8, 16, 32, 64, 128, 256, 512, 1024
export GPUS_PER_EXPERT=4
# 2, 4, 8, 16, 32, 64, 128
export NUM_CLUSTERS=4
for CLUSTER_NUMBER in $(seq 0 $((NUM_CLUSTERS-1))); do 
    FOLDER=${DATASET}.${MODEL_ARCH}.${NUM_CLUSTERS}_clusters.cluster${CLUSTER_NUMBER}.ngpu${GPUS_PER_EXPERT}/
    mkdir -p ${PRETRAINED_MODELS_DIR}/cbtm/$FOLDER
    wget -O ${PRETRAINED_MODELS_DIR}/cbtm/$FOLDER/consolidated.pt https://dl.fbaipublicfiles.com/cbtm/cbtm_models/$DATASET/${MODEL_ARCH}/${NUM_CLUSTERS}_clusters/ngpu${GPUS_PER_EXPERT}/${CLUSTER_NUMBER}/consolidated.pt
done
```

All models for which we show results in the core results section of the paper can be downloaded here. The highest budget (168B token) 1.3B parameter models have `${NUM_CLUSTERS} * $GPUS_PER_EXPERT = 1024`, resulting in `(${NUM_CLUSTERS}, $GPUS_PER_EXPERT) = (1, 1024), (2, 512) ... (128, 8)`. Settings for smaller compute budget models can be found with a proportionally small product. E.g., for 42B token 1.3B parameter models, we have `${NUM_CLUSTERS} * $GPUS_PER_EXPERT = 256`, resulting in `(${NUM_CLUSTERS}, $GPUS_PER_EXPERT) = (1, 256), (2, 128) ... (64, 4)`.

These models can be evaluated using the command given above, under "Cluster BTM Training and Evaluation". However, to continue training any of these experts using model parallel, you will need to reshard them, check the `metaseq` library to learn more on how to do this.

See this [README](https://l.facebook.com/l.php?u=https%3A%2F%2Fdl.fbaipublicfiles.com%2FREADME&h=AT1F-a_xXIhZseEwKETbvCNYQIJlBJOLEU2_MkOfjxpaCML8sQz-hm7qGMpUAwJ-Zd3F-P3x3ZfrPCnxP2gME5jGSGZo8c7pCXB_NP1CyxkxJYQWGGPm2ZiTsur2Qt29FxgAF4V4IAoCDStOkr8y8hH0) for details on rate-limiting when downloading models.

### Converting from metaseq to Huggingface

We have provided a script to convert all metaseq models to Huggingface transformers compatible checkpoints.

First, you have to install huggingface transformers [from source](https://huggingface.co/docs/transformers/installation#install-from-source), and make sure you consolidate your metaseq checkpoint using the `consolidate_fsdp_shards.sh` first (see above).

Then run:

```
PATH_TO_TRANSFORMERS=/path/to/installed/transformers/library/
# path to directory containing consolidated.pt checkpoint
INPUT_DIR=/path/to/metaseq/model/dir
OUTPUT_DIR=my_huggingface_model

bash metaseq/scripts/convert_hf.sh $PATH_TO_TRANSFORMERS $INPUT_DIR $OUTPUT_DIR
```

This will output a checkpoint in $OUTPUT_DIR you can use in any huggingface transformers pipeline.

