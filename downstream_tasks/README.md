# C-BTM Downstream Task Evalutions

### Installing extra dependencies
```
pip install openai accelerate
```

## Converting models to HuggingFace format

Our downstream task evaluation setup assumes that models are saved in a format compatible with the HuggingFace libraries. We've provided a script which, given a directory containing our models, will iterate through all subdirectories looking for files named `$FAIRSEQ_FILE_NAME`, which we default to `consolidated.pt`, and converting those into Huggingface-compatible checkpoint files named `pytorch_model.bin` (the default determined by the transformers library). The are saved in `$HF_MODEL_DIR`, maintaining the folder structure of `$MODEL_FOLDER`. Any of the defined SLURM constants may be modified to suit your setup or preference:

```
CBTM_PATH=/private/home/margaretli/cbtm_metaseq
FAIRSEQ_FILE_NAME=consolidated.pt
HF_MODEL_DIR=/checkpoint/margaretli/cbtm/s2orc/16_clusters/hf
MODEL_FOLDER=/checkpoint/margaretli/cbtm/s2orc/16_clusters

CBTM_PATH=/path/to/cbtm/code
FAIRSEQ_FILE_NAME=consolidated.pt
HF_MODEL_DIR=/path/to/dir/for/saving/hf/models
MODEL_FOLDER=/path/to/dir/with/models

python $CBTM_PATH/downstream_tasks/convert_ckpt_to_hf.py --fairseq-path $MODEL_FOLDER \
--fairseq-file-name $FAIRSEQ_FILE_NAME --pytorch-dump-folder-path $HF_MODEL_DIR
```

The above script serially converts all checkpoints in `$MODEL_FOLDER`. This is fine for just 1 or 2; however, if you need to convert a large number of checkpoints, we parallelize this process for you with the [gnu-parallel](https://www.gnu.org/software/parallel/) package:

```
ORIGINAL_MODEL_PATHS=$(find $MODEL_FOLDER -name $FAIRSEQ_FILE_NAME)
parallel --link --ungroup --jobs 10 \
 python $CBTM_PATH/downstream_tasks/convert_ckpt_to_hf.py --fairseq-path $MODEL_FOLDER \
 --fairseq-file-name $FAIRSEQ_FILE_NAME --pytorch-dump-folder-path $HF_MODEL_DIR \
 --fairseq-model {1} ::: $ORIGINAL_MODEL_PATHS 
```

## Data preprocessing

Our data formats build upon [MetaICL](), which in turn borrows from [CrossFit](). MetaICL's data downloading and preprocessing code requires `datasets==1.4.0`, which is incompatible with the version of the `transformers` package required for their other code. We do not use their training code, but to avoid potential conflicts, you may want to create a conda environment just to do these steps, and then switch back to your general C-BTM for the rest of this evaluation procedure.
To download and preprocess the tasks:

```
# recommended
conda create -n cbtm_downstream_data 
conda activate cbtm_downstream_data

pip install datasets==1.4.0 wget

git clone https://github.com/facebookresearch/MetaICL.git
cd MetaICL

python preprocess/_build_gym.py --build --n_proc=40 --do_test --test_k 8
```

We provide commands to prepend `k` demonstrations in-context from the training data and reformat this data. MetaICL hardcodes their random seeds; if you'd like to use other values as your random seed, you'll need to modify the MetaICL code for the particular task of interest under their `preprocess` folder.

```
DATA_PATH=
RAW_DATA_PATH=$DATA_PATH/raw
PROCESSED_DATA_PATH=$DATA_PATH/processed
seed=13 # must be one of [100, 13, 21, 42, 87], unless MetaICL code is modified
k=8
split=test
DATASET=ag_news

DATA_PATH=$CBTM_PATH/downstream_tasks/data
RAW_DATA_PATH=$DATA_PATH/raw
PROCESSED_DATA_PATH=$DATA_PATH/processed
seed=13 # must be one of [100, 13, 21, 42, 87], unless MetaICL code is modified
k=8
split=test
DATASET=ag_news

mv ./data $DATA_PATH/raw    # move data to the cbtm folder

python $CBTM_PATH/downstream_tasks/process_data.py --dataset $DATASET \
 --data-write-path $PROCESSED_DATA_PATH --data-read-path $RAW_DATA_PATH \
 --random-seed $seed --n-shot $k 
```


## Cluster data

In order to perform our cluster-based routing, we process the task data according to the cluster centroids trained for the C-BTM model. This step is not necessary if you're only evaluating a dense (1-cluster) model.

```
MIXTURE_FOLDER=$HF_MODEL_DIR/$DATASET/$split/${k}shot_seed${seed}
DATASET_DIR=$PROCESSED_DATA_PATH/$DATASET/$split
CLUSTERER=/checkpoint/margaretli/cbtm/clusterers/s2orc/16
NUM_CLUSTERS=16
COLUMN=input

MIXTURE_FOLDER=$HF_MODEL_DIR/$DATASET/$split/${k}shot_seed${seed}
DATASET_DIR=$DATA_PATH/processed/$DATASET/$split
CLUSTERER=/checkpoint/margaretli/cbtm/clusterers/c4/16
NUM_CLUSTERS=16
COLUMN=input

python $CBTM_PATH/downstream_tasks/estimate_cluster.py --eval-file $DATASET_DIR/${k}shot_${seed}.jsonl \
 --path-to-clusterer $CLUSTERER --mixture-file-name $MIXTURE_FOLDER/cluster.npy --column $COLUMN
```


## Evaluate experts and save outputs

Each expert now needs to be evaluated on the dataset split. We do this asynchronously for each expert:

```
MODEL_PATH=/path/to/hf/expert/checkpoint
MIXTURE_FOLDER=$HF_MODEL_DIR/$DATASET/$split/${k}shot_seed${seed}
EXPERT_OUTPUT_PATH=$MIXTURE_FOLDER/output
python $CBTM_PATH/downstream_tasks/score.py --dataset $DATASET --data-dir $DATASET-DIR \
 --save-dir $EXPERT_OUTPUT_PATH --n-shot $k --split $split --seed $seed \
 --mixture-folder $MIXTURE_FOLDER/output --model-path $MODEL_PATH
```

This results in files saved to `$MIXTURE_FOLDER/output/{expert_folder_name}`, including a file named `predictions_list.jsonl` which will contain the predictions made by that expert.

The above command needs to be called on each expert separately. However, for convenience, we also wrote a script which launches a slurm grid which includes a job for each file found in `$HF_MODELS_DIR` with the filename `pytorch_model.bin`.

```
python $CBTM_PATH/downstream_tasks/score_grid.py --dataset $DATASET --data-dir $DATASET_DIR \
 --models-parent-folder $HF_MODEL_DIR --model-file-name pytorch_model.bin --n-shot $k --seed $seed \
 --mixture-folder $MIXTURE_FOLDER/output --script $CBTM_PATH/downstream_tasks/score.py \
 -p score/${DATASET} -g 1 --checkpoints-dir $HF_MODEL_DIR --no-wandb --no-tensorboard \
 --use-jobarray --jobarray-name score_$DATASET_${HF_MODEL_DIR}
```

## Ensemble experts

If you're evaluating a dense (1-cluster) model, or only one expert, the prediction files generated in the last step are sufficient. However, if you would like to ensemble more than one expert:

```
topk=-1 #number of experts to use, -1 indicates using all experts
python $CBTM_PATH/downstream_tasks/ensemble.py $DATASET --expert-outputs-dir $MIXTURE_FOLDER/output \
 --mixture-file $MIXTURE_FOLDER/cluster.npy --topk $topk --method standard --num-clusters $NUM_CLUSTERS
```

This will also produce a predictions file under the name `predictions_list.jsonl` in `$MIXTURE_FOLDER/output/ensemble/standard/top{topk}`.