# C-BTM Downstream Task Evalutions

### Installing extra dependencies
```
pip install openai promptsource accelerate
```

## Converting models to HuggingFace format

Our downstream task evaluation setup assumes that models are saved in a format compatible with the HuggingFace libraries. We've provided a script which, given a directory containing our models, will iterate through all subdirectories looking for files named `$FAIRSEQ_FILE_NAME`, which we default to `consolidated.pt`, and converting those into Huggingface-compatible checkpoint files named `pytorch_model.bin` (the default determined by the transformers library). These are saved in `$HF_MODEL_DIR`, maintaining the folder structure of `$MODEL_FOLDER`. Any of the defined SLURM constants may be modified to suit your setup or preference:

```
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

Our data formats build upon [MetaICL](https://github.com/facebookresearch/MetaICL.git), which in turn borrows from [CrossFit](). MetaICL's data downloading and preprocessing code requires `datasets==1.4.0`, which is incompatible with the version of the `transformers` package required for their other code. We do not use their training code, but to avoid potential conflicts, you may want to create a conda environment just to do these steps, and then switch back to your general C-BTM for the rest of this evaluation procedure.
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
DATA_PATH=$CBTM_PATH/downstream_tasks/data
RAW_DATA_PATH=$DATA_PATH/raw
PROCESSED_DATA_PATH=$DATA_PATH/processed
seed=(100 13 21 42 87)
k=8
split=test
DATASET=ag_news

mv ./data $DATA_PATH/raw    # move data to the cbtm folder

python $CBTM_PATH/downstream_tasks/process_data.py --tasks $DATASET \
 --data-write-path $PROCESSED_DATA_PATH --data-read-path $RAW_DATA_PATH \
 --random-seed ${seed[@]} --n-shot $k 
```

The above command can be run with any number of tasks, just pass more than one argument to `--tasks`, e.g. `--tasks ag_news dbpedia_14`. If `--task` is not set, this command will process all tasks under `$RAW_DATA_PATH`.

```
python $CBTM_PATH/downstream_tasks/process_data.py \
 --data-write-path $PROCESSED_DATA_PATH --data-read-path $RAW_DATA_PATH \
 --random-seed ${seed[@]} --n-shot $k 
```

## Cluster data

In order to perform our cluster-based routing, we process the task data according to the cluster centroids trained for the C-BTM model, which can be downloaded according to instructions in the main [README](). This step is not necessary if you're only evaluating a dense (1-cluster) model.

```
NUM_CLUSTERS=16 # 1, 2, 4, etc.
MIXTURE_FOLDER=$HF_DIR/$DATASET/$split
DATASET_DIR=$PROCESSED_DATA_PATH/$DATASET/$split
CLUSTERER=/path/to/clusterer
COLUMN=input

python $CBTM_PATH/downstream_tasks/estimate_cluster.py --dataset-dir $DATASET_DIR \
 --path-to-clusterer $CLUSTERER_DIR --mixture-folder $MIXTURE_FOLDER \
 --column $COLUMN --seed ${seed[@]} --n-shot $k
```

This command works on one clusterer and one dataset at a time, running through seeds serially. For most people, this is a feature, not a bug, as running more than one clustering process is likely to cause a CUDA out of memory error on many machines, and this allows us to only load the clusterer once for any number of seeds. However, if you have substantially more memory and would like to use gnu-parallel to shorten wall-clock time:

```
parallel --link --ungroup --jobs 4 \
 python $CBTM_PATH/downstream_tasks/estimate_cluster.py --dataset-dir $DATASET_DIR \
 --path-to-clusterer $CLUSTERER_DIR --mixture-folder $MIXTURE_FOLDER \
 --column $COLUMN --seed {1} --n-shot $k ::: ${seed[@]}
 ```

## Evaluate experts and save outputs

Each expert now needs to be evaluated on the dataset split. We do this asynchronously for each expert:

```
MODEL_PATH=/path/to/hf/expert/checkpoint
MIXTURE_FOLDER=$HF_DIR/$DATASET/$split
EXPERT_OUTPUT_PATH=$MIXTURE_FOLDER/output
python $CBTM_PATH/downstream_tasks/score.py --dataset $DATASET --data-dir $DATASET_DIR \
 --save-dir $EXPERT_OUTPUT_PATH --n-shot $k --split $split --seeds ${seed[@]} \
 --mixture-folder $MIXTURE_FOLDER --model-path $MODEL_PATH
```

This results in files saved to `$MIXTURE_FOLDER/${k}shot_seed${seed}/output/{expert_folder_name}`, including a file named `predictions_list.jsonl` which will contain the predictions made by that expert.

The above command needs to be called on each expert separately. However, for convenience, we also wrote a script which launches a slurm grid which includes a job for each file found in `$HF_MODELS_DIR` with the filename `pytorch_model.bin`.

```
python $CBTM_PATH/downstream_tasks/score_grid.py --dataset $DATASET --data-dir $DATASET_DIR \
 --models-parent-folder $HF_MODEL_DIR --model-file-name pytorch_model.bin --n-shot $k --seeds ${seed[@]} \
 --mixture-folder $MIXTURE_FOLDER --script $CBTM_PATH/downstream_tasks/score.py \
 -p score/${DATASET} -g 1 --checkpoints-dir $HF_MODEL_DIR --no-wandb --no-tensorboard \
 --use-jobarray --jobarray-name score_${DATASET}_${HF_MODEL_DIR}
```

## Ensemble experts

If you're evaluating a dense (1-cluster) model, or only one expert, the prediction files generated in the last step are sufficient. However, if you would like to ensemble more than one expert:

```
topk=-1 #number of experts to activate, -1 indicates using all experts
python $CBTM_PATH/downstream_tasks/ensemble.py $DATASET --expert-outputs-dir output \
 --mixture-folder $MIXTURE_FOLDER --topk $topk --seed ${seed[@]} --n-shot $k --method standard --num-clusters $NUM_CLUSTERS
```

This will also produce a predictions file under the name `predictions_list.jsonl` in `$MIXTURE_FOLDER/output/ensemble/standard/top{topk}`.

If you would like to eval for multiple values of `--topk`, you can pass in multiple values, e.g. `--topk 1 2 4`. Alternatively, if you would like to evaluate for all power-of-2 values of `--topk` up to `$NUM_CLUSTERS` (e.g., for `$NUM_CLUSTERS=16`, `--topk 1 2 4 8 16`), simply set `--topk all`:

```
topk=all
python $CBTM_PATH/downstream_tasks/ensemble.py $DATASET --expert-outputs-dir output \
 --mixture-folder $MIXTURE_FOLDER --topk $topk --seeds ${seed[@]} \
 --n-shot $k --method standard --num-clusters $NUM_CLUSTERS
```