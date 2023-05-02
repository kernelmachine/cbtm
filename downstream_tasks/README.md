# cbtm
### Downstream task evaluation
Note that Downstream task evaluation requires Huggingface checkpoints. 

#### Save the prediction file for each expert
The first step is to evaluate each expert on the dataset and save the prediction file. 

```
MODEL_PATH=/path/to/each_expert_huggingface_checkpoint
RESULT_CACHE_PATH=/path/to/output # e.g., ./output/numclusters_16/0 could be the output path for expert 0.
python score.py $DATASET --model $MODEL_PATH --output $RESULT_CACHE_PATH --n-shot $k --data_seed $seed
```
#### Estimate cluster assignment
Then you estimiate cluster assignment for each example by running the following command
```
OUTPUT=/path/to/output
DATA_DIR=./data_test
DATA=task_name
seed=0
CLUSTER=/path/to/cluster_centroid
COLUMN=input
python estimate_cluster.py --eval-file ${DATA_DIR}/${DATA}/8shot_${seed}.jsonl --path-to-clusterer $CLUSTER --output-file $OUTPUT/cluster.npy --column $COLUMN
```

#### Ensemble
After saving the prediction file for each expert model and the ensemble weights, 
```
DATA=task_name
RESULT_CACHE_DIR=/path/to/directory_of_RESULT_CACHE_PATH # e.g., ./output/numclusters_16
OUTPUT=/path/to/output
NUM_CLUSTER=number of cluster for the cbtm #  e.g., 16
topk=4 # number of experts used for the ensemble method
python ensemble.py $DATA --model $RESULT_CACHE_DIR --output $OUTPUT --topk $topk --method standard --num_cluster $NUM_CLUSTER
```
