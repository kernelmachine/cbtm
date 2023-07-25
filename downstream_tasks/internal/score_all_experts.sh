DATASETS=(ag_news amazon_polarity dbpedia_14 glue-sst2 tweet_eval-offensive)
NUM_CLUSTERS_ARR=(1 2 4 8 16 32 64 128)
seed=(13 100 21 42 87)
TRAIN_DATA=c4
COMPUTE="168b"
k=8
split=test

CBTM_PATH=/private/home/margaretli/cbtm_metaseq
DATA_PATH=$CBTM_PATH/downstream_tasks/data
PROCESSED_DATA_PATH=$DATA_PATH/processed

for DATASET in "${DATASETS[@]}"; do
    for NUM_CLUSTERS in "${NUM_CLUSTERS_ARR[@]}"; do
        DATASET_DIR=$PROCESSED_DATA_PATH/$DATASET/$split
        HF_DIR=/checkpoint/margaretli/cbtm/${TRAIN_DATA}/${COMPUTE}/${NUM_CLUSTERS}_clusters/hf
        HF_MODEL_DIR=${HF_DIR}/models
        MIXTURE_FOLDER=$HF_DIR/$DATASET/$split
        python $CBTM_PATH/downstream_tasks/score_grid.py --dataset $DATASET --data-dir $DATASET_DIR \
        --models-parent-folder $HF_MODEL_DIR --model-file-name pytorch_model.bin --n-shot $k --seeds ${seed[@]} \
        --mixture-folder $MIXTURE_FOLDER --script $CBTM_PATH/downstream_tasks/score.py \
        -p score/${DATASET} -g 1 --checkpoints-dir $HF_MODEL_DIR --no-wandb --no-tensorboard \
        --use-jobarray --jobarray-name score_${DATASET}_${HF_MODEL_DIR}
    done;
done;        