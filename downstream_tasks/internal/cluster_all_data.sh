DATASETS=(ag_news amazon_polarity dbpedia_14 financial_phrasebank glue-sst2 tweet_eval-offensive)
NUM_CLUSTERS_ARR=(2 4 8 16 32 64 128)
seed=(13 100 21 42 87)
TRAIN_DATA=c4
COMPUTE="168b"

CBTM_PATH=/private/home/margaretli/cbtm_metaseq
FAIRSEQ_FILE_NAME=consolidated.pt

DATA_PATH=$CBTM_PATH/downstream_tasks/data
RAW_DATA_PATH=$DATA_PATH/raw
PROCESSED_DATA_PATH=$DATA_PATH/processed
k=8
split=test

for NUM_CLUSTERS in "${NUM_CLUSTERS_ARR[@]}"; do
    MODEL_FOLDER=/checkpoint/margaretli/cbtm/${TRAIN_DATA}/${COMPUTE}/${NUM_CLUSTERS}_clusters
    HF_DIR=$MODEL_FOLDER/hf
    HF_MODEL_DIR=${HF_DIR}/models

    for DATASET in "${DATASETS[@]}"; do
        MIXTURE_FOLDER=$HF_DIR/$DATASET/$split
        DATASET_DIR=$PROCESSED_DATA_PATH/$DATASET/$split
        CLUSTERER_DIR=/checkpoint/margaretli/cbtm/clusterers/${TRAIN_DATA}/$NUM_CLUSTERS
        COLUMN=input

        python $CBTM_PATH/downstream_tasks/estimate_cluster.py --dataset-dir $DATASET_DIR \
        --path-to-clusterer $CLUSTERER_DIR --mixture-folder $MIXTURE_FOLDER \
        --column $COLUMN --seed ${seed[@]} --n-shot $k
    done;
done;
