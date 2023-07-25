# DATASETS=$1
# NUM_CLUSTERS=$2

DATASETS=(ag_news amazon_polarity dbpedia_14 financial_phrasebank glue-sst2 tweet_eval-offensive)
NUM_CLUSTERS_ARR=(2 4 8 16 32 64 128)
seed=(13 100 21 42 87)
TRAIN_DATA=c4
COMPUTE="168b"

topk=all
split=test
k=8
CBTM_PATH=/private/home/margaretli/cbtm_metaseq
FAIRSEQ_FILE_NAME=consolidated.pt

for DATASET in "${DATASETS[@]}"; do
    for NUM_CLUSTERS in "${NUM_CLUSTERS_ARR[@]}"; do
        MODEL_FOLDER=/checkpoint/margaretli/cbtm/${TRAIN_DATA}/${COMPUTE}/${NUM_CLUSTERS}_clusters
        HF_DIR=$MODEL_FOLDER/hf
        MIXTURE_FOLDER=$HF_DIR/$DATASET/$split
        python $CBTM_PATH/downstream_tasks/ensemble.py $DATASET --expert-outputs-dir output \
        --mixture-folder $MIXTURE_FOLDER --topk $topk --seeds ${seed[@]} \
        --n-shot $k --method standard --num-clusters $NUM_CLUSTERS
    done;
done;

