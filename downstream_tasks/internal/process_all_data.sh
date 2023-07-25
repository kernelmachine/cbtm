# DATASETS=(ag_news amazon_polarity financial_phrasebank glue-sst2 tweet_eval-offensive)
DATASETS=(ag_news amazon_polarity dbpedia_14 glue-sst2 tweet_eval-offensive)
# DATASETS=(financial_phrasebank)
# DATASETS=(dbpedia_14)
seed=(13 100 21 42 87)

CBTM_PATH=/private/home/margaretli/cbtm_metaseq
FAIRSEQ_FILE_NAME=consolidated.pt

DATA_PATH=$CBTM_PATH/downstream_tasks/data
RAW_DATA_PATH=$DATA_PATH/raw
PROCESSED_DATA_PATH=$DATA_PATH/processed
k=8
split=test

cd $CBTM_PATH;
for DATASET in "${DATASETS[@]}"; do
    DATASET_DIR=$PROCESSED_DATA_PATH/$DATASET/$split
    
    python downstream_tasks/process_data.py --tasks $DATASET \
    --data-write-path $PROCESSED_DATA_PATH --data-read-path $RAW_DATA_PATH \
    --random-seed ${seed[@]} --n-shot $k  --use-hardcoded-prompt
done;
