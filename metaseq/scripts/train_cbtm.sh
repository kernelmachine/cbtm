NUM_NODES=$1
NUM_GPUS=$2
MODEL_SIZE=$3
PATH_TO_CLUSTERS_DIR=$4
NUM_CLUSTERS=$5
TRAIN_CLUSTER=$6
SLURM=$7
LR=$8
UPDATE_FREQ=${9}
BATCH_SIZE=${10}
RANDOM_CLUSTERS=${11}
CLUSTER_TAG=${12}
DATA=${13}
INSTRUCTION=${14}
SUBSET=${15}
TRAIN_SUBSET=${16}
VALID_SUBSET=${17}
PATH_TO_DATA=${18}
PATH_TO_CBTM=${19}
PARTITION=${20}
ACCOUNT=${21}
CONSTRAINT=${22}
MAX_STEPS=${23}
ADD_CLUSTER_TOKEN=${24}

if [ $TRAIN_CLUSTER == "None" ]; then
    TRAIN_CLUSTER="";
    NCLUSTER=$(( $NUM_CLUSTERS - 1 ));
    for i in $(seq 0 $NCLUSTER ); do TRAIN_CLUSTER+="${i},"; done;
    TRAIN_CLUSTER="${TRAIN_CLUSTER%?}";
fi

if [ $NUM_CLUSTERS == 1 ]; then
    MOD_PHRASE="";
else
    MOD_PHRASE="--path-to-clusters-dir $PATH_TO_CLUSTERS_DIR --train-cluster $TRAIN_CLUSTER";
fi;

CHECKPOINT_DIR="/$SERIALIZATION_DIR/${NUM_CLUSTERS}_clusters";

PROJECT="cbtm.$DATA"


if [ $CLUSTER_TAG != "None" ]; then
    CLUSTER_TAG_PHRASE="--cluster-tag $CLUSTER_TAG";
else CLUSTER_TAG_PHRASE="";
fi;

if [ $SLURM == "slurm" ]; then
    LOCAL_PHRASE="";
else
    LOCAL_PHRASE="--dry-run --local";
fi;

if [ $RANDOM_CLUSTERS == "true" ]; then
    RANDOM_CLUSTERS_PHRASE="--random-clusters --num-clusters $NUM_CLUSTERS";
else
    RANDOM_CLUSTERS_PHRASE="--num-clusters $NUM_CLUSTERS";
fi;

if [ $ADD_CLUSTER_TOKEN == "true" ]; then
    CLUSTER_TOKEN_PHRASE="--add-cluster-token";
else
    CLUSTER_TOKEN_PHRASE="";
fi;


# JOBARRAY_NAME="${NUM_CLUSTERS}_clusters";
# JOBARRAY_NAME="${DATA}/${MODEL_SIZE}/numclusters${NUM_CLUSTERS}/uf${UPDATE_FREQ}.ngpu{$NUM_NODES*$NUM_GPUS}";
JOBARRAY_NAME="${NUM_CLUSTERS}_cluster"
if [ $SLURM == "slurm" ]; then
    JOBARRAY_PHRASE="--use-jobarray --jobarray-name $JOBARRAY_NAME ";
else
    JOBARRAY_PHRASE="";
fi;

if [ $INSTRUCTION == "true" ]; then
    LABEL_LOSS_PHRASE="--label-loss";
else
    LABEL_LOSS_PHRASE="";
fi;

SAVE_INTERVAL_UPDATES=200;
if [ $UPDATE_FREQ != "1" ]; then
    SAVE_INTERVAL_UPDATES=`expr $SAVE_INTERVAL_UPDATES / $UPDATE_FREQ `;
fi;


python -m metaseq.fb_sweep.ft_stream \
    -n $NUM_NODES \
    -g $NUM_GPUS \
    -p $PROJECT \
    --checkpoints-dir $CHECKPOINT_DIR \
    $MOD_PHRASE \
    --model-size $MODEL_SIZE \
    --data-type ft_data \
    --data-dir $PATH_TO_DATA \
    --valid-subset $VALID_SUBSET \
    --train-subset $TRAIN_SUBSET \
    --lr $LR \
    --warmup-update 0 \
    --max-update $MAX_STEPS \
    --no-tensorboard \
    --no-wandb \
    --interval 500 \
    --save-interval-updates $SAVE_INTERVAL_UPDATES \
    --keep-interval-updates 2 \
    --bs $BATCH_SIZE \
    --uf $UPDATE_FREQ \
    --patience 10000 \
    --sbm none \
    $LABEL_LOSS_PHRASE \
    --partition $PARTITION \
    --account $ACCOUNT \
    --exclusive \
    --resume-failed \
    $RANDOM_CLUSTERS_PHRASE \
    $CLUSTER_TAG_PHRASE \
    $CLUSTER_TOKEN_PHRASE \
    --fair \
    --max-valid-steps 100 \
    $LOCAL_PHRASE \
    $JOBARRAY_PHRASE \
    --script ${PATH_TO_CBTM}/metaseq_cli/train.py \
    --constraint $CONSTRAINT \
    --subset $SUBSET