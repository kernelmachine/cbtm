CMD='
trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_JOB_ID
     scontrol requeue $SLURM_JOB_ID
   fi
}

# Install signal handler
trap '"'"'trap_handler USR1'"'"' USR1
trap '"'"'trap_handler TERM'"'"' TERM

'

RANDOM_PORT=$(shuf -i 2000-65000 -n 1)

for CLUSTER_NUMBER in $(seq 0 $((NUM_CLUSTERS-1))); do 
    CMD+='if [ "$SLURM_ARRAY_TASK_ID" =' 
    CMD+=" \"${CLUSTER_NUMBER}\" ]; then 
    srun --job-name ${JOB_NAME}/${CLUSTER_NUMBER} \
    --output ${SERIALIZATION_DIR}/${JOB_NAME}/${CLUSTER_NUMBER}/train.log \
    --error ${SERIALIZATION_DIR}/${JOB_NAME}/${CLUSTER_NUMBER}/train.stderr.%A_%a \
    --open-mode append --unbuffered --cpu-bind=none \
    python -u metaseq_cli.train ${DATA_DIR}/${DATASET} \
    --distributed-world-size $NUM_GPUS --distributed-port $RANDOM_PORT \
    --save-dir ${SERIALIZATION_DIR}/${JOB_NAME}/${CLUSTER_NUMBER} \
    --finetune-from-model ${SEED_MODEL} --no-epoch-checkpoints --no-best-checkpoints \
    --save-interval 200 --save-interval-updates 200 \
    --keep-interval-updates 1 --validate-interval-updates 500 --best-checkpoint-metric loss \
    --task streaming_finetune_language_modeling --num-clusters ${NUM_CLUSTERS} \
    --path-to-clusters-dir ${CLUSTERS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
    --train-cluster ${CLUSTER_NUMBER} \
    --vocab-filename $VOCAB_DIR/gpt2-vocab.json --merges-filename $VOCAB_DIR/gpt2-merges.txt \
    --valid-subset $VALID_DATASET_SLICE --train-subset $TRAIN_DATASET_SLICE \
    --tensor-parallel-init-model-on-gpu --model-parallel-size 2 \
    --criterion vocab_parallel_cross_entropy --distribute-checkpointed-activations \
    --arch transformer_lm_megatron --activation-fn relu --decoder-learned-pos \
    --share-decoder-input-output-embed --no-emb-dropout --checkpoint-activations \
    --use-sharded-state --gradient-predivide-factor 32.0 --no-scale-embedding \
    --full-megatron-init --megatron-init-sigma 0.006 \
    --tokens-per-sample 2048 --sample-break-mode none \
    --ddp-backend fully_sharded --save-async --quiet --max-valid-steps 100 \
    --decoder-layers 24 --decoder-embed-dim 2048 --decoder-ffn-embed-dim 8192 \
    --decoder-attention-heads 32 --max-update $TOTAL_UPDATES --total-num-update $TOTAL_UPDATES \
    --warmup-updates 0 --required-batch-size-multiple 1 --batch-size 8 --batch-size-valid 4 \
    --update-freq 1 --dropout 0.1 --attention-dropout 0.1 \
    --adam-betas \"(0.9, 0.95)\" --adam-eps 1e-06 --clip-norm 1.0 \
    --optimizer adam --lr-scheduler polynomial_decay --lr 0.0002 --end-learning-rate 0.0 \
    --fp16 --fp16-init-scale 128 --num-workers 0 --num-workers-valid 0 \
    --log-format json --log-interval 10 --patience -1 --seed 1234 \
    ;fi ; "

done


sbatch --job-name $JOB_NAME --gpus-per-node $NUM_GPUS_PER_NODE --nodes $NUM_NODES --ntasks-per-node $NUM_GPUS_PER_NODE --cpus-per-task 10 \
--output "${SERIALIZATION_DIR}/${JOB_NAME}/%a/slrm_stdout.%A_%a" --error "${SERIALIZATION_DIR}/${JOB_NAME}/%a/slrm_stderr.%A_%a" \
--open-mode append --signal B:USR1@180 --array 0-$(($NUM_CLUSTERS-1)) --partition $SLURM_PARTITION --account $SLURM_ACCOUNT --time 4320 --mem 0 \
--wrap "$CMD"
