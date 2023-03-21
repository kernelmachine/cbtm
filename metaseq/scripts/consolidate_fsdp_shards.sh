# path to serialized models
SERIALIZATION_DIR=$1
# model folder pattern (e.g., "*ngpu4")
PATTERN=$2

CONSOLIDATED_MODEL_PREFIXES=;
ORIGINAL_MODEL_PATHS=;
mapfile -t MODEL_FOLDERS < <(find ${SERIALIZATION_DIR} -type d -name $PATTERN -name "*cluster*" -printf "%f|%p\n" | sort -t "|" -k1,1 -t "|" -k2,2 | cut -d "|" -f 2)
echo "Number of models found: ${#MODEL_FOLDERS[@]}"

count=0
for folder in "${MODEL_FOLDERS[@]}"; do
    # check if there are any consolidated.pt files in the model folder 
    if ! test -f "${folder}/consolidated.pt"; then
        ((count++))
        CONSOLIDATED_MODEL_PREFIXES+="${folder}/consolidated ";
        ORIGINAL_MODEL_PATHS+="${folder}/checkpoint_last.pt ";
    fi;
done

echo "Consolidating ${count} models"

/gscratch/zlab/sg01/bin/parallel --link --ungroup --jobs 10 python -m metaseq.scripts.consolidate_fsdp_shards {1} --save-prefix {2} --new-arch-name transformer_lm_gpt ::: $ORIGINAL_MODEL_PATHS ::: $CONSOLIDATED_MODEL_PREFIXES