INPUT_DIR=$1
OUTPUT_DIR=$2

python /private/home/suching/metaseq/metaseq/scripts/generate_hf_config.py $OUTPUT_DIR

python /private/home/suching/transformers/src/transformers/models/opt/convert_opt_original_pytorch_checkpoint_to_pytorch.py \
--fairseq_path $INPUT_DIR/consolidated.pt --pytorch_dump_folder_path $OUTPUT_DIR --hf_config $OUTPUT_DIR/config.json