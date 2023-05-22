import argparse
import os

from transformers.models.opt.convert_opt_original_pytorch_checkpoint_to_pytorch import convert_opt_checkpoint
from transformers import OPTConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fairseq-model",
        type=str,
        help=(
            "path to fairseq checkpoint in correct format. You can find all checkpoints in the correct format here:"
            " https://huggingface.co/models?other=opt_metasq"
        ),
        default=None
    )
    parser.add_argument(
        "--fairseq-path",
        type=str,
        help=(
            "path to fairseq checkpoint in correct format. You can find all checkpoints in the correct format here:"
            " https://huggingface.co/models?other=opt_metasq"
        ),
    )
    parser.add_argument('--fairseq-file-name', type=str, default='consolidated.pt')
    parser.add_argument("--pytorch-dump-folder-path", type=str, required=True, help="Path to the output PyTorch model.")
    args = parser.parse_args()

    if args.fairseq_model:
        fairseq_paths = [os.path.dirname(args.fairseq_model)]
    else:
        fairseq_paths = []
        for name, _folders, files in os.walk(args.fairseq_path):
            # regex = re.compile(re_string) if re_string else None
            if args.fairseq_file_name not in files:
                continue
            fairseq_paths.append(name)
    print(fairseq_paths)

    num_layers = 24
    num_heads = 32
    d_model = 2048

    for path in fairseq_paths:
        pytorch_dump_folder_path = os.path.join(
            args.pytorch_dump_folder_path, 
            os.path.relpath(path, args.fairseq_path)
        )
        os.makedirs(pytorch_dump_folder_path, exist_ok=True)
        config = OPTConfig(hidden_size=d_model, num_attention_heads=num_heads, num_hidden_layers=num_layers, ffn_dim=4*d_model)
        config.save_pretrained(pytorch_dump_folder_path)  # <- this will create a `config.json` file

        convert_opt_checkpoint(
            os.path.join(path, args.fairseq_file_name),
            pytorch_dump_folder_path, 
            config=os.path.join(pytorch_dump_folder_path, 'config.json')
        )
