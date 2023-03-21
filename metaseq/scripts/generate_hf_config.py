from transformers import OPTConfig
import sys


num_layers = 24
num_heads = 32
d_model = 2048

config = OPTConfig(hidden_size=d_model, num_attention_heads=num_heads, num_hidden_layers=num_layers, ffn_dim=4*d_model)
config.save_pretrained(sys.argv[1])  # <- this will create a `config.json` in your current folder
