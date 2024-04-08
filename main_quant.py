import argparse
import os
import math
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from attn_viewer.core import view_attention
from qllm_eval.quantization.quant_wrapper import quantize_model
from qllm_eval.utils import build_model_and_enc

# parse arguments
parser = argparse.ArgumentParser()
# model config
parser.add_argument('--model_path', required=True, help='the path of the model')
parser.add_argument('--model_id', required=True, help='the name you give to the model')
# input config
parser.add_argument('--prompt', default='Summer is warm. Winter is cold.\n')
# saving and loading of attention scores
parser.add_argument('--save_attention_scores', action='store_true', help='whether to store the attention scores')
parser.add_argument('--save_attention_scores_path', default='./attn_scores')
parser.add_argument('--load_attention_scores_path', default=None, help='if specified, would just load the stored attention scores and plot')
# visualization
parser.add_argument('--plot_figs_per_head', action='store_true', help='whether to plot heatmap for each head')
parser.add_argument('--save_fig_path', default='./vis')
parser.add_argument('--num_figs_per_row', type=int, default=4)
# advance: quantization
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument("--a_group_size", type=int, default=128)
parser.add_argument("--a_bit", type=int, default=16)
parser.add_argument("--kv_group_size", type=int, default=128)
parser.add_argument("--kv_bit", type=int, default=16)
args = parser.parse_args()

if __name__ == "__main__":

        # load model and tokenizer
        model, tokenizer = build_model_and_enc(args.model_path, False, args.kv_bit, args.kv_group_size)
        model = quantize_model(model, args)

        # visualize attention
        view_attention(
            model=model,  # the model object
            model_id=args.model_id,
            tokenizer=tokenizer,
            prompt=args.prompt,
            save_attention_scores=args.save_attention_scores,
            save_attention_scores_path=args.save_attention_scores_path,
            load_attention_scores_path=args.load_attention_scores_path,
            plot_figs_per_head=args.plot_figs_per_head,
            save_fig_path=args.save_fig_path,
            num_figs_per_row=args.num_figs_per_row
        )

        
