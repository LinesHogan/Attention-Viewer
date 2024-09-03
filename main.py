import argparse
import os
import math
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from attn_viewer.core import view_attention

# parse arguments
parser = argparse.ArgumentParser()
# model config
parser.add_argument('--model_path', required=True, help='the path of the model')
parser.add_argument('--model_id', required=True, help='the name you give to the model, used for the folder name to save the attention scores')
parser.add_argument('--model_adapter', default=None, help='the path of the adapter')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
# input config
parser.add_argument('--prompt', default=None, help='the prompt to visualize. If you are using chat model, you need to provide prompt in code rather than command line')
parser.add_argument('--ignore_first_token', action='store_true', help='whether to ignore the start token when plotting')
# saving and loading of attention scores
parser.add_argument('--save_attention_scores', action='store_true', help='whether to store the attention scores')
parser.add_argument('--save_attention_scores_path', default='./attn_scores')
parser.add_argument('--load_attention_scores_path', default=None, help='if specified, would just load the stored attention scores and plot')
# visualization
parser.add_argument('--plot_figs_per_head', action='store_true', help='whether to plot heatmap for each head')
parser.add_argument('--save_fig_path', default='./vis')
parser.add_argument('--num_figs_per_row', type=int, default=4)
parser.add_argument('--start_layer', type=int, default=0)
parser.add_argument('--end_layer', type=int, default=-1)
parser.add_argument('--token_focus', action='store_true', help='whether to plot attention for each token')
parser.add_argument('--numerical_norm', default='log', choices=['log', 'linear', 'power'], help='the normalization method for the heatmap')
args = parser.parse_args()

prompt =  [
    [
    {'role': 'user', 'content': "What is the ultimate answer to life, the universe, and everything?"},
    {"role": "assistant", "content": "It's 42."},
           ],
            ]

torch.manual_seed(args.seed)

def prompt_preprocess(prompt):
    prompt = tokenizer.apply_chat_template(prompt, return_tensors="pt")[0][1:]
    prompt = tokenizer.decode(prompt)
    return prompt

if __name__ == "__main__":

        model_dir = "/root/autodl-tmp/llama2"
        if args.model_path:
            model_dir = args.model_path
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, attn_implementation="eager", **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        prompt = prompt_preprocess(prompt)
        
        if args.model_adapter:
            model.load_adapter(args.model_adapter)
        
        if args.prompt:
            prompt = args.prompt
        
        # visualize attention
        view_attention(
            model=model,  # the model object
            model_id=args.model_id,
            tokenizer=tokenizer,
            prompt=prompt,
            save_attention_scores=args.save_attention_scores,
            save_attention_scores_path=args.save_attention_scores_path,
            load_attention_scores_path=args.load_attention_scores_path,
            plot_figs_per_head=args.plot_figs_per_head,
            save_fig_path=args.save_fig_path,
            ignore_first_token=args.ignore_first_token,
            num_figs_per_row=args.num_figs_per_row,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            per_token_focus=args.token_focus,
            numerical_norm=args.numerical_norm,
        )

        
