import os
import math
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.colors as colors

def plot_heatmap(attention_scores, model_id, plot_figs_per_head, save_fig_path, tokens_list=None, ignore_first_token=False, num_figs_per_row=4, start_layer=0, per_token_focus=False, numerical_norm='log'):
    save_fig_path_model = os.path.join(save_fig_path, model_id)
    os.makedirs(save_fig_path_model, exist_ok=True)

    if ignore_first_token:
        mask_first_token = 1
        attention_scores = [attention_scores[i][:, :, mask_first_token: , mask_first_token: ] for i in range(len(attention_scores))]
        tokens_list = tokens_list[mask_first_token: ]
        print(tokens_list)

    num_layers = len(attention_scores)
    num_tokens = len(tokens_list)
    
    if per_token_focus:
        for token_index, token in enumerate(tokens_list):
            print(f'Plotting attention for token: {token}')
            
            # Create a matrix to hold attention scores for this token across all layers
            token_attention = torch.zeros(num_layers, num_tokens)
            
            for layer_idx in range(num_layers):
                avg_attention_scores = attention_scores[layer_idx][0].mean(dim=0)
                token_attention[layer_idx] = avg_attention_scores[token_index, :]
            
            # Plot the heatmap for this token
            plt.figure(figsize=(14, 8))
            vmin = token_attention[token_attention > 0].min().item()
            vmax = token_attention.max().item()
            if numerical_norm == 'log':
                norm = colors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
            elif numerical_norm == 'linear':
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
            elif numerical_norm == 'power':
                norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
            sns.heatmap(token_attention.numpy(), 
                        cmap='RdBu_r', 
                        norm=norm,
                        xticklabels=tokens_list,
                        yticklabels=range(num_layers))
            
            plt.title(f'Attention for token: {token}')
            plt.xlabel('Tokens')
            plt.ylabel('Layers')
            plt.tight_layout()
            plt.savefig(os.path.join(save_fig_path_model, f'token_{token_index}_attention.jpg'), dpi=300, bbox_inches='tight')
            plt.close()

        return

    num_heads = len(attention_scores)
    num_rows = math.ceil(num_heads / num_figs_per_row)
    print(f'plotting a figure for layers {start_layer} - {start_layer + num_heads}...')
    fig, axes = plt.subplots(num_rows, num_figs_per_row, figsize=(len(tokens_list) * 2, 0.5 * num_rows * len(tokens_list)))
    for layer_idx in tqdm(range(len(attention_scores))):
        row, col = layer_idx // num_figs_per_row, layer_idx % num_figs_per_row
        avg_attention_scores = attention_scores[layer_idx][0].mean(dim=0)
        avg_attention_scores = torch.where(torch.isnan(avg_attention_scores) | torch.isinf(avg_attention_scores), torch.tensor(0.1), avg_attention_scores)  # Replace NaN or Inf with 0.1
        vmin = avg_attention_scores[avg_attention_scores > 0].min().item()  # smallest positive value
        vmax = avg_attention_scores.max().item()
        if numerical_norm == 'log':
            norm = colors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
        elif numerical_norm == 'linear':
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        elif numerical_norm == 'power':
            norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        mask = torch.triu(torch.ones_like(avg_attention_scores, dtype=torch.bool), diagonal=1)
        sns.heatmap(avg_attention_scores.numpy(), mask=mask.numpy(), cmap='RdBu_r', square=True, xticklabels=tokens_list, yticklabels=tokens_list, ax=axes[row, col], norm=norm)
        axes[row, col].set_title(f'layer {layer_idx}')

    plt.suptitle(f'layers {start_layer} - {start_layer + num_heads} avg')
    plt.savefig(os.path.join(save_fig_path_model, f'{start_layer}-{start_layer + num_heads}_layers_avg.jpg'))
    plt.close()

    if not plot_figs_per_head:
        return

    for layer_idx in range(len(attention_scores)):
        print(f'plotting layer {layer_idx} ...')
        num_heads = attention_scores[layer_idx].shape[1]
        num_rows = math.ceil(num_heads / num_figs_per_row)
        fig, axes = plt.subplots(num_rows, num_figs_per_row, figsize=(len(tokens_list) * 2, 0.5 * num_rows * len(tokens_list)))
        for head_idx in tqdm(range(num_heads)):
            row, col = head_idx // num_figs_per_row, head_idx % num_figs_per_row
            head_attention_scores = attention_scores[layer_idx][0][head_idx]
            mask = torch.triu(torch.ones_like(head_attention_scores, dtype=torch.bool), diagonal=1)
            if numerical_norm == 'log':
                norm = colors.LogNorm(vmin=head_attention_scores.min(), vmax=head_attention_scores.max())
            elif numerical_norm == 'linear':
                norm = colors.Normalize(vmin=head_attention_scores.min(), vmax=head_attention_scores.max())
            elif numerical_norm == 'power':
                norm = colors.PowerNorm(gamma=0.5, vmin=head_attention_scores.min(), vmax=head_attention_scores.max())
            sns.heatmap(head_attention_scores.numpy(), mask=mask.numpy(), cmap='RdBu_r', square=True, xticklabels=tokens_list, yticklabels=tokens_list, ax=axes[row, col], norm=norm)
            axes[row, col].set_title(f'head {head_idx}')

        plt.suptitle(f'layer_{layer_idx}')
        plt.savefig(os.path.join(save_fig_path_model, f'layer_{layer_idx}.jpg'))
        plt.close()
